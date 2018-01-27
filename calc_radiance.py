import numpy as N
import math

from super_resample import super_resample

def calc_radiance(paths,SZA,SunElliptic,resp_func, plot_flag = False):
    '''


    '''

    # paths[0] = modpath
    # paths[1] = irradpath
    # paths[2] = HICO_REFL_Path
    mod_dat = dict()

    # Load MODTRAN Irradiance
    irrad = N.genfromtxt(paths[1],skip_header = 3)
    # Convert to wavelength increments
    irrad[:,0] = 1e7 / irrad[:,0]
    irrad[:,1] = irrad[:,1] * (1 / (irrad[:,0] ** 2)) * 1e11
    irrad = irrad[0:-1,:]
    # I0 = I0(I0(:,1)<3000,:)
    # Convolve to NEON Wavelength
    mu = math.cos((SZA/180)*math.pi)
    mod_dat['I0'] = SunElliptic * super_resample(irrad[:,1], irrad[:,0], wvl, resp_func['fwhm']) * 1000
    # *1000 converts form W/m^2/nm to W/m^2/um
    mod_path = paths[0]
    # Load Transmittances
    acd_dat = load_acd(mod_path + '.acd');
    mod_wvl = 1e7/(acd_dat['freq']); # Transfer Wavenumber to Wavelength

    mod_dat['sph_alb'] = super_resample(acd_dat['sph'],mod_wvl,resp_func['wvl'],resp_func['fwhm']);
    mod_dat['ts'] = super_resample(acd_dat['ts'],mod_wvl,resp_func['wvl'],resp_func['fwhm']);
    mod_dat['Ts'] = super_resample(acd_dat['Ts'],mod_wvl,resp_func['wvl'],resp_func['fwhm']);
    mod_dat['t'] = super_resample(acd_dat['t'],mod_wvl,resp_func['wvl'],resp_func['fwhm']);
    mod_dat['T'] = super_resample(acd_dat['T'],mod_wvl,resp_func['wvl'],resp_func['fwhm']);

    # Load MODTRAN Radiance
    data = N.loadtxt(mod_path + '.psc')
    mod_dat['rad0'] = super_resample(data[:,1],data[:,0],resp_func['wvl'],resp_func['fwhm']) * 10

    # Load Reference Spectra
    #   ASTER Library
    HICO_refl = sio.loadmat(paths[2])
    #   Cloud, Altocumulus Cloud
    cloud_refl = N.array([N.arange(360,960,10),
                        [0.659, 0.665, 0.666, 0.663, 0.658, 0.657, 0.659,
                         0.663, 0.664, 0.668, 0.673, 0.673, 0.678, 0.684,
                         0.684, 0.682, 0.680, 0.678, 0.674, 0.674, 0.676,
                         0.677, 0.684, 0.686, 0.688, 0.689, 0.689, 0.688,
                         0.687, 0.687, 0.687, 0.687, 0.685, 0.653, 0.678,
                         0.676, 0.653, 0.681, 0.693, 0.696, 0.587, 0.689,
                         0.680, 0.666, 0.647, 0.622, 0.620, 0.634, 0.620,
                         0.607, 0.603, 0.570, 0.546, 0.479, 0.477, 0.458,
                         0.369, 0.391, 0.408, 0.412]])

    mod_dat['sigma_bar'] = ((mod_dat['Ts'] + mod_dat['ts']) * (mod_dat['T'] +mod_dat['t']) * mod_dat['sph_alb']) / ((mod_dat['Ts'] + mod_dat['ts']) * (mod_dat['T'] +mod_dat['t']))



    def rad_from_refl(refl,mu,I0,mod_dat):
        R = ((((mod_dat['mu'] * mod_dat['I0'] * (mod_dat['Ts'] + mod_dat['ts']) * mod_dat['T']) / math.pi) * refl) + (((mod_dat['mu'] * mod_dat['I0'] * (mod_dat['Ts'] + mod_dat['ts']) * mod_dat['t']) / math.pi) * refl)) / (1 - mod_dat['sigma_bar'] * refl)
        return R
    # Calculate Radiances
    Refl_asphalt = super_resample(N.squeeze(HICO_refl['asphalt_refl']) / 100,hypercube.resp_func['wvl'],resp_func['wvl'],resp_func['fwhm'])
    R_asphalt = rad_from_refl(Refl_asphalt,mod_dat)

    Refl_conc = super_resample(N.squeeze(HICO_refl['conc_refl']) / 100,hypercube.resp_func['wvl'],resp_func['wvl'],resp_func['fwhm'])
    R_conc = rad_from_refl(Refl_asphalt,mod_dat)

    Refl_snow = super_resample(N.squeeze(HICO_refl['snow_refl']) / 100,hypercube.resp_func['wvl'],resp_func['wvl'],resp_func['fwhm'])
    R_snow = R_asphalt = rad_from_refl(Refl_asphalt,mod_dat)

    Refl_soil = super_resample(N.squeeze(HICO_refl['soil_refl']) / 100,hypercube.resp_func['wvl'],resp_func['wvl'],resp_func['fwhm'])
    R_soil = R_asphalt = rad_from_refl(Refl_asphalt,mod_dat)

    Refl_veg = super_resample(N.squeeze(HICO_refl['veg_refl']) / 100,hypercube.resp_func['wvl'],resp_func['wvl'],resp_func['fwhm'])
    R_veg = R_asphalt = rad_from_refl(Refl_asphalt,mod_dat)

    Refl_water = super_resample(N.squeeze(HICO_refl['water_refl']) / 100,hypercube.resp_func['wvl'],resp_func['wvl'],resp_func['fwhm'])
    R_water = R_asphalt = rad_from_refl(Refl_asphalt,mod_dat)

    rho = super_resample(cloud_refl[1,:],cloud_refl[0,:],resp_func['wvl'],resp_func['fwhm'])
    R_cloud = rad_from_refl(Refl_cloud,mod_dat)

    R_atmo = mod_dat['rad0']

    if inputs['plot_flag'] is True:
        plt.figure()
        h_asphalt = plt.plot(wvl,R_asphalt,color = 'k', label = 'Asphalt')
        h_conc = plt.plot(wvl,R_conc,color = 'k', linestyle = '--', label = 'Concrete')
        h_snow = plt.plot(wvl,R_snow,color = 'm', label = 'Snow')
        h_soil = plt.plot(wvl,R_soil,color = 'r', label = 'Soil')
        h_veg = plt.plot(wvl,R_veg,color = 'g', label = 'Vegetation')
        h_water = plt.plot(wvl,R_water,color = 'b', label = 'Water')
        h_cloud = plt.plot(wvl,R_cloud,color = 'm', linestyle = '--', label = 'Cloud')
        h_atmo = plt.plot(wvl,R_atmo,color = 'b', linestyle = '--', label = 'Atmospheric Scattering')
        plt.legend()

    refl_endmembers = dict([('Asphalt',R_asphalt), ('Concrete',R_conc),
                          ('Snow',R_snow), ('Soil',R_soil),
                          ('Vegetation',R_veg), ('Water',R_water),
                          ('Cloud',R_cloud), ('Atmosphere',R_atmo)])
    sio.savemat(file + '_HICO_REFL.mat', refl_endmembers)
