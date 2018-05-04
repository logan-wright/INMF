# -*- coding: utf-8 -*-
'''
load_HICO.py - Reads in HICO HDF5 files and outputs the relevant data

Version 1.2
Created on: May, 19, 2016
Last Modified: June, 14, 2017
Author: Logan Wright, logan.wright@colorado.edu

Description:
 Version 1.2 Notes:
 Added ability to load in positional data [lat, lon, sol_azi, sol_zen,
 hico_azi, hico_zen]

 -Reads Hyperspectral Imager for Coastal Ocean (HICO) HDF5 files and outputs
  the scene datacube, wavelengths, fwhm, and radiance units. The function
  includes the ability to interpolate over saturated or missing values.
 -The function also produces an RGB image of the scene.
 -Also included in the output are the data quality flags. Flags is a structure
  of the format flags.FLAGNAME. A value of 1 indicates that the flag is set.
 -All outputs are returned with a single object have the properties: data_cube,
  resp_func, nav, flags, rgb

 Inputs:
       filepath = A single string including path to the file and the filename
       **kwds
       fill_saturated = A boolean flag to indicate whether or not the function
           should attempt to interpolate to fill saturated values that are
           missing from the datacube. Flag is True by default.
 Outputs:
       datacube = Hyperspectral datacube of the scene
           - format: numpy array, dtype = float64
           - dimensions: [along_track, across_track, wvl]
           - datacube is scaled to match the reported units
           - datacube edge is obscured by slit, pixels 1-10 are affected
       resp_func = dictionary including: wvl, fwhm and units
           wvl = wavelength vector corresponding to the datacube
               - format: numpy array, dtype = float32
               - dimensions: [wvl,1]
           fwhm = fwhm vector again corresponding to the datacube
               - format: numpy array, dtype = float32
               - dimensions: [wvl,1]
           units = string reporting the units of the scaled data cube
       rgb = a "true-color" image derived from the data cube using a 2%
               linear stretch, (see lines 76-131).
           - format: numpy array, dtype = float64
           - dimensions: [along_track,across_track,3]
               Red =   channel 51, 638.928 nm
               Blue =  channel 36, 553.008 nm
               Green = channel 20, 461.360 nm
       flags = HICO processing flags in a mask format
           - format: numpy masked array, dtype = bool
           - dimensions: [along_track,across_track]
           - logical: True (1) or False (0)
           flags.LAND = land (or possibly glint or clouds) (?NIR > 0.02)
           flags.NAVWARN = latitude or longitude out of bounds
           flags.NAVFAIL = navigation is rough (currently always set to 1)
           flags.HISATZEN = satellite view angle > 60 deg
           flags.HISOLZEN = solar zenith angle at estimated position > 75 deg
           flags.SATURATE = pixel has ? 1 saturated bands
           flags.CALFAIL = pixel has ? bands from a dropped packet
           flags.CLOUD = rough cloud mask (R_NIR > 0.05 and R_RED > 0.5) or
                         (0.8 < R_NIR/R_RED < 1.1)
       nav = HICO navigation/position/observational geometry data
           - In format [along_track, across_track, 6]
           - 6 dimensions: [lat, lon, sol_azi, sol_zen, hico_azi, hico_zen]
'''

# Import Required Modules
import h5py
import bitget
import numpy as N
import scipy.interpolate as sci_interp
from skimage import exposure
from datetime import datetime

# Define the "datacube" object
class datacube(object):
    def __init__(self,filepath,data_cube,resp_func,nav,flags,rgb):
        self.name = filepath.split('/')[-1]
        self.data_cube = data_cube
        self.resp_func = resp_func
        self.nav = nav
        self.flags = flags
        self.rgb = rgb

def load_hico(filepath,fill_saturated = True):
    load_flag = True
    # If asked to interpolate over saturated data loader will first check is
    # there is an existing npz containing a filled datacube
    if fill_saturated:
        try:
            reloaded = N.load(filepath[:-8] + '_unsat.npz')
            scaled_data = reloaded['scaled_data']
            load_flag = False
        except:
            print('No Saturation Correction HICO File Exists for this Scene')

    # Open the pointed to HICO hdf5 file
    f = h5py.File(filepath,'r')

    # Load Response Function
    units = str(f['/products/Lt'].attrs['units'])
    wvl = f['/products/Lt'].attrs['wavelengths']
    fwhm = f['/products/Lt'].attrs['fwhm']
    resp = dict([('units',units),('wvl',wvl),('fwhm',fwhm)])

    # Load Navigation Data
    lat = f['/navigation/latitudes']
    lon = f['/navigation/longitudes']
    sol_azi = f['/navigation/solar_azimuth']
    sol_zen = f['/navigation/solar_zenith']
    hico_azi =f['/navigation/sensor_azimuth']
    hico_zen = f['/navigation/sensor_zenith']

    # Pull Collect Time from Filename
    time = datetime.strptime(filepath[-21:-8],'%Y%j%H%M%S')

    # Assemble nav data into a dictionary
    nav = dict([('lat',lat[:]),('lon',lon[:]),('sol_azi',sol_azi[:]),
       ('sol_zen',sol_zen[:]),('hico_azi',hico_azi[:]),
        ('hico_zen',hico_zen[:]),('time',time)])

    # Load Data Quality Flags
    flag = f['/quality/flags']
    flags = flag[:]
    flags = dict([('Land',N.ma.MaskedArray(bitget.bitget(flags,0),
                    mask = bitget.bitget(flags,0) == 0, dtype = 'bool')),
                  ('NavWarn',N.ma.MaskedArray(bitget.bitget(flags,1),
                    mask = bitget.bitget(flags,1) == 0, dtype = 'bool')),
                  ('NavFail',N.ma.MaskedArray(bitget.bitget(flags,2),
                    mask = bitget.bitget(flags,2) == 0, dtype = 'bool')),
                  ('HiSatZen',N.ma.MaskedArray(bitget.bitget(flags,3),
                    mask = bitget.bitget(flags,3) == 0, dtype = 'bool')),
                  ('HiSolZen',N.ma.MaskedArray(bitget.bitget(flags,4),
                    mask = bitget.bitget(flags,4) == 0, dtype = 'bool')),
                  ('Saturate',N.ma.MaskedArray(bitget.bitget(flags,5),
                    mask = bitget.bitget(flags,5) == 0, dtype = 'bool')),
                  ('CalFail',N.ma.MaskedArray(bitget.bitget(flags,6),
                    mask = bitget.bitget(flags,6) == 0, dtype = 'bool')),
                  ('Cloud',N.ma.MaskedArray(bitget.bitget(flags,7),
                    mask = bitget.bitget(flags,7) == 0, dtype = 'bool'))])
    # load_flag determines whether  to create data_cube from the hdf5 file, it
    # is only set to false if a previous generated .npz file is detected
    if load_flag:
        # Load Core Data Product - Datacube and associated values
        data = f['/products/Lt']
        scale = f['/products/Lt'].attrs['slope']

        # Apply Scaling Factor
        scaled_data = data[:] * scale

        # If fill_saturated flag is True, fill in the saturated pixel with a
        # cubic interpolation
        if fill_saturated and N.sum(flags['Saturate']) > 0:

            # Fill Saturated Values in Array
            mask = N.fliplr((flags['Saturate'] == 1))
            I,J,L = scaled_data.shape
            filled_data = N.zeros((I,J,L))
            for l in range(L):
                single_wvl = scaled_data[:,:,l]
                single_wvl[mask] = sci_interp.griddata((nav['lon'][~mask],
                          nav['lat'][~mask]), single_wvl[~mask],
                          (nav['lon'][mask], nav['lat'][mask]), method='cubic')
                filled_data[:,:,l] = single_wvl
            scaled_data = filled_data

            # Save Data so we can avoid having to interpolate every time
            N.savez_compressed(filepath[:-8] + '_unsat.npz',
                               scaled_data = scaled_data)
        else:
            print('No Saturation Correction Required')

    # Close file
    f.close()

    # Produce and RGB "Quasi-truecolor" preview image
    # Apply 2% Linear Stretch to "True Color"
    R = scaled_data[:,:,50]/N.amax(scaled_data[:,:,50])
    p2, p98 = N.percentile(R, (2, 98))
    Rscl = exposure.rescale_intensity(R, in_range=(p2, p98))

    G = scaled_data[:,:,35]/N.amax(scaled_data[:,:,35])
    p2, p98 = N.percentile(G,(2, 98))
    Gscl = exposure.rescale_intensity(G, in_range=(p2, p98))

    B = scaled_data[:,:,19]/N.amax(scaled_data[:,:,19])
    p2, p98 = N.percentile(B, (2, 98))
    Bscl = exposure.rescale_intensity(B, in_range=(p2, p98))

    rgb = N.dstack((Rscl,Gscl,Bscl))

    # Return
    ret_val = datacube(scaled_data,resp,nav,flags,rgb)
    return ret_val


def disp_cube(data):
    '''
    load_HICO.disp_cube - Plots a hypercube from a loaded datacube object

    Version 1.0
    Created on: May, 19, 2016
    Last Modified: June, 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu

    Description:
     -Takes a "datacube" object, which is created and returned by
      load_hico.load_hico and plots as a 3-Dimensional hypercube.

    WARNING: This function takes a long time to create the hypercube image

     Inputs:
           data = a datacube object returned by the load_hico function
     Outputs:
           No Returned Value
           - A hypercube figure is produced and saved
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    imdata = N.fliplr(data.rgb)
    dims = data.data_cube.shape
    res = 1

    Xi, Zi = N.meshgrid(N.arange(dims[1]),N.arange(dims[0]))
    Yi = N.full((dims[0],dims[1]),0,dtype = 'int')

    X, Y = N.meshgrid(N.arange(dims[1]),N.arange(dims[2]) * 4)
    Z = N.full((dims[2],dims[1]),dims[0],dtype = 'int')
    C = N.fliplr(N.transpose(data.data_cube[-1,:,:]))
    Cnorm = mpl.colors.Normalize(vmin = 1.2 * C.min(), vmax = .8 * C.max())

    Z2, Y2 = N.meshgrid(N.arange(dims[0]),N.arange(dims[2]) * 4)
    X2 = N.full((dims[2],dims[0]),dims[1],dtype = 'int')
    C2 = N.transpose(data.data_cube[:,0,:])
    C2norm = mpl.colors.Normalize(vmin = 1.2 * C2.min(), vmax = .8 * C2.max())

    hypercube = plt.figure(figsize = (2,5))
    ax = hypercube.gca(projection = '3d',aspect = 'equal')
    ax.set_xlabel('X')
#    ax.set_xlim(0, 250)
    ax.set_ylabel('Y')
#    ax.set_ylim(0, 1300)
    ax.set_zlabel('Z')
#    ax.set_zlim(0, 100)

    ax.plot_surface(Xi, Yi, Zi, rstride = res, cstride = res, facecolors = imdata, linewidth=0, shade = False, antialiased = False)
    ax.plot_surface(X, Y, Z, rstride = res, cstride = res, facecolors = mpl.cm.inferno(Cnorm(C)), linewidth=0, shade = False, antialiased=False)
    ax.plot_surface(X2, Y2, Z2, rstride = res, cstride = res, facecolors = mpl.cm.inferno(C2norm(C2)), linewidth=0, shade = False, antialiased=False)

    ax.azim = 285
    ax.elev = 15
    scaling = N.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']);
    ax.auto_scale_xyz(*[[N.min(scaling), N.max(scaling)]]*3)
#    ax.auto_scale_xyz([0,250],[0,100],[0,1200])
    plt.axis('off')
    plt.savefig('TestHypercube.png',format = 'png',dpi = 1800,bbox_inches = 'tight',transparent=True)

def truecolor(filepath):
    '''
    load_HICO.truecolor - Produces an rgb "truecolor" images

    Version 1.0
    Created on: May, 19, 2016
    Last Modified: June, 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu

    Description:
     -Will produce a true color rgb image of all HICO L1B scenes in the given
      filepath.

     Inputs:
           filepath = string, path to a directory containing 1 or more HICO
                      level 1 B (*.L1B_ISS) scenes
     Outputs:
           No Returned Value
           - A rgb images are produced and saved
    '''
    import glob
    import matplotlib.pyplot as plt

    hico_files = glob.glob(filepath + '*.L1B_ISS')
    for file in hico_files:
        rgb_file = glob.glob(filepath + file[-22:-8] + '_rgb.png')
        if not rgb_file:
            hypercube = load_hico(file, fill_saturated = True)
            I,J,L = hypercube.rgb.shape
            # Show Quasi-truecolor Preview
            plt.figure(figsize = ((J/I*10),10))
#            mask = N.dstack((N.ones((2000,512)),N.zeros((2000,512)),N.ones((2000,512))))
#            maskalpha = N.zeros((2000,512))
#            maskalpha[hypercube.flags['Saturate'] == 1] = 1
#            mask = N.dstack((mask,maskalpha))
            plt.imshow(N.flipud(N.fliplr(hypercube.rgb)))
            #    plt.imshow(N.flipud(mask))
            plt.xticks([],[])
            plt.yticks([],[])
            plt.savefig(filepath + file[-22:-8] + '_rgb.png', dpi=300, bbox_inches = 'tight', Transparent = True)
