#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inmf_master.py - master script needed to run the INMF code

Version 2.1
Created on: Oct, 13, 2016
Last Modified: June, 16, 2017
Author: Logan Wright, logan.wright@colorado.edu

Description:
 Version 2.1 Notes:
     -Added the multiprocessing ability if the input file contains multiple INMF
      processing requests.
     -Added built-in handling to initialized a perturbed ensemble
 
 Version 2.0 Notes:
     Seperated master script into subfunctions

 - Reads in a hyperspectral data scene,
 - Generates intial guess radiance spectra using MODTRAN inputs
 - Calls the INMF routine
 - Saves and plots results

 Inputs:
       inputfile = name of the inputfile to be used, as a string
        EXAMPLE: python3 nmf_master 'SampleINMF.in'
 Outputs:
       INMF results file
       Automatic plots
"""
# Import Modules
import os
import sys
import nmf
import math
import numpy as N
import scipy.io as sio
import matplotlib as mpl
mpl.use('Agg')
import multiprocessing as mp
import matplotlib.pyplot as plt

# Import individual functions
from datetime import datetime
from load_hico import load_hico    
    
def load_input_file(filepath):
    """
    inmf_master.load_input_file - parses INMF input file

    Version 1.0
    Created on: Oct, 13, 2016
    Last Modified: June, 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu

     - Reads in an INMF input file like "SampleINMF.in"
     - Extracts the filename, and parameters need to initialize the INMF master
         script
     - Returns these parameters as a dictionary to the INMF master script

    Inputs:
        inputfile = name of the inputfile to be used, as a string
        EXAMPLE: load_input_file('SampleINMF.in')
    Outputs:
        inputs = dictionary of input parameters
    """
    f = open(filepath, 'r')    # Open File and read each line
    contents = f.readlines()
    inputs = dict()
    # Loop through each line to parse contents
    for line in range(len(contents)):
        temp = contents[line]
        if temp[0] is not '#':    # Ignore Commmented Lines
            strs = temp.split('=')    # Divide based on position of "="
            key = strs[0].strip(' \n\t')
            val =  strs[1].split(',')    # Split comma separated values
            if len(val) > 1:    # Processes if there is more than one value, 
                for n in range(len(val)):
                    val[n] = str(val[n].strip(' \n\t'))
            else:
                val = val[0].strip(' \n\t')
            inputs[key] = val    # Save parsed values in a dictionary
    f.close()
    
    # Check for required inputs and format
    # Converts wvl_ind into integers, if no value is given uses the full range
    try:
        inputs['wvl_ind'] = N.array(inputs['wvl_ind'], dtype = int)
    except ValueError:
        inputs['wvl_ind'] = N.array([0,128], dtype = int)
        
    # Converts SZA into integer, if no value is given, requests an input from the user
    try: 
        inputs['SZA'] = float(inputs['SZA'])
    except NameError:
        inputs['SZA'] = input('No Solar Zenith Angle Provided! Enter the SZA now:')
        
    # Converts the Sun Ellipticity correction into a float, if no value is given assumes 1
    try: 
        inputs['SunElliptic'] = float(inputs['SunElliptic'])
    except NameError:
        inputs['SunElliptic'] = 1
    
    # Converts the allowed maximum number of iterations an integer, if none is present, sets the value to 500
    try:
        inputs['max_i'] = int(inputs['max_i'])
    except NameError:
        inputs['max_i'] = 500
    # Sets the 'perturb" keyword to None, if not set
    try:
        inputs['perturb']
    except KeyError:
        inputs['perturb'] = None
    # Sets other required inputs if they are not given in the inout file
    if 'name' not in inputs.keys():
        print('ERROR: No file name given')
        inputs['name'] = input('Input filename now:')
    if 'plot_flag' not in inputs.keys():
        inputs['plot_flag'] = True
    if 'rad_flag' not in inputs.keys():
        inputs['rad_flag'] = True
   
    return inputs

def load_acd(filepath):
    """
    inmf_master.load_acd() - load data contained in a MODTRAN .acd output file
    
    Version 1.0
    Created on: Oct, 13, 2016
    Last Modified: June, 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu

     - Reads in the atmospheric correction data contained in a MODTRAN acd file
     - Extracts the filename, and parameters need to initialize the INMF master
         script
     - Returns these parameters as a dictionary to the INMF master script

    Inputs:
        filepath = path to, and name of the acd file to be used.
        Example: '/home/Modtran/Nameofacdfile.acd'
    Outputs:
        ret_dict = contents of the acd file as a dictionary
    """
    rawdata = N.loadtxt(filepath,skiprows = 5)
    ret_dict = dict([('freq',rawdata[:,0]),
                     ('los',rawdata[:,1]),
                     ('kint',rawdata[:,2]),
                     ('kweight',rawdata[:,3]),
                     ('ts',rawdata[:,4]),
                     ('Ts', rawdata[:,5]),
                     ('t', rawdata[:,6]),
                     ('T', rawdata[:,7]),
                     ('sph', rawdata[:,8])])
    
    return ret_dict
    
def gauss_conv(in_spec,in_wvl,center,fwhm):
    """
    inmf_master.gauss_conv() - load data contained in a MODTRAN .acd output file
    
    Version 1.0
    Created on: Oct, 13, 2016
    Last Modified: June, 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu

     - Gaussian reference function, calculated based on the inputs of center
       and fwhm
     - Uses the input spectrum (in_spec) and wavelengths (in_wvl) with the
       Gaussian reference to calculated the resample spectrum value at the new
       center wavelength
     - Returns this new value as 'out'
     - Wavelength units are irrelevant as long as all wavelength and fwhm
       inputs are in the SAME units

    Inputs:
        in_spec = input spectrum to be resampled
        in_wvl = center wavelengths of the input spectrum
        center = center wavelength of the new response function
        fwhm = Full-Width-Half-Max of the new response function at this center
    Outputs:
        out = value of the new spectrum at this center wavelength
    """
    gaus = N.zeros(len(in_spec));
    gaus = N.exp((-4 * N.log(2) * ((in_wvl-center) / fwhm) ** 2))
    
    totinwvl = len(in_wvl)
    diff = N.zeros(totinwvl)
    diff[1:totinwvl-1] = (in_wvl[2:totinwvl]-in_wvl[0:totinwvl-2])/2
    diff[0] = (in_wvl[1]-in_wvl[0])
    diff[totinwvl-1] = (in_wvl[totinwvl-1]-in_wvl[totinwvl-2])
    gaus = gaus * diff

    out = sum(in_spec * gaus) / sum(gaus)
    return out
    
def super_resample(in_spec,in_wvl,out_wvl,out_fwhm):
    """
    inmf_master.super_resample() - Resamples a spectrum to a new response function
    
    Version 1.0
    Created on: Apr, 16, 2016
    Last Modified: June, 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu

     - Given an input wavelength array and spectrum, and output response 
       function, wavelength and fwhm resample the spectraum into the output response function.

    Inputs:
        in_spec - input spectrum to be resampled
                - format: numpy array, dtype = float
                - dimensions: [oldwvl,1] or [1,oldwvl]
        in_wvl - center wavelengths of the input spectrum
               - format: numpy array, dtype = float
               - dimensions: [oldwvl,1] or [1,oldwvl]
        out_wvl - center wavelength of the new response function
                - format: numpy array, dtype = float
                - dimensions: [newwvl,1] or [1,newwvl]
        fwhm - Full-Width-Half-Max of the new response function at this center
             - format: numpy array, dtype = float
             - dimensions: [newwvl,1] or [1,newwvl]
    Outputs:
        out - value of the new spectrum at this center wavelength
            - format: numpy array, dtype = float
            - dimensions: [newwvl,1] or [1,newwvl]
    """
    
    n = len(out_wvl);
    out_spec = N.zeros(n)
#    bp = N.zeros(n,len(in_spec))
    if len(out_fwhm) == 1:
        for i in range(n):
            out_spec[i] = gauss_conv(in_spec,in_wvl,out_wvl[i],out_fwhm)
    else:
        for i in range(n):
            out_spec[i] = gauss_conv(in_spec,in_wvl,out_wvl[i],out_fwhm[i]);

    return out_spec

def plot_nmf(output, W1, name, titles, K, cmaps = dict([('Asphalt','Greys'),
                                                        ('Concrete', 'Greys'), 
                                                        ('Snow','Greys'), 
                                                        ('Soil', 'Oranges'), 
                                                        ('Vegetation','Greens'), 
                                                        ('Water', 'Blues'), 
                                                        ('Cloud','RdPu'), 
                                                        ('Atmosphere', 'Purples')])):
    """
    inmf_master.plot_nmf() - plots and saves figures based on the output of the
                             INMF algorithm
    
    Version 1.0
    Created on: Mar, 16, 2017
    Last Modified: June, 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu

     - Given an input wavelength array and spectrum, and output response 
       function, wavelength and fwhm resample the spectraum into the output response function.

    Inputs:
        output - output of the INMF algorithm, generaterated by the nmf_output
                 function
        W1 - Initial Endmember Spectra
        name - Name of the scene, used in saving figures
        titles - Titles corresponding to each endmember, # of titles, must be
                 equal to K, the number of endmembers
        K - number of endmembers
        cmaps - A dictionary of colormaps, where the key corresponds to the
                title of an endmember and the value is a Matplotlib Colormap
    Outputs:
        Saved figure of the endmember spectra, abundanes and the residual
    """
    # Get Current Time
    timenow = datetime.now()
    timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
    
    # Get Size of array
    I, J, K = output['abundances'].shape
    
    # Plot Cost Function Descent
    plt.figure('cost')
    labels = ['Default','Smoothing','ASO']
    for i in range(len(output['cost'])):
        plt.plot(output['cost'][i,:],label = labels[i])
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost Function Value')
    plt.legend(loc = 'best')
    plt.savefig(timenow_str + '_' + name + '_cost.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)
    plt.close('cost')
    
    plt.figure('soln',figsize = (5.3,4))
    plt.xlabel('Wavelength [$nm$]')
    plt.ylabel('Radiance [$W m^{-2} sr^{-1} \mu m^{-1}$]')
    for i in range(K):
        endmember_color = mpl.cm.get_cmap(cmaps[titles[i]],128)
        plt.plot(output['wavelengths'],output['endmembers'][:,i],linestyle = '-',linewidth = 2, color = endmember_color(0.75),label = titles[i])

    x01,xn1,y01,yn1 = plt.axis()
    plt.legend(ncol = 2, columnspacing = 1, handletextpad = 0)
    
    plt.figure('init',figsize = (5.3,4))
    plt.xlabel('Wavelength [$nm$]')
    plt.ylabel('Radiance [$W m^{-2} sr^{-1} \mu m^{-1}$]')
    for i in range(K):
        endmember_color = mpl.cm.get_cmap(cmaps[titles[i]],128)
        plt.plot(output['wavelengths'],W1[:,i],linestyle = '-',linewidth = 2, color = endmember_color(0.75), label = titles[i])
    x02,xn2,y02,yn2 = plt.axis()
    plt.legend(ncol = 2, columnspacing = 1, handletextpad = 0)
    
    if yn2 > yn1:
        plt.axis([390,950,0,yn2])
        plt.savefig(timenow_str + '_' + name + '_init_endmembers.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)

        plt.figure('soln')
        plt.axis([390,950,0,yn2])
        plt.savefig(timenow_str + '_' + name + '_soln_endmembers.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)

    else:
        plt.axis([390,950,0,yn1])
        plt.savefig(timenow_str + '_' + name + '_init_endmembers.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)

        plt.figure('soln')
        plt.axis([390,950,0,yn1])
        plt.savefig(timenow_str + '_' + name + '_soln_endmembers.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)

    plt.close('soln')
    plt.close('init')
        
    for i in range(K):
        plt.figure(figsize = ((J/I*10),11.9))
        cax = plt.contourf(N.fliplr(output['abundances'][:,:,i]), cmap = cmaps[titles[i]], levels = N.linspace(0,0.8,num = 100), extend = 'max')
        plt.xticks([],[])
        plt.yticks([],[])
        cbar = plt.colorbar(cax, ticks = [0, 0.2, 0.4, 0.6, 0.8], orientation = 'horizontal', pad = 0.025, aspect = 7, shrink = 0.9)
        cbar.ax.tick_params(labelsize = 14)     
    
        plt.savefig(timenow_str + '_' + name + '_' + titles[i] + '.png', format = 'png', dpi = 300,bbox_inches = 'tight', transparent = True)
        plt.close()
        
    plt.figure('resid',figsize = ((J/I*10),10))
    plt.contourf(output['residual'],cmap = 'inferno')
    plt.colorbar()
    plt.savefig(timenow_str + '_' + name + '_resid' + '.png', format = 'png', dpi = 300,bbox_inches = 'tight', transparent = True)
    plt.close('resid')
    
def nmf_output(output):
    """
    inmf_master.nmf_output() - Takes NMF algorithm output and creates and saves 
                               all the relevant details for analysis
    
    Version 1.0
    Created on: Mar, 16, 2017
    Last Modified: June, 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu

     - Given an the NMF output and initialization data this function produces a
       new dictionary containing the result endmember spectra, abundances, 
       number of iterations, cost function value, title of the endmembers, and
       the initial guess of endmember spectra.
     - The new dictionary is then saved as a matlab '.mat' files
     relevant input wavelength array and spectrum, and output response 
       function, wavelength and fwhm resample the spectraum into the output response function.

    Inputs:
        output - output of the INMF algorithm, generaterated by the nmf_output
                 function
    Outputs:
        new_output - dictionary combining NMF initialization and output data
        TIME_NAME_results.mat - a ".mat" file saving the new_output dictionary
    """
    new_output = dict([('endmembers',output['W']), 
                        ('iterations',output['n_iter']),
                        ('titles',output['titles']),
                        ('wavelengths',N.squeeze(output['wavelengths'])),
                        ('cost',output['cost']),
                        ('initialW',output['W1'])])
    # Calculate Residual
    recon = N.reshape(N.dot(output['W'],output['H']), (output['dims'][0],output['dims'][1],output['dims'][2]))
    new_output['residual'] = N.sum(N.sqrt((output['datacube'] - recon) ** 2), 2)

    # Reshape Spatial Abundances
    new_output['abundances'] = N.reshape(N.transpose(output['H']), (output['dims'][0],output['dims'][1],output['dims'][3]))
    
    # Export Data as .mat files
    timenow = datetime.now()
    timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
    filename_tosave = timenow_str + '_' + output['fname'] + '_RESULTS.mat'
    
    sio.savemat(filename_tosave, new_output)
    
    plot_nmf(new_output,output['W1'],output['fname'],output['titles'],output['dims'][3])
    
    return new_output
    
'''

-------------------------------------------------------------------------------
NMF Master Script

'''
params = load_input_file('PARAMS.txt')

inputfile = sys.argv[1]
inputs = load_input_file(inputfile)
wvl_ind = inputs['wvl_ind']
titles = inputs['members']
name = inputs['name']
file = inputs['file']

hypercube = load_hico(params['path'] + file + '.L1B_ISS', fill_saturated = True)

# Slice datacube
#   This step remove bad wavelengths
#   Removes slit from side of image
wvl = hypercube.resp_func['wvl']
wvl = wvl[wvl_ind[0]:wvl_ind[1]]
fwhm = hypercube.resp_func['fwhm']
fwhm = fwhm[wvl_ind[0]:wvl_ind[1]]
resp_func = dict()
resp_func['wvl'] = wvl
resp_func['fwhm'] = fwhm
try: 
    subset = N.array(inputs['subset'],dtype = 'int')
    print(subset)
    hypercube.data_cube = hypercube.data_cube[subset[0]:subset[1],subset[2]:subset[3],wvl_ind[0]:wvl_ind[1]]
except KeyError:
    # For just removing side of slit and bad wavelengths use:
    #   hypercube.data_cube = hypercube.data_cube[:,9:509,wvl_ind[0]:wvl_ind[1]]
    # Remove side of slit, bad wavelengths, and subset spatial domain
    hypercube.data_cube = hypercube.data_cube[:,9:,wvl_ind[0]:wvl_ind[1]]

# Load and apply HICO Vicarious Calibration Gains
gains = N.genfromtxt(params['caldat'])
hypercube.data_cube = hypercube.data_cube * gains[wvl_ind[0]:wvl_ind[1]]

I,J,L = hypercube.data_cube.shape
K = len(titles)

# Generate Library Radiance Spectra
if os.path.isfile(file + '_HICO_REFL.mat') == False:
    
    # Load MODTRAN Irradiance
    irrad = N.genfromtxt(params['irradpath'],skip_header = 3)
    # Convert to wavelength increments
    irrad[:,0] = 1e7 / irrad[:,0]
    irrad[:,1] = irrad[:,1] * (1 / (irrad[:,0] ** 2)) * 1e11
    irrad = irrad[0:-1,:]
    # I0 = I0(I0(:,1)<3000,:)
    # Convolve to NEON Wavelength
    mu = math.cos((inputs['SZA']/180)*math.pi)
    I0 = inputs['SunElliptic'] * super_resample(irrad[:,1], irrad[:,0], wvl, fwhm) * 1000
    # *1000 converts form W/m^2/nm to W/m^2/um
    
    mod_path = params['modpath'] + file
    
    # Load Transmittances
    acd_dat = load_acd(mod_path + '.acd');
    mod_wvl = 1e7/(acd_dat['freq']); # Transfer Wavenumber to Wavelength

    sph_alb = super_resample(acd_dat['sph'],mod_wvl,wvl,fwhm);
    ts = super_resample(acd_dat['ts'],mod_wvl,wvl,fwhm);
    Ts = super_resample(acd_dat['Ts'],mod_wvl,wvl,fwhm);
    t = super_resample(acd_dat['t'],mod_wvl,wvl,fwhm);
    T = super_resample(acd_dat['T'],mod_wvl,wvl,fwhm);
    
    # Load MODTRAN Radiance
    data = N.loadtxt(mod_path + '.psc')
    rad0 = super_resample(data[:,1],data[:,0],wvl,fwhm) * 10

    # Load Reference Spectra
    #   ASTER Library
    HICO_refl = sio.loadmat(params['hicoreflpath'])
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

    sigma_bar = ((Ts + ts) * (T + t) * sph_alb) / ((Ts + ts) * (T + t))

    # Calculate Radiances
    rho = super_resample(N.squeeze(HICO_refl['asphalt_refl']) / 100,hypercube.resp_func['wvl'],wvl,fwhm)
    R_asphalt = ((((mu * I0 * (Ts + ts) * T) / math.pi) * rho) + (((mu * I0 * (Ts + ts) * t) / math.pi) * rho)) / (1 - sigma_bar * rho)
        
    rho = super_resample(N.squeeze(HICO_refl['conc_refl']) / 100,hypercube.resp_func['wvl'],wvl,fwhm)
    R_conc = ((((mu * I0 * (Ts + ts) * T) / math.pi) * rho) + (((mu * I0 * (Ts + ts) * t) / math.pi) * rho)) / (1 - sigma_bar * rho)
    
    rho = super_resample(N.squeeze(HICO_refl['snow_refl']) / 100,hypercube.resp_func['wvl'],wvl,fwhm)
    R_snow = ((((mu * I0 * (Ts + ts) * T) / math.pi) * rho) + (((mu * I0 * (Ts + ts) * t) / math.pi) * rho)) / (1 - sigma_bar * rho)
    
    rho = super_resample(N.squeeze(HICO_refl['soil_refl']) / 100,hypercube.resp_func['wvl'],wvl,fwhm)
    R_soil = ((((mu * I0 * (Ts + ts) * T) / math.pi) * rho) + (((mu * I0 * (Ts + ts) * t) / math.pi) * rho)) / (1 - sigma_bar * rho)
    
    rho = super_resample(N.squeeze(HICO_refl['veg_refl']) / 100,hypercube.resp_func['wvl'],wvl,fwhm)
    R_veg = ((((mu * I0 * (Ts + ts) * T) / math.pi) * rho) + (((mu * I0 * (Ts + ts) * t) / math.pi) * rho)) / (1 - sigma_bar * rho)

    rho = super_resample(N.squeeze(HICO_refl['water_refl']) / 100,hypercube.resp_func['wvl'],wvl,fwhm)
    R_water = ((((mu * I0 * (Ts + ts) * T) / math.pi) * rho) + (((mu * I0 * (Ts + ts) * t) / math.pi) * rho)) / (1 - sigma_bar * rho)
    
    rho = super_resample(cloud_refl[1,:],cloud_refl[0,:],wvl,fwhm)
    R_cloud = ((((mu * I0 * (Ts + ts) * T) / math.pi) * rho) + (((mu * I0 * (Ts + ts) * t) / math.pi) * rho)) / (1 - sigma_bar * rho)

    R_atmo = rad0
    
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

# Load Previously Generated HICO Endmembers
refl_endmembers = sio.loadmat(file + '_HICO_REFL.mat')
glintwater = sio.loadmat('WaterGlint_Initial.mat')

#refl_endmembers['Water'] = glintwater['Water_Refl']
W1 = N.empty((L,K))
    
for j in range(len(titles)):
    W1[:,j] = N.squeeze(refl_endmembers[titles[j]])
    
cmaps = dict([('Asphalt','Greys'), ('Concrete', 'Greys'), ('Snow','Greys'),
              ('Soil', 'Oranges'), ('Vegetation','Greens'), ('Water', 'Blues'),
              ('Cloud','RdPu'), ('Atmosphere', 'Purples')])

H1 = N.ones((K,I*J)) / K;

# Set NMF Parameters
win_W = N.ones(K, dtype = 'int') * 11
win_H = N.ones(K-1, dtype = 'int') * 5
win_H = N.append(win_H, 11)

if inputs['perturb'] is not None:
    # Perturbing Initial Conditions
    import itertools
    prod = itertools.product(N.array(inputs['perturb'], dtype = 'float') + 1, repeat = K)
    f = open(inputs['name'] + '_PerturbationGuide.txt','w')
    out = titles.copy()
    out.insert(0,'#')
    f.write(('{}\t'*(K+1)).format(*out))
    count = 0
    perturb = list()
    for perm in prod:
        perturb.append(N.tile(perm,[L,1]))
        f.write('\n')
        out = N.insert(N.array(perm),0,count)
        f.write(('{:0.2f}\t'*(K+1)).format(*out))
        count += 1
    f.close()

    # Construct input arguments
    kwargs = {'K':K, 'windowW':win_W, 'windowH':win_H, 'maxiter':inputs['max_i']}
    print('Number of Processes: ',params['procnum'])
    pool = mp.Pool(processes = params['procnum'])
    count = 0
    for scl in perturb:
        info = {'fname':inputs['name'] + str(count) + '_inmf','titles':titles,'dims':(I, J, L, K)}
        print(count, info['fname'])
        pool.apply_async(nmf.INMF, (hypercube.data_cube,W1*scl,H1,resp_func,info), kwargs, callback = nmf_output)
        count +=1
    pool.close()
    pool.join()

else:
    # Run INMF   
    info = {'fname':inputs['name'] + '_inmf','titles':titles,'dims':(I, J, L, K)}
    inmf_out = nmf.INMF(hypercube.data_cube, W1, H1, resp_func, info, K = K, windowW = win_W, 
                      windowH = win_H, maxiter = inputs['max_i'])
    inmf_out['wavelengths'] = wvl
    inmf_out = nmf_output(inmf_out)
    # Plot Results
    if inputs['plot_flag'] is True:
        plot_nmf(inmf_out, W1, inputs['name'] + '_inmf', titles, K)
    print('Completed INMF')
