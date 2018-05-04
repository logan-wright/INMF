#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inmf_master.py - master script needed to run the INMF code

Version 2.3
Created on: Oct 13, 2016
Last Modified: Apr 19, 2018
Author: Logan Wright, logan.wright@colorado.edu

Description:
 Version 2.3 Notes:
    - Added Normalization Options (See readme for descriptions)
        - None (default)
        - Top of Atmosphere Reflectance
        - Abundance-Sum-to-One or ASO
        - Pixel Signal Weighted
        - Wavelength Signal Weighted

 Version 2.2 Notes:
     -Improved the ability to use initialization files

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
# inputs = load_input_file

# # Define the NMF object
# class NMF_obj(object):
#     def __init__(self,inital,result,meta):
#         self.data_cube = data_cube
#         self.resp_func = resp_func
#         self.nav = nav
#         self.flags = flags
#         self.rgb = rgb
'''

-------------------------------------------------------------------------------
NMF Master Script

'''
import os
import sys
import subprocess

# Check if arguments are passed to the inmf_master function if not, open the GUI
try:
    sys.argv[1]
except:
    subprocess.run(['python','inmf_gui.py'])

# Import Modules
import nmf
import math
import numpy as np
import scipy.io as sio
import matplotlib as mpl
#mpl.use('Agg')
import multiprocessing as mp
import matplotlib.pyplot as plt

# Import individual functions
from datetime import datetime
from load_hico import load_hico
import nmf_output
from calc_radiance import calc_radiance
from loading import load_input_file
from super_resample import super_resample

# Read Parameter and Input Files
params = load_input_file('PARAMS.txt')
inputs = load_input_file(sys.argv[1])

# wvl_rng = inputs['wvl_rng']
# titles = inputs['members']
# name = inputs['inputs['name']']
# file = inputs['file']
ID = inputs['file'].split('.')[0]

# Load HICO Scene
hypercube = load_hico(params['path'] + inputs['file'], fill_saturated = True)

# Slice datacube
#   This step remove bad wavelengths
# wvl = hypercube.resp_func['wvl']
# wvl = wvl[inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]
# fwhm = hypercube.resp_func['fwhm']
# fwhm = fwhm[inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]
# resp_func = {'wvl':wvl,'fwhm':fwhm}

wvl = hypercube.resp_func['wvl'][inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]
fwhm = hypercube.resp_func['fwhm'][inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]
resp_func = {'wvl':wvl,'fwhm':fwhm}

try:
    subset = np.array(inputs['subset'],dtype = 'int')
    print(subset)
    # Remove side of slit, bad wavelengths, and subset spatial domain
    hypercube.data_cube = hypercube.data_cube[subset[0]:subset[1],subset[2]:subset[3],inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]
except KeyError:
    # For just removing side of HICO slit and bad wavelengths use:
    hypercube.data_cube = hypercube.data_cube[:,9:,inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]

# Load and apply HICO Vicarious Calibration Gains
gains = np.genfromtxt(params['caldat'])[:,1]
hypercube.data_cube = hypercube.data_cube * gains[inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]

I,J,L = hypercube.data_cube.shape
K = len(inputs['members'])

# Deal with the endmembers
 #### PARSE MEMBERS HERE

# Generate Library Radiance Spectra
if os.path.isfile(ID + '_HICO_REFL.mat') == False:
    paths = [params['modpath'] + ID,params['irradpath'],params['hicoreflpath']]
    calc_radiance(paths,inputs['SZA'],inputs['SunElliptic'],resp_func,hypercube.resp_func,plot_flag = 1)

# Load Previously Generated HICO Endmembers
refl_endmembers = sio.loadmat(params['modpath'] + ID + '_HICO_REFL.mat')
glintwater = sio.loadmat('/Users/wrightad/Dropbox/LASP/NMF/py_pro/WaterGlint_Initial.mat')

#refl_endmembers['Water'] = glintwater['Water_Refl']
W1 = np.empty((L,K))

for j in range(len(inputs['members'])):
    W1[:,j] = np.squeeze(refl_endmembers[inputs['members'][j]])

cmaps = dict([('Asphalt','Greys'), ('Concrete', 'Greys'), ('Snow','Greys'),
              ('Soil', 'Oranges'), ('Vegetation','Greens'), ('Water', 'Blues'),
              ('Cloud','RdPu'), ('Atmosphere', 'Purples')])

H1 = np.ones((K,I*J)) / K;

# Set NMF Parameters
win_W = np.ones(K, dtype = 'int') * 11
win_H = np.ones(K-1, dtype = 'int') * 5
win_H = np.append(win_H, 11)

# Normalization
if inputs['norm'] == 'none':

elif inputs['norm'] == 'aso':
    print('Applying Abundance-Sum-to-One Normalization')
elif inputs['norm'] == 'refl':
    print('Applying Top-of-Atmosphere Reflectance Normalization')
elif inputs['norm'] == 'pixel':
    print('Applying Pixel-by-Pixel Spatial Normalization')
elif inputs['norm'] == 'spectral':
    print('Applying Spectral Normalization')

else:
    print('No Normalization Specified')


# Run NMF
if inputs['perturb'] is not None:
    # Perturbing Initial Conditions
    import itertools
    prod = itertools.product(np.array(inputs['perturb'], dtype = 'float') + 1, repeat = K)
    f = open(inputs['name'] + '_PerturbationGuide.txt','w')
    out = inputs['members'].copy()
    out.insert(0,'#')
    f.write(('{}\t'*(K+1)).format(*out))
    count = 0
    perturb = list()
    for perm in prod:
        perturb.append(np.tile(perm,[L,1]))
        f.write('\n')
        out = np.insert(np.array(perm),0,count)
        f.write(('{:0.2f}\t'*(K+1)).format(*out))
        count += 1
    f.close()

    # Construct input arguments
    kwargs = {'K':K, 'windowW':win_W, 'windowH':win_H, 'maxiter':inputs['max_i']}
    print('Number of Processes: ',params['procnum'])
    pool = mp.Pool(processes = params['procnum'])
    count = 0
    for scl in perturb:
        info = {'fname':inputs['name'] + str(count) + '_inmf','titles':inputs['members'],'dims':(I, J, L, K)}
        print(count, info['fname'])
        pool.apply_async(nmf.INMF, (hypercube.data_cube,W1*scl,H1,resp_func,info), kwargs, callback = nmf_output.nmf_output)
        count +=1
    pool.close()
    pool.join()

else:
    # Run INMF
    info = {'fname':inputs['name'] + '_inmf','titles':inputs['members'],'dims':(I, J, L, K)}
    inmf_out = nmf.INMF(hypercube.data_cube, W1, H1, resp_func, info, K = K, windowW = win_W,
                      windowH = win_H, maxiter = inputs['max_i'])
    inmf_out['wavelengths'] = wvl
    inmf_out = nmf_output.nmf_output(inmf_out)
    # Plot Results
    if inputs['plot_flag'] is True:
        nmf_output.plot_nmf(inmf_out, W1, inputs['name'] + '_inmf', inputs['members'], K)
    print('Completed INMF')
