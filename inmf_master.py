#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inmf_master.py - master script needed to run the INMF code

Version 2.2
Created on: Oct, 13, 2016
Last Modified: Jan, 27 2018
Author: Logan Wright, logan.wright@colorado.edu

Description:
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

import nmf_output
from calc_radiance import calc_radiance
from loading import load_input_file
from super_resample import super_resample

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
# Read Parameter and Input Files
params = load_input_file('PARAMS.txt')

inputfile = sys.argv[1]
inputs = load_input_file(inputfile)
wvl_rng = inputs['wvl_rng']
titles = inputs['members']
name = inputs['name']
file = inputs['file']

# Load HICO Scene
hypercube = load_hico(params['path'] + file, fill_saturated = True)

# Slice datacube
#   This step remove bad wavelengths
#   Removes slit from side of image
wvl = hypercube.resp_func['wvl']
wvl = wvl[wvl_rng[0]:wvl_rng[1]]
fwhm = hypercube.resp_func['fwhm']
fwhm = fwhm[wvl_rng[0]:wvl_rng[1]]
resp_func = {'wvl':wvl,'fwhm':fwhm}

try:
    subset = N.array(inputs['subset'],dtype = 'int')
    print(subset)
    hypercube.data_cube = hypercube.data_cube[subset[0]:subset[1],subset[2]:subset[3],wvl_rng[0]:wvl_rng[1]]
except KeyError:
    # For just removing side of slit and bad wavelengths use:
    #   hypercube.data_cube = hypercube.data_cube[:,9:509,wvl_rng[0]:wvl_rng[1]]
    # Remove side of slit, bad wavelengths, and subset spatial domain
    hypercube.data_cube = hypercube.data_cube[:,9:,wvl_rng[0]:wvl_rng[1]]

# Load and apply HICO Vicarious Calibration Gains
gains = N.genfromtxt(params['caldat'])
hypercube.data_cube = hypercube.data_cube * gains[wvl_rng[0]:wvl_rng[1]]

I,J,L = hypercube.data_cube.shape
K = len(titles)

# Generate Library Radiance Spectra
if os.path.isfile(file + '_HICO_REFL.mat') == False:
    paths = [params['modpath'],params['irradpath'],params['hicoreflpath']]
    print(paths)
    calc_radiance(paths,inputs['SZA'],inputs['SunElliptic'],resp_func,plot_flag = 1)

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
        pool.apply_async(nmf.INMF, (hypercube.data_cube,W1*scl,H1,resp_func,info), kwargs, callback = nmf_output.nmf_output)
        count +=1
    pool.close()
    pool.join()

else:
    # Run INMF
    info = {'fname':inputs['name'] + '_inmf','titles':titles,'dims':(I, J, L, K)}
    inmf_out = nmf.INMF(hypercube.data_cube, W1, H1, resp_func, info, K = K, windowW = win_W,
                      windowH = win_H, maxiter = inputs['max_i'])
    inmf_out['wavelengths'] = wvl
    inmf_out = nmf_output.nmf_output(inmf_out)
    # Plot Results
    if inputs['plot_flag'] is True:
        nmf_output.plot_nmf(inmf_out, W1, inputs['name'] + '_inmf', titles, K)
    print('Completed INMF')
