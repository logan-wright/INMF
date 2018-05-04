#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inmf_master.py - master script to run the INMF code

 - Reads in a hyperspectral data scene, (Written for HICO data)
 - Reads INMF settings and parameters from input (*.in) and PARAMS.txt files
 - Performs INMF on the scene with the specified settings
 - Saves and plots results

 X Generates intial guess radiance spectra using MODTRAN inputs (Version 2.3
   Removed this functionality to the standalone function "calc_radiance.py")

 Inputs:
        inputfile = name of the inputfile to be used, as a string
        EXAMPLE: python nmf_master.py 'SampleINMF.in'
    OR:
        Calling without an input file will prompt the user to create an input
        files with a graphical interface.
        EXAMPLE: python nmf_master.py
    OR:
        Alternatively, the GUI may be access directly by running inmf_gui.py
        EXAMPLE: inmf_gui.py

 Outputs:
       INMF results file
       Automatic plots

Version 2.3
Created on: Oct 13, 2016
Last Modified: May 4, 2018
Author: Logan Wright, logan.wright@colorado.edu

Description:
 Version 2.3 Notes:
    - Added Normalization Options (See readme for descriptions)
        - None (default)
        - Top of Atmosphere Reflectance
        - Abundance-Sum-to-One or ASO
        - Pixel Signal Weighted
        - Wavelength Signal Weighted
    - Re-organized code to be Object-Oriented
    - Added GUI

 Version 2.2 Notes:
     -Improved the ability to use initialization files

 Version 2.1 Notes:
     -Added the multiprocessing ability if the input file contains multiple INMF
      processing requests.
     -Added built-in handling to initialized a perturbed ensemble

 Version 2.0 Notes:
     Seperated master script into subfunctions

-------------------------------------------------------------------------------
NMF Master Script

"""
# Import system interface modules
import os
import sys
import subprocess

# Check if arguments are passed to inmf_master. If not, open the GUI
try:
    sys.argv[1]
except:
    subprocess.run(['python','inmf_gui.py'])

# Import Modules
import nmf
import copy
import math
import numpy as np
import scipy.io as sio
import matplotlib as mpl
# import multiprocessing as mp
import matplotlib.pyplot as plt

# Import individual functions
from datetime import datetime
from load_hico import load_hico
import nmf_output
from calc_radiance import calc_radiance
from loading import load_input_file
from super_resample import super_resample

# Define the NMF object
class NMF_obj(object):
    def __init__(self,hico_cube,inputs):
        self.scene = copy.copy(hico_cube)
        self.inputs = inputs
        self.name = inputs['name']
        self.results = ['No Results Computed']

        self.subset(inputs['subset'])

        # Load and apply HICO Vicarious Calibration Gains
        gains = np.genfromtxt(params['caldat'])[:,1]
        self.scene.data_cube = hico_cube.data_cube * gains[inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]

        # Get dimensions of image to be decomposed
        #   I,J are the along and across-track spatial dimensions,
        #   L is the spectral dimension
        #   K is the number of endmembers to be used in the decomposition
        I,J,L = self.data_cube.shape
        K = len(inputs['members'])
        self.scenesize = [I,J,L,K]

    def __str__(self):
        return 'NMF Object, {0} operating on HICO scene: {1} \n \
                {2}'.format(self.name,self.scene.name,self.results)

    def subset(self,subset):
        # Slice datacube
        self.scene.resp_func['wvl'] = self.scene.resp_func['wvl'][inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]
        self.scene.resp_func['fwhm'] = self.scene.resp_func['fwhm'][inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]

        try:
            subset = np.array(subset,dtype = 'int')
            print(subset)
            # Remove side of slit, bad wavelengths, and subset spatial domain
            self.scene.data_cube = self.scene.data_cube[subset[0]:subset[1],subset[2]:subset[3],inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]
        except KeyError:
            # For just removing side of HICO slit and bad wavelengths use:
            self.scene.data_cube = self.scene.data_cube[:,9:,inputs['wvl_rng'][0]:inputs['wvl_rng'][1]]

    def INMF(self):
        # Do stuff
        results = nmf.INMF(self)
        self.results[0] = 'Results Computed!'
        self.results[1] = results

    def plot():

        nmf_output.plot_nmf(self)

    def output():

        nmf_output.nmf_output(self)


# Read Parameter and Input Files
params = load_input_file('PARAMS.txt')
inputs = load_input_file(sys.argv[1])

# ID = inputs['file'].split('.')[0]

# Load HICO Scene
hypercube = load_hico(params['path'] + inputs['file'], fill_saturated = True)

# Create NMF object
INMF_processing = NMF_obj(hypercube,inputs)

# Deal with the endmembers
# Load in Previously Generated Endmembers
INMF_processing.endmembers = sio.loadmat(params['endmember_file'])
 #### PARSE MEMBERS HERE

# Normalization
if inputs['norm'] == 'none':
    solar_irrad = 5

    # hico_refl = hypercube.data_cube/
elif inputs['norm'] == 'aso':
    print('Applying Abundance-Sum-to-One Normalization')
    delta_vec = delta*inputs['aso_vec']


elif inputs['norm'] == 'refl':
    print('Applying Top-of-Atmosphere Reflectance Normalization')


elif inputs['norm'] == 'pixel':
    print('Applying Pixel-by-Pixel Spatial Normalization')


elif inputs['norm'] == 'spectral':
    print('Applying Spectral Normalization')


else:
    print('No Normalization Specified')

# Run NMF

INMF_processing.INMF()

# info = {'fname':inputs['name'] + '_inmf','titles':inputs['members'],'dims':(I, J, L, K)}
# inmf_out = nmf.INMF(hypercube.data_cube, W1, H1, resp_func, info, K = K, windowW = inputs['spectral_win'],
#                   windowH = inputs['spatial_win'], maxiter = inputs['max_i'])
# inmf_out['wavelengths'] = wvl
# inmf_out = nmf_output.nmf_output(inmf_out)
# # Plot Results
# if inputs['plot_flag'] is True:
#     nmf_output.plot_nmf(inmf_out, W1, inputs['name'] + '_inmf', inputs['members'], K)
print('Completed INMF')

'''
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
    '''
