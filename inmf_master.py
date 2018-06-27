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
    inputfile = sys.argv[1]
except:
    subprocess.run(['python','inmf_gui.py'])

# Import Modules
import nmf
import copy
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Import individual functions
from datetime import datetime
from load_hico import load_hico
import nmf_output
from calc_radiance import calc_radiance
from loading import load_input_file
from super_resample import super_resample

class NMF_result(object):
    def __init__(self,data,W,H):
        [I,J,L] = data.shape
        self.A = np.transpose(np.reshape(data,(I*J,L)))
        self.W = W
        self.H = H
        self.cost = [None]
        self.residual = [None]
        self.iter = [None]
        self.norm = [None]

# Define the NMF object
class NMF_obj(object):
    def __init__(self,hico_cube,inputs,params):
        self.name = inputs['name']
        self.scene = copy.copy(hico_cube)
        self.inputs = inputs
        self.parameters = params
        self.endmembers = []
        self.status = ['No Results Computed']
        self.results = []

        # Get dimensions of image to be decomposed
        #   I,J are the along and across-track spatial dimensions,
        #   L is the spectral dimension
        #   K is the number of endmembers to be used in the decomposition
        I,J,L = self.scene.data_cube.shape
        K = len(inputs['members'])

        # Subset the scene given the inputs for spatial and wavelength ranges
        try:
            # Selects a user defined subset
            subset = np.array(inputs['subset'],dtype = 'int')
        except KeyError:
            # If no subset is specified only remove side of slit, and bad wavelengths
            subset = np.array([0,I+1,9,J+1])
        self.subset(subset)

        # Recalculate Dimensions after subsetting
        I,J,L = self.scene.data_cube.shape
        self.scenesize = [I,J,K,L]

    def __str__(self):
        return 'NMF Object, {0} operating on HICO scene: {1} \n \
                {2}'.format(self.name,self.scene.name,self.results)

    def subset(self,subset):
        # Slice datacube to a user defined subset
        self.scene.resp_func['wvl'] = self.scene.resp_func['wvl'][self.inputs['wvl_rng'][0]:self.inputs['wvl_rng'][1]]
        self.scene.resp_func['fwhm'] = self.scene.resp_func['fwhm'][self.inputs['wvl_rng'][0]:self.inputs['wvl_rng'][1]]
        self.scene.data_cube = self.scene.data_cube[subset[0]:subset[1],subset[2]:subset[3],self.inputs['wvl_rng'][0]:self.inputs['wvl_rng'][1]]

    def cal(self,cal_file):
        # Applies a calibration file
        gains = np.genfromtxt(cal_file)[:,1]
        self.scene.data_cube = self.scene.data_cube * gains[self.inputs['wvl_rng'][0]:self.inputs['wvl_rng'][1]]

    def initialize(self):
        # Generates initial Endmember Spectra and Abundances
        # Load Endmembers
        possible_endmembers = sio.loadmat(self.inputs['endmember_file'])
        titles = list()
        spectra = list()

        for endmember in inputs['members']:
            spectra.append(possible_endmembers[endmember])
            titles.append(endmember)

        self.endmembers = {'spectra':np.vstack(np.transpose(spectra)),'titles':titles}
        W_init = self.endmembers['spectra']

        [I,J,K,L] = self.scenesize
        H_init = np.ones((K,I*J)) / K
        self.results = NMF_result(self.scene.data_cube,W_init, H_init)

        self.inputs['delta_vec'] = np.full((1,K),0)

    def NMF(self):
        nmf.NMF(self)

    def INMF(self):

        # Do INMF iteration
        nmf.INMF(self)

        # Use Normalization Factor to Return to Radiance Units (if neccesary)
        if self.results.norm is not None:
            dim = self.results.norm.shape

            if dim == self.scenesize[2]:
                INMF_processing.results.A = np.transpose(np.transpose(INMF_processing.results.A)*INMF_processing.results.norm)
                INMF_processing.results.W = np.transpose(np.transpose(INMF_processing.results.W)*INMF_processing.results.norm)
            elif dim == self.scenesize[0]*self.scenesize[1]:
                INMF_processing.results.A = INMF_processing.results.A*INMF_processing.results.norm
            else:
                print('Unable to Recogize Normalization Dimensions')

        # Calculate Residual
        recon = np.reshape(np.dot(self.results.W,self.results.H), (self.scenesize[0],self.scenesize[1],self.scenesize[3]))
        self.results.residual = np.sum(np.sqrt((self.scene.data_cube - recon) ** 2), 2)

        # Reshape Spatial Abundances
        self.results.H = np.reshape(np.transpose(self.results.H), (self.scenesize[0],self.scenesize[1],self.scenesize[2]))
        self.status = 'Results Computed!'
        print(self.status)

    def plot(self):
        '''
        nmf_output.plot_nmf() - plots and saves figures of INMF results,
            self.INMF() must be called before this function
        '''
        # Check status
        if self.status is not 'Results Computed!':
            print('INMF results not yet computed')
            return
        else:
            nmf_output.plot_nmf(self)

    def output():
        '''
        nmf_output.output_nmf() - saves INMF run data for future analysis,
            self.INMF() must be called before this function
        '''
        # Check status
        if self.status is not 'Results Computed!':
            print('INMF results not yet computed')
            return
        else:
            nmf_output.nmf_output(self)


# Read Parameter and Input Files
print(sys.argv,inputfile,'line218')

params = load_input_file('PARAMS.txt')
inputs = load_input_file(inputfile)

print(inputs)

# ID = inputs['file'].split('.')[0]

# Load HICO Scene
hypercube = load_hico(os.path.abspath(inputs['file']), fill_saturated = True)

# Create NMF object
INMF_processing = NMF_obj(hypercube,inputs,params)

# Load and apply HICO Vicarious Calibration Gains
INMF_processing.cal(os.path.abspath(params['caldat']))

# Deal with the endmembers
# Load in Previously Generated Endmembers
INMF_processing.initialize()
 #### PARSE MEMBERS HERE

# Normalization
if inputs['norm'] == 'none':
    print('No Normalization')

elif inputs['norm'] == 'aso':
    print('Applying Abundance-Sum-to-One Normalization')
    try:
        delta_vec = INMF_processing.inputs['delta']*np.array(INMF_processing.inputs['aso_vec'],ndmin = 2)
    except KeyError:
        delta_vec = np.full((1,K),INMF_processing.inputs['delta'])
    INMF_processing.inputs['delta_vec'] = delta_vec

elif inputs['norm'] == 'refl':
    print('Applying Top-of-Atmosphere Reflectance Normalization')
    # Load Irradiance Data and Create Fit Function for Fixed Atmosphere section
    # Using standard TOA Irradiance Spectrum
    irrad = np.genfromtxt('/Users/wrightad/Documents/MODTRAN/DATA/SUN01med2irradwnNormt.dat',skip_header = 3)
    # Convert to wavelength increments
    irrad[:,0] = 1e7 / irrad[:,0]
    irrad[:,1] = irrad[:,1] * (1 / (irrad[:,0] ** 2)) * 1e11
    irrad = irrad[0:-1,:]
    # Resample to HICO wavelengths
    solar_irrad = super_resample(irrad[:,1],irrad[:,0],INMF_processing.scene.resp_func['wvl'],INMF_processing.scene.resp_func['fwhm'])

    INMF_processing.results.norm = solar_irrad
    INMF_processing.results.A = np.transpose(np.transpose(INMF_processing.results.A)/solar_irrad)
    INMF_processing.results.W = np.transpose(np.transpose(INMF_processing.results.W)/solar_irrad)

elif inputs['norm'] == 'pixel':
    print('Applying Pixel-by-Pixel Spatial Normalization')

    mean_val = np.mean(INMF_processing.results.A, axis = 0)

    print(mean_val.shape)

    INMF_processing.results.norm = np.mean(INMF_processing.results.A, axis = 0)
    INMF_processing.results.A = INMF_processing.results.A/INMF_processing.results.norm

    print(INMF_processing.results.A.shape)

elif inputs['norm'] == 'spectral':
    print('Applying Spectral Normalization')

    INMF_processing.results.norm = np.mean(INMF_processing.results.A, axis = 1)
    INMF_processing.results.A = np.transpose(np.transpose(INMF_processing.results.A)/INMF_processing.results.norm)
    INMF_processing.results.W = np.transpose(np.transpose(INMF_processing.results.W)/INMF_processing.results.norm)

else:
    print('No Normalization Specified')

# Run NMF

print(INMF_processing)

INMF_processing.INMF()
INMF_processing.plot()

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
