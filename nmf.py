#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmf.py - contains 3 NMF routines

Created on Sun Jan 22 23:41:47 2017
Last Modified on Jun, 16, 2017
author: Logan Wright

Description:
    This file contains three NMF routines:
        nmf.NMF, Canonical NMF as described by Lee & Seung, 2000
        nmf.INMF, Informed NMF routine developed by the author
        nmf.PSnsNMF, Piecewise Smoothess  "nonsmooth" NMF Described by Jia and
                     Qian, 2009 and Pascual-Montano, A. et al., 2006
"""
import numpy as np
import numpy.matlib
import scipy.ndimage as scimg
from datetime import datetime
import nmf_eval
import super_resample

def NMF(data, W, H, K = 5, maxiter = 500, debug = False):
    '''
    Lee & Seung Multiplicative Update NMF
    Created By: Logan Wright
    Created On: Feb 14, 2017
    Last Modified: Feb 20, 2017

    Description:
        An NMF routine using a multiplicative update rule for H and W. A
        Frobenius Norm cost function is used. Routine is described in
        Lee and Seung, 2001, Section 4.

    Reference:
        Lee, D. D., and H. S. Seung (2001), Algorithms for Non-negative Matrix
        Factorization, in Advances in Neural Information Processing Systems 13,
        edited by T. K. Leen, T. G. Dietterich, and V. Tresp, pp. 556–562,
        MIT Press.

    Inputs:
        data, a numpy array of a hyperspectral data cube of the form [X by Y by Z] where the third (Z) dimension is the wavelength axis.
        init_W, spectral endmembers initial guess, numpy array, should have dimensions of [Z, K]
        init_H, spatial abundance initial guess, numpy array, should have dimensions of [X*Y,K]

        params, dictionary may include:
            K = 5
            maxiter = 500
    '''
    # Define Cost Function for Lee & Seung
    #   Cost function is Frobenius Norm of difference between A and W*H
    def cost_function(A, W, H):
        cost = np.linalg.norm(A - np.dot(W, H), ord = 'fro')
        return cost

    # Define Ending Conditions, these are dependent on the data value
    #   Change end condition
    d_cost = 1

    # Flatten data to 2 dimensions
    I,J,L = data.shape
    A = np.transpose(np.reshape(data,(I * J, L)))

    # Initialize looping counters and flags
    count = 1           # Iteration Counter
    stop_criteria = 0   # Stop_Criteria Flag, stops iterations when equal to 1

    # Calculate Initial Cost
    cost = cost_function(A, W, H)

    # If debug option is turned on output initial data
    if debug:
        debugpath = '/Users/wrightad/Documents/Data/NMF_DebugFiles/'
        timenow = datetime.now()
        timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
        # Reshape Spatial Abundances
        H_recon = np.reshape(np.transpose(H), (I, J, K))
        np.savez(debugpath + 'NMF_DebugOutput_' + timenow_str, W = W, H = H_recon, count = count)

    # Lee & Seung Multiplicative Update Algorithm While Loop
    while stop_criteria == 0:
        # Update Spatial Abundances
        H = H * (np.dot(np.transpose(W), A)) / (np.dot(np.dot(np.transpose(W), W), H) + 10e-9)
        # Update Spectral Endmembers
        W = W * (np.dot(A, np.transpose(H))) / (np.dot(np.dot(W, H), np.transpose(H))+ 10e-9)

        cost = np.append(cost,cost_function(A, W, H))
        count += 1 # Update iteration counter

        # Stop Criteria: If either stop criteria is met iteration will cease.
        #   1. Have we achieved an acceptable convergence? (change < d_cost)
        #   2. Have we exceeded the max number of iterations?
        if abs(cost[-1] - cost[-2]) < d_cost:
            stop_criteria = 1
            print('NMF: Cost function change is less than d_cost')
            print(count)
        elif count >= maxiter:
            stop_criteria = 1
            print('NMF: Number of iterations exceeded the maxmimum allowed')
            print(count)

        # If debug option is turned on output data every 25 iterations
        if debug and count % 25 == 0:
            timenow = datetime.now()
            timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
            # Reshape Spatial Abundances
            H_recon = np.reshape(np.transpose(H), (I,J,K))
            np.savez(debugpath + 'NMF_DebugOutput_' + timenow_str, W = W, H = H_recon, count = count)

    # Reformat cost function into standard format
    cost = np.reshape(cost, (1, count))

    # If debug option is turned on output final data before returning
    if debug:
        timenow = datetime.now()
        timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
        # Reshape Spatial Abundances
        H_recon = np.reshape(np.transpose(H), (I, J, K))
        np.savez(debugpath + 'NMF_DebugOutput_' + timenow_str + '_Final', W = W, H = H_recon,
                cost = cost, count = count)

    ret_dict = dict([('W',W), ('H',H), ('cost',cost), ('n_iter',count)])
    return ret_dict
##---------------------------------------------------------------------------##

def INMF(NMF_object):
    '''
    Informed Non-negative Matrix Factorization Routine
    Created On: Oct 20, 2016
    Last Modified: May, 4, 2018
    Author: Logan Wright, logan.wright@colorado.edu

    Description:
        An NMF routine using a multiplicative update rule for H and W. This
        routine adds smoothness and abundance-sum-to-one constraints.
            - Smoothness criteria are applied to spatial abundances (H) and
                endmember spectra (W)
            - An Abundance-Sum-to-One (ASO) constraints forces abundances to 1,
                allowing a fractional, or sum-of-parts interpretation
        Routine is based on the piecewise smoothness NMF from Jia and Qian,
        2009 Section IIIB
    Reference:
        Jia, S., and Y. Qian (2009), Constrained Nonnegative Matrix
        Factorization for Hyperspectral Unmixing, Geosci. Remote Sensing,
        IEEE Trans., 47(1), 161–173, doi:10.1109/TGRS.2008.2002882.

    Inputs:
        data, a numpy array of a hyperspectral data cube of the form [X by Y by Z] where the third (Z) dimension is the wavelength axis.
        init_w, spectral endmembers initial guess, numpy array, should have dimensions of [Z, K]
        init_h, spatial abundance initial guess, numpy array, should have dimensions of [X*Y,K]
        params, dictionary may include:
            delta = 100
            alpha = 1.0
            beta = 5.0
            yw = 0.1
            yh = 0.1
            window_w = [11, 11, 11, 11, 11]
            window_h = [5, 5, 5, 5, 11]
            K = 5
            maxiter = 500
    '''
    t1 = datetime.now()
    # Unpack NMF_object
    [I,J,K,L] = NMF_object.scenesize
    # maxiter = NMF_object.inputs['max_i']

    # Define Cost Function for Lee & Seung
    #   Cost function is Frobenius Norm of difference between A and W*H
    def cost_function(A,W,Af,Wf,H,g_W,g_H,alpha,beta):
        # Calculate the cost of the smoothness portion of the cost function
        cost2 = alpha * np.sum(g_W) + beta * np.sum(g_H)
        # Calculate the total cost function
        cost1 = np.linalg.norm(A - np.dot(W,H), ord = 'fro') + cost2
        # Calculate the ASO Component
        cost3 = np.linalg.norm(Af - np.dot(Wf,H), ord = 'fro') + cost2
        cost = [[cost1], [cost2], [cost3]]
        return cost

    # Check input data for consistency
    if np.min(np.mod(NMF_object.inputs['spatial_win'],2)) == 0:
        print('Window sizes MUST be Odd')

    # Define Ending Conditions, these are dependent on the data value
    #   Change end condition
    d_cost = 2e-10 * np.sum(NMF_object.scene.data_cube)
#    print('Minimum Change per Iteration End Condition: ',d_cost)

    # Flatten data to 2 dimensions
    I,J,L = NMF_object.scene.data_cube.shape
    # A = np.transpose(np.reshape(NMF_object.scene.data_cube,(I*J,L)))
    W1 = NMF_object.results.W

    # Initialize Data
    P = I*J
    count = 0           # Iteration Counter
    stop_criteria = 0   # Stop_Criteria Flag, stops iterations when equal to 1
    stable_flag = 0     # Stable_Flag, determines when spectral iteration begins

    # Load Irradiance Data and Create Fit Function for Fixed Atmosphere section
    irrad = np.genfromtxt('/Users/wrightad/Documents/MODTRAN/DATA/SUN01med2irradwnNormt.dat',skip_header = 3)
    # Convert to wavelength increments
    irrad[:,0] = 1e7 / irrad[:,0]
    irrad[:,1] = irrad[:,1] * (1 / (irrad[:,0] ** 2)) * 1e11
    irrad = irrad[0:-1,:]
    # Convolve to NEON Wavelength
    I0 = super_resample.super_resample(irrad[:,1], irrad[:,0], NMF_object.scene.resp_func['wvl'], NMF_object.scene.resp_func['fwhm']) * 1000
    tr = nmf_eval.bodhaine(NMF_object.scene.resp_func['wvl']/1000)
    fit = np.reshape(I0*tr,(1,-1))

    # Preallocate indices and Arrays for finding smoothness parameters
    g_W = np.zeros((K,1))
    h_W = np.zeros((L,K))
    gp_W = np.zeros((L,K))
    g_H = np.zeros((K,1))
    h_H = np.zeros((K,I*J))
    gp_H = np.zeros((K,I*J))

    indexW0 = list(range(K))
    indexW = list(range(K))

    indexH0 = list(range(K))
    indexH = list(range(K))

    for k in range(K):
        # Create Indices for Spectral Endmembers
        indexW0[k] = np.reshape(np.matlib.repmat(np.arange(0, L, 1), NMF_object.inputs['spectral_win'][k] - 1, 1), (1, L * (NMF_object.inputs['spectral_win'][k] - 1)), order = 'F')
        modw = np.arange(-(NMF_object.inputs['spectral_win'][k] // 2), (NMF_object.inputs['spectral_win'][k] / 2), dtype = int)
        modw = np.delete(modw,NMF_object.inputs['spectral_win'][k] // 2)
        indexW[k] = indexW0[k] + np.matlib.repmat(modw, 1, L)
        # Replace out of range values with center value (indexW0 value)
        outofrange = np.logical_or(indexW[k] < 0, indexW[k] >= L)
        indexW[k][outofrange] = indexW0[k][outofrange]

        # Create Indices for Spatial Abundances
        n1 = (NMF_object.inputs['spatial_win'][k] // 2)
        n2 = ((NMF_object.inputs['spatial_win'][k] ** 2) // 2) # Number of points in Neighborhood

        temp_HI0 = np.matlib.repmat(np.arange(0,I,1, dtype = int), n2, J)
        temp_HJ0 = np.transpose(np.reshape(np.matlib.repmat(np.arange(0, J, 1, dtype = int), I, n2), (I * J, n2), order = 'F'))

        struct = scimg.iterate_structure(scimg.generate_binary_structure(2, 1), n1)
        indices = np.asarray(np.nonzero(struct)) - n1;
        indices = np.delete(indices, n2/2, axis = 1)

        modi = np.reshape(indices[0,:],(n2,1))
        modj = np.reshape(indices[1,:],(n2,1))

        temp_HI = np.add(temp_HI0,modi)
        outofrange = np.logical_or(temp_HI < 0,temp_HI >= I)
        temp_HI[outofrange] = temp_HI0[outofrange]

        temp_HJ = np.add(temp_HJ0,modj)
        outofrange = np.logical_or(temp_HJ < 0,temp_HJ >= J)
        temp_HJ[outofrange] = temp_HJ0[outofrange]

        # Change 2D Matric Subscripts to Linear Indicies
        indexH[k] = np.ravel_multi_index((temp_HI, temp_HJ), (I,J))
        indexH0[k] = np.ravel_multi_index((temp_HI0, temp_HJ0), (I,J))

    # Clear the large placeholding variables
    del(temp_HI,temp_HI0,temp_HJ,temp_HJ0)

    # Calculate Initial Cost
    for k in range(K):
        Wdiff = NMF_object.results.W[indexW0[k], k] - NMF_object.results.W[indexW[k], k]
        g_W[k] = np.sum(-np.exp((-(Wdiff) ** 2) / NMF_object.inputs['spectral_gamma']) + 1)

        Hdiff = NMF_object.results.H[k,indexH0[k]] - NMF_object.results.H[k,indexH[k]]
        g_H[k] = np.sum(-np.exp((-(Hdiff) ** 2) / NMF_object.inputs['spatial_gamma']) + 1)
    Wf = np.vstack((NMF_object.results.W, NMF_object.inputs['delta_vec']))
    Af = np.vstack((NMF_object.results.A, NMF_object.inputs['delta'] * np.ones((1,P))))

    cost = cost_function(NMF_object.results.A, NMF_object.results.W, Af, Wf, NMF_object.results.H, g_W, g_H, NMF_object.inputs['spectral_strength'], NMF_object.inputs['spatial_strength'])

    print('INMF Initialization Time:',datetime.now()-t1)

    # Update Algorithm While Loop
    while stop_criteria == 0:
        t2 = datetime.now()
        # Upated W and A with the ASO constraint row.
        Wf = np.vstack((NMF_object.results.W, NMF_object.inputs['delta_vec']))
        Af = np.vstack((NMF_object.results.A, NMF_object.inputs['delta'] * np.ones((1,P))))

        for k in range(K):
            # Generate Spectral Smoothness Functions
            Wdiff = NMF_object.results.W[indexW0[k], k] - NMF_object.results.W[indexW[k], k]
            g_W[k] = np.sum(-np.exp((-(Wdiff) ** 2) / NMF_object.inputs['spectral_gamma']) + 1)
            h_W_temp = 2 / NMF_object.inputs['spectral_gamma'] * np.reshape(np.exp((-(Wdiff) ** 2) / NMF_object.inputs['spectral_gamma']), (NMF_object.inputs['spectral_win'][k] - 1, L))
            h_W[:,k] = np.sum(h_W_temp, 0)
            gp_W_temp = 2 / NMF_object.inputs['spectral_gamma'] *  np.reshape(Wdiff * np.exp((-(Wdiff) ** 2) / NMF_object.inputs['spectral_gamma']), (NMF_object.inputs['spectral_win'][k] - 1, L))
            gp_W[:,k] = np.sum(gp_W_temp, 0)

            # Generate Spatial Smoothness Functions
            Hdiff = NMF_object.results.H[k,indexH0[k]] - NMF_object.results.H[k,indexH[k]]
            g_H[k] = np.sum(-np.exp((-(Hdiff) ** 2) / NMF_object.inputs['spatial_gamma']) + 1)
            h_H_temp = np.exp((-(Hdiff) ** 2) / NMF_object.inputs['spatial_gamma'])
            h_H[k,:] = 2 / NMF_object.inputs['spatial_gamma'] * np.sum(h_H_temp, 0)
            gp_H_temp = Hdiff * np.exp((-(Hdiff) ** 2) / NMF_object.inputs['spatial_gamma'])
            gp_H[k,:] = 2 / NMF_object.inputs['spatial_gamma'] * np.sum(gp_H_temp, 0)

        # Calculate Cost Function before update (i)
        if count != 0:
            cost = np.append(cost, cost_function(NMF_object.results.A, NMF_object.results.W, Af, Wf, NMF_object.results.H, g_W, g_H, NMF_object.inputs['spectral_strength'], NMF_object.inputs['spatial_strength']), axis = 1)

            # Stop Criteria: If either stop criteria is met iteration will cease.
            #   1. Have we achieved an acceptable convergence? (change < d_cost)
            #   2. Have we exceeded the max number of iterations?
            if abs(cost[0,-1] - cost[0,-2]) < d_cost:
                stop_criteria = 1
                print('PSNMF: Cost function change is less than d_cost',count)
            elif count >= NMF_object.inputs['max_i']:
                stop_criteria = 1
                print('PSNMF: Number of iterations exceeded the maxmimum allowed',count)

        NMF_object.results.H = NMF_object.results.H * (np.dot(np.transpose(Wf), Af) + NMF_object.inputs['spatial_strength'] * (NMF_object.results.H * h_H - gp_H)) / (np.dot(np.dot(np.transpose(Wf), Wf),NMF_object.results.H) + NMF_object.inputs['spatial_strength'] * NMF_object.results.H * h_H + 10e-9) # Spatial Abundances

        # Check if stable
        if stable_flag == 0 and count >= 5 and (cost[0,-1] - cost[0,-2]) < 0:
                stable_flag = 1

        if stable_flag == 1:
            NMF_object.results.W = NMF_object.results.W * (np.dot(NMF_object.results.A, np.transpose(NMF_object.results.H)) + NMF_object.inputs['spectral_strength'] * (NMF_object.results.W * h_W - gp_W)) / (np.dot(np.dot(NMF_object.results.W,NMF_object.results.H), np.transpose(NMF_object.results.H)) + NMF_object.inputs['spectral_strength'] * NMF_object.results.W * h_W + 10e-9) # Spectral Endmembers
            a = nmf_eval.scattering_fit(NMF_object.results.W[:,-1], fit)

            NMF_object.results.W[:,-1] = a
        count += 1 # Update iteration counter

        t3 = datetime.now()
        print(t3-t2)
    # NMF_object.results.W = W
    # NMF_object.results.H = H
    NMF_object.results.cost = cost
    NMF_object.results.iter = count

    # return ret_dict
##---------------------------------------------------------------------------##
