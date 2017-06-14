#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:41:47 2017
Last Modified on Feb, 20, 2017
@author: Logan Wright

"""
import numpy as N
import numpy.matlib
import scipy.ndimage as scimg
from datetime import datetime
import matplotlib.pyplot as plt
import nmf_eval

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
        cost = N.linalg.norm(A - N.dot(W, H), ord = 'fro')
        return cost
    
    # Define Ending Conditions, these are dependent on the data value
    #   Change end condition
    d_cost = 1
        
    # Flatten data to 2 dimensions
    I,J,L = data.shape
    A = N.transpose(N.reshape(data,(I * J, L)))
    
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
        H_recon = N.reshape(N.transpose(H), (I, J, K))
        N.savez(debugpath + 'NMF_DebugOutput_' + timenow_str, W = W, H = H_recon, count = count)
    
    # Lee & Seung Multiplicative Update Algorithm While Loop
    while stop_criteria == 0:
        # Update Spatial Abundances
        H = H * (N.dot(N.transpose(W), A)) / (N.dot(N.dot(N.transpose(W), W), H) + 10e-9)
        # Update Spectral Endmembers
        W = W * (N.dot(A, N.transpose(H))) / (N.dot(N.dot(W, H), N.transpose(H))+ 10e-9)
                       
        cost = N.append(cost,cost_function(A, W, H))
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
            H_recon = N.reshape(N.transpose(H), (I,J,K))
            N.savez(debugpath + 'NMF_DebugOutput_' + timenow_str, W = W, H = H_recon, count = count)
            
    # Reformat cost function into standard format
    cost = N.reshape(cost, (1, count))

    # If debug option is turned on output final data before returning
    if debug:
        timenow = datetime.now()
        timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
        # Reshape Spatial Abundances
        H_recon = N.reshape(N.transpose(H), (I, J, K))
        N.savez(debugpath + 'NMF_DebugOutput_' + timenow_str + '_Final', W = W, H = H_recon,
                cost = cost, count = count)
    
    ret_dict = dict([('W',W), ('H',H), ('cost',cost), ('n_iter',count)])
    return ret_dict
##---------------------------------------------------------------------------##

def PSNMF(data, W, H, resp_func, info, delta = 100., alpha = 0.5, beta = 0.1, yw = 0.01, ys = 0.5,
        windowW = [11,11,11,11,11], windowH = [5,5,5,5,11], K = 5,
        maxiter = 500, debug = False):
    '''
    Piecewise Smoothness Non-negative Matrix Factorization Routine
    Created By: Logan Wright
    Created On: Oct 20, 2016
    Last Modified: Feb, 20, 2017
            
    Description:
        An NMF routine using a multiplicative update rule for H and W. This 
        routine adds smoothness and abundance-sum-to-one constraints.
            - Smoothness criteria are applied to spatial abundances (H) and
                endmember spectra (W)
            - An Abundance-Sum-to-One (ASO) constraints forces abundances to 1,
                allowing a fractional, or sum-of-parts interpretation
        Routine is based on Jia and Qian, 2009 Section IIIB
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
    # Define Cost Function for Lee & Seung
    #   Cost function is Frobenius Norm of difference between A and W*H
    def cost_function(A,W,Af,Wf,H,g_W,g_H,alpha,beta):
        # Calculate the cost of the smoothness portion of the cost function
        cost2 = alpha * N.sum(g_W) + beta * N.sum(g_H)
        # Calculate the total cost function
        cost1 = N.linalg.norm(A - N.dot(W,H), ord = 'fro') + cost2
        # Calculate the ASO Component
        cost3 = N.linalg.norm(Af - N.dot(Wf,H), ord = 'fro') + cost2
        cost = [[cost1], [cost2], [cost3]]
        return cost
        
    # Check input data for consistency
    if N.min(N.mod(windowH,2)) == 0:
        print('Window sizes MUST be Odd')
        
    # Define Ending Conditions, these are dependent on the data value
    #   Change end condition
    d_cost = 2e-10 * N.sum(data)
#    print('Minimum Change per Iteration End Condition: ',d_cost)
        
    # Flatten data to 2 dimensions
    I,J,L = data.shape
    A = N.transpose(N.reshape(data,(I*J,L)))
    W1 = W
    
    # Initialize Data
    P = I*J
    count = 0           # Iteration Counter
    stop_criteria = 0   # Stop_Criteria Flag, stops iterations when equal to 1
    stable_flag = 0     # Stable_Flag, determines when spectral iteration begins
    
    # Load Irradiance Data and Create Fit Function for Fixed Atmosphere section
    irrad = N.genfromtxt('/home/wrightla/../../data/wrightla/MODTRAN/SUN01med2irradwnNormt.dat',skip_header = 3)
    # Convert to wavelength increments
    irrad[:,0] = 1e7 / irrad[:,0]
    irrad[:,1] = irrad[:,1] * (1 / (irrad[:,0] ** 2)) * 1e11
    irrad = irrad[0:-1,:]
    # Convolve to NEON Wavelength
    I0 = nmf_eval.super_resample(irrad[:,1], irrad[:,0], resp_func['wvl'], resp_func['fwhm']) * 1000
    tr = nmf_eval.bodhaine(resp_func['wvl']/1000)
    fit = N.reshape(I0*tr,(1,-1))

    # Preallocate indices and Arrays for finding smoothness parameters
    g_W = N.zeros((K,1))
    h_W = N.zeros((L,K))
    gp_W = N.zeros((L,K))
    g_H = N.zeros((K,1))
    h_H = N.zeros((K,I*J))
    gp_H = N.zeros((K,I*J))

    indexW0 = list(range(K))
    indexW = list(range(K))
    
    indexH0 = list(range(K))
    indexH = list(range(K))    
    
    for k in range(K):
        # Create Indices for Spectral Endmembers
        indexW0[k] = N.reshape(N.matlib.repmat(N.arange(0, L, 1), windowW[k] - 1, 1), (1, L * (windowW[k] - 1)), order = 'F')
        modw = N.arange(-(windowW[k] // 2), (windowW[k] / 2), dtype = int)
        modw = N.delete(modw,windowW[k] // 2)
        indexW[k] = indexW0[k] + N.matlib.repmat(modw, 1, L)
        # Replace out of range values with center value (indexW0 value)
        outofrange = N.logical_or(indexW[k] < 0, indexW[k] >= L)        
        indexW[k][outofrange] = indexW0[k][outofrange]

        # Create Indices for Spatial Abundances
        n1 = (windowH[k] // 2)
        n2 = ((windowH[k] ** 2) // 2) # Number of points in Neighborhood
        
        temp_HI0 = N.matlib.repmat(N.arange(0,I,1, dtype = int), n2, J)       
        temp_HJ0 = N.transpose(N.reshape(N.matlib.repmat(N.arange(0, J, 1, dtype = int), I, n2), (I * J, n2), order = 'F'))

        struct = scimg.iterate_structure(scimg.generate_binary_structure(2, 1), n1)
        indices = N.asarray(N.nonzero(struct)) - n1;
        indices = N.delete(indices, n2/2, axis = 1)
        
        modi = N.reshape(indices[0,:],(n2,1))
        modj = N.reshape(indices[1,:],(n2,1))
        
        temp_HI = N.add(temp_HI0,modi)
        outofrange = N.logical_or(temp_HI < 0,temp_HI >= I)
        temp_HI[outofrange] = temp_HI0[outofrange]
    
        temp_HJ = N.add(temp_HJ0,modj)
        outofrange = N.logical_or(temp_HJ < 0,temp_HJ >= J)
        temp_HJ[outofrange] = temp_HJ0[outofrange]

        # Change 2D Matric Subscripts to Linear Indicies
        indexH[k] = N.ravel_multi_index((temp_HI, temp_HJ), (I,J))
        indexH0[k] = N.ravel_multi_index((temp_HI0, temp_HJ0), (I,J))

    # Clear the large placeholding variables
    del(temp_HI,temp_HI0,temp_HJ,temp_HJ0)
        
    # Calculate Initial Cost
    for k in range(K):
        Wdiff = W[indexW0[k], k] - W[indexW[k], k]
        g_W[k] = N.sum(-N.exp((-(Wdiff) ** 2) / yw) + 1)
    
        Hdiff = H[k,indexH0[k]] - H[k,indexH[k]]
        g_H[k] = N.sum(-N.exp((-(Hdiff) ** 2) / ys) + 1)
    
    Wf = N.vstack((W, N.append(delta * N.ones((1,K-1)),0)))
    Af = N.vstack((A, delta * N.ones((1,P))))
    
    cost = cost_function(A, W, Af, Wf, H, g_W, g_H, alpha, beta)
    
    # If debug option is turned on output initial data
    if debug:
        debugpath = '/Users/wrightad/Documents/Data/NMF_DebugFiles/'
    
    # Update Algorithm While Loop
    while stop_criteria == 0:
        # Upated W and A with the ASO constraint row.
        Wf = N.vstack((W, N.append(delta * N.ones((1,K-1)),0)))
        Af = N.vstack((A, delta * N.ones((1,P))))
        
        for k in range(K):
            # Generate Spectral Smoothness Functions
            Wdiff = W[indexW0[k], k] - W[indexW[k], k]
            g_W[k] = N.sum(-N.exp((-(Wdiff) ** 2) / yw) + 1)
            h_W_temp = 2 / yw * N.reshape(N.exp((-(Wdiff) ** 2) / yw), (windowW[k] - 1, L))
            h_W[:,k] = N.sum(h_W_temp, 0)
            gp_W_temp = 2 / yw *  N.reshape(Wdiff * N.exp((-(Wdiff) ** 2) / yw), (windowW[k] - 1, L))
            gp_W[:,k] = N.sum(gp_W_temp, 0)
        
            # Generate Spatial Smoothness Functions
            Hdiff = H[k,indexH0[k]] - H[k,indexH[k]]
            g_H[k] = N.sum(-N.exp((-(Hdiff) ** 2) / ys) + 1)
            h_H_temp = N.exp((-(Hdiff) ** 2) / ys)
            h_H[k,:] = 2 / ys * N.sum(h_H_temp, 0)
            gp_H_temp = Hdiff * N.exp((-(Hdiff) ** 2) / ys)
            gp_H[k,:] = 2 / ys * N.sum(gp_H_temp, 0)  
        
        # Calculate Cost Function before update (i)
        if count != 0:
            cost = N.append(cost, cost_function(A, W, Af, Wf, H, g_W, g_H, alpha, beta), axis = 1)
                            
            # Stop Criteria: If either stop criteria is met iteration will cease.
            #   1. Have we achieved an acceptable convergence? (change < d_cost)
            #   2. Have we exceeded the max number of iterations?
            if abs(cost[0,-1] - cost[0,-2]) < d_cost:
                stop_criteria = 1
                print('PSNMF: Cost function change is less than d_cost',count)
            elif count >= maxiter:
                stop_criteria = 1
                print('PSNMF: Number of iterations exceeded the maxmimum allowed',count)
        
        # If debug option is turned on output data every 25 iterations
        if debug and count % 25 == 0:
            timenow = datetime.now()
            timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
            # Reshape Spatial Abundances
            H_recon = N.reshape(N.transpose(H), (I,J,K))
            N.savez(debugpath + 'NMF_DebugOutput_' + timenow_str, W = W, H = H_recon, count = count)
        
        H = H * (N.dot(N.transpose(Wf), Af) + beta * (H * h_H - gp_H)) / (N.dot(N.dot(N.transpose(Wf), Wf),H) + beta * H * h_H + 10e-9) # Spatial Abundances

        # Check if stable
        if stable_flag == 0 and count >= 5 and (cost[0,-1] - cost[0,-2]) < 0:
                stable_flag = 1

        if stable_flag == 1:
            W = W * (N.dot(A, N.transpose(H)) + alpha * (W * h_W - gp_W)) / (N.dot(N.dot(W,H), N.transpose(H)) + alpha * W * h_W + 10e-9) # Spectral Endmembers
            a = nmf_eval.scattering_fit(W[:,-1], fit)

            W[:,-1] = a
        count += 1 # Update iteration counter  
    
    # If debug option is turned on output final data before returning
    if debug:
        timenow = datetime.now()
        timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
        # Reshape Spatial Abundances
        H_recon = N.reshape(N.transpose(H), (I, J, K))
        N.savez(debugpath + 'NMF_DebugOutput_' + timenow_str + '_Final', W = W, H = H_recon,
                cost = cost, count = count)
        
    ret_dict = dict([('W',W), ('H',H), ('cost',cost), ('n_iter',count), 
                     ('wavelengths',resp_func['wvl']), ('dims',info['dims']), 
                     ('datacube',data), ('fname',info['fname']), 
                     ('titles',info['titles']), ('W1',W1)])
    return ret_dict
##---------------------------------------------------------------------------##



def PSnsNMF(data, W, H, delta = 100., alpha = 0.5, beta = 0.1, yw = 0.01, ys = 0.5,
        windowW = [11,11,11,11,11], windowH = [5,5,5,5,11], K = 5, theta = 0.2,
        maxiter = 500, debug = False):
    '''
    Piecewise Smoothness Nonsmooth Non-negative Matrix Factorization Routine
    Created By: Logan Wright
    Created On: Feb 20, 2016
    Last Modified: Feb, 20, 2017
            
    Description:
        An NMF routine using a multiplicative update rule for H and W. This 
        routine adds smoothness and abundance-sum-to-one constraints.
            - Smoothness criteria are applied to spatial abundances (H) and
                endmember spectra (W)
            - An Abundance-Sum-to-One (ASO) constraints forces abundances to 1,
                allowing a fractional, or sum-of-parts interpretation
            - A "smoothing matrix" based sparsity term described 
        Routine is based on Jia and Qian, 2009 Section IIIC
    References:
        Jia, S., and Y. Qian (2009), Constrained Nonnegative Matrix 
        Factorization for Hyperspectral Unmixing, Geosci. Remote Sensing, 
        IEEE Trans., 47(1), 161–173, doi:10.1109/TGRS.2008.2002882.
        
        Pascual-Montano, A., J. M. Carazo, K. Kochi, D. Lehmann,
        and R. D. Pascual-Marqui (2006), Nonsmooth nonnegative matrix
        factorization, Pami, 28(3), 403–15, doi:10.1109/TPAMI.2006.60.
        
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
    # Define Cost Function for Lee & Seung
    #   Cost function is Frobenius Norm of difference between A and W*H
    def cost_function(A,W,Af,Wf,H,C,g_W,g_H,alpha,beta):
        # Calculate the cost of the smoothness portion of the cost function
        cost2 = alpha * N.sum(g_W) + beta * N.sum(g_H)
        # Calculate the total cost function
        cost1 = N.linalg.norm(A - N.dot(N.dot(W,C),H), ord = 'fro') + cost2         
        # Calculate the ASO Component
        cost3 = N.linalg.norm(Af - N.dot(N.dot(Wf,C),H), ord = 'fro') + cost2
        cost = [[cost1], [cost2], [cost3]]
        return cost
        
    # Check input data for consistency
    if N.min(N.mod(windowH,2)) == 0:
        print('Window sizes MUST be Odd')
        
    # Define Ending Conditions, these are dependent on the data value
    #   Change end condition
    d_cost = 1
        
    # Flatten data to 2 dimensions
    I,J,L = data.shape
    A = N.transpose(N.reshape(data,(I*J,L)))
    
    # Initialize Data
    P = I*J
    count = 0           # Iteration Counter
    stop_criteria = 0   # Stop_Criteria Flag, stops iterations when equal to 1
    stable_flag = 0     # Stable_Flag, determines when spectral iteration begins

    # Preallocate indices and Arrays for finding smoothness parameters
    g_W = N.zeros((K,1))
    h_W = N.zeros((L,K))
    gp_W = N.zeros((L,K))
    g_H = N.zeros((K,1))
    h_H = N.zeros((K,I*J))
    gp_H = N.zeros((K,I*J))

    indexW0 = list(range(K))
    indexW = list(range(K))
    
    indexH0 = list(range(K))
    indexH = list(range(K))    
    
    for k in range(K):
        # Create Indices for Spectral Endmembers
        indexW0[k] = N.reshape(N.matlib.repmat(N.arange(0, L, 1), windowW[k] - 1, 1), (1, L * (windowW[k] - 1)), order = 'F')
        modw = N.arange(-(windowW[k] // 2), (windowW[k] / 2), dtype = int)
        modw = N.delete(modw,windowW[k] // 2)
        indexW[k] = indexW0[k] + N.matlib.repmat(modw, 1, L)
        # Replace out of range values with center value (indexW0 value)
        outofrange = N.logical_or(indexW[k] < 0, indexW[k] >= L)        
        indexW[k][outofrange] = indexW0[k][outofrange]

        # Create Indices for Spatial Abundances
        n1 = (windowH[k] // 2)
        n2 = ((windowH[k] ** 2) // 2) # Number of points in Neighborhood
        
        temp_HI0 = N.matlib.repmat(N.arange(0,I,1, dtype = int), n2, J)       
        temp_HJ0 = N.transpose(N.reshape(N.matlib.repmat(N.arange(0, J, 1, dtype = int), I, n2), (I * J, n2), order = 'F'))

        struct = scimg.iterate_structure(scimg.generate_binary_structure(2, 1), n1)
        indices = N.asarray(N.nonzero(struct)) - n1;
        indices = N.delete(indices, n2/2, axis = 1)
        
        modi = N.reshape(indices[0,:],(n2,1))
        modj = N.reshape(indices[1,:],(n2,1))
        
        temp_HI = N.add(temp_HI0,modi)
        outofrange = N.logical_or(temp_HI < 0,temp_HI >= I)
        temp_HI[outofrange] = temp_HI0[outofrange]
    
        temp_HJ = N.add(temp_HJ0,modj)
        outofrange = N.logical_or(temp_HJ < 0,temp_HJ >= J)
        temp_HJ[outofrange] = temp_HJ0[outofrange]

        # Change 2D Matric Subscripts to Linear Indicies
        indexH[k] = N.ravel_multi_index((temp_HI, temp_HJ), (I,J))
        indexH0[k] = N.ravel_multi_index((temp_HI0, temp_HJ0), (I,J))

    # Clear the large placeholding variables
    del(temp_HI,temp_HI0,temp_HJ,temp_HJ0)
        
    # Calculate Initial Cost
    for k in range(K):
        Wdiff = W[indexW0[k], k] - W[indexW[k], k]
        g_W[k] = N.sum(-N.exp((-(Wdiff) ** 2) / yw) + 1)
    
        Hdiff = H[k,indexH0[k]] - H[k,indexH[k]]
        g_H[k] = N.sum(-N.exp((-(Hdiff) ** 2) / ys) + 1)
    
    Wf = N.vstack((W, N.append(delta * N.ones((1,K-1)),0)))
    Af = N.vstack((A, delta * N.ones((1,P))))
    C = (1 - theta) * N.identity(K) + (theta / P) * N.ones((K,K))
    
    cost = cost_function(A, W, Af, Wf, H, C, g_W, g_H, alpha, beta)
    
    # If debug option is turned on set debugoutput file path
    if debug:
        debugpath = '/Users/wrightad/Documents/Data/NMF_DebugFiles/'
    
    # Update Algorithm While Loop
    while stop_criteria == 0:
        # Upated W and A with the ASO constraint row.
        Wf = N.vstack((W, N.append(delta * N.ones((1,K-1)),0)))
        Af = N.vstack((A, delta * N.ones((1,P))))
        
        for k in range(K):
            # Generate Spectral Smoothness Functions
            Wdiff = W[indexW0[k], k] - W[indexW[k], k]
            g_W[k] = N.sum(-N.exp((-(Wdiff) ** 2) / yw) + 1)
            h_W_temp = 2 / yw * N.reshape(N.exp((-(Wdiff) ** 2) / yw), (windowW[k] - 1, L))
            h_W[:,k] = N.sum(h_W_temp, 0)
            gp_W_temp = 2 / yw *  N.reshape(Wdiff * N.exp((-(Wdiff) ** 2) / yw), (windowW[k] - 1, L))
            gp_W[:,k] = N.sum(gp_W_temp, 0)
        
            # Generate Spatial Smoothness Functions
            Hdiff = H[k,indexH0[k]] - H[k,indexH[k]]
            g_H[k] = N.sum(-N.exp((-(Hdiff) ** 2) / ys) + 1)
            h_H_temp = N.exp((-(Hdiff) ** 2) / ys)
            h_H[k,:] = 2 / ys * N.sum(h_H_temp, 0)
            gp_H_temp = Hdiff * N.exp((-(Hdiff) ** 2) / ys)
            gp_H[k,:] = 2 / ys * N.sum(gp_H_temp, 0)  
        
        # Generate Smoothness Matrix, C
        C = (1 - theta) * N.identity(K) + (theta / P) * N.ones((K,K))
        
        # Calculate Cost Function before update (i)
        if count != 0:
            cost = N.append(cost, cost_function(A, W, Af, Wf, H, C, g_W, g_H, alpha, beta), axis = 1)
                            
            # Stop Criteria: If either stop criteria is met iteration will cease.
            #   1. Have we achieved an acceptable convergence? (change < d_cost)
            #   2. Have we exceeded the max number of iterations?
            if abs(cost[0,-1] - cost[0,-2]) < d_cost:
                stop_criteria = 1
                print('PSnsNMF: Cost function change is less than d_cost',count)
            elif count >= maxiter:
                stop_criteria = 1
                print('PSnsNMF: Number of iterations exceeded the maxmimum allowed',count)
        
        # If debug option is turned on output data every 25 iterations
        if debug and count % 25 == 0:
            timenow = datetime.now()
            timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
            # Reshape Spatial Abundances
            H_recon = N.reshape(N.transpose(H), (I,J,K))
            N.savez(debugpath + 'NMF_DebugOutput_' + timenow_str, W = W, H = H_recon, count = count)
        
        H = H * (N.dot(N.transpose(N.dot(Wf,C)), Af) + beta * (H * h_H - gp_H)) / (N.dot(N.dot(N.dot(N.transpose(N.dot(Wf,C)), Wf), C), H) + beta * H * h_H + 10e-9) # Spatial Abundances

        # Check if stable
        if stable_flag == 0 and count >= 5 and (cost[0,-1] - cost[0,-2]) < 0:
                stable_flag = 1
        
        if stable_flag == 1:
            W = W * (N.dot(A, N.transpose(H)) + alpha * (W * h_W - gp_W)) / (N.dot(N.dot(W,H), N.transpose(H)) + alpha * W * h_W + 10e-9) # Spectral Endmembers
    
        count += 1 # Update iteration counter  
    # If debug option is turned on output final data before returning
    if debug:
        timenow = datetime.now()
        timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
        # Reshape Spatial Abundances
        H_recon = N.reshape(N.transpose(H), (I, J, K))
        N.savez(debugpath + 'NMF_DebugOutput_' + timenow_str + '_Final', W = W, H = H_recon,
                cost = cost, count = count)
        
    ret_dict = dict([('W',W), ('H',H), ('cost',cost), ('n_iter',count)])
    return ret_dict
