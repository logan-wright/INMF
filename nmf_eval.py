#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:35:49 2017

@author: wrightad
"""

import numpy as N
import matplotlib.pyplot as plt

def rmse(v1,v2):
    '''
    rmse(v1,v2) - Calculates the root mean square error between two vectors 
    Version 1.0
    Created On: Apr 17, 2017
    Last Modified: Jun 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu
    
    Description:
        - Calculates the Root-Mean-Square-Error between two vectors
        - Vectors must be the same length
          
    Inputs:
        v1 - Numpy 1-dimensional array of arbitrary length
        v2 - Numpy 1-dimensional array with a length equal to that of v1
    Output:        
        rmse, the rmse value for the comparison of the two vectors
    ''' 
    dims1 = v1.shape
    dims2 = v2.shape
    if dims1 == dims2:
        diff = v1 - v2
        err = N.sum(diff**2)/dims1[0]
        rms = N.sqrt(err)
    else:
        print('Dimension Mismatch: v1.shape ~= v2.shape!')
        rms = None
        
    return rms

def sid(v1,v2):
    '''
    sid(v1,v2) - Calculates the spectral information divergence (SID) between
                 two vectors 
    Version 1.0
    Created On: Apr 17, 2017
    Last Modified: Jun 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu
    
    Description:
        - Calculates the Spectral Information Divergence between two vectors
        - Vectors must be the same length
        
    Reference:
        Chang, C.-I. (2000), An information-theoretic approach to spectral 
        variability, similarity, and discrimination for hyperspectral image 
        analysis, Inf. Theory, IEEE Trans., 46(5), 1927–1932, 
        doi:10.1109/18.857802.
        
    Inputs:
        v1 - Numpy 1-dimensional array of arbitrary length
        v2 - Numpy 1-dimensional array with a length equal to that of v1
    Output:        
        SID, the SID value for the comparison of the two vectors
    ''' 
    p = v1 / N.sum(v1)
    q = v2 / N.sum(v2)
    
    D1 = N.sum(p * N.log(p / q))    
    D2 = N.sum(q * N.log(q / p))
    
    D_sum = D1 + D2
    
    return D_sum

def scattering_fit(data, function, sigma = 1e-9):
    '''
    Linear least-squares fit for a function of the form y = a * f(x)
    Version 1.0
    Created On: Apr 17, 2017
    Last Modified: Apr 17, 2017
    Author: Logan Wright, logan.wright@colorado.edu
    
    Description:
        
    Reference:
    
    Inputs:
        wvl, wavelength in NANOMETERS, must be same length as data and function
        data, the y data that the function is to be fit to. Should be a vector 
            (N,) or a 2D array with one single dimension.
        function, the function to be scaled with a linear factor to fit the 
            data. Again it should be a vector (N,) or a 2D array with one 
            single dimension. data and function must be the same length.
        OPTIONAL:
        sigma, the value small value that determines when iteration stops
        
    Output:        
        a, a single scalar describing the best-fit value of "a" 
    ''' 
    # Initialize parametrs, including change and the initial minimum
    change = 100    # Arbitrary value greater than sigma
    minval = N.sum((data - function) ** 2)  # Initial Min
    
    # Calculate the intial multiplicative factor between the data and function,
    #   and use to set range for calculating minimums
    Amin = 0
    Amax = (data/function).max()
    
    # Iterate
    while change > sigma:
        # Create Array of Amplitudes for the fit
        Arr = N.linspace(Amin,Amax,100)
        Test = N.matmul(N.reshape(Arr,(-1,1)),function)
        
        # Calculate the square difference between the data and the fit guess
        diff = Test - N.matlib.repmat(N.reshape(data,(1,-1)),100,1)
    
        # Find Minimum, and calculate the change and difference.
        val = N.sum(diff ** 2, axis = 1)
        vali = N.argmin(val)
        change = minval - val.min()
        minval = val.min()
    
        # Calculate New range of "a" for next iteration
        Amin = Arr[max(vali-2,0)]
        Amax = Arr[min(vali+2,len(Arr)-1)]
    result = N.squeeze(Arr[vali] * function)
    return result

def bodhaine(wvl):
    '''
    bodhaine(wvl) - Calculates the Bodhaine aproximation of rayleigh optical depth
    
    Version 1.0
    Created On: Apr 17, 2017
    Last Modified: June 14, 2017
    Author: Logan Wright, logan.wright@colorado.edu
    
    Description:
        
    Reference:
        Bodhaine, B. A., N. B. Wood, E. G. Dutton, and J. R. Slusser (1999),
        On Rayleigh optical depth calculations, J. Atmos. Ocean. Technol.,
        16(11 PART 2), 1854–1861, 
        doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2.
    
    Inputs:
        wvl - a vector of wavelengths at which to calculate the rayleigh optical
              depth. Wavelength sould be in MICROMETERS         
    Output:        
        tr - vector of rayleigh optical depths corresponding to wavelengths from the input vectora single scalar describing the best-fit value of "a" 
    '''
    s = 0.0021520
    a = 1.0455996
    b = 341.29061
    c = 0.90230850
    d = 0.0027059889 
    e = 85.968563
    
    tr = s * (a - b * wvl ** -2 - c * wvl ** 2)/(1 + d * wvl ** -2 - e * wvl ** 2)
    return tr