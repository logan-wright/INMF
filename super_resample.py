import numpy as N

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
