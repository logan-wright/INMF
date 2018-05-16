import numpy as np
import scipy.io as sio
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_nmf(NMF_object, cmaps = dict([('Asphalt','Greys'),
                                                        ('Concrete', 'Greys'),
                                                        ('Snow','Greys'),
                                                        ('Soil', 'Oranges'),
                                                        ('Vegetation','Greens'),
                                                        ('Water', 'Blues'),
                                                        ('Cloud','RdPu'),
                                                        ('Atmosphere', 'Purples')])):
    """
    nmf_output.plot_nmf() - plots and saves figures based on the output of the
                             INMF algorithm

    Version 1.1
    Created on: Mar, 16, 2017
    Last Modified: May, 16, 2016
    Author: Logan Wright, logan.wright@colorado.edu

    Version 1.1 Update Notes:
        Function updated to accept NMF_object and be more pythonic

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
    # Get Size of array
    I, J, K, L = NMF_object.scenesize

    # Plot Cost Function Descent
    plt.figure('cost')
    labels = ['Default','Smoothing','ASO']
    # for i in range(len(NMF_object.results.cost)):
    plt.plot(NMF_object.results.cost[i,:])
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost Function Value')
    plt.legend(labels, loc = 'best')
    plt.savefig(NMF_object.inputs['name'] + '_cost.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)
    plt.close('cost')

    plt.figure('soln',figsize = (5.3,4))
    plt.xlabel('Wavelength [$nm$]')
    plt.ylabel('Radiance [$W m^{-2} sr^{-1} \mu m^{-1}$]')
    for i in range(K):
        endmember_color = mpl.cm.get_cmap(cmaps[NMF_object.endmembers['titles'][i]],128)
        plt.plot(NMF_object.scene.resp_func['wvl'],NMF_object.endmembers['spectra'][:,i],linestyle = '-',linewidth = 2, color = endmember_color(0.75),label = NMF_object.endmembers['titles'][i])

    x01,xn1,y01,yn1 = plt.axis()
    plt.legend(ncol = 2, columnspacing = 1, handletextpad = 0)

    plt.figure('init',figsize = (5.3,4))
    plt.xlabel('Wavelength [$nm$]')
    plt.ylabel('Radiance [$W m^{-2} sr^{-1} \mu m^{-1}$]')
    for i in range(K):
        endmember_color = mpl.cm.get_cmap(cmaps[NMF_object.endmembers['titles'][i]],128)
        plt.plot(NMF_object.scene.resp_func['wvl'],NMF_object.results.W[:,i],linestyle = '-',linewidth = 2, color = endmember_color(0.75), label = NMF_object.endmembers['titles'][i])
    x02,xn2,y02,yn2 = plt.axis()
    plt.legend(ncol = 2, columnspacing = 1, handletextpad = 0)

    if yn2 > yn1:
        plt.axis([390,950,0,yn2])
        plt.savefig(NMF_object.inputs['name'] + '_init_endmembers.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)

        plt.figure('soln')
        plt.axis([390,950,0,yn2])
        plt.savefig(NMF_object.inputs['name'] + '_soln_endmembers.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)

    else:
        plt.axis([390,950,0,yn1])
        plt.savefig(NMF_object.inputs['name'] + '_init_endmembers.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)

        plt.figure('soln')
        plt.axis([390,950,0,yn1])
        plt.savefig(NMF_object.inputs['name'] + '_soln_endmembers.png', format = 'png', dpi = 300, bbox_inches = 'tight', transparent = True)

    plt.close('soln')
    plt.close('init')

    for i in range(K):
        plt.figure(figsize = ((J/I*10),11.9))
        cax = plt.contourf(np.fliplr(NMF_object.results.H[:,:,i]), cmap = cmaps[NMF_object.endmembers['titles'][i]], levels = np.linspace(0,0.8,num = 100), extend = 'max')
        plt.xticks([],[])
        plt.yticks([],[])
        cbar = plt.colorbar(cax, ticks = [0, 0.2, 0.4, 0.6, 0.8], orientation = 'horizontal', pad = 0.025, aspect = 7, shrink = 0.9)
        cbar.ax.tick_params(labelsize = 14)

        plt.savefig(timenow_str + '_' + NMF_object.inputs['name'] + '_' + NMF_object.endmembers['titles'][i] + '.png', format = 'png', dpi = 300,bbox_inches = 'tight', transparent = True)
        plt.close()

    # plt.figure('resid',figsize = ((J/I*10),10))
    # plt.contourf(output['residual'],cmap = 'inferno')
    # plt.colorbar()
    # plt.savefig(timenow_str + '_' + NMF_object.inputs['name'] + '_resid' + '.png', format = 'png', dpi = 300,bbox_inches = 'tight', transparent = True)
    # plt.close('resid')

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
                        ('wavelengths',np.squeeze(output['wavelengths'])),
                        ('cost',output['cost']),
                        ('initialW',output['W1'])])
    # Calculate Residual
    recon = np.reshape(np.dot(output['W'],output['H']), (output['dims'][0],output['dims'][1],output['dims'][2]))
    new_output['residual'] = np.sum(np.sqrt((output['datacube'] - recon) ** 2), 2)

    # Reshape Spatial Abundances
    new_output['abundances'] = np.reshape(np.transpose(output['H']), (output['dims'][0],output['dims'][1],output['dims'][3]))

    # Export Data as .mat files
    timenow = datetime.now()
    timenow_str = timenow.strftime('%Y_%m_%d_%H%M')
    filename_tosave = timenow_str + '_' + output['fname'] + '_RESULTS.mat'

    sio.savemat(filename_tosave, new_output)

    plot_nmf(new_output,output['W1'],output['fname'],output['titles'],output['dims'][3])

    return new_output
