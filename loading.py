import numpy as N

def load_input_file(filepath):
    """
    inmf_master.load_input_file - parses INMF input file

    Created on: Oct, 13, 2016
    Last Modified: Jan 27, 2018
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
        if (temp[0] is not '#') and (temp[0] is not '\n'):    # Ignore Commmented Lines
            strs = temp.split('=')    # Divide based on position of "="
            key = strs[0].strip(' \n\t')
            val =  strs[1].split(',')    # Split comma separated values
            if len(val) > 1:    # Processes if there is more than one value,
                for n in range(len(val)):
                    val[n] = str(val[n].strip(' \n\t'))
                try
            else:
                val = val[0].strip(' \n\t')
                
            
            inputs[key] = val    # Save parsed values in a dictionary
    f.close()
    
    
#    # Check for required inputs and format
#    if filepath == 'PARAMS.txt':
#        inputs['max_i'] = int(inputs['max_i'])
#    else:
#        # Converts wvl_ind into integers, if no value is given uses the full range
#        try:
#            inputs['wvl_rng'] = N.array(inputs['wvl_rng'], dtype = int)
#        except ValueError:
#            inputs['wvl_rng'] = N.array([0,128], dtype = int)
#        # Converts SZA into integer, if no value is given, requests an input from the user
#        try:
#            inputs['SZA'] = float(inputs['SZA'])
#        except NameError:
#            inputs['SZA'] = input('No Solar Zenith Angle Provided! Enter the SZA now:')
#
#        # Converts the Sun Ellipticity correction into a float, if no value is given assumes 1
#        try:
#            inputs['SunElliptic'] = float(inputs['SunElliptic'])
#        except NameError:
#            inputs['SunElliptic'] = 1
#
#        # Converts the allowed maximum number of iterations an integer, if none is present, sets the value to 500
#        try:
#            inputs['max_i'] = int(inputs['max_i'])
#        except NameError:
#            inputs['max_i'] = 500
#        # Sets the 'perturb" keyword to None, if not set
#        try:
#            inputs['perturb']
#        except KeyError:
#            inputs['perturb'] = None
#        # Sets other required inputs if they are not given in the inout file
#        if 'name' not in inputs.keys():
#            print('ERROR: No file name given')
#            inputs['name'] = input('Input filename now:')
#        if 'plot_flag' not in inputs.keys():
#            inputs['plot_flag'] = True
#        if 'rad_flag' not in inputs.keys():
#            inputs['rad_flag'] = True

    return inputs

def load_acd(filepath):
    """
    inmf_master.load_acd() - load data contained in a MODTRAN .acd output file

    Version 1.0
    Created on: Oct, 13, 2016
    Last Modified: Jan, 27 2018
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
