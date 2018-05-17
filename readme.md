# Informed Non-Negative Matrix Factorization

## Dependencies:
- Python3
- NumPy
- SciPy
- Matplotlib

## Description
Takes a hyperspectral image and a plain text inputs file and uses the settings
from the input file to perform INMF spectral unmixing on the image. The result
is a set of endmember spectra and maps of the spatial abundance of each
endmember.

##
- Developed for application to data from the Hyperspectral Imager for the Coastal Ocean (HICO)
- 2 NMF algorithms are implemented:
	- NMF (Lee & Seung, 2001)
  - INMF (Wright et al, 2018)

Inputs:
- See example input file ‘SampleINMF.in’ for a full description of input options
- HICO image (HICO images can be downloaded from the NASA Ocean Color Website [https://oceancolor.gsfc.nasa.gov/])
- Initial endmember spectra (Currently this data should be in a MATLAB save file
    format [.mat]) This can be generated using the included 'calc_radiance'
    function with a reflectance spectra, modtran atmospheric correction output
    [.acd] and a top of atmosphere irradiance spectrum.

Output:
- Result Plots
- MATLAB save file of Endmember spectra, abundances, and cost function

## Using the Code
### With a text input file:
The code can be called from the command by providing the inmf_master script with
a string of the location of the desired input file

    python3 inmf_master ‘SampleINMF.in’
### Using a GUI for input:
Alternatively, calling the code with no input file will open a GUI that prompts
the user to enter all of input data

    python3 inmf_master
This approach will generate an input file based on user inputs and then run the
INMF code using those inputs.
