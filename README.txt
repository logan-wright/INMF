README.txt

Informed Non-Negative Matrix Factorization Code
Version 1.1


- Developed for application to data from the Hyperspectral Imager for the Coastal Ocean (HICO)
- 3 Implemented include:
	- NMF (Lee & Seung, 2001)
	- Constrained NMF or psNMF (Jia & Qian, 2009)
	- INMF (Wright et al, 2018)

Example of calling program 

	python3 inmf_master ‘SampleINMF.in’

Input:
- See example input file ‘SampleINMF.in’
- HICO image
- MODTRAN results for HICO geometry

Output:
- Result Plots
- ‘.mat’ matlab save file of Endmember spectra, abundances, and cost function