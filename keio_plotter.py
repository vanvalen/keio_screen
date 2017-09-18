#Import packages
import numpy as np 
import os
import tifffile as tiff
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle
from SLIP_functions import analyze_well, analyze_plate, segment_SLIP, plot_slip_well, plot_slip_wells, plot_slip_wells_gmm, plot_slip_wells_lysis_posterior, plot_slip_wells_MOI_posterior
from SLIP_functions import plot_slip_joint_plot, fit_kde, compute_p_values
from SLIP_functions import classify_infections, compute_p_lysis_posterior, compute_MOI_posterior
from keio_names import get_keio_names, pos_to_strain
import seaborn as sns
import pandas as pd
import pymc3 as pm

direc = "/media/vanvalen/fe0ceb60-f921-4184-a484-b7de12c1eea6/keio_screen/08.03.2017/"
data_08032017 = [os.path.join(direc, 'keio1_1'), os.path.join(direc, 'keio1_2'), os.path.join(direc, 'keio9_1')]

direc = "/media/vanvalen/693d2597-3dbf-41bb-b919-341f714e3199/keio_screen/08.14.2017/"
data_08152017 = [os.path.join(direc, 'keio3'), os.path.join(direc, 'keio5'), os.path.join(direc, 'keio7'), os.path.join(direc, 'keio11')]

direc = "/media/vanvalen/693d2597-3dbf-41bb-b919-341f714e3199/keio_screen/09.04.2017/"
data_09042017 = [os.path.join(direc, 'keio13'), os.path.join(direc, 'keio15'), os.path.join(direc, 'keio23'), os.path.join(direc, 'keio25'), os.path.join(direc, 'keio27'), os.path.join(direc, 'keio29'), os.path.join(direc, 'keio31')]

data = data_08032017 + data_08152017 + data_09042017
plate_numbers = [1, 1, 9, 3, 5, 7, 11, 13, 15, 23, 25, 27, 29, 31]

for root_direc, plate_number in zip([data[-1]], [plate_numbers[-1]]):
	print root_direc
	#Define directory path to infection data (all positions)
	data_direc = os.path.join(root_direc, 'data')

	#Define directory path to control data (all positions)
	control_direc = os.path.join(root_direc, 'control')

	#Define directory path to where you want to store neural net outputs. 
	#mask directories must exist at run time!
	mask_direc = os.path.join(root_direc, 'masks')
	control_mask_direc = os.path.join(root_direc,'control_masks')

	#Define which wells were used
	row_control = ['A']
	col_control = [1]

	row_data = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
	col_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

	#Load saved data
	mean_FITC_name = os.path.join(root_direc, 'mean_FITC.pkl')
	mean_cherry_name = os.path.join(root_direc, 'mean_cherry.pkl')
	mean_FITC = pickle.load(open(mean_FITC_name, 'rb'))
	mean_cherry = pickle.load(open(mean_cherry_name, 'rb'))

	#Plot the scatter plot of intensities
	wells = []
	titles = []
	keio_names_array = get_keio_names()

	for row in row_data:
		for col in col_data:
			well = row + str(col)
			wells += [well]
			titles += [pos_to_strain(keio_names_array, plate_number, well)]

	if plate_number == 29:
		plot_slip_wells(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, save_fig = True)
		plot_slip_wells_gmm(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = ['F7', 'F8', 'F9'])
		plot_slip_wells_lysis_posterior(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = ['F7', 'F8', 'F9'])
		plot_slip_wells_MOI_posterior(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = ['F7', 'F8', 'F9'])

	if plate_number == 31:
		plot_slip_wells(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, save_fig = True)
		plot_slip_wells_gmm(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = ['H5', 'F4', 'D2'])
		plot_slip_wells_lysis_posterior(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = ['H5', 'F4', 'D2'])
		plot_slip_wells_MOI_posterior(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = ['H5', 'F4', 'D2'])

	else:
		plot_slip_wells(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, save_fig = True)
		plot_slip_wells_gmm(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number)
		plot_slip_wells_lysis_posterior(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number)
		plot_slip_wells_MOI_posterior(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number)
