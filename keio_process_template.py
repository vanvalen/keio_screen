#Import packages
import numpy as np 
import os
import tifffile as tiff
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle
from SLIP_functions import analyze_well, analyze_plate, segment_SLIP, plot_slip_well, plot_slip_wells
from SLIP_functions import plot_slip_joint_plot, fit_kde, compute_p_values
from SLIP_functions import classify_infections, compute_p_lysis_posterior, compute_MOI_posterior
from keio_names import get_keio_names, pos_to_strain
import seaborn as sns
import pandas as pd
import pymc3 as pm

#Define root directory path

#direc = "/home/vanvalen/Data/keio_screen/07.11.2017/"
#data_7112017 = [os.path.join(direc,'keio_7'), os.path.join(direc, 'keio_11'), os.path.join(direc, 'keio_13')] 

direc = "/media/vanvalen/fe0ceb60-f921-4184-a484-b7de12c1eea6/keio_screen/07.13.2017/"
data_7132017 = [os.path.join(direc,'keio_1'), os.path.join(direc,'keio_3'), os.path.join(direc,'keio_5'), os.path.join(direc,'keio_9')]
for root_direc in data_7132017:
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

	#Segment the control wells
	segment_SLIP(control_direc, control_mask_direc, alphabet = row_control, columns= col_control)

	#Segment the infected wells
	segment_SLIP(data_direc, mask_direc, alphabet = row_data, columns= col_data)

	# Quantify the data from the control wells
	mean_FITC_control, mean_cherry_control = analyze_plate(control_direc, control_mask_direc, pos_list = range(5), row_names = row_control, col_names = col_control)
	mean_FITC_control_name = os.path.join(root_direc, 'mean_FITC_control.pkl')
	mean_cherry_control_name = os.path.join(root_direc, 'mean_cherry_control.pkl')
	pickle.dump(mean_FITC_control, open(mean_FITC_control_name, 'wb'))
	pickle.dump(mean_cherry_control, open(mean_cherry_control_name, 'wb'))

	# Quantify the data from the infection wells
	mean_FITC, mean_cherry = analyze_plate(data_direc, mask_direc, pos_list = range(9), row_names = row_data, col_names = col_data)
	mean_FITC_name = os.path.join(root_direc, 'mean_FITC.pkl')
	mean_cherry_name = os.path.join(root_direc, 'mean_cherry.pkl')
	pickle.dump(mean_FITC, open(mean_FITC_name, 'wb'))
	pickle.dump(mean_cherry, open(mean_cherry_name, 'wb'))

	#Load saved data
	mean_FITC_name = os.path.join(root_direc, 'mean_FITC.pkl')
	mean_cherry_name = os.path.join(root_direc, 'mean_cherry.pkl')
	mean_FITC = pickle.load(open(mean_FITC_name, 'rb'))
	mean_cherry = pickle.load(open(mean_cherry_name, 'rb'))

	mean_FITC_control_name = os.path.join(root_direc, 'mean_FITC_control.pkl')
	mean_cherry_control_name = os.path.join(root_direc, 'mean_cherry_control.pkl')
	mean_FITC_control = pickle.load(open(mean_FITC_control_name, 'rb'))
	mean_cherry_control = pickle.load(open(mean_cherry_control_name, 'rb'))

	# Fit a KDE estimator to the no infection control
	kernel = fit_kde(mean_FITC_control, mean_cherry_control, 'A1')

	# Compute the probability of observing each data point assuming there was no infection
	p_values_dict = {}
	for well in mean_FITC.keys():
		p_values_dict[well] = np.array(compute_p_values(mean_FITC, mean_cherry, well, kernel))

	# Save p values dictionary
	p_values_name = os.path.join(root_direc, 'p_values_dict.pkl')
	pickle.dump(p_values_dict, open(p_values_name, 'wb'))
