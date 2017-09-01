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
from SLIP_functions import classify_infections, classify_infections_gmm, compute_p_lysis_posterior, compute_MOI_posterior
from keio_names import get_keio_names, pos_to_strain
import seaborn as sns
import pandas as pd
import pymc3 as pm

#Define root directory path

#direc = "/home/vanvalen/Data/keio_screen/07.11.2017/"
#data_7112017 = [os.path.join(direc,'keio_7'), os.path.join(direc, 'keio_11'), os.path.join(direc, 'keio_13')] 

#direc = "/media/vanvalen/fe0ceb60-f921-4184-a484-b7de12c1eea6/keio_screen/07.13.2017/"
#data_7132017 = [os.path.join(direc,'keio_1'), os.path.join(direc,'keio_3'), os.path.join(direc,'keio_5'), os.path.join(direc,'keio_9')]

# direc = "/media/vanvalen/fe0ceb60-f921-4184-a484-b7de12c1eea6/keio_screen/08.03.2017/"
# data_08032017 = [os.path.join(direc, 'keio1_1'), os.path.join(direc, 'keio1_2'), os.path.join(direc, 'keio9_1')]
# data_08032017 = [os.path.join(direc, 'keio9_1')]
# data_08032017 = [os.path.join(direc, 'keio1_1'), os.path.join(direc, 'keio1_2'), os.path.join(direc, 'keio9_1')]

direc = "/media/vanvalen/693d2597-3dbf-41bb-b919-341f714e3199/keio_screen/08.14.2017/"
data_08152017 = [os.path.join(direc, 'titer')]

data = data_08152017

for root_direc in data:
	plate_number = 0
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
	col_data = [5, 6, 7, 8, 9, 10, 11, 12]

	#Segment the control wells
	# segment_SLIP(control_direc, control_mask_direc, alphabet = row_control, columns= col_control)

	#Segment the infected wells
	# segment_SLIP(data_direc, mask_direc, alphabet = row_data, columns= col_data)

	# Quantify the data from the control wells
	# mean_FITC_control, mean_cherry_control = analyze_plate(control_direc, control_mask_direc, pos_list = range(25), row_names = row_control, col_names = col_control)
	# mean_FITC_control_name = os.path.join(root_direc, 'mean_FITC_control.pkl')
	# mean_cherry_control_name = os.path.join(root_direc, 'mean_cherry_control.pkl')
	# pickle.dump(mean_FITC_control, open(mean_FITC_control_name, 'wb'))
	# pickle.dump(mean_cherry_control, open(mean_cherry_control_name, 'wb'))

	# Quantify the data from the infection wells
	# mean_FITC, mean_cherry = analyze_plate(data_direc, mask_direc, pos_list = range(25), row_names = row_data, col_names = col_data)
	# mean_FITC_name = os.path.join(root_direc, 'mean_FITC.pkl')
	# mean_cherry_name = os.path.join(root_direc, 'mean_cherry.pkl')
	# pickle.dump(mean_FITC, open(mean_FITC_name, 'wb'))
	# pickle.dump(mean_cherry, open(mean_cherry_name, 'wb'))

	#Load saved data
	mean_FITC_name = os.path.join(root_direc, 'mean_FITC.pkl')
	mean_cherry_name = os.path.join(root_direc, 'mean_cherry.pkl')
	mean_FITC = pickle.load(open(mean_FITC_name, 'rb'))
	mean_cherry = pickle.load(open(mean_cherry_name, 'rb'))

	mean_FITC_control_name = os.path.join(root_direc, 'mean_FITC_control.pkl')
	mean_cherry_control_name = os.path.join(root_direc, 'mean_cherry_control.pkl')
	mean_FITC_control = pickle.load(open(mean_FITC_control_name, 'rb'))
	mean_cherry_control = pickle.load(open(mean_cherry_control_name, 'rb'))

	#Plot the scatter plot of intensities
	wells = []
	classification_wells = []
	titles = []
	keio_names_array = get_keio_names()

	for row in row_data:
		for col in col_data:
			well = row + str(col)
			wells += [well]
			titles += [well]

	for row in row_data:
		for col in [11, 12]:
			well = row + str(col)
			classification_wells += [well]
			titles += [well]
	lytic_dict, lysogenic_dict, uninfected_dict = classify_infections_gmm(mean_FITC, mean_cherry, wells = wells, classification_wells = classification_wells)

	for well in wells:
		fraction_infected = np.float(lytic_dict[well].shape[0] + lysogenic_dict[well].shape[0])/np.float(lytic_dict[well].shape[0] + lysogenic_dict[well].shape[0] + uninfected_dict[well].shape[0])
		# print well + ' lytic ' + str(lytic_dict[well].shape[0])
		# print well + ' lysogenic ' + str(lysogenic_dict[well].shape[0])
		# print well + ' uninfected ' + str(uninfected_dict[well].shape[0])
		print well + ' fraction infected ' + str(fraction_infected)

	plot_slip_wells_gmm(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = classification_wells)
	plot_slip_wells_lysis_posterior(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = classification_wells)
	plot_slip_wells_MOI_posterior(mean_FITC, mean_cherry, wells = wells, titles = titles, plate_number = plate_number, classification_wells = classification_wells)

