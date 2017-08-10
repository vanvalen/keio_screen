import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, segment_nuclei, segment_cytoplasm, dice_jaccard_indices, run_model
from model_zoo import sparse_bn_feature_net_61x61 as fn 

import os
import numpy as np
from skimage.io import imread
from scipy.stats import mode
from skimage.measure import label, regionprops
import scipy
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import pymc3 as pm
import math


def segment_SLIP(data_direc, mask_direc, alphabet = ['A','B','C','D','E','F','G','H'], columns = range(1,13)):

	print alphabet, columns

	for row in alphabet:
		for column in columns:
			pos = row + str(column)
			print "Segmenting Position " + pos
			current_data_direc = os.path.join(data_direc, pos)
			current_mask_direc = os.path.join(mask_direc, pos)

			#make position directory for masks if necessary
			try:
				os.stat(current_mask_direc)
			except:
				os.mkdir(current_mask_direc)

			segment_SLIP_plate(current_data_direc, current_mask_direc)
			
def segment_SLIP_plate(data_direc, mask_direc):
	data_location = data_direc
	phase_location = mask_direc

	phase_channel_names = ['Phase']#['channel000']

	trained_network_phase_directory = "/home/vanvalen/DeepCell/trained_networks/slip"   

	phase_prefix = "2017-06-06_slip_61x61_bn_feature_net_61x61_"
	#"2017-02-12_ecoli_90x_31x31_ecoli_90x_feature_net_31x31_"

	win_phase = 30

	image_size_x, image_size_y = get_image_sizes(data_location, phase_channel_names)
	image_size_x /= 2
	image_size_y /= 2

	list_of_phase_weights = []
	for j in xrange(3):
		phase_weights = os.path.join(trained_network_phase_directory,  phase_prefix + str(j) + ".h5")
		list_of_phase_weights += [phase_weights]

	phase_predictions = run_models_on_directory(data_location, phase_channel_names, phase_location, model_fn = fn, 
		list_of_weights = list_of_phase_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
		win_x = win_phase, win_y = win_phase, std = False, split = False)

def screen_masks(list_of_masks, confidence = 0.75, area_filter = True, eccen_filter = True, minor_axis_filter = False, major_axis_filter = False, solidity_filter = True):
	mask_area = []
	mask_ecc = []
	mask_minor_axis = []
	mask_major_axis = []
	mask_solidity = []
	list_of_screened_masks = []

	for mask in list_of_masks:
		mask = mask > confidence
		label_mask = label(mask)
		mask_props = regionprops(label_mask, mask)

		for prop in mask_props:
			mask_area.append(prop.area)
			mask_ecc.append(prop.eccentricity)
			mask_minor_axis.append(prop.minor_axis_length)
			mask_major_axis.append(prop.major_axis_length)
			mask_solidity.append(prop.solidity)

	area_limit = [np.mean(mask_area) - np.std(mask_area), np.mean(mask_area) + np.std(mask_area)]
	ecc_limit = [np.mean(mask_ecc) - np.std(mask_ecc), np.mean(mask_ecc) + np.std(mask_ecc)]
	minor_limit = [np.mean(mask_minor_axis) - np.std(mask_minor_axis), np.mean(mask_minor_axis) + np.std(mask_minor_axis)]
	major_limit = [np.mean(mask_major_axis) - np.std(mask_major_axis), np.mean(mask_major_axis) + np.std(mask_major_axis)]
	solidity_limit = [np.mean(mask_solidity) - np.std(mask_solidity), np.mean(mask_solidity) + np.std(mask_solidity)]
	
	for mask in list_of_masks:
		mask = mask > confidence
		label_mask = label(mask)
		mask_props = regionprops(label_mask)
		for prop in mask_props:
			if area_filter:
				if prop.area < area_limit[0] or prop.area > area_limit[1]:
					mask[label_mask == prop.label] = 0
			if eccen_filter:
				if prop.eccentricity < ecc_limit[0] or prop.eccentricity > ecc_limit[1]:
					mask[label_mask == prop.label] = 0
			if minor_axis_filter:
				if prop.minor_axis_length < minor_limit[0] or prop.minor_axis_length > minor_limit[1]:
					mask[label_mask == prop.label] = 0
			if major_axis_filter:
				if prop.major_axis_length < major_limit[0] or prop.major_axis_length > major_limit[1]:
					mask[label_mask == prop.label] = 0
			if solidity_filter:
				if prop.solidity < solidity_limit[0] or prop.solidity > solidity_limit[1]:
					mask[label_mask == prop.label] = 0
		list_of_screened_masks.append(mask)

	return list_of_screened_masks

def background_subtraction(image):
	background = np.median(image.flatten())
	return image - background

def analyze_well(data_direc, mask_direc, pos_list, panorama = True):
	if panorama is True:
		list_of_masks = []
		FITC_list = []
		cherry_list = []
		phase_list = []

		for pos in pos_list:
			mask_name = os.path.join(mask_direc, 'feature_1_frame_' + str(pos) + '.tif')
			FITC_name = os.path.join(data_direc, 'img_000000000_EGFP_' +  str(pos).zfill(3) + '.tif')
			cherry_name = os.path.join(data_direc, 'img_000000000_mCherry_' +  str(pos).zfill(3) + '.tif')
			phase_name = os.path.join(data_direc, 'img_000000000_Phase_' +  str(pos).zfill(3) + '.tif')

			mask = np.float32(imread(mask_name))[40:-40, 140:-140]
			FITC = np.float32(imread(FITC_name))[40:-40, 140:-140]
			cherry = np.float32(imread(cherry_name))[40:-40, 140:-140]
			phase = np.float32(imread(phase_name))[40:-40, 140:-140]

			FITC_norm = background_subtraction(FITC)
			cherry_norm = background_subtraction(cherry)

			list_of_masks.append(mask)
			FITC_list.append(FITC_norm)
			cherry_list.append(cherry_norm)
			phase_list.append(phase)

		list_of_screened_masks = screen_masks(list_of_masks)

		# Check the stitching parameters - if off, use pre computed stitching parameters
		mask_pan, h, v = merge_images_v2(list_of_masks)

		h_pre = [490, 498, 491, 499, 490, 499, 10, 11]
		v_pre = [9, 9, 9, 9, 9, 8, 595, 596]

		replace = False
		for h_c, h_p, v_c, v_p in zip(h, h_pre, v, v_pre):
			if np.abs(h_c-h_p) > 10 or np.abs(h_c-h_p) > 10:
				replace = True

		if replace == True:
			h = h_pre
			v = v_pre

		mask_panorama = merge_images_v2(list_of_screened_masks, h = h, v = v)[0]
		phase_panorama = merge_images_v2(phase_list, h = h, v = v)[0]
		fitc_panorama = merge_images_v2(FITC_list, h = h, v = v)[0]
		cherry_panorama = merge_images_v2(cherry_list, h = h, v = v)[0]

		# Collect data points
		mean_FITC = []
		mean_cherry = []

		label_mask = label(mask_panorama)
		FITC_props = regionprops(label_mask, fitc_panorama)
		cherry_props = regionprops(label_mask, cherry_panorama)

		for props in FITC_props:
			mean_FITC.append(props.mean_intensity)

		for props in cherry_props:
			mean_cherry.append(props.mean_intensity)


	if panorama is False:
		list_of_masks = []
		FITC_list = []
		cherry_list = []
		for pos in pos_list:
			mask_name = os.path.join(mask_direc, 'feature_1_frame_' + str(pos) + '.tif')
			FITC_name = os.path.join(data_direc, 'img_000000000_EGFP_' +  str(pos).zfill(3) + '.tif')
			cherry_name = os.path.join(data_direc, 'img_000000000_mCherry_' +  str(pos).zfill(3) + '.tif')

			mask = np.float32(imread(mask_name))
			FITC = np.float32(imread(FITC_name))
			cherry = np.float32(imread(cherry_name))

			FITC_norm = background_subtraction(FITC)
			cherry_norm = background_subtraction(cherry)

			list_of_masks.append(mask)
			FITC_list.append(FITC_norm)
			cherry_list.append(cherry_norm)

		list_of_screened_masks = screen_masks(list_of_masks)

		# Save screened masks
		for pos, mask in zip(pos_list, list_of_screened_masks):
			mask_name = os.path.join(mask_direc, 'mask_' + str(pos) + '.tif')
			tiff.imsave(mask_name, np.float32(mask))

		# Collect data points
		mean_FITC = []
		mean_cherry = []

		for mask, FITC, cherry in zip(list_of_screened_masks, FITC_list, cherry_list):
			label_mask = label(mask)

			FITC_props = regionprops(label_mask, FITC)
			cherry_props = regionprops(label_mask, cherry)

			for props in FITC_props:
				mean_FITC.append(props.mean_intensity)

			for props in cherry_props:
				mean_cherry.append(props.mean_intensity)

	return mean_FITC, mean_cherry

def analyze_plate(data_direc, mask_direc, pos_list, row_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], col_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']):
	mean_FITC = {}
	mean_cherry = {}
	for row in row_names:
		for col in col_names:
			well = row + str(col)
			print 'Processing well ' + well
			data_directory = os.path.join(data_direc, well)
			mask_directory = os.path.join(mask_direc, well)
			fitc, cherry = analyze_well(data_directory, mask_directory, pos_list)
			mean_FITC[well] = fitc
			mean_cherry[well] = cherry
	return mean_FITC, mean_cherry

def plot_slip_well(fitc_dict, cherry_dict, well, title, infected_cells = None,):
	if infected_cells is None:
		fitc_list = fitc_dict[well]
		cherry_list = cherry_dict[well]
	else:
		fitc_list = np.array(fitc_dict[well])[infected_cells]
		cherry_list = np.array(cherry_dict[well])[infected_cells]

	xmax = max(fitc_list)
	xmin = min(fitc_list)
	ymax = max(cherry_list)
	ymin = min(cherry_list)

	fig = plt.figure()
	plt.plot(fitc_list, cherry_list,'o')
	plt.axis([xmin, xmax, ymin, ymax])
	plt.xlabel('FITC Pixel Intensity')
	plt.ylabel('Cherry Pixel Intensity')
	plt.title(title)

	return fig

def plot_slip_wells(fitc_dict, cherry_dict, wells, titles, infected_cells = None,):
	fig, axes = plt.subplots(8,12, figsize = (4*12, 4*8))

	xmax = 0
	xmin = 0
	ymax = 0
	ymin = 0

	for well in wells:
		if infected_cells is None:
			fitc_list = fitc_dict[well]
			cherry_list = cherry_dict[well]
		else:
			fitc_list = np.array(fitc_dict[well])[infected_cells[well]]
			cherry_list = np.array(cherry_dict[well])[infected_cells[well]]
	
		# if max(fitc_list) > xmax:
		# 	xmax = max(fitc_list)
		# if min(fitc_list) < xmin:
		# 	xmin = min(fitc_list)
		# if max(cherry_list) > ymax:
		# 	ymax = max(cherry_list)
		# if min(cherry_list) < ymin:
		# 	ymin = min(cherry_list)

	for well, title in zip(wells, titles):
		if infected_cells is None:
			fitc_list = fitc_dict[well]
			cherry_list = cherry_dict[well]
		else:
			fitc_list = np.array(fitc_dict[well])[infected_cells[well]]
			cherry_list = np.array(cherry_dict[well])[infected_cells[well]]

		if len(fitc_list) > 0:
			alphabet = ['A','B','C','D','E','F','G','H']
			chars = list(well)
			row = alphabet.index(chars[0])
			if len(chars) == 2:
				column = int(chars[1])-1
			if len(chars) == 3:
				column = int(chars[1] + chars[2])-1

			axes[row,column].plot(fitc_list, cherry_list,'o')
			axes[row,column].set_xlim([min(fitc_list), max(fitc_list)])
			axes[row,column].set_ylim([min(cherry_list), max(cherry_list)]) 
			axes[row,column].set_xlabel('FITC Pixel Intensity')
			axes[row,column].set_ylabel('Cherry Pixel Intensity')
			axes[row,column].set_title(title)
	plt.tight_layout()
	return fig, axes

def fit_kde(fitc_dict, cherry_dict, well):
	fitc_array = np.array(fitc_dict[well])
	cherry_array = np.array(cherry_dict[well])

	values = np.vstack([fitc_array, cherry_array])

	kernel = stats.gaussian_kde(values)
	return kernel

def plot_slip_joint_plot(fitc_dict, cherry_dict, well, title):
	sns.set_style('dark')
	fitc_array = np.array(fitc_dict[well])
	cherry_array = np.array(cherry_dict[well])

	g = sns.jointplot(fitc_array, cherry_array, kind = 'scatter', color = 'b')
	g.plot_joint(plt.kde_plot, c = 'b', bw = 'silverman')
	g.ax_joint.collections[0].set_alpha(0)
	g.set_axis_labels('FITC Pixel Intensity', 'Cherry Pixel Intensity')
	sns.plt.suptitle(title)
	return None

def compute_p_values(fitc_dict, cherry_dict, well, kernel, max_val = 1e6):
	fitc_list = fitc_dict[well]
	cherry_list = cherry_dict[well]

	p_values = []
	for fitc, cherry in zip(fitc_list, cherry_list):
		low_values = [fitc, cherry]
		high_values = [max_val, max_val]
		p = kernel.integrate_box(low_values, high_values)
		p_values += [p]

	return np.array(p_values)

def compute_p_lysis_posterior(N_lysis, N_lysogeny):
	observed_lysis = np.ones(N_lysis)
	observed_lysogeny = np.zeros(N_lysogeny)
	observed_data = np.concatenate([observed_lysis, observed_lysogeny], axis = 1)
	np.random.shuffle(observed_data)

	with pm.Model() as model:
		theta = pm.Uniform('theta', 0, 1)
		x = pm.Bernoulli('x', p = theta, observed = observed_data)

	with model:
		trace = pm.sample(2000)

	return trace[500:]

def compute_MOI_posterior(N_cells, N_infected):

	with pm.Model() as model:
		theta = pm.Uniform('theta', 0, 1)
		empty_bins = pm.Binomial('empty_bins', p = theta, n = N_cells, observed = N_infected)
	
	with model:
		trace = pm.sample(2000)

	return trace[500:]

def classify_infections(kernel_fitc, kernel_cherry, mean_FITC, mean_cherry, p_values_dict, rows, cols):
	return_dict = {}
	for row in rows:
		for col in cols:
			well = row + str(col)
			p_value = p_values_dict[well]
			infected_cells = np.where(p_value < 0.01)[0]
			fitc_list = np.array(mean_FITC[well])[infected_cells]
			cherry_list = np.array(mean_cherry[well])[infected_cells]

			p_fitc = []
			p_cherry = []
			for fitc, cherry in zip(fitc_list, cherry_list):
				p_fitc += [1-kernel_fitc.integrate_box_1d(fitc, 1e6)]
				p_cherry += [1-kernel_cherry.integrate_box_1d(cherry, 1e6)]

			# Remove double positives
			p_fitc_new = []
			p_cherry_new = []
			fitc_list_new = []
			cherry_list_new = []
			for p_f, p_c, f, c in zip(p_fitc, p_cherry, fitc_list, cherry_list):
				if (p_f < 0.85 and c > 50) or p_c < 0.85:
					p_fitc_new += [p_f]
					p_cherry_new += [p_c]
					fitc_list_new += [f]
					cherry_list_new += [c]

			p_fitc = p_fitc_new
			p_cherry = p_cherry_new
			fitc_list = fitc_list_new
			cherry_list = cherry_list_new

			d = {'FITC Probability': p_fitc, 'Cherry Probability': p_cherry, 'FITC Intensity': fitc_list, 'Cherry Intensity': cherry_list}

			return_dict[well] = d

	return return_dict

def cross_corr(im0, im1):
	from numpy.fft import fft2, ifft2
	f0 = fft2(im0)
	f1 = fft2(im1)
	shape = im0.shape
	ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
	t0, t1 = np.unravel_index(np.argmax(ir), shape)
	return t0, t1

def merge_images(img_list, h = None, v = None):
	if h is None:
		h0, v0 = cross_corr(img_list[0], img_list[1])
		h1, v1 = cross_corr(img_list[1], img_list[2])
	else:
		h0 = h[0]
		v0 = v[0]
		h1 = h[1]
		v1 = v[1]

	temp = np.concatenate((img_list[1][:-h0,:-v0], img_list[0][:,v0:]), axis = 0)
	temp = np.concatenate((img_list[2][:-h1,:-v1-v0],temp[:,v1:]), axis = 0)

	if h is None:
		h2, v2 = cross_corr(img_list[5], img_list[4])
		h3, v3 = cross_corr(img_list[4], img_list[3])
	else:
		h2 = h[2]
		v2 = v[2]
		h3 = h[3]
		v3 = v[3]


	temp2 = np.concatenate((img_list[4][:-h2,:-v2], img_list[5][:,v2:]), axis = 0)
	temp2 = np.concatenate((img_list[3][:-h3,:-v3-v2], temp2[:,v3:]), axis = 0)

	if h is None:
		h4, v4 = cross_corr(img_list[6], img_list[7])
		h5, v5 = cross_corr(img_list[7], img_list[8])
	else:
		h4 = h[4]
		v4 = v[4]
		h5 = h[5]
		v5 = v[5]

	temp3 = np.concatenate((img_list[7][:-h4,:-v4], img_list[6][:,v4:]), axis = 0)
	temp3 = np.concatenate((img_list[8][:-h5,:-v5-v4],temp3[:,v5:]), axis = 0)

	xmin = min([temp.shape[0], temp2.shape[0], temp3.shape[0]])
	ymin = min([temp.shape[1], temp2.shape[1], temp3.shape[1]])

	temp = temp[0:xmin,0:ymin]
	temp2 = temp2[0:xmin,0:ymin]
	temp3 = temp3[0:xmin,0:ymin]

	if h is None:
		h6, v6 = cross_corr(temp2, temp)
		h7, v7 = cross_corr(temp3, temp2)
	else:
		h6 = h[6]
		v6 = v[6]
		h7 = h[7]
		v7 = v[7]

	temp4 = np.concatenate((temp2[h6:,:v6],temp[:-h6,:]), axis = 1)
	temp5 = np.concatenate((temp3[h7+h6:,:v7], temp4[:-h7,:]), axis = 1)

	h = [h0, h1, h2, h3, h4, h5, h6, h7]
	v = [v0, v1, v2, v3, v4, v5, v6, v7]
	
	return temp5, h, v

def merge_images_v2(img_list, h = None, v = None):
	if is_square(len(img_list)) is True:
		num = np.int(math.sqrt(len(img_list))+0.5)

		cols = []
		for col in xrange(num):
			imgs_to_merge = []
			for row in xrange(num):
				imgs_to_merge += [img_list[row + col*num]]

			if col % 2 == 1:
				imgs_to_merge.reverse()

			cols += [imgs_to_merge]

		is_h_none = h is None

		if is_h_none is True:
			h = []
			v = []
			for col in cols:
				for row in xrange(num-1):
					h_temp, v_temp = cross_corr(col[row], col[row+1])
					h += [h_temp]
					v += [v_temp]

		# Merge rows using the offsets
		merged_cols = []
		for j in xrange(num):
			merged_col = cols[j][0]
			h_temp = h[j*(num-1):j*(num-1) + num-1]
			v_temp = v[j*(num-1):j*(num-1) + num-1]

			for row in xrange(num-1):
				merged_col = np.concatenate((cols[j][row+1][:-h_temp[row],:-np.sum(v_temp[0:row+1])], merged_col[:,v_temp[row]:]), axis = 0)
			merged_cols += [merged_col]

		xmins = [col.shape[0] for col in merged_cols]
		ymins = [col.shape[1] for col in merged_cols]

		xmin = min(xmins)
		ymin = min(ymins)

		print xmins, ymins

		merged_cols_v2 = [merged_col[0:xmin,0:ymin] for merged_col in merged_cols]
		merged_cols = merged_cols_v2

		if is_h_none is True:
			for j in xrange(num-1):
				h_temp, v_temp = cross_corr(merged_cols[j+1], merged_cols[j])
				h +=[h_temp]
				v +=[v_temp]

		# Merge the merged rows by column together using the offsets
		merged = merged_cols[0]
		h_temp = h[num*(num-1):]
		v_temp = v[num*(num-1):]

		print h,v
		print h_temp, v_temp
		print merged.shape
		for j in xrange(num-1):
			print j
			print merged_cols[j+1][np.sum(h_temp[0:j+1]):,:v_temp[j]].shape
			print merged[:-h_temp[j],:].shape
			merged = np.concatenate((merged_cols[j+1][np.sum(h_temp[0:j+1]):,:v_temp[j]],merged[:-h_temp[j],:]), axis = 1)
			print merged.shape

		return merged, h, v

	else:
		print "Not a square grid!"







