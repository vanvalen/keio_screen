#Import packages
import numpy as np 
import os
import tifffile as tiff
import math
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle
from SLIP_functions import analyze_well, analyze_plate, segment_SLIP, plot_slip_well, plot_slip_wells
from SLIP_functions import plot_slip_joint_plot, fit_kde, compute_p_values, background_subtraction
from SLIP_functions import classify_infections, compute_p_lysis_posterior, compute_MOI_posterior
from keio_names import get_keio_names, pos_to_strain
import seaborn as sns
import pandas as pd
import pymc3 as pm

def cross_corr(im0, im1):
	from numpy.fft import fft2, ifft2
	f0 = fft2(im0)
	f1 = fft2(im1)
	shape = im0.shape
	ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
	t0, t1 = np.unravel_index(np.argmax(ir), shape)
	return t0, t1

def is_square(integer):
	root = math.sqrt(integer)
	if int(root + 0.5) ** 2 == integer:
		return True
	else:
		return False

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
					if h_temp == 0:
						h_temp += 1
					if v_temp == 0:
						v_temp += 1
					h += [h_temp]
					v += [v_temp]

		# Merge rows using the offsets
		merged_cols = []

		print h,v
		for j in xrange(num):
			merged_col = cols[j][0]
			h_temp = h[j*(num-1):j*(num-1) + num-1]
			v_temp = v[j*(num-1):j*(num-1) + num-1]

			for row in xrange(num-1):
				print merged_col.shape
				print v_temp[row]
				print merged_col[:,v_temp[row]:].shape
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
				if merged_cols[j].shape[0] == 0 or merged_cols[j].shape[1] == 0:
					h_temp = 1
					v_temp = 1
				if merged_cols[j+1].shape[0] == 0 or merged_cols[j+1].shape[1] == 0:
					h_temp = 1
					v_temp = 1
				else:
	 				h_temp, v_temp = cross_corr(merged_cols[j+1], merged_cols[j])
				if h_temp == 0:
					h_temp += 1
				if v_temp == 0:
					v_temp += 1
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

	print h4, v4, h5, v5

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

	h = [h0, h1, h2, h3, h4, h5, h6, h7]
	v = [v0, v1, v2, v3, v4, v5, v6, v7]

	temp4 = np.concatenate((temp2[h6:,:v6],temp[:-h6,:]), axis = 1)
	temp5 = np.concatenate((temp3[h7+h6:,:v7], temp4[:-h7,:]), axis = 1)
	
	print h, v

	return temp5, h, v

#Define root directory path
root_direc = '/media/vanvalen/fe0ceb60-f921-4184-a484-b7de12c1eea6/keio_screen/08.03.2017/keio9_1/'

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
col_control = [12]

row_data = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
col_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

pos_list = range(25)
list_of_masks = []
FITC_list = []
cherry_list = []
phase_list = []

mask_direc_well = os.path.join(mask_direc, 'F12')
data_direc_well = os.path.join(data_direc, 'F12')

for pos in pos_list:
	mask_name = os.path.join(mask_direc_well, 'feature_1_frame_' + str(pos) + '.tif')
	FITC_name = os.path.join(data_direc_well, 'img_000000000_EGFP_' +  str(pos).zfill(3) + '.tif')
	cherry_name = os.path.join(data_direc_well, 'img_000000000_mCherry_' +  str(pos).zfill(3) + '.tif')
	phase_name = os.path.join(data_direc_well, 'img_000000000_Phase_' +  str(pos).zfill(3) + '.tif')

	mask = np.float32(imread(mask_name))
	FITC = np.float32(imread(FITC_name))
	cherry = np.float32(imread(cherry_name))
	phase = np.float32(imread(phase_name))

	FITC_norm = background_subtraction(FITC)
	cherry_norm = background_subtraction(cherry)

	list_of_masks.append(mask[40:-40, 140:-140])
	FITC_list.append(FITC_norm[40:-40, 140:-140])
	cherry_list.append(cherry_norm[40:-40, 140:-140])
	phase_list.append(phase[40:-40, 140:-140])

# mask_panorama, h, v = merge_images(list_of_masks)
mask_panorama, h, v = merge_images_v2(list_of_masks)

tiff.imsave('mask_panorama.tif', mask_panorama)

phase_panorama = merge_images_v2(phase_list,  h = h, v = v)[0]
tiff.imsave('phase_panorama.tif', phase_panorama)

fitc_panorama = merge_images_v2(FITC_list, h = h, v = v)[0]
tiff.imsave('fitc_panorama.tif', fitc_panorama)

cherry_panorama = merge_images_v2(cherry_list, h = h, v = v)[0]
tiff.imsave('cherry_panorama.tif', cherry_panorama)

	# phase_names = ['temp.tif', 'temp2.tif', 'temp3.tif', 'temp4.tif', 'temp5.tif']
	# tiff.imsave(phase_names[0], temp)
	# tiff.imsave(phase_names[1], temp2)
	# tiff.imsave(phase_names[2], temp3)



