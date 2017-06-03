import numpy as np
import os
import tifffile as tiff
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
from gaussian_model import extract_infected, gaussian_classifier, gmm_2D
from keio_names import get_keio_names, pos_to_strain

strain_matrix = get_keio_names()

plt.ioff()

#analyzes a single frame of SLIP data
#inputs: data_direc = path to 
def analyze_slip_frame(data_direc, mask_direc, frame, threshold = 0.75, multiplier = 1.5):
	mask_name = os.path.join(mask_direc, 'feature_1_frame_' + str(frame) + '.tif')
	#Assume channel001 is FITC
	FITC_name = os.path.join(data_direc, 'img_000000000_EGFP_' + str(frame).zfill(3) + '.tif')
	cherry_name = os.path.join(data_direc, 'img_000000000_mCherry_' + str(frame).zfill(3) + '.tif')
	mask = np.float32(imread(mask_name))
	mask = mask > confidence

	FITC = np.float32(imread(FITC_name)) 
	cherry = np.float32(imread(cherry_name))

	norm_FITC = FITC - np.mean(np.invert(mask)*FITC)
	norm_cherry = cherry - np.mean(np.invert(mask)*cherry)

	label_mask = label(mask)

	mask_props = regionprops(label_mask, mask)
	mask_area = []
	for props in mask_props:
		mask_area.append(props.area)

	mask_ecc = []
	for props in mask_props:
		mask_ecc.append(props.eccentricity)

	iqr_area = np.subtract(*np.percentile(mask_area,[75, 25]))
	iqr_ecc = np.subtract(*np.percentile(mask_ecc,[75, 25]))

	ecc_max = np.percentile(mask_ecc, 75) + multiplier*iqr_ecc
	ecc_min = np.percentile(mask_ecc, 25) - multiplier*iqr_ecc
	area_max = np.percentile(mask_area, 75) + multiplier*iqr_area
	area_min = np.percentile(mask_area, 25) - multiplier*iqr_area

	FITC_props = regionprops(label_mask, norm_FITC)
	cherry_props = regionprops(label_mask, norm_cherry)

	mean_FITC = []
	mean_cherry = []
	for props in FITC_props:
		if ((props.area > area_max) or (props.area < area_min)):
			continue
		if ((props.eccentricity > ecc_max) or (props.eccentricity < ecc_min)):
			continue
		mean_FITC.append(props.mean_intensity)

	for props in cherry_props:
		if ((props.area > area_max) or (props.area < area_min)):
			continue
		if ((props.eccentricity > ecc_max) or (props.eccentricity < ecc_min)):
			continue
		mean_cherry.append(props.mean_intensity)

	return zip(mean_FITC, mean_cherry)

def analyze_slip_pos(infect_direc, control_direc, mask_direc, pos, num_pos = 25, threshold = 0.75, gaussian_confidence = 1.0, multiplier = 1.5, verbose=False, plot=False, save_direc = None):
	data_direc_infect = os.path.join(infect_direc, pos)
	data_direc_noinfect = os.path.join(control_direc, pos)
	pos_mask_direc = os.path.join(mask_direc, pos + "_mask")
	FITC_mean = []
	cherry_mean = []
	FITC_control = []
	cherry_control = []
	for i in range(0, num_pos):
		print "Analyzing Position " + pos + " Frame " + str(i)
		means = analyze_slip_frame(data_direc_infect, pos_mask_direc, i, confidence, multiplier)
		means = zip(*means)
		means_control = analyze_slip_frame(data_direc_noinfect, pos_mask_direc, i, confidence, multiplier)
		means_control = zip(*means_control)
		FITC_mean = FITC_mean+list(means[0])
		cherry_mean = cherry_mean+list(means[1])
		FITC_control = FITC_control+list(means[0])
		cherry_control = cherry_control+list(means[1])

	strain = None
	if verbose or plot:
		strain = pos_to_strain(strain_matrix, pos)

	return extract_lysis_ratio(FITC_mean, cherry_mean, FITC_control, cherry_control, gaussian_confidence, verbose, plot, save_direc, strain)

def analyze_slip_plate(infect_direc, control_direc, mask_direc, plate_num, num_pos = 25, threshold=0.75, gaussian_confidence = 0.99999, multiplier = 1.5, verbose=False, plot=False, save_direc=None):
	alphabet = ['A','B','C','D','E','F','G','H']
	columns = range(1,13)
	ratio_matrix = np.zeros([8, 12])
	
	for row in range(0,8):
		for column in columns:
			pos = alphabet[row] + str(column)
			print "Analyzing Plate #" + str(plate_num)
			ratio_matrix[row, column] = analyze_slip_pos(infect_direc, control_direc, mask_direc, pos, num_pos, threshold, gaussian_confidence, multiplier, verbose, plot, save_direc)
	
	return ratio_matrix

def extract_lysis_ratio(FITC_mean, cherry_mean, FITC_control, cherry_control, confidence, verbose = False, plot = False, plot_save_direc=None, strain=None):
	FITC_mean_np = np.asarray(FITC_mean)
	cherry_mean_np = np.asarray(cherry_mean)
	FITC_control_np = np.asarray(FITC_control)
	cherry_control_np = np.asarray(cherry_control)

	FITC_infected = gaussian_classifier(FITC_control_np.reshape(-1,1), FITC_mean_np.reshape(-1,1), confidence)
	cherry_infected = gaussian_classifier(cherry_control_np.reshape(-1,1), cherry_mean_np.reshape(-1,1), confidence)

	lysis_FITC = []
	lysis_cherry = []
	lyso_FITC = []
	lyso_cherry = []
	for i in FITC_infected:
		lysis_index = FITC_mean.index(i)
		if FITC_mean[lysis_index] > cherry_mean[lysis_index]:
			lysis_FITC.append(FITC_mean[lysis_index])
			lysis_cherry.append(cherry_mean[lysis_index])

	for i in cherry_infected:
		lyso_index = cherry_mean.index(i)
		if FITC_mean[lyso_index] < cherry_mean[lyso_index]:
			lyso_FITC.append(FITC_mean[lyso_index])
			lyso_cherry.append(cherry_mean[lyso_index])

	zipped_all = zip(FITC_mean, cherry_mean)
	zipped_lysis = zip(lysis_FITC,lysis_cherry)
	zipped_lyso = zip(lyso_FITC,lyso_cherry)

	uninfected = set(zipped_all) - set(zipped_lysis) - set(zipped_lyso)

	uninfected = zip(*uninfected)
	
	if verbose == True:
		#print uninfected
		n_lysis = float(len(lysis_FITC))
		n_lyso = float(len(lyso_FITC))
		print('Lysis Count: ' + str(n_lysis))
		print('Lysogeny Count: ' + str(n_lyso))
		if (n_lyso + n_lysis) == 0:
			print('Lysis ratio = ' + str(float(0)))
		else:
			print('Lysis ratio = ' + str(n_lysis/(n_lysis+n_lyso)))

	if plot == True: 
		fig = plt.figure()	
		plt.plot(uninfected[0], uninfected[1],'bo')
		plt.plot(lysis_FITC, lysis_cherry,'go')
		plt.plot(lyso_FITC, lyso_cherry, 'ro')
		plt.xlabel('FITC Fluorescence')
		plt.ylabel('Cherry Fluorescence')
		filename = os.path.join(plot_save_direc, strain + '_classified_gcc' + '.png')
		plt.savefig(filename, format = 'png')
		plt.close(fig)

	return n_lysis/(n_lysis+n_lyso)



