import numpy as np
import os
import tifffile as tiff
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
from gaussian_model import extract_infected, gaussian_classifier, gmm_2D

plt.ioff()

#analyzes a single frame of SLIP data
#inputs: data_direc = path to 
def analyze_slip_frame(data_direc, mask_direc, frame, confidence = 0.75, multiplier = 1.5):
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

def analyze_slip_pos(infect_direc, control_direc, save_direc, mask_direc, pos, num_pos = 25, confidence = 0.75, gaussian_confidence = 1.0, multiplier = 1.5):
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

	print_gaussian_ratio_no_infect(FITC_mean, cherry_mean, FITC_control, cherry_control, gaussian_confidence, save_direc, pos)
		




def analyze_ecoli(data_direc, mask_direc, pos, frame, confidence = 0.75, multiplier = 1.5, FITC_channel = 1, cherry_channel = 2):
	mask_name = os.path.join(mask_direc, 'feature_1_frame_' + str(frame) + '.tif')
	#Assume channel001 is FITC
	FITC_name = os.path.join(data_direc, 'img_channel' + str(FITC_channel).zfill(3) + '_position' + str(pos).zfill(3) + '_time' + str(frame).zfill(9) + '_z000.tif')
	cherry_name = os.path.join(data_direc, 'img_channel' + str(cherry_channel).zfill(3) + '_position' + str(pos).zfill(3) + '_time' + str(frame).zfill(9) + '_z000.tif')
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


def plot_intensity_scatter(mask_direc, data_direc, save_direc, pos, step, confidence = 0.75, multiplier = 1.5, FITC_channel = 1, cherry_channel = 2):
	num_masks = len([name for name in os.listdir(mask_direc)]) - 1
	print('Number of masks: ' + str(num_masks))
	means = analyze_ecoli(data_direc = data_direc, mask_direc = mask_direc, pos = pos, frame = num_masks)
	means = zip(*means)
	xmax = max(means[0])
	xmin = min(means[0])
	ymax = max(means[1])
	ymin = min(means[1])

	for frame in range(0, num_masks, step):
		means = analyze_ecoli(data_direc = data_direc, mask_direc = mask_direc, pos = pos, frame = frame, confidence = confidence, multiplier = multiplier, FITC_channel = FITC_channel, cherry_channel = cherry_channel)
		means = zip(*means)
		fig = plt.figure()
		plt.plot(means[0], means[1],'o')
		plt.axis([xmin, xmax, ymin, ymax])
		plt.xlabel('FITC Pixel Intensity')
		plt.ylabel('Cherry Pixel Intensity')
		plt.title('Pos ' + str(pos) + ' Frame ' + str(frame))
		filename = os.path.join(save_direc, 'pos' + str(pos) + '_frame' + str(frame) + '.png')
		print('Processing position ' + str(pos) + ' frame ' + str(frame))
		plt.savefig(filename, format = 'png')
		plt.close(fig)

def extract_lysis_ratio(mask_direc, data_direc, pos, frame, cutoff_multiplier = 1.5, confidence = 0.75, multiplier = 1.5, FITC_channel = 1, cherry_channel = 2):
	means = analyze_ecoli(data_direc = data_direc, mask_direc = mask_direc, pos = pos, frame = frame, confidence = confidence, multiplier = multiplier, FITC_channel = FITC_channel, cherry_channel = cherry_channel)
	means = zip(*means)
	FITC = means[0]
	cherry = means[1]
	iqr_cherry = np.subtract(*np.percentile(cherry,[75, 25]))
	iqr_FITC = np.subtract(*np.percentile(FITC,[75, 25]))
	threshold_cherry = np.percentile(cherry,75) + cutoff_multiplier*iqr_cherry
	threshold_FITC = np.percentile(FITC,75) + cutoff_multiplier*iqr_FITC
	lysis = [i for i in cherry if i >= threshold_cherry]
	lyso = [i for i in FITC if i >= threshold_FITC]
	if (len(lysis) + len(lyso) == 0):
		return float(0)
	#print(len(lysis))
	#print(len(lyso))
	return float(len(lysis))/(float(len(lyso))+float(len(lysis)))

def print_ratio(FITC_mean,cherry_mean,cutoff_multiplier):
	FITC = FITC_mean
	cherry = cherry_mean
	iqr_cherry = np.subtract(*np.percentile(cherry,[75, 25]))
	iqr_FITC = np.subtract(*np.percentile(FITC,[75, 25]))
	threshold_cherry = np.percentile(cherry,75) + cutoff_multiplier*iqr_cherry
	threshold_FITC = np.percentile(FITC,75) + cutoff_multiplier*iqr_FITC
	lysis = [i for i in FITC if i >= threshold_cherry]
	lyso = [i for i in cherry if i >= threshold_FITC]
	print('Lysis Count: ' + str(len(lysis)))
	print('Lysogeny Count: ' + str(len(lyso)))
	if (len(lysis) + len(lyso) == 0):
		print('Lysis ratio = ' + str(float(0)))
	#print(len(lysis))
	#print(len(lyso))
	else: 
		print('Lysis ratio = ' + str(float(len(lysis))/(float(len(lyso))+float(len(lysis)))))

def print_gaussian_ratio(FITC_mean, cherry_mean, confidence, plot_save_direc, media):
	FITC_mean_np = np.asarray(FITC_mean)
	cherry_mean_np = np.asarray(cherry_mean)
	FITC_infected = extract_infected(FITC_mean_np.reshape(-1,1), confidence)
	cherry_infected = extract_infected(cherry_mean_np.reshape(-1,1), confidence)

	lysis_FITC = []
	lysis_cherry = []
	lyso_FITC = []
	lyso_cherry = []
	for i in FITC_infected:
		lysis_index = FITC_mean.index(i)
		if FITC_mean[lysis_index] > cherry_mean[lysis_index]:
			lysis_FITC.append(FITC_mean[lysis_index])
			lysis_cherry.append(cherry_mean[lysis_index])
		#else:  #to prevent misclassification
	#		lyso_FITC.append(FITC_mean[lysis_index])  
#			lyso_cherry.append(cherry_mean[lysis_index])

	for i in cherry_infected:
		lyso_index = cherry_mean.index(i)
		if FITC_mean[lyso_index] < cherry_mean[lyso_index]:
			lyso_FITC.append(FITC_mean[lyso_index])
			lyso_cherry.append(cherry_mean[lyso_index])
	#	else:  #to prevent misclassification
#			lysis_FITC.append(FITC_mean[lyso_index])  
#			lysis_cherry.append(cherry_mean[lyso_index])

	zipped_all = zip(FITC_mean, cherry_mean)
	zipped_lysis = zip(lysis_FITC,lysis_cherry)
	zipped_lyso = zip(lyso_FITC,lyso_cherry)

	uninfected = set(zipped_all) - set(zipped_lysis) - set(zipped_lyso)

	uninfected = zip(*uninfected)
	
	#print uninfected
	n_lysis = float(len(lysis_FITC))
	n_lyso = float(len(lyso_FITC))
	print('Lysis Count: ' + str(n_lysis))
	print('Lysogeny Count: ' + str(n_lyso))
	if (n_lyso + n_lysis) == 0:
		print('Lysis ratio = ' + str(float(0)))
	else:
		print('Lysis ratio = ' + str(n_lysis/(n_lysis+n_lyso)))

	fig = plt.figure()
	#plt.plot(lysis_FITC, lysis_cherry, 'o')
	#plt.plot(lyso_FITC, lyso_cherry, 'o')
	plt.plot(uninfected[0], uninfected[1],'bo')
	plt.plot(lysis_FITC, lysis_cherry,'go')
	plt.plot(lyso_FITC, lyso_cherry, 'ro')
	plt.xlabel('FITC Fluorescence')
	plt.ylabel('Cherry Fluorescence')
	filename = os.path.join(plot_save_direc, media + '_classified' + '.png')
	#print('Processing position ' + str(pos) + ' frame ' + str(frame))
	plt.savefig(filename, format = 'png')
	plt.close(fig)
	
def print_gaussian_ratio_2D(FITC_mean, cherry_mean, confidence, plot_save_direc, media):
	FITC_mean_np = np.asarray(FITC_mean)
	cherry_mean_np = np.asarray(cherry_mean)
	data = gmm_2D(FITC_mean_np.reshape(-1,1), cherry_mean_np.reshape(-1,1), confidence)
	#print data.shape
	data0 = []
	data1 = []
	data2 = []
	for i in range(0, data.shape[0]):
		if data[i, 2] == 0: 
			data0.append([data[i,0], data[i,1]])
		elif data[i,2] == 1:
			data1.append([data[i,0], data[i,1]])
		else:
			data2.append([data[i,0], data[i,1]])

	data0 = np.asarray(data0)
	data1 = np.asarray(data1)
	data2 = np.asarray(data2)
	mean0 = np.mean(data0, axis=0)
	mean1 = np.mean(data1, axis=0)
	mean2 = np.mean(data2, axis=0)
	mean_all = np.vstack([mean0, mean1, mean2])
	max_means = np.argmax(mean_all, axis=0)

	lysis = []
	lyso = []
	no_infect = []

	if np.all(max_means == [0, 1]):
		lysis = data0
		lyso = data1
		no_infect = data2
	elif np.all(max_means == [0, 2]):
		lysis = data0
		lyso = data2
		no_infect = data1
	elif np.all(max_means == [1, 0]):
		lysis = data1
		lyso = data0
		no_infect = data2
	elif np.all(max_means == [1, 2]):
		lysis = data1
		lyso = data2
		no_infect = data0
	elif np.all(max_means == [2, 0]):
		lysis = data2
		lyso = data0
		no_infect = data1
	else:
		lysis = data2
		lyso = data1
		no_infect = data0

	n_lysis = float(len(lysis))
	n_lyso = float(len(lyso))
	print('Lysis Count: ' + str(n_lysis))
	print('Lysogeny Count: ' + str(n_lyso))
	if (n_lyso + n_lysis) == 0:
		print('Lysis ratio = ' + str(float(0)))
	else:
		print('Lysis ratio = ' + str(n_lysis/(n_lysis+n_lyso)))

	lysis = np.asarray(lysis)
	lyso = np.asarray(lyso)
	no_infect = np.asarray(no_infect)
	fig = plt.figure()
	plt.plot(lysis[:,0], lysis[:,1], 'go')
	plt.plot(lyso[:,0], lyso[:,1], 'ro')
	plt.plot(no_infect[:,0], no_infect[:,1], 'bo')
	plt.xlabel('FITC Fluorescence')
	plt.ylabel('Cherry Fluorescence')
	filename = os.path.join(plot_save_direc, media + '_classified_gmm2D' + '.png')
	plt.savefig(filename, format = 'png')
	plt.close(fig)



def print_gaussian_ratio_no_infect(FITC_mean, cherry_mean, FITC_control, cherry_control, confidence, plot_save_direc, media):
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
		#else:  #to prevent misclassification
	#		lyso_FITC.append(FITC_mean[lysis_index])  
#			lyso_cherry.append(cherry_mean[lysis_index])

	for i in cherry_infected:
		lyso_index = cherry_mean.index(i)
		if FITC_mean[lyso_index] < cherry_mean[lyso_index]:
			lyso_FITC.append(FITC_mean[lyso_index])
			lyso_cherry.append(cherry_mean[lyso_index])
	#	else:  #to prevent misclassification
#			lysis_FITC.append(FITC_mean[lyso_index])  
#			lysis_cherry.append(cherry_mean[lyso_index])

	zipped_all = zip(FITC_mean, cherry_mean)
	zipped_lysis = zip(lysis_FITC,lysis_cherry)
	zipped_lyso = zip(lyso_FITC,lyso_cherry)

	uninfected = set(zipped_all) - set(zipped_lysis) - set(zipped_lyso)

	uninfected = zip(*uninfected)
	
	#print uninfected
	n_lysis = float(len(lysis_FITC))
	n_lyso = float(len(lyso_FITC))
	print('Lysis Count: ' + str(n_lysis))
	print('Lysogeny Count: ' + str(n_lyso))
	if (n_lyso + n_lysis) == 0:
		print('Lysis ratio = ' + str(float(0)))
	else:
		print('Lysis ratio = ' + str(n_lysis/(n_lysis+n_lyso)))

	fig = plt.figure()
	#plt.plot(lysis_FITC, lysis_cherry, 'o')
	#plt.plot(lyso_FITC, lyso_cherry, 'o')
	plt.plot(uninfected[0], uninfected[1],'bo')
	plt.plot(lysis_FITC, lysis_cherry,'go')
	plt.plot(lyso_FITC, lyso_cherry, 'ro')
	plt.xlabel('FITC Fluorescence')
	plt.ylabel('Cherry Fluorescence')
	filename = os.path.join(plot_save_direc, media + '_classified_gcc' + '.png')
	#print('Processing position ' + str(pos) + ' frame ' + str(frame))
	plt.savefig(filename, format = 'png')
	plt.close(fig)

def extract_lyso_ratio(mask_direc, data_direc, pos, frame, cutoff_multiplier = 1.5, confidence = 0.75, multiplier = 1.5, FITC_channel = 1, cherry_channel = 2):
	means = analyze_ecoli(data_direc = data_direc, mask_direc = mask_direc, pos = pos, frame = frame, confidence = confidence, multiplier = multiplier, FITC_channel = FITC_channel, cherry_channel = cherry_channel)
	means = zip(*means)
	FITC = means[0]
	cherry = means[1]
	iqr_cherry = np.subtract(*np.percentile(cherry,[75, 25]))
	iqr_FITC = np.subtract(*np.percentile(FITC,[75, 25]))
	threshold_cherry = np.percentile(cherry,75) + cutoff_multiplier*iqr_cherry
	threshold_FITC = np.percentile(FITC,75) + cutoff_multiplier*iqr_FITC
	lysis = [i for i in cherry if i >= threshold_cherry]
	lyso = [i for i in FITC if i >= threshold_FITC]
	if (len(lysis) + len(lyso) == 0):
		return float(0)
	#print(len(lysis))
	#print(len(lyso))
	return float(len(lyso))/(float(len(lyso))+float(len(lysis)))

'''

plt.figure(1)
plt.plot(range(1,20),total,'k-')
plt.xlabel('IQR multiplier')
plt.ylabel('Total infections counted')
filename = '/home/nquach/DeepCell2/prototypes/plots/21317_plots/total.pdf'
plt.savefig(filename, format = 'pdf')
plt.close

threshold_cherry = np.percentile(mean_cherry,75)+10*iqr_cherry
threshold_FTIC = np.percentile(mean_FTIC,75)+10*iqr_FTIC

lysis = [i for i in mean_cherry if i >= threshold_cherry]
lyso = [i for i in mean_FTIC if i >= threshold_FTIC]
print(len(lysis))
print(len(lyso))
print(float(len(lysis))/(float(len(lyso))+float(len(lysis))))
'''

'''
plt.figure(0)
plt.plot(mean_FTIC, mean_cherry, 'o')
plt.plot((0,2500),(threshold_cherry, threshold_cherry), 'k-')
plt.plot((threshold_FTIC, threshold_FTIC),(0,50000),'k-')
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Mean FTIC Intensity')
plt.ylabel('Mean Cherry Intensity')
plt.title('Pos 0 Frame 58')
filename = '/home/nquach/DeepCell2/prototypes/plots/21317_plots/correlation.pdf'
plt.savefig(filename, format = 'pdf')
plt.close

plt.figure(1)
binwidth = 10
plt.hist(mean_FTIC, bins=np.arange(min(mean_FTIC), max(mean_FTIC) + binwidth, binwidth))
#plt.hist(mean_FTIC)
#plt.xscale('log')
plt.xlabel('Mean Pixel Intensity')
plt.ylabel('Frequency')
plt.title('FTIC Channel')
filename = '/home/nquach/DeepCell2/prototypes/plots/21317_plots/FTIC_hist.pdf'
plt.savefig(filename, format = 'pdf')
plt.close

plt.figure(2)
binwidth = 10
plt.hist(mean_cherry, bins=np.arange(min(mean_cherry), max(mean_cherry) + binwidth, binwidth))
#plt.xscale('log')
plt.xlabel('Mean Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Cherry Channel')
filename = '/home/nquach/DeepCell2/prototypes/plots/21317_plots/cherry_hist.pdf'
plt.savefig(filename, format = 'pdf')
plt.close

'''
