import numpy as np
import os
import tifffile as tiff
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle
from SLIP_functions import analyze_well, analyze_plate
#from analyze import analyze_slip_pos
#from gaussian_model import extract_infected
#from sklearn import mixture
#from keio_names import get_keio_names, loc_to_strain, pos_to_strain

root_direc = '/home/vanvalen/Data/keio_screen/plate_9/'
infect_direc = os.path.join(root_direc,'data')
control_direc = os.path.join(root_direc,'control')
mask_direc = os.path.join(root_direc, 'masks')
control_mask_direc = os.path.join(root_direc, 'control_masks')

#pos = 'A7'
plate_num = 9
plate = (plate_num-1)/2

data_directory = os.path.join(control_direc)
mask_directory = os.path.join(control_mask_direc)
# mean_FITC, mean_cherry = analyze_plate(data_directory, mask_directory, pos_list = range(9))

mean_FITC_name = os.path.join(root_direc, 'mean_FITC_control.pkl')
mean_cherry_name = os.path.join(root_direc, 'mean_cherry_control.pkl')

# pickle.dump(mean_FITC, open(mean_FITC_name, 'wb'))
# pickle.dump(mean_cherry, open(mean_cherry_name, 'wb'))

mean_FITC = pickle.load(open(mean_FITC_name, 'rb'))
mean_cherry = pickle.load(open(mean_cherry_name, 'rb'))

for key in mean_FITC.keys():
	print key
	print len(mean_FITC[key])
	# print np.mean(mean_FITC[key]), np.std(mean_FITC[key])
	# print np.mean(mean_cherry[key]), np.std(mean_cherry[key])
#gaussian_confidence = 0.99999

#ratio_matrix = analyze_slip_plate(infect_direc, control_direc, save_direc, mask_direc, pos, num_pos = 25, confidence = 0.75, gaussian_confidence = gaussian_confidence, multiplier = 1.5)

#ratio_list = list(ratio_matrix.flatten())

#strain_matrix = get_keio_names()

#strain_list = list(strain_matrix[plate,:,:].flatten())