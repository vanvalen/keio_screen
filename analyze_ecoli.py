import numpy as np
import os
import tifffile as tiff
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
from functions import analyze_ecoli, plot_intensity_scatter, extract_lysis_ratio, print_ratio, print_gaussian_ratio, print_gaussian_ratio_no_infect, print_gaussian_ratio_2D
from gaussian_model import extract_infected
from sklearn import mixture

media = 'maltose_trial3'
control = 'maltose_noinfection'
cutoff_multiplier = 1.5
confidence = 1.0
confidence_2 = 0.9999999  #for no infection control gaussian classifier
print('Media tested: ' + media)

###### CONTROL ###########
pos=0
frame=0
data_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/4.19.17/BW25113-NQ009-' + control + '_1/Pos' + str(pos)
mask_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/4.19.17/BW25113-NQ009-' + control + '_1/Pos' + str(pos)
save_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/plots/5.2.17/'
means = analyze_ecoli(data_direc, mask_direc, pos=pos, frame=frame, confidence = 0.75, multiplier = 1.5, FITC_channel = 1, cherry_channel = 2)
means = zip(*means)
FITC_control = list(means[0])
cherry_control = list(means[1])

#### DATA #####
pos=0
frame=0
data_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/4.19.17/BW25113-NQ009-' + media + '_1/Pos' + str(pos)
mask_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/4.19.17/BW25113-NQ009-' + media + '_1/Pos' + str(pos)
save_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/plots/5.2.17/'
means = analyze_ecoli(data_direc, mask_direc, pos=pos, frame=frame, confidence = 0.75, multiplier = 1.5, FITC_channel = 1, cherry_channel = 2)
means = zip(*means)
FITC_mean = list(means[0])
cherry_mean = list(means[1])


pos=1
frame=0
data_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/4.19.17/BW25113-NQ009-' + media + '_1/Pos' + str(pos)
mask_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/4.19.17/BW25113-NQ009-' + media + '_1/Pos' + str(pos)
save_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/plots/5.2.17/'
means = analyze_ecoli(data_direc, mask_direc, pos=pos, frame=frame, confidence = 0.75, multiplier = 1.5, FITC_channel = 1, cherry_channel = 2)

means = zip(*means)
FITC_mean = FITC_mean+list(means[0])
cherry_mean = cherry_mean+list(means[1])

pos=2
frame=0
data_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/4.19.17/BW25113-NQ009-' + media + '_1/Pos' + str(pos)
mask_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/4.19.17/BW25113-NQ009-' + media + '_1/Pos' + str(pos)
save_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/plots/5.2.17/'
means = analyze_ecoli(data_direc, mask_direc, pos=pos, frame=frame, confidence = 0.75, multiplier = 1.5, FITC_channel = 1, cherry_channel = 2)

means = zip(*means)
FITC_mean = FITC_mean+list(means[0])
cherry_mean = cherry_mean+list(means[1])

means = zip(*means)
xmax = max(FITC_mean)
xmin = min(FITC_mean)
ymax = max(cherry_mean)
ymin = min(cherry_mean)
fig = plt.figure()
plt.plot(FITC_mean, cherry_mean,'o')
plt.axis([xmin, xmax, ymin, ymax])
plt.xlabel('FITC Pixel Intensity')
plt.ylabel('Cherry Pixel Intensity')
plt.title('Test: ' + media)
filename = os.path.join(save_direc, media + '_all' + '.png')
#print('Processing position ' + str(pos) + ' frame ' + str(frame))
plt.savefig(filename, format = 'png')
plt.close(fig)
print_ratio(FITC_mean, cherry_mean, cutoff_multiplier)
print 'Gaussian Mixture Model:'
print 'Confidence: ' + str(confidence)

print_gaussian_ratio(FITC_mean, cherry_mean, confidence, save_direc, media)

print 'Gaussian Classifier:'
print 'Confidence: ' + str(confidence_2)

print_gaussian_ratio_no_infect(FITC_mean, cherry_mean, FITC_control, cherry_control, confidence_2, save_direc, media)

print 'Gaussian Mixture Model 2D:'
print 'Confidence: ' + str(confidence)
print_gaussian_ratio_2D(FITC_mean, cherry_mean, confidence, save_direc, media)

#FITC_mean = np.asarray(FITC_mean)
#infected = count_infected(FITC_mean.reshape(-1,1), 0.75)
#print infected


'''
lysis_ratio = []
for frame in range(0,46, 1):
	lysis = extract_lysis_ratio(mask_direc, data_direc, pos = pos, frame = frame, cutoff_multiplier = 1.5, confidence = 0.75, multiplier = 5, FITC_channel = 1, cherry_channel = 2)
	lysis_ratio.append(lysis)

fig = plt.figure()
plt.plot(range(0, 46, 1), lysis_ratio)
plt.xlabel('Frame')
plt.ylabel('Lysis ratio')
filename = os.path.join(save_direc, 'lysis_ratio.pdf')
plt.savefig(filename, format = 'pdf')
plt.close(fig)
'''
