import numpy as np
import os
import tifffile as tiff
from skimage.io import imread
from skimage.measure import label, regionprops
import scipy
import matplotlib.pyplot as plt
from analyze import analyze_slip_pos
from gaussian_model import extract_infected
from sklearn import mixture

infect_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/5.15.17_SLIP/infection/'
control_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/5.15.17_SLIP/no_infection/'
mask_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/microscope/5.15.17_SLIP/masks/'
save_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/plots/5.22.17/'
pos = 'A7'

gaussian_confidence = 0.99999

analyze_slip_pos(infect_direc, control_direc, save_direc, mask_direc, pos, num_pos = 25, confidence = 0.75, gaussian_confidence = gaussian_confidence, multiplier = 1.5)

