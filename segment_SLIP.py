from SLIP_functions import segment_SLIP
import os

#Define root directory path
root_direc = None
#define directory path to infection data (all positions)
data_direc = os.path.join(root_direc, 'data')
#define directory path to control data (all positions)
control_direc = os.path.join(root_direc, 'control')
#define directory path to where you want to store neural net outputs. 
#mask directories must exist at run time!
mask_direc = os.path.join(root_direc, 'masks')
control_mask_direc = os.path.join(root_direc,'control_masks')

segment_SLIP(data_direc, control_mask_direc, mask_direc, control_mask_direc)