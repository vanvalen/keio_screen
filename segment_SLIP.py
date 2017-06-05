from SLIP_functions import segment_SLIP

#define directory path to infection data (all positions)
data_direc = None
#define directory path to control data (all positions)
control_direc = None
#define directory path to where you want to store neural net outputs
mask_direc = None
control_mask_direc = None

segment_SLIP(data_direc, control_mask_direc, mask_direc, control_mask_direc)