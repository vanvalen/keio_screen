import numpy as np
import os
import matplotlib.pyplot as plt 
import openpyxl as xls

#Convert keio plate map excel into 3D matrix of names (plate,row, column)
def get_keio_names():

	data_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/keio_screen/'
	data_path = os.path.join(data_direc,'keio_map.xlsx')

	wb = xls.load_workbook(data_path)
	sheet = wb.get_sheet_by_name('Sheet1')

	keio_names = np.empty((47,8,12), dtype=object)

	for plate in range(0, 47):
		for row in range(1,9):
			for column in range(1,13):
				keio_names[plate ,row-1, column-1] = sheet.cell(row=row*2 + plate*16, column=column).value
				#print plate, row-1, column-1, keio_names[plate ,row-1, column-1]

	return keio_names

#Given the strain name matrix (strains), plate, row and column, return the strain name
#plate, row and column are assumed to be 1 indexed
def loc_to_strain(strains, plate, row, column):
	plate = (plate-1)/2
	return strains[plate, row-1, column-1]

def pos_to_strain(strains, plate, pos):
	alphabet = ['A','B','C','D','E','F','G','H']
	chars = list(pos)
	row = alphabet.index(chars[0]) + 1
	print row
	column = int(chars[1])
	plate = (plate-1)/2
	return strains[plate, row-1, column-1]

