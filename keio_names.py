import numpy as np
import os
import matplotlib.pyplot as plt 
import openpyxl as xls

#Convert keio plate map excel into 3D matrix of names (plate,row, column)
def get_keio_names():

	#Define path to keio_map.xlsx file
	data_direc = '/home/vanvalen/keio_screen/'
	data_path = os.path.join(data_direc,'keio_map.xlsx')

	#Load excel sheet
	wb = xls.load_workbook(data_path)
	sheet = wb.get_sheet_by_name('Sheet1')

	#will store names here
	keio_names = np.empty((47,8,12), dtype=object)

	#Parse excel sheet
	for plate in range(0, 47):
		for row in range(1,9):
			for column in range(1,13):
				keio_names[plate ,row-1, column-1] = sheet.cell(row=row*2 + plate*16, column=column).value

	return keio_names

#Given the strain name matrix (strains), plate, row and column, return the strain name
#plate, row and column are assumed to be 1 indexed
def loc_to_strain(strains, plate, row, column):
	plate = (plate-1)/2
	return strains[plate, row-1, column-1]

#Converts a position (like 'A7') into a strain name
def pos_to_strain(strains, plate, pos):
	alphabet = ['A','B','C','D','E','F','G','H']
	chars = list(pos)
	row = alphabet.index(chars[0]) + 1
	if len(chars) == 2:
		column = 13 - int(chars[1])
	if len(chars) == 3:
		column = 13 - int(chars[1] + chars[2])
	plate = (plate-1)/2
	return strains[plate, row-1, column-1]

