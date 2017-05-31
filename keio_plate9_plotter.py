import numpy as np
import os
import matplotlib.pyplot as plt 
import openpyxl as xls
from keio_names import get_keio_names

plate = 9

plate = (plate-1)/2

keio_strains = get_keio_names()

data_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/phage_engr/'
data_path = os.path.join(data_direc,'keio_plate9_try3.xlsx')
plot_direc = '/Users/nicolasquach/Documents/stanford/covert_lab/phage_engr/plots/keio_plate9_try3'

wb = xls.load_workbook(data_path)
sheet = wb.get_sheet_by_name('Sheet1')

nsteps = 83

proc_data = np.zeros((nsteps, 96))

for i in range(0, nsteps):
	for j in range(1, 97):
		 proc_data[i, j-1] = sheet.cell(row=i*96+j, column=2).value

t = range(0,nsteps*5, 5)

names = ['A','B','C','D','E','F','G','H']
style = ['-r','-b','-g','-m','-k','--r','--b','--g','--m','--k','+k','+b']

#A1-A12

for m in range(0,8):
	legend_tag = []
	alpha_range = range(0+m,12+m)
	fig = plt.figure()
	for k in alpha_range:
		plt.plot(t, np.subtract(proc_data[:,k], proc_data[0,k]), style[k-m])
		legend_tag.append(keio_strains[plate, m, k-m])
	plt.xlabel('Time (min)')
	plt.ylabel('OD600')
	title = 'Wells ' + names[m] + str(1) + '-' + names[m] + str(12)
	plt.title(title)
	plt.legend(legend_tag ,loc='upper left')
	save_path = os.path.join(plot_direc, names[m]+'_curves.pdf')
	plt.savefig(save_path, format='pdf')
	plt.close(fig)

'''

fig0 = plt.figure()
for k in range(0, 11):
	plt.plot(t,proc_data[:,k])

save_path = os.path.join(plot_direc, 'A_curves.pdf')
plt.savefig(save_path, format='pdf')
plt.close(fig0)
'''

