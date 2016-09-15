import numpy as np
from numpy import *
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict


font = {'family' : 'serif', 'serif' : ['Times'], 'size' : '35'}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times'
matplotlib.rcParams['mathtext.it'] = 'Times'
plt.rc('legend', **{'fontsize':30})


#################################################################################################################
###             																							  ###
### THIS CODE READS self.vol, .zone, AND .void FILES TO GET ALL ZONES, VOIDS, AND VOLUMES FROM A ZOBOV OUTPUT ###
###																											  ###
#################################################################################################################

def read_zone(filename):
	#open text file that was output of jozov, which contains all the zones per particle
	f = open(filename,'r') 

	#Create array for zones 
	zone = []

	rows = f.readlines()

	#Get number of particles and zones
	firstrow = rows[0].split()
	numpart = firstrow[0]
	numzones = firstrow[1]

	#Create loop to populate arrays for given the output file
	for row in rows[1:]:
		zone.append(row)

	#Convert all values to floats
	zone = np.array(zone).astype(np.float)
	numpart = float(numpart)
	numzones = float(numzones)

	return numpart, numzones, zone 

def read_vol(filename, Lbox):
	#Open .txt vol file and extract volume of each cell in units of Mpc/h^3
	f = open(filename, 'r')

	vol = []

	rows = f.readlines()

	#Create loop to read volume for each cell
	for row in rows:
		data = row.split() #split rows to get the cell id and volume
		vol.append(data[1])

	numpart_vol = len(vol)
	vol = np.array(vol).astype(np.float) #convert volume to floats

	# The following converts the volume to comoving unit of Mpc/h^3
	vol = vol*(Lbox**3.)/numpart_vol


	return numpart_vol, vol	

def read_vol_zone_txt(filename):
	# Open file that contains 9 columns with and ID, volume weighted xyz, den min of xyz, void radius [Mpc/h], and void density contrast
	f = open(filename, 'r')

	ID = []
	x_vol = []
	y_vol = []
	z_vol = []
	x_denmin = []
	y_denmin = []
	z_denmin = []
	zone_rad = [] # Mpc/h
	dencon = []

	rows = f.readlines()

	# Loop through all the columns and extract each parameter
	for row in rows[1:]:
		data = row.split()
		ID.append(data[0])
		x_vol.append(data[1])
		y_vol.append(data[2])
		z_vol.append(data[3])
		x_denmin.append(data[4])
		y_denmin.append(data[5])
		z_denmin.append(data[6])
		zone_rad.append(data[7])
		dencon.append(data[8])

	ID = np.array(ID).astype(np.float)
	x_vol = np.array(x_vol).astype(np.float)
	y_vol = np.array(y_vol).astype(np.float)
	z_vol = np.array(z_vol).astype(np.float)
	x_denmin = np.array(x_denmin).astype(np.float)
	y_denmin = np.array(y_denmin).astype(np.float)
	z_denmin = np.array(z_denmin).astype(np.float)
	zone_rad = np.array(zone_rad).astype(np.float)
	dencon = np.array(dencon).astype(np.float)


	return ID, x_vol, y_vol, z_vol, x_denmin, y_denmin, z_denmin, zone_rad, dencon

def read_void(filename):
	f = open(filename, 'r')

	rows = f.readlines()

	void_id = []
	num_voids = rows[0]

	for row in rows[1:]:
		# Split line
		new_line = [float(i) for i in row.split()]
		
		# Get line id, split again
		line_id = [new_line[0]]
		new_line = new_line[1:]

		# Get first num of substructures
		sub_next = new_line[0]
		new_line = new_line[1:]

		structs_to_add = []
		while sub_next != 0:
			structs_to_add = structs_to_add + new_line[1:int(sub_next)+1]
			sub_next_temp = sub_next
			sub_next = new_line[int(sub_next)+1]
			new_line = new_line[int(sub_next_temp)+2:]

		final_line = line_id + structs_to_add
		void_id.append(final_line)


	return num_voids, void_id

def read_AllHalo(filename):
	### Read the AllHalo file and return x,y,z coordinates of all halos in Mpc/h

	f = open(filename,'r')

	x = []
	y = []
	z = []

	rows = f.readlines()
	for row in rows[1:]:
		data = row.split()
		x.append(data[0])
		y.append(data[1])
		z.append(data[2])

	x = np.array(x).astype(np.float)
	y = np.array(y).astype(np.float)
	z = np.array(z).astype(np.float)

	return x,y,z

def read_adj(filename):
	# Read adjacency file and sort by placing each halo with it's adjacent halos
	# Note that this file is sorted such that if a halo number is adjacent to a halo with a smaller number than it,
	# e.g. if 5 and 13 are adjacent, then '5' wont show up in '13' because it was already accounted for in '5'

	f = open(filename,'r') 

	# Create dictionary for each halo to contain it's adjacent halos
	d = defaultdict(list)

	rows = f.readlines()

	for i,row in enumerate(rows):
		if i%2 == 0:
			halo = np.int(row.split()[0]) # Get which halo adjacencies are for

			adj_halos = []
			adj_halos_temp = rows[i+1]

			# Split and loop over each value in adjacent halos to create a list with their values
			split_halos = adj_halos_temp.split()
			for x in split_halos:
				if x != '#':
					adj_halos.append(np.int(x))
				d[halo] = adj_halos # Add all adjacent halos to current halo
	
	# particle = (len(rows)/2)-1
	# last_adj = [key for key in d.keys() if particle -1 in d[key]] #returns array of keys which contains a certain value ie 'particle'
	
	# Loop over dictionary and add adjacencies that are below the current halo
	for v in range(0,len(rows)/2):
		
		# Array of adjacent values for particle 'v'
		arr = d[v]

		for part in arr:
			if part > v:
				d[part].extend([v])
				
	return d


def find_nearest(array,value):
	#returns index of value closest to input value in given array
	idx = (np.abs(array-value)).argmin()
	return idx

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None












