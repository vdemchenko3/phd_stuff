import numpy as np
from numpy import *
import matplotlib
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/Vasiliy/Desktop/PhD/Scripts/ZOBOV')
from read_vol_zone_void import *
from scipy import spatial
from periodic_kdtree import PeriodicCKDTree
import time
from itertools import cycle



font = {'family' : 'serif', 'serif' : ['Times'], 'size' : '35'}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times'
matplotlib.rcParams['mathtext.it'] = 'Times'
plt.rc('legend', **{'fontsize':30})

t_begin = time.time()

numpart, numzones, zone = read_zone('L1000.zone')
numpart_vol, vol = read_vol('L1000.vol.txt', 1000)
x,y,z = read_AllHalo('AllHalo_1000Mpc_a1.0_M12.8.txt')
ID, x_vol, y_vol, z_vol, x_denmin, y_denmin, z_denmin, zone_rad, dencon = read_vol_zone_txt('vol.zone.txt')
adj_dict = read_adj('adj.txt')


Lbox = 1000

def dist(x_orig, y_orig, z_orig, x_pt, y_pt, z_pt):
	d = sqrt( (x_orig - x_pt)**2 + (y_orig - y_pt)**2 + (z_orig - z_pt)**2)
	return d

def rm_duplicate(seq):
	# Removes duplicates in array while preserving order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def percent_diff(x,y):
	# Both values mean the same kind of thing (one value is not obviously older or better than the other
	percent_diff = abs((x-y)/((x+y)/2.))*100.
	return percent_diff

def percent_err(exact,est):
	# The difference between Approximate and Exact Values, as a percentage of the Exact Value.

	percent_err = abs((est-exact)/exact)*100.
	return percent_err

def percent_change(new,old):
	# Percentage of old value to new value 

	percent_change = ((new-old)/abs(old))*100.
	return percent_change

def coord_rot(coord,val,Lbox):
	# Rotate box to put a particular coordinate in the middle
	# This requires subtracting the coordinates of the cell from all coordinates making it at (0,0,0)
	# Then adding half the box size to each coordinates
	# Then adding the box size and take the np.mod of box size of all coordinates to get the desired
	# point at the center
	coord_rot = np.mod(((coord-val) + (Lbox+(Lbox/2.))),Lbox)
	return coord_rot

def spherical_stk(zone_rad_stk,x_zone_stk,y_zone_stk,z_zone_stk,numzone_stk):
	for r in range(0,(len(zone_rad_stk))):
		# Create array from ~0 to 2 in order for it to be the same for all different sized zones
		R_stk = np.linspace(0,2,20)
		# Array of volumes for each consecutive sphere for R values
		V = ((4.*pi)/3.)*(R_stk*zone_rad_stk[r])**3

		# Find index of halo representing min value of zone
		ind_x_stk = find_nearest(x,x_zone_stk[r])
		ind_y_stk = find_nearest(y,y_zone_stk[r])
		ind_z_stk = find_nearest(z,z_zone_stk[r])

		count_stk = []
		nden_stk = []
		for n in range(0,len(R_stk)):
			# This gives me a number density in each shell
			# looks for number of particles within a volume given by input radius
			if n==0:
				count_temp1 = len(periodic_tree.query_ball_point([x[ind_x_stk], y[ind_y_stk], z[ind_z_stk]], R_stk[n]*zone_rad_stk[r]))
				nden_temp1 = count_temp1/V[n]
			else:
				count_temp11 = len(periodic_tree.query_ball_point([x[ind_x_stk], y[ind_y_stk], z[ind_z_stk]], R_stk[n]*zone_rad_stk[r]))
				count_temp12 = len(periodic_tree.query_ball_point([x[ind_x_stk], y[ind_y_stk], z[ind_z_stk]], R_stk[n-1]*zone_rad_stk[r]))
				count_temp1 = count_temp11-count_temp12
		 		nden_temp1 = count_temp1/(V[n]-V[n-1])

			# Count of halos and number density in a each shell
		 	count_stk.append(count_temp1)
		 	nden_stk.append(nden_temp1)
		# Add elements of each zone's number count and number density per shell 
		if r==0:
			avg_count_temp = count_stk
			avg_nden_temp = nden_stk
		else:
			avg_count_temp = [a+b for a,b in zip(avg_count_temp,count_stk)]
			avg_nden_temp = [c+d for c,d in zip(avg_nden_temp,nden_stk)]

		# Divide total counts and number densities in shells by number of voids
		avg_cnt = np.array(avg_count_temp)/numzone_stk
		avg_nden = np.array(avg_nden_temp)/numzone_stk

	return avg_cnt, avg_nden

def vol_avg_center(x_cell,y_cell,z_cell, x_adj,y_adj,z_adj, v_cell,v_adj):
	# Gets volume weighted average location of a cell and it's adjacency
	# The collection of these volume weighted locations is the location of the boundary for the zone
	# Volume weighted average is obtained by this formula: (x_cell*vol_cell + x_adj[i]*vol_adj[i])/(vol_cell+vol_adj[i])
	# Note that x_adj[i] and vol_adj[i] may contain more than one value so this is a summation
	# Same procedure for y and z

	x_vol_avg = []
	y_vol_avg = []
	z_vol_avg = []

	# Loop over all cells that are on the boundary of a zone
	for l in range(0,len(v_cell)):
		# Numerator for xyz
		num_x_adj = sum([x*volx for x,volx in zip(x_adj[l],v_adj[l])])
		num_x = num_x_adj + (x_cell[l]*v_cell[l])

		num_y_adj = sum([y*voly for y,voly in zip(y_adj[l],v_adj[l])])
		num_y = num_y_adj + (y_cell[l]*v_cell[l])
		
		num_z_adj = sum([z*volz for z,volz in zip(z_adj[l],v_adj[l])])
		num_z = num_z_adj + (z_cell[l]*v_cell[l])

		# Denominator
		dem = v_cell[l]+sum(v_adj[l])

		x_vol_avg.append(num_x/dem)
		y_vol_avg.append(num_y/dem)
		z_vol_avg.append(num_z/dem)

	return x_vol_avg, y_vol_avg, z_vol_avg

def boundary_bin(vm, dist, min_bin, max_bin):
	# Function to bin the boundary distance profile
	# den is an array of densities for each particle in a zone
	# dist is an array of distance of each particle to the closest "boundary particle"
	# min and max are the upper and lower bounds of the bin

	cnt = 0.
	bin_vol_temp = []
	bin_dist = []
	for i,t in enumerate(dist):
		if t > min_bin and t <= max_bin:
			bin_dist.append(t)
			if not isnan(vm[i]):
				bin_vol_temp.append(vm[i])
				cnt += 1.
			else:
				bin_vol_temp.append(0)
			# cnt += 1.

	if sum(bin_vol_temp) != 0.:
		bin_den = ((1./sum(bin_vol_temp))*cnt)/(np.int(numpart)/(Lbox**3.))
	else:
		bin_den = 0.
	
	return bin_den, bin_dist, cnt

def bin_zone(x_den,y_den,z_den, x_vol,y_vol,z_vol, zone_rad, dcon, min_bin, max_bin):
	# This function returns in xyz centers (volume averaged and density), zone radius, density contrast, 
	# and zone ID for particles in a specific range of zone radii

	x_zone_stk = []
	y_zone_stk = []
	z_zone_stk = []
	x_vol_zone_stk = []
	y_vol_zone_stk = []
	z_vol_zone_stk = []
	zone_rad_stk = []
	zone_dencon_stk = []
	zn = []

	# Loop over all zones and retain xyz and radius for those that have given size
	for i in range(0,len(ID)):
		if zone_rad[i] > min_bin and zone_rad[i] <= max_bin:	
			x_zone_stk.append(x_den[i])
			y_zone_stk.append(y_den[i])
			z_zone_stk.append(z_den[i])
			x_vol_zone_stk.append(x_vol[i])
			y_vol_zone_stk.append(y_vol[i])
			z_vol_zone_stk.append(z_vol[i])
			zone_rad_stk.append(zone_rad[i])
			zone_dencon_stk.append(dcon[i])
			zn.append(zone[i])

	return x_zone_stk, y_zone_stk, z_zone_stk, x_vol_zone_stk, y_vol_zone_stk, z_vol_zone_stk, zone_rad_stk, zone_dencon_stk, zn

def adj_particles(same_zone_adj_bn):
	# This function takes in an array of adjacencies for a zone and 
	# finds the cells and their xyz and vol if any of their adjacencies
	# are not in the same zone as the cell itself

	x_non_zone_adj = []
	y_non_zone_adj = []
	z_non_zone_adj = []
	x_non_zone_adj_slice = []
	y_non_zone_adj_slice = []
	z_non_zone_adj_slice = []
	x_non_zone_adj_tot = []
	y_non_zone_adj_tot = []
	z_non_zone_adj_tot = []
	x_non_zone_adj_arr_slice = []
	y_non_zone_adj_arr_slice = []
	z_non_zone_adj_arr_slice = []
	x_non_zone_adj_arr_tot = []
	y_non_zone_adj_arr_tot = []
	z_non_zone_adj_arr_tot = []
	adj_cell_vol = []
	adj_cell_vol_tot = []
	cell_on_boundary_temp = [] # ID of cells that have adjacent cells not in the same zone
	cell_on_boundary_temp_tot = []

	for a in range(0, len(same_zone_adj_bn)):
		adj_cell_vol_temp = []
		x_non_zone_adj_temp = []
		y_non_zone_adj_temp = []
		z_non_zone_adj_temp = []

		adj_cell_vol_temp_tot = []
		x_non_zone_adj_temp_tot = []
		y_non_zone_adj_temp_tot = []
		z_non_zone_adj_temp_tot = []
		for b in same_zone_adj_bn[a]:
			if zone[b] != zn:
				# xyz as array of arrays for each adjacency 
				x_non_zone_adj_temp_tot.append(x[b])
				y_non_zone_adj_temp_tot.append(y[b])
				z_non_zone_adj_temp_tot.append(z[b])

				# Get volume for each adjacency
				adj_cell_vol_temp_tot.append(vol[b])

				# Get ID of cell thats on a boundary
				cell_on_boundary_temp_tot.append(same_zone_id_bn[0][a])

			if zone[b] != zn and z[b] <= slice_max/mult and z[b] >= slice_min/mult:
				# xyz in order for plots
				x_non_zone_adj.append(x[b])
				y_non_zone_adj.append(y[b])
				z_non_zone_adj.append(z[b])

				# xyz as array of arrays for each adjacency 
				x_non_zone_adj_temp.append(x[b])
				y_non_zone_adj_temp.append(y[b])
				z_non_zone_adj_temp.append(z[b])

				# Get volume for each adjacency
				adj_cell_vol_temp.append(vol[b])

				# Get ID of cell thats on a boundary
				cell_on_boundary_temp.append(same_zone_id_bn[0][a])
				
		# Gets array of arrays which contain volumes for each adjacent cell not part of zone for each particle in same_zone_adj
		if adj_cell_vol_temp != []:
			adj_cell_vol.append(adj_cell_vol_temp) 
			x_non_zone_adj_arr_slice.append(x_non_zone_adj_temp)
			y_non_zone_adj_arr_slice.append(y_non_zone_adj_temp)
			z_non_zone_adj_arr_slice.append(z_non_zone_adj_temp)

			# Gets array of arrays which contain volumes for each adjacent cell not part of zone for each particle in same_zone_adj
		if adj_cell_vol_temp_tot != []:
			adj_cell_vol_tot.append(adj_cell_vol_temp_tot) 
			x_non_zone_adj_arr_tot.append(x_non_zone_adj_temp_tot)
			y_non_zone_adj_arr_tot.append(y_non_zone_adj_temp_tot)
			z_non_zone_adj_arr_tot.append(z_non_zone_adj_temp_tot)

	# Remove duplicate IDs of cells that are on a boundary
	cell_on_boundary = rm_duplicate(cell_on_boundary_temp)
	cell_on_boundary_tot = rm_duplicate(cell_on_boundary_temp_tot)

	return cell_on_boundary_tot, cell_on_boundary, x_non_zone_adj_arr_tot, y_non_zone_adj_arr_tot, z_non_zone_adj_arr_tot, adj_cell_vol_tot

def boundary_stk(xvol, yvol, zvol, x_same_zone_bn, y_same_zone_bn, z_same_zone_bn, vol_same_zone_bn, zn, rad_val):
	# This function takes in the location of the boundary points (likely volume averaged) and
	# bins them to find the profile

	# Create tree of volume weighted boundary points
	boundary_pts = zip(xvol, yvol, zvol) 
	periodic_tree_boundary = PeriodicCKDTree(bounds, boundary_pts)

	x_part = []
	y_part = []
	z_part = []

	# Find particles within rad_val of zone radius that are not in zone
	idx = periodic_tree.query_ball_point([x_vol[zn],y_vol[zn],z_vol[zn]],rad_val)

	new_idx = [] # index of particles not in zone, but within 2*R_eff of zone
	for i in idx:
		if zone[i] != zn:
			new_idx.append(i)
			x_part.append(x[i])
			y_part.append(y[i])
			z_part.append(z[i])

	cls_dist = []
	cls_idx = []
	cls_dist_non_zn = []
	cls_idx_non_zn = []

	# Find closest distance for each particle in a zone to the boundary particle
	for i in range(0,len(x_same_zone_bn)):
		cls_dist.append(periodic_tree_boundary.query([x_same_zone_bn[i],y_same_zone_bn[i],z_same_zone_bn[i]])[0])
		cls_idx.append(periodic_tree_boundary.query([x_same_zone_bn[i],y_same_zone_bn[i],z_same_zone_bn[i]])[1])

	# Find closest distance for each particle not in a zone to the boundary particle
	for i in range(0,len(x_part)):
		cls_dist_non_zn.append(periodic_tree_boundary.query([x_part[i],y_part[i],z_part[i]])[0])
		cls_idx_non_zn.append(periodic_tree_boundary.query([x_part[i],y_part[i],z_part[i]])[1])

	# Calculate density for each cell in the zone
	# den_same_zone_bn = [(1./volume) for volume in vol_same_zone_bn[0]]
	vol_same_zone_bn = [(volume) for volume in vol_same_zone_bn[0]]

	# Calculate density for each particle 
	vol_non_zn_part = []
	for i in new_idx:
		# den_non_zn_part.append(1./vol[i])
		vol_non_zn_part.append(vol[i])
	
	# Bin the density, distance, and number counts from 0 to 2.5.  This binning is normalized to effective radius of each zone 
	den_bins = [] 
	dist_bins = []
	ncnt_bins = []

	den_bins_non_zn = []
	dist_bins_non_zn = []
	ncnt_bins_non_zn = []

	# Find den, dist, cnt for particles in zone
	for i in range(0,len(bins)-1):
		# Density, distance, and num counts of for each bin.  Bins are normalized to the effective radius of each zone
		den_temp, dist_temp, ncnt_temp = boundary_bin(vol_same_zone_bn, cls_dist, bins[i], bins[i+1])

		# Make arrays of den, dist, num counts of for bins
		if den_temp != np.nan:
			den_bins.append(den_temp)
		else:
			den_bins.append(0)

		if dist_temp != np.nan:
			dist_bins.append(dist_temp)
		else:
			dist_bins.append(0)
		if ncnt_temp != np.nan:
			ncnt_bins.append(ncnt_temp)
		else:
			ncnt_bins.append(0)

	# Find den, dist, cnt for particle not in zone
	for i in range(0,len(bins)-1):
		# Density, distance, and num counts of for each bin.  Bins are normalized to the effective radius of each zone
		den_temp2, dist_temp2, ncnt_temp2 = boundary_bin(vol_non_zn_part, cls_dist_non_zn, bins[i], bins[i+1])

		# Make arrays of den, dist, num counts of for bins
		if den_temp2 != np.nan:
			den_bins_non_zn.append(den_temp2)
		else:
			den_bins_non_zn.append(0)

		if dist_temp2 != np.nan:
			dist_bins_non_zn.append(dist_temp2)
		else:
			dist_bins.append(0)
		if ncnt_temp2 != np.nan:
			ncnt_bins_non_zn.append(ncnt_temp2)
		else:
			ncnt_bins_non_zn.append(0)

	return den_bins, dist_bins, ncnt_bins, den_bins_non_zn, dist_bins_non_zn, ncnt_bins_non_zn
	
# Array of effective radius of each cell
r_eff = []
for v in vol:
	r = (v*(3./(4*pi)))**(1./3.)
	r_eff.append(r)


# Find index of void with largest radius
max_ind = np.int(find_nearest(r_eff, max(r_eff)))

# Find index of void with second largest radius
sec_lrg_ind = np.int(find_nearest(r_eff, second_largest(r_eff)))

# Find index of void with smallest radius
min_ind = np.int(find_nearest(r_eff, min(r_eff)))

# Find index of void with arbitrary radius
arb_ind = np.int(find_nearest(np.asarray(r_eff), 15))

# Find index of max zone radius from vol.zone.txt file
max_rad_idx = np.int(find_nearest(x_denmin, x[max_ind]))

### FIND ALL PARTICLES THAT BELONG TO ZONE #####################################################

same_zone_id = []

# Loop over all zones and get index for each particle in that zone
for (i,t) in enumerate(zone):
	if t == zone[arb_ind]:
		same_zone_id.append(i)

### FIND TOTAL VOLUME AND EFFECTIVE RADIUS OF A ZONE ###########################################################

tot_zone_vol = 0
vol_same_zone = []

for n in same_zone_id:
	tot_zone_vol += vol[n]
	vol_same_zone.append(vol[n])

r_eff_zone_tot = (tot_zone_vol*(3./(4*pi)))**(1./3.)

### FIND HISTOGRAM OF ZONE RADII ###########################################################

zone_rad_hist, zone_rad_hist_bin = np.histogram(log10(zone_rad))

######################################################################################################


### GET COORDINATES, RADIUS, AND ADJACENCIES OF PARTICLES THAT BELONG TO ZONE BELONG TO ZONE #####################################################

x_same_zone = []
y_same_zone = []
z_same_zone = []
r_eff_same_zone = []
same_zone_adj = []

for valx in same_zone_id:
	x_same_zone.append(x[valx])

for valy in same_zone_id:
	y_same_zone.append(y[valy])

for valz in same_zone_id:
	z_same_zone.append(z[valz])

for valrad in same_zone_id:
	r_eff_same_zone.append(r_eff[valrad])

for valadj in same_zone_id:
	# Gets array of indices of adjacent particles to each particle in a specific zone
	same_zone_adj.append(adj_dict[valadj])

### CREATE SLICE IN THE Z DIMENSION ########################################################

x_slice = []
y_slice = []
slice_idx = []

x_slice_zone = []
y_slice_zone = []
slice_zone_idx = []
r_eff_zone_slice = []

mult = 4.5 #for the slice of zone
mult2 = 4.5 #for the slice of the boundary particles

slice_max = (mult2*r_eff[arb_ind]+z[arb_ind])
slice_min = (z[arb_ind]-mult2*r_eff[arb_ind])


# Loop over halo z values and create a slice in the cube with the size of the effective radius 
# of chosen void
for (i,t) in enumerate(z):
	if t <= slice_max and t >= slice_min:
		slice_idx.append(i)
		x_slice.append(x[i])
		y_slice.append(y[i])

for (i,t) in enumerate(z_same_zone):
	if t <= slice_max and t >= slice_min:
		slice_zone_idx.append(i)
		x_slice_zone.append(x_same_zone[i])
		y_slice_zone.append(y_same_zone[i])

# Effective radii of each cell in slice that's within a zone
for i in slice_zone_idx:
	r_eff_zone_slice.append(r_eff_same_zone[i])

denminrad = np.int(find_nearest(x_slice_zone, x_denmin[np.int(zone[arb_ind])]))

#################################################################################

### CREATE TREE FOR X,Y,Z COORDINATES FOR ALL HALOS #########################

# Create tree with period boundary conditions
halos = zip(x.ravel(), y.ravel(), z.ravel()) #makes x,y,z single arrays
bounds = np.array([Lbox,Lbox,Lbox])
periodic_tree = PeriodicCKDTree(bounds, halos)

#############################################################################


### CREATE SPHERICAL DENSITY PROFILE FOR A SINGLE ZONE ###############################

# This will take any void and build shells around it up to 2*R_v
# and find the number density per shell using the volume of the shell.

R_shell = np.linspace(0.001, 2*zone_rad[np.int(zone[arb_ind])], 20) #shells from ~0 to 2*R_v in units of Mpc/h
V_shell = ((4.*pi)/3.)*R_shell**3. #volume of each shell in units of Mpc**3 
tot_numden = numpart/(Lbox**3.)


count = []
count_void = []
nden = []


for i in R_shell:
	# Find number of halos in each concetric sphere of radius given by array R
	count_void.append(len(periodic_tree.query_ball_point([x_denmin[np.int(zone[arb_ind])], y_denmin[np.int(zone[arb_ind])], z_denmin[np.int(zone[arb_ind])]], i)))

for i in range(0,len(R_shell)):

	# This gives me a number density in each shell
	# looks for number of particles within a volume given by input radius
	if i==0:
		count_temp = len(periodic_tree.query_ball_point([x_denmin[np.int(zone[arb_ind])], y_denmin[np.int(zone[arb_ind])], z_denmin[np.int(zone[arb_ind])]], R_shell[i]))
		nden_temp = count_temp/V_shell[i]
	else:		
		count_temp1 = len(periodic_tree.query_ball_point([x_denmin[np.int(zone[arb_ind])], y_denmin[np.int(zone[arb_ind])], z_denmin[np.int(zone[arb_ind])]], R_shell[i]))
		count_temp2 = len(periodic_tree.query_ball_point([x_denmin[np.int(zone[arb_ind])], y_denmin[np.int(zone[arb_ind])], z_denmin[np.int(zone[arb_ind])]], R_shell[i-1]))
		count_temp = count_temp1-count_temp2
 		nden_temp = count_temp/(V_shell[i]-V_shell[i-1])

 	count.append(count_temp)
 	nden.append(nden_temp)



##########################################################################################################################################
###																																	   ###		
###		GET SPHERICAL PROFILE OF ALL ZONES BINNED BY EFFECTIVE RADIUS OF EACH ZONE 													   ###
###																																	   ###		
##########################################################################################################################################

# Split zones into 20 bins from 0 to maximum radius of each zone
zone_bins = np.linspace(0,max(zone_rad),10)

# Create dictionary to store values for each output from the binning procedure
x_den_stk = {}
y_den_stk = {}
z_den_stk = {}
x_vol_stk = {}
y_vol_stk = {}
z_vol_stk = {}
zone_rad_stk = {}
zone_dencon_stk = {}
zn_stk = {}
num_zn_stk = {}


for i in range(0,len(zone_bins)-1):
	# Loops over all the bins and puts xyz (den and vol), zone radii, zone dencon, num of zones in bin, and zone ID into dictionary
	# Note that values of the dictionary are referred to as the upper value of the bin
	# eg there will be no '0' entry, but will be a 'max(zone_radius)' entry
	x_den_stk[i],y_den_stk[i],z_den_stk[i],x_vol_stk[i],y_vol_stk[i],z_vol_stk[i],zone_rad_stk[i],zone_dencon_stk[i],zn_stk[i] = bin_zone(x_denmin,y_denmin,z_denmin, x_vol,y_vol,z_vol, zone_rad, dencon, zone_bins[i], zone_bins[i+1])
	num_zn_stk[i] = len(zone_rad_stk[i])


# Create dictionaries for average counts, num den, and R_stk for each bins
avg_count_sph = {}
avg_nden_sph = {}

# # Get avg counts and nden for each shell in each bin
# for i in range(0,len(zone_bins)-1):
# 	if zone_rad_stk[i] != []:
# 		avg_count_sph[i], avg_nden_sph[i] = spherical_stk(zone_rad_stk[i],x_vol_stk[i],y_vol_stk[i],z_vol_stk[i],num_zn_stk[i])
# 		np.save('sph_count_nden_%2d' % zone_bins[i+1], (avg_count_sph[i], avg_nden_sph[i]))



# Range of spherical shells is from 0 to 2*R_eff of each zone in bin
R_stk = np.linspace(0,2,20)


# Load spherical avg counts and avg nden
for i in range(0,len(zone_bins)-1):
	avg_count_sph[i], avg_nden_sph[i] = np.load('sph_count_nden_%2d.npy' % zone_bins[i+1])

################################################################################################################################################
###																																			 ###	
###				LOOP OVER ALL ZONES TO CREATE STACKED BOUNDARY PROFILE 																		 ###
###																																			 ###
################################################################################################################################################


# Bins and bin mids to be used for each zone.  These will be normalized to the effective zone radius
# bins = np.linspace(0,3.0,20)
bins = np.arange(0,150.,5)


bins_mid_zn = []
bins_mid_non_zn = []
for i in range(0,len(bins)-1):
	bins_mid_zn.append(-(bins[i]+bins[i+1])/2)
	bins_mid_non_zn.append((bins[i]+bins[i+1])/2)

# Adding the negative and positive bin values for particles inside and outside zone
bins_mid_full = np.append(bins_mid_zn[::-1],bins_mid_non_zn)

# Create den, dist, and num count arrays to append later
den_bins_tot = np.zeros(len(bins))
dist_bins_tot = np.zeros(len(bins))
ncnt_bins_tot = np.zeros(len(bins))

den_bins_tot_non_zn = np.zeros(len(bins))
dist_bins_tot_non_zn = np.zeros(len(bins))
ncnt_bins_tot_non_zn = np.zeros(len(bins))
zn_cnt = 0

# Create dict of all the bins to access for plotting
den_bins_all_dict = {}

# for i in range(0,len(zone_bins)-1):
# 	print 'num of zones in %d' % i, len(zn_stk[i])
# 	for zn in zn_stk[i]:

# 	# Loop over all zones and get index for each particle in that zone

# 		same_zone_id_bn = np.where(zone == zn)

# 		tot_zone_vol_bn = 0
# 		vol_same_zone_bn = []

# 		for n in same_zone_id_bn:
# 			tot_zone_vol_bn += vol[n]
# 			vol_same_zone_bn.append(vol[n])

# 		r_eff_zone_tot_bn = (tot_zone_vol*(3./(4*pi)))**(1./3.)

# 		x_same_zone_bn = []
# 		y_same_zone_bn = []
# 		z_same_zone_bn = []
# 		r_eff_same_zone_bn = []
# 		same_zone_adj_bn = []

# 		for valx in same_zone_id_bn[0]:
# 			x_same_zone_bn.append(x[valx])

# 		for valy in same_zone_id_bn[0]:
# 			y_same_zone_bn.append(y[valy])

# 		for valz in same_zone_id_bn[0]:
# 			z_same_zone_bn.append(z[valz])

# 		for valrad in same_zone_id_bn[0]:
# 			r_eff_same_zone_bn.append(r_eff[valrad])

# 		for valadj in same_zone_id_bn[0]:
# 			# Gets array of indices of adjacent particles to each particle in a specific zone
# 			same_zone_adj_bn.append(adj_dict[valadj])



# 		#### GET ADJACENT PARTICLES THAT ARE NOT PART OF THE ZONE ###########################
# 		cell_on_boundary_tot, cell_on_boundary, x_non_zone_adj_arr_tot, y_non_zone_adj_arr_tot, z_non_zone_adj_arr_tot, adj_cell_vol_tot = adj_particles(same_zone_adj_bn)

# 		# Volume weighted average of boundary
# 		x_vol_avg, y_vol_avg, z_vol_avg = vol_avg_center(x[cell_on_boundary_tot],y[cell_on_boundary_tot],z[cell_on_boundary_tot], x_non_zone_adj_arr_tot,y_non_zone_adj_arr_tot,z_non_zone_adj_arr_tot, vol[cell_on_boundary_tot],adj_cell_vol_tot)
# 		# x_vol_avg_plot, y_vol_avg_plot, z_vol_avg_plot = vol_avg_center(x[cell_on_boundary],y[cell_on_boundary],z[cell_on_boundary], x_non_zone_adj_arr_slice,y_non_zone_adj_arr_slice,z_non_zone_adj_arr_slice, vol[cell_on_boundary],adj_cell_vol)


# 		### CREATE BOUNDARY DISTANCE PROFILE FOR A SINGLE ZONE AND APPEND TO STACKED VALUE ###############################
# 		den_bins, dist_bins, ncnt_bins, den_bins_non_zn, dist_bins_non_zn, ncnt_bins_non_zn = boundary_stk(x_vol_avg,y_vol_avg,z_vol_avg, x_same_zone_bn,y_same_zone_bn,z_same_zone_bn, vol_same_zone_bn, zn, 150)


# 		# Add the values of density, distance, and num counts to each bin
# 		for l in range(0, len(den_bins)):
# 			if den_bins[l] == np.nan:
# 				den_bins_tot[l] += 0.
# 			else:
# 				den_bins_tot[l] += den_bins[l]
# 			# dist_bins_tot[l] += dist_bins[l]
# 			if ncnt_bins[l] != np.nan:
# 				ncnt_bins_tot[l] += ncnt_bins[l]

# 		for l in range(0, len(den_bins_non_zn)):
# 			if den_bins_non_zn[l] == np.nan:
# 				den_bins_tot_non_zn[l] += 0.
# 			else:
# 				den_bins_tot_non_zn[l] += den_bins_non_zn[l]
# 			# dist_bins_tot[l] += dist_bins[l]
# 			if ncnt_bins_tot_non_zn[l] != np.nan:
# 				ncnt_bins_tot_non_zn[l] += ncnt_bins_non_zn[l]

# 	# Divide each value in the density, distance, and number counts by the number of zones
# 	den_bins_tot = den_bins_tot/len(zn_stk[i])
# 	dist_bins_tot = dist_bins_tot/len(zn_stk[i])
# 	ncnt_bins_tot = ncnt_bins_tot/len(zn_stk[i])

# 	den_bins_tot_non_zn = den_bins_tot_non_zn/len(zn_stk[i])
# 	dist_bins_tot_non_zn = dist_bins_tot_non_zn/len(zn_stk[i])
# 	ncnt_bins_tot_non_zn = ncnt_bins_tot_non_zn/len(zn_stk[i])

# 	# Reverse den_bins_tot to add 
# 	den_bins_tot_rev = den_bins_tot[::-1]

# 	den_bins_all = np.append(den_bins_tot_rev[1:],den_bins_tot_non_zn[:-1])
# 	den_bins_all_dict[i] = den_bins_all

	# Save den_bins_all for whichever bin 
	# np.save('den_bins_all_69_test', (den_bins_all))
	# np.save('den_bins_all_%2d' % zone_bins[i+1], (den_bins_all))
	# db_all = np.load('den_bins_all_69_test.npy')


# den_bins_all = np.load('den_bins_all_35_45.npy')

# Load filesfor boundary stacked bins
for i in range(0,len(zone_bins)-1):
	den_bins_all_dict[i] = np.load('den_bins_all_%2d.npy' % zone_bins[i+1])

### PRINT STATEMENTS ##############################################################################

# print 'x,y,z', x[arb_ind], y[arb_ind], z[arb_ind]
# print 'x,y,z of den min of largest zone', x_denmin[np.int(zone[arb_ind])], y_denmin[np.int(zone[arb_ind])], z_denmin[np.int(zone[arb_ind])]
print 'radius of largest cell in zone slice', max(r_eff_same_zone)
# print 'vol',vol[arb_ind]
print 'radius of density min cell of zone', r_eff_zone_slice[denminrad]
print 'zone_id', zone[arb_ind]
print ''

print 'Num of same zone IDs',  len(same_zone_id)
print 'Particles in slice with same zone', len(x_slice_zone)
print ''

print 'zone radius from file', zone_rad[np.int(zone[arb_ind])]
print 'zone radius from adding cell volumes', r_eff_zone_tot
print 'max zone radius', max(zone_rad)
print ''

# print 'num of boundary cells', len(cell_on_boundary)

t_end = time.time()
print 'time of code: \t%g minutes' % ((t_end-t_begin)/60.)

###################################################################################################

### CREATE PLOTS ###

###################################################################################################

# fig1 = plt.figure(figsize=(12,10))
# fig2 = plt.figure(figsize=(12,10))
# fig3 = plt.figure(figsize=(12,10))
# fig4 = plt.figure(figsize=(12,10))
# fig5 = plt.figure(figsize=(12,10))
fig6 = plt.figure(figsize=(12,10))
# fig7 = plt.figure(figsize=(12,10))

# Create ability to cycle through linestyles
lines = ["-","--","-.",":"]
linecycler = cycle(lines)


# Create scatter plot of all halos in slice and all halos representing void centers
# ax1 = fig1.add_subplot(111)
# ax1.set_xlim(x_vol[np.int(zone[arb_ind])]-150,x_vol[np.int(zone[arb_ind])]+150)
# ax1.set_ylim(y_vol[np.int(zone[arb_ind])]-150,y_vol[np.int(zone[arb_ind])]+150)
# ax1.set_xlabel(r'$\mathrm{x}$')
# ax1.set_ylabel(r'$\mathrm{y}$')
# ax1.scatter(x_slice, y_slice, color='red', marker='o')
# ax1.scatter(x_slice_zone, y_slice_zone, color='green', marker='s')
# # ax1.scatter(x[arb_ind], y[arb_ind], color='black', marker='d', s=4)
# circ2=plt.Circle((x_denmin[np.int(zone[arb_ind])],y_denmin[np.int(zone[arb_ind])]),zone_rad[np.int(zone[arb_ind])], fill=None)
# for i in range(0,len(slice_zone_idx)):
# 	circ=plt.Circle((x_slice_zone[i],y_slice_zone[i]),r_eff_zone_slice[i]*0.5, color='b', alpha=0.5)
# 	ax1.add_artist(circ)
# # circ=plt.Circle((x[arb_ind],y[arb_ind]),r_eff[arb_ind], color='black', alpha=0.5)
# ax1.scatter(x_denmin[np.int(zone[arb_ind])],y_denmin[np.int(zone[arb_ind])], color='black', marker='d', s=100)
# ax1.scatter(x_vol[np.int(zone[arb_ind])],y_vol[np.int(zone[arb_ind])], color='black', marker='+', s=100)
# ax1.add_artist(circ)
# ax1.add_artist(circ2)
# ax1.spines['top'].set_linewidth(2.3)
# ax1.spines['left'].set_linewidth(2.3)
# ax1.spines['right'].set_linewidth(2.3)
# ax1.spines['bottom'].set_linewidth(2.3)
# for tick in ax1.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax1.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)   
# # fig1.savefig('Zone_2131', format='pdf')

# # Create density contrast vs R_v for single void up to 2*R_v
# ax2 = fig2.add_subplot(111)
# #ax2.set_xlim(0,33)
# ax2.set_ylim(0,2)
# ax2.set_xlabel(r'$\mathrm{R_v}$')
# ax2.set_ylabel(r'$\mathrm{\delta(r)+1}$')
# ax2.plot(R_shell, (np.array(nden)/tot_numden), linewidth=3)
# ax2.axvline(zone_rad[zone[arb_ind]], linewidth=2, linestyle='--', color='black')
# ax2.spines['top'].set_linewidth(2.3)
# ax2.spines['left'].set_linewidth(2.3)
# ax2.spines['right'].set_linewidth(2.3)
# ax2.spines['bottom'].set_linewidth(2.3)
# for tick in ax2.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax2.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)


# Create density contrast vs R_v for stacked voids
# label = [r'$\mathrm{R_{zone}}<=35$',r'$35<\mathrm{R_{zone}}<=50$',r'$\mathrm{R_{zone}}>50$']
# ax3 = fig3.add_subplot(111)
# #ax3.set_xlim(0,33)
# ax3.set_ylim(0,3)
# ax3.set_xlabel(r'$\mathrm{r/R_{zone}}$')
# ax3.set_ylabel(r'$\mathrm{\delta(r)+1}$')
# for i in range(0,len(zone_bins)-1):
# 	# if zone_rad_stk[i] != []:
# 	ax3.plot(R_stk, (np.array(avg_nden_sph[i])/tot_numden), linewidth=3, label='%2d' % zone_bins[i+1], linestyle = next(linecycler))
# # ax3.plot(R_stk, (np.array(avg_nden_sph[1])/tot_numden), linewidth=3, label='%2d' % zone_bins[2])
# # ax3.plot(R_stk, (np.array(avg_nden_sph[2])/tot_numden), linewidth=3, label='%2d' % zone_bins[3])
# # ax3.plot(R_stk, (np.array(avg_nden_sph[3])/tot_numden), linewidth=3, label='%2d' % zone_bins[4])
# ax3.spines['top'].set_linewidth(2.3)
# ax3.spines['left'].set_linewidth(2.3)
# ax3.spines['right'].set_linewidth(2.3)
# ax3.spines['bottom'].set_linewidth(2.3)
# for tick in ax3.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax3.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# ax3.legend(loc='best', fancybox = True, shadow = True)
# fig3.savefig('L1000_spherical_prof_stk_bin', format='pdf')


# # Create density contrast vs R_v for stacked voids with vol avg center
# label = [r'$\mathrm{R_{zone}}<=35$',r'$35<\mathrm{R_{zone}}<=50$',r'$\mathrm{R_{zone}}>50$']
# ax4 = fig4.add_subplot(111)
# #ax4.set_xlim(0,33)
# ax4.set_ylim(0,2)
# ax4.set_xlabel(r'$\mathrm{r/R_{zone}}$')
# ax4.set_ylabel(r'$\mathrm{\delta(r)+1}$')
# ax4.plot(R_stk, (np.array(avg_nden_sm_vol)/tot_numden), linewidth=3)
# ax4.plot(R_stk, (np.array(avg_nden_md_vol)/tot_numden), linewidth=3, linestyle='--')
# ax4.plot(R_stk, (np.array(avg_nden_lg_vol)/tot_numden), linewidth=3, linestyle='-.')
# ax4.spines['top'].set_linewidth(2.3)
# ax4.spines['left'].set_linewidth(2.3)
# ax4.spines['right'].set_linewidth(2.3)
# ax4.spines['bottom'].set_linewidth(2.3)
# for tick in ax4.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax4.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# ax4.legend(label, loc='best', fancybox = True, shadow = True)
# # fig4.savefig('L1000_tot_zone_spherical_prof_multi_bin_vol_center', format='pdf')


# Plot of adjacencies
# ax5 = fig5.add_subplot(111)
# ax5.set_xlim(x_vol[np.int(zone[arb_ind])]-150,x_vol[np.int(zone[arb_ind])]+150)
# ax5.set_ylim(y_vol[np.int(zone[arb_ind])]-150,y_vol[np.int(zone[arb_ind])]+150)
# ax5.set_xlabel(r'$\mathrm{x}$')
# ax5.set_ylabel(r'$\mathrm{y}$')
# ax5.scatter(x_slice, y_slice, color='red', marker='o')
# ax5.scatter(x_slice_zone, y_slice_zone, color='green', marker='s')
# circ2=plt.Circle((x_denmin[np.int(zone[arb_ind])],y_denmin[np.int(zone[arb_ind])]),zone_rad[np.int(zone[arb_ind])], fill=None)
# # circ=plt.Circle((x[arb_ind],y[arb_ind]),r_eff[arb_ind], color='black', alpha=0.5)
# ax5.scatter(x_non_zone_adj_slice, y_non_zone_adj_slice, color='blue', marker='*', s=100)
# ax5.scatter(x_vol_avg_plot, y_vol_avg_plot, color='purple', marker='^', s=100)
# ax5.scatter(x_denmin[np.int(zone[arb_ind])],y_denmin[np.int(zone[arb_ind])], color='black', marker='d', s=100)
# ax5.scatter(x_vol[np.int(zone[arb_ind])],y_vol[np.int(zone[arb_ind])], color='black', marker='+', s=100)
# ax5.add_artist(circ2)
# ax5.spines['top'].set_linewidth(2.3)
# ax5.spines['left'].set_linewidth(2.3)
# ax5.spines['right'].set_linewidth(2.3)
# ax5.spines['bottom'].set_linewidth(2.3)
# for tick in ax5.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax5.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)   
# fig5.savefig('Zone_2131_adj_halos', format='pdf')

# Boundary distance profile
ax6 = fig6.add_subplot(111)
ax6.set_xlim(-50,50)
# ax6.set_ylim(0,2)
ax6.set_xlabel(r'$\mathrm{{\it D} \hspace{0.5cm} [Mpc]}$')
ax6.set_ylabel(r'$\mathrm{\delta({\it D})+1}$') 
# ax6.plot(bins_mid_full, den_bins_all, linewidth=3, linestyle = next(linecycler))
for i in range(0,len(zone_bins)-1):
	# if zone_rad_stk[i] != []:
	ax6.plot(bins_mid_full, den_bins_all_dict[i], linewidth=3, label='%2d' % zone_bins[i+1], linestyle = next(linecycler))
ax6.spines['top'].set_linewidth(2.3)
ax6.spines['left'].set_linewidth(2.3)
ax6.spines['right'].set_linewidth(2.3)
ax6.spines['bottom'].set_linewidth(2.3)
for tick in ax6.xaxis.get_major_ticks():
    tick.label.set_fontsize(27)
for tick in ax6.yaxis.get_major_ticks():
    tick.label.set_fontsize(27)
ax6.legend(loc='best', fancybox = True, shadow = True)
# fig6.savefig('Boundary_prof_stk_bin', format='pdf')

# Histogram of zone radii
# ax7 = fig7.add_subplot(111)
# # ax7.set_xlim(0,33)
# # ax7.set_ylim(0,2)
# ax7.set_xlabel(r'$\mathrm{R_{eff}}$')
# ax7.set_ylabel(r'$\mathrm{dn}$') 
# ax7.hist(zone_rad, bins='auto')#, linewidth=3)
# # ax7.set_xscale('log')
# ax7.spines['top'].set_linewidth(2.3)
# ax7.spines['left'].set_linewidth(2.3)
# ax7.spines['right'].set_linewidth(2.3)
# ax7.spines['bottom'].set_linewidth(2.3)
# for tick in ax7.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax7.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# fig7.savefig('Zones_rad_hist', format='pdf')


plt.show()