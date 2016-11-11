import numpy as np
from numpy import *
import matplotlib
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/Vasiliy/Desktop/PhD/Scripts/ZOBOV')
sys.path.append('/Users/Vasiliy/Desktop/PhD/Scripts')
from read_vol_zone_void import *
from scipy.spatial import cKDTree
from periodic_kdtree import PeriodicCKDTree
import time
from itertools import cycle
from Cosmology_class import *
from astropy.io import fits
import matplotlib.colors as mclr
from scipy.integrate import quad
from mpl_toolkits.axes_grid1 import make_axes_locatable


font = {'family' : 'serif', 'serif' : ['Times'], 'size' : '35'}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times'
matplotlib.rcParams['mathtext.it'] = 'Times'
plt.rc('legend', **{'fontsize':30})

t_begin = time.time()

LC = LC_cosmology()

Lbox = 252.5
pix_count = 1549#7745#

# WMAP9+SN+BAO cosmology
LC.H0 = 68.98 #km s^{-1} Mpc^{-1} 
h = LC.H0/100.
c = 2.99792E5 #km/s
LC.omega_m = 0.2905
LC.omega_L = 0.7095
LC.z_box = 0.525
z_box_317 = 0.317
LC.Lbox_full = 505. #Mpc/h this is the size of the simulation box that the light cone was in, but only half of it is used so it's 252.5 Mpc/h
LC.theta = 10. #degrees on sky of each mock
LC.Lcube = 265.
nc = 3072.
D_H = c/LC.H0 #Hubble distance in units of Mpc

def comoving_dist(z_box,omega_m,omega_L,H0,h,c,theta,Lbox):
	### FINDING COMOVING DISTANCE TO LOWER, MIDDLE, AND UPPER LIGHTCONE IN Z 
	### AND FINDING UPPER AND LOWER ANGULAR DIAMETER DISTANCE FOR X AND Y

	# Find comoving distance to center of box and then to the z_max and z_min in order to find x_min(y_min)
	# and x_max(y_max) at z_min and z_max
	def integrand(z):
	    return (1./sqrt(omega_m*(1+z)**3 + omega_L))

	ans, err = quad(integrand, 0, z_box)
	D_box = h*D_H*ans #This is the comoving distance to the 'middle' of the light cone in units of Mpc/h

	# Distance to 'lower' and 'upper' section of the light cone
	# Lbox is divided by 4 because the lightcone already is taking up Lbox/2
	D_low = D_box - Lbox/4. 
	D_upper = D_box + Lbox/4. 

	# Angular distance of 'lower' and 'upper' section of lightcone
	DA_low = 2.*D_low*tan(radians(theta/2.))
	DA_upper = 2.*D_upper*tan(radians(theta/2.))

	return D_box, D_low, D_upper, DA_low, DA_upper

# Get comoving distances of middle of box, lower end, and upper end as well as angular distances of lower and upper edge of cone
D_box, D_low, D_upper, DA_low, DA_upper = LC.comoving_dist()
D_box_317, D_low_317, D_upper_317, DA_low_317, DA_upper_317 = comoving_dist(z_box_317,LC.omega_m,LC.omega_L,LC.H0,h,c,LC.theta,LC.Lbox_full)

### VOLUMES ###############################################################

# Volume of light cone
# vol_cone = (1./3.)*(DA_low**2 + (DA_low*DA_upper) + DA_upper**2)*(LC.Lbox_full/2.)
vol_cone = (1./3.)*DA_upper**2*D_upper # This is the volume if you have a whole light cone from z_i = 0 to some z 


### LOAD DATA ###############################################################

numpart, numzones, zone = read_zone('KiDS_Mocks_upto_z_525.zone')
# numpart, numzones, zone = read_zone('Test_L1000/AllHalo_1000Mpc_wrapped_cai.zone')
numpart_vol, vol, non_zero_idx = read_vol('KiDS_Mocks_upto_z_525.vol.txt', vol_cone)
x,y,z = read_LC_gal_pos('KiDS_Mocks_upto_z_525_LC.txt')
ID_all, x_vol_all, y_vol_all, z_vol_all, x_denmin_all, y_denmin_all, z_denmin_all, zone_rad_all, dencon_all = read_vol_zone_txt('KiDS_Mocks_upto_z_525.vol.zone.txt')
adj_dict = read_adj('KiDS_Mocks_upto_z_525.adj.txt')
numpart_0_042, x_gal_0_042,y_gal_0_042,z_gal_0_042, x_halo_0_042,y_halo_0_042,z_halo_0_042, m200c_0_042, r200c_0_042, Rs_0_042, c_0_042 = read_HOD('L505Mpc_HOD+0.042.dat')
numpart_0_130, x_gal_0_130,y_gal_0_130,z_gal_0_130, x_halo_0_130,y_halo_0_130,z_halo_0_130, m200c_0_130, r200c_0_130, Rs_0_130, c_0_130 = read_HOD('L505Mpc_HOD+0.130.dat')
numpart_0_221, x_gal_0_221,y_gal_0_221,z_gal_0_221, x_halo_0_221,y_halo_0_221,z_halo_0_221, m200c_0_221, r200c_0_221, Rs_0_221, c_0_221 = read_HOD('L505Mpc_HOD+0.221.dat')
numpart_0_317, x_gal_0_317,y_gal_0_317,z_gal_0_317, x_halo_0_317,y_halo_0_317,z_halo_0_317, m200c_0_317, r200c_0_317, Rs_0_317, c_0_317 = read_HOD('L505Mpc_HOD+0.317.dat')
numpart_0_418, x_gal_0_418,y_gal_0_418,z_gal_0_418, x_halo_0_418,y_halo_0_418,z_halo_0_418, m200c_0_418, r200c_0_418, Rs_0_418, c_0_418 = read_HOD('L505Mpc_HOD+0.418.dat')
numpart_0_525, x_gal_0_525,y_gal_0_525,z_gal_0_525, x_halo_0_525,y_halo_0_525,z_halo_0_525, m200c_0_525, r200c_0_525, Rs_0_525, c_0_525 = read_HOD('L505Mpc_HOD+0.525.dat')
# kappa_fits = fits.open('../kappa_0.582_mass.dat_LOS500.fits')
kappa_LoRes = fits.open('../kappa_0.582_mass.dat_LOS500_LoRes.fits')
# xgal_kappa, ygal_kappa, gal_kappa = loadtxt('Source_galaxies_for_kappa_z_0_582.txt', unpack=True)

### DOWNGRADE KAPPA RESOLUTION ##############################################

# Open some Fits image called kappa_fits 
# kappa_fits has dimensions of 7745*7745 --> I am lowering its resolution by factor of 5 on each axis

# Now lower the resolution of this thing to 1549*1549

# kappa_LoRes = fits.PrimaryHDU() # Make a new fits image object
# kappa_LoRes.data = np.empty([1549, 1549])
	

# Averaging the pixels in blocks of (4)^2 pxls^2 
# jump = int(len(kappa_fits[0].data[0,:])/len(kappa_LoRes.data[0,:])) # =5
	

# For loop goes through pxls of lo res mask: 1549*1549
# for x in range(0, len(kappa_LoRes.data[0,:]) ):	
# 	for y in range(0, len(kappa_LoRes.data[:,0]) ):
# 		avg_pxls = np.sum(kappa_fits[0].data[y*jump:(y+1)*jump, x*jump:(x+1)*jump])/(jump*jump)
# 		kappa_LoRes.data[y,x] = avg_pxls
# kappa_LoRes.writeto('../kappa_0.582_mass.dat_LOS500_LoRes.fits', output_verify='ignore', clobber=True)

############################################################################################

### GET KAPPA VALUES AND PIXELS ##############################################
# kappa = kappa_fits[0].data # Regular resolution kappa
kappa = kappa_LoRes[0].data # Lower resolution kappa
x_pix = np.arange(0,pix_count,1)
y_pix = np.arange(0,pix_count,1)
kappa_loc = [x_pix for i in range(pix_count)]

kappa_dict = {}
for i in xrange(len(kappa)):
	kappa_dict[i] = {}
	kappa_dict[i] = kappa[i]

# Create dictionary of random kappa values pulled from a guassian distribution
kappa_rand = np.random.normal(0, (0.3)**2, (pix_count,pix_count))


################################################################################################

# delta_map = read_delta_map('../0.221delta.dat_bicubic_LOS500.txt',pix_count)

### COMBINE ALL XYZ OF GAL AND HALOS UP TO Z=0.525 AND MASS #############################################

x_gal_upto_z_525 = hstack((x_gal_0_042,x_gal_0_130,x_gal_0_221,x_gal_0_317,x_gal_0_418,x_gal_0_525))
y_gal_upto_z_525 = hstack((y_gal_0_042,y_gal_0_130,y_gal_0_221,y_gal_0_317,y_gal_0_418,y_gal_0_525))
z_gal_upto_z_525 = hstack((z_gal_0_042,z_gal_0_130,z_gal_0_221,z_gal_0_317,z_gal_0_418,z_gal_0_525))

x_halo_upto_z_525 = hstack((x_halo_0_042,x_halo_0_130,x_halo_0_221,x_halo_0_317,x_halo_0_418,x_halo_0_525))
y_halo_upto_z_525 = hstack((y_halo_0_042,y_halo_0_130,y_halo_0_221,y_halo_0_317,y_halo_0_418,y_halo_0_525))
z_halo_upto_z_525 = hstack((z_halo_0_042,z_halo_0_130,z_halo_0_221,z_halo_0_317,z_halo_0_418,z_halo_0_525))

m200c_upto_z_525 = hstack((m200c_0_042,m200c_0_130,m200c_0_221,m200c_0_317,m200c_0_418,m200c_0_525))

################################################################################################

# Average density of survey
tot_numden = numpart_vol/(vol_cone)

# Zone per particle that has a non zero volume
zone_nonzero = [zone[q] for q in non_zero_idx]
void_idx_all = rm_duplicate(zone_nonzero)

# Get vol that isn't zero
vol = [vol[q] for q in non_zero_idx]

# Use only xyz of non zero volume particles
x = [x[q] for q in non_zero_idx]
y = [y[q] for q in non_zero_idx]
z = [z[q] for q in non_zero_idx]

# Find voids that are in overdensities
num_v_in_c = 0
void_in_cloud_idx = []
zone_vol = [(4./3.)*pi*zr**3 for zr in zone_rad_all]
for i,t in enumerate(zone_vol):
	if 1./t >= tot_numden:
		num_v_in_c += 1
		void_in_cloud_idx.append(i)

x_cloud = [x_vol_all[q] for q in void_in_cloud_idx]
y_cloud = [y_vol_all[q] for q in void_in_cloud_idx]
z_cloud = [z_vol_all[q] for q in void_in_cloud_idx]

some_idx = np.arange(0,len(ID_all),1)
# Get rid of void in cloud 
ID = [ID_all[i] for i in some_idx if i not in void_in_cloud_idx]
x_denmin = [x_denmin_all[i] for i in some_idx if i not in void_in_cloud_idx]
y_denmin = [y_denmin_all[i] for i in some_idx if i not in void_in_cloud_idx]
z_denmin = [z_denmin_all[i] for i in some_idx if i not in void_in_cloud_idx]
x_vol = [x_vol_all[i] for i in some_idx if i not in void_in_cloud_idx]
y_vol = [y_vol_all[i] for i in some_idx if i not in void_in_cloud_idx]
z_vol = [z_vol_all[i] for i in some_idx if i not in void_in_cloud_idx]
zone_rad = [zone_rad_all[i] for i in some_idx if i not in void_in_cloud_idx]
dencon = [dencon_all[i] for i in some_idx if i not in void_in_cloud_idx]
void_idx = [void_idx_all[i] for i in some_idx if i not in void_in_cloud_idx]


### FIND ZONES WITH RADIUS LARGER THAN 10 MPC
zone_rad_cut = []
x_vol_cut = []
y_vol_cut = []
z_vol_cut = []
x_denmin_cut = []
y_denmin_cut = []
z_denmin_cut = []
void_idx_cut = []

for i,t in enumerate(zone_rad):
	if t > 10.:
		zone_rad_cut.append(zone_rad[i])
		x_vol_cut.append(x_vol[i])
		y_vol_cut.append(y_vol[i])
		z_vol_cut.append(z_vol[i])
		x_denmin_cut.append(x_denmin[i])
		y_denmin_cut.append(y_denmin[i])
		z_denmin_cut.append(z_denmin[i])
		void_idx_cut.append(void_idx[i])

# f = open('LOS500_vol_avg_void_centers.txt', 'w')
# for i in xrange(len(x_vol_cut)):
# 	f.write("{} {} {}\n".format(x_vol_cut[i],y_vol_cut[i],z_vol_cut[i]))
# f.close()

### FIND NUMBER OF HALOS WITH MASS OF MAX HALO AND THEIR INDEX 

mass_idx = np.where(m200c_0_221==max(m200c_0_221))[0]

##############################################################

def interpolate2D(X, Y, grid): 
	# linear 2D interpolation
	Xi = X.astype(np.int)
	Yi = Y.astype(np.int) # these round down to integer

	VAL_XYlo = grid[Yi, Xi] + (X - Xi)*( grid[Yi, Xi+1] - grid[Yi, Xi] )
	VAL_XYhi = grid[Yi+1,Xi] + (X - Xi)*( grid[Yi+1,Xi+1] - grid[Yi+1, Xi] )
	VAL_XY = VAL_XYlo + (Y - Yi)*( VAL_XYhi - VAL_XYlo )  
	
	return VAL_XY

def kappa_pix_to_mpc(coord,D_upper):
	mpc_val = [D_upper*tan((value*(LC.theta/pix_count)*(pi/180.))-5*(pi/180.)) for value in coord]
	return mpc_val

def mpc_to_pix(coord,z_coord):
	# Converts Mpc to pixel values for KiDS Mocks configuration created by Joachim
	pix_val = [(arctan(val/valz) + 5.*(pi/180.))*(pix_count/LC.theta)*(180./pi) for (val,valz) in zip(coord,z_coord)]
	return pix_val

def spherical_stk(zone_rad_stk,x_zone_stk,y_zone_stk,z_zone_stk,numzone_stk):
	delta_stk = [[] for i in xrange(numzone_stk)]
	for r in range(0,(len(zone_rad_stk))):
		# Create array from ~0 to 2 in order for it to be the same for all different sized zones
		# R_stk = np.linspace(0,2,9)
		# Array of volumes for each consecutive sphere for R values
		V = ((4.*pi)/3.)*(R_stk*zone_rad_stk[r])**3

		# Find index of halo representing min value of zone
		ind_x_stk = find_nearest(x,x_zone_stk[r])
		ind_y_stk = find_nearest(y,y_zone_stk[r])
		ind_z_stk = find_nearest(z,z_zone_stk[r])
			
		count_stk = []
		nden_stk = []
		den_stk = []

		for n in range(0,len(R_stk)):
			# This gives me a number density in each shell
			# looks for number of particles within a volume given by input radius
			if n==0:
				# Density part of loop
				loc_temp1 = tree.query_ball_point([x[ind_x_stk], y[ind_y_stk], z[ind_z_stk]], R_stk[n]*zone_rad_stk[r])
				count_temp1 = len(loc_temp1)
				# nden_temp1 = count_temp1/V[n]
				# vol_temp1 = [vol[j] for j in loc_temp1]
				sum_vol1 = sum([vol[j] for j in loc_temp1])
				if sum_vol1 != 0.0:
					den_temp1 = ((1./sum_vol1)*count_temp1)/(numpart_vol/vol_cone)
				else:
					den_temp1 = 0.0

			else:
				# Density part of loop
				delta_r = (R_stk[n-1]*zone_rad_stk[r]) - (R_stk[n]*zone_rad_stk[r])
				loc_temp11 = tree.query_ball_point([x[ind_x_stk], y[ind_y_stk], z[ind_z_stk]], R_stk[n]*zone_rad_stk[r])
				loc_temp12 = tree.query_ball_point([x[ind_x_stk], y[ind_y_stk], z[ind_z_stk]], R_stk[n-1]*zone_rad_stk[r])

				count_temp11 = len(loc_temp11)
				count_temp12 = len(loc_temp12)
				count_temp1 = count_temp11-count_temp12
				
				# Index of particles in shell
				# shell_loc = [i for i in loc_temp11 if i not in loc_temp12]
				shell_loc = list(set(loc_temp11) - set(loc_temp12))

				if len(shell_loc) != count_temp1:
					print 'number of particles in shell doesnt add up!'
		 		# nden_temp1 = count_temp1/(V[n]-V[n-1])
		 		# vol_temp1 = [vol[j] for j in shell_loc]
		 		sum_vol1 = sum([vol[j] for j in shell_loc])
		 		if sum_vol1 != 0.0:
			 		den_temp1 = ((1./sum_vol1)*count_temp1)/(numpart_vol/vol_cone)
			 	else: 
			 		den_temp1 = 0.0

			# Count of halos and number density in a each shell
		 	count_stk.append(count_temp1)
		 	# nden_stk.append(nden_temp1)
		 	den_stk.append(den_temp1)
		 	delta_stk[r] = den_stk
		
		# Add elements of each zone's number count and number density per shell 
		if r==0:
			avg_count_temp = count_stk
			# avg_nden_temp = nden_stk
			avg_den_temp = den_stk
			
		else:
			avg_count_temp = [a+b for a,b in zip(avg_count_temp,count_stk)]
			# avg_nden_temp = [c+d for c,d in zip(avg_nden_temp,nden_stk)]
			avg_den_temp = [c+d for c,d in zip(avg_den_temp,den_stk)]
			

		# Divide total counts and number densities in shells by number of voids
		avg_cnt = np.array(avg_count_temp)/numzone_stk
		# avg_nden = np.array(avg_nden_temp)/numzone_stk
		avg_den = np.array(avg_den_temp)/numzone_stk
		

	return avg_cnt, avg_den, delta_stk#,avg_nden

def sph_kappa_stk(zn_rad_pix_stk,xpix_zone_stk,ypix_zone_stk,numzone_stk):
	kappa_stk = [[] for i in xrange(numzone_stk)]
	w = [[] for i in xrange(numzone_stk)]
	for r in range(0,(len(zn_rad_pix_stk))):
		print r
		# Create array from ~0 to 2 in order for it to be the same for all different sized zones
		# R_stk = np.linspace(0,2,9)

		# Get XY location of void on map projected at z where the kappa map is
		ind_x_pix = find_nearest(x_pix,xpix_zone_stk[r])
		ind_y_pix = find_nearest(y_pix,ypix_zone_stk[r])

		# Create array of X and Y that spans 2x zone radius
		# x_zone_all = np.arange(x_pix[ind_x_pix]-R_mult*int(zn_rad_pix_stk[r]),x_pix[ind_x_pix]+R_mult*int(zn_rad_pix_stk[r]),1)
		# y_zone_all = np.arange(y_pix[ind_y_pix]-R_mult*int(zn_rad_pix_stk[r]),y_pix[ind_y_pix]+R_mult*int(zn_rad_pix_stk[r]),1)

		# # Non-grid values
		x_zone_all = np.arange(x_pix[ind_x_pix]-R_mult*zn_rad_pix_stk[r],x_pix[ind_x_pix]+R_mult*zn_rad_pix_stk[r],1)
		y_zone_all = np.arange(y_pix[ind_y_pix]-R_mult*zn_rad_pix_stk[r],y_pix[ind_y_pix]+R_mult*zn_rad_pix_stk[r],1)

		# The '-2' on pix_count is to account for the interpolation that goes to X(Y) + 1 for indexing purposes
		x_zone = x_zone_all[np.where(np.logical_and(x_zone_all>=0, x_zone_all<=pix_count-2))[0]]
		y_zone = y_zone_all[np.where(np.logical_and(y_zone_all>=0, y_zone_all<=pix_count-2))[0]]


		# print 'dist start'
		t_dist1 = time.time()
		# Find distance to each pixel from the void center 
		# Creates len(x_zone) x len(y_zone) array where the dist[3][1] is location (3,1) in xy coords x[:,None]
		# dist = [[sqrt((i-x_pix[ind_x_pix])**2. + (j-y_pix[ind_y_pix])**2.) for j in y_zone] for i in x_zone]
		dist = sqrt((x_zone[:,None]-x_pix[ind_x_pix])**2. + (y_zone[None,:]-y_pix[ind_y_pix])**2.)
		t_dist2 = time.time()
	
	
		# print 'dist done.  It took \t%g minutes' % ((t_dist2-t_dist1)/60.) 

		# print ''


		# print 'kappa start'
		t_kappa1 = time.time()
		# Get all kappa within 2x zone rad for a particular zone
		kappa_seg = {}
		for i,t in enumerate(x_zone):
			# Need to put 'y' value first in 2D arrays
			# kappa_seg[i] = [kappa_dict[j][t] for j in y_zone]
			kappa_seg[i] = interpolate2D(t,y_zone,kappa)
		# kappa_seg = [[kappa_dict[i][j] for j in y_zone] for i in x_zone]
		t_kappa2 = time.time()
		# print 'kappa done.  It took \t%g minutes' % ((t_kappa2-t_kappa1)/60.)
		
		t_sort1 = time.time()
		# Get index of the bin where each distance would fall if it were sorted
		dist_sort_idx = {}
		for j in xrange(len(dist)):
			dist_sort_idx[j] = np.searchsorted(R_stk*zn_rad_pix_stk[r], dist[j])
		# dist_sort_idx = [np.searchsorted(R_stk*zn_rad_pix_stk[r], dist[j]) for j in xrange(len(dist))]
		t_sort2 = time.time()
		# print 'dist sort done.  It took \t%g minutes' % ((t_sort2-t_sort1)/60.)
		# Sort the values of the kappa segment based on the sorting of the distances
		# kappa_seg_sorted = [[x for y,x in sorted(zip(dist[i],kappa_seg[i]))] for i in xrange(len(dist))]
		

		k_rav = np.array(kappa_seg.values()).ravel()

		dist_idx_rav = np.array(dist_sort_idx.values()).ravel()
		
		t_bin1 = time.time()
		# Put arrays of kappa into radii bins
		kappa_sum_temp = [k_rav[np.where(dist_idx_rav==i)[0]] for i in xrange(len(R_stk))]# if k_rav[np.where(dist_idx_rav==i)[0]] not []]

		t_bin2 = time.time()
		# print 'kappa bin done.  It took \t%g minutes' % ((t_bin2-t_bin1)/60.)
		
		# Mean of kappa value for each radial bin
		kappa_stk[r] = [nanmean(k) for k in kappa_sum_temp]
		w[r] = [1 if isfinite(kappa_stk[r][i]) else 0 for i in xrange(len(kappa_stk[r]))]

		

		# if r ==0:
		# 	avg_kappa_temp = kappa_stk[r]
		# else:
		# 	avg_kappa_temp = [e+f for e,f in zip(avg_kappa_temp,kappa_stk[r])]

	# avg_kappa = [np.nanmean(np.array(avg_kappa_temp[i])) for i in xrange(len(R_stk))]

	return kappa_stk#, w

# def sph_kappa_stk(zn_rad_pix_stk,xpix_zone_stk,ypix_zone_stk,numzone_stk):
	# for r in range(0,(len(zn_rad_pix_stk))):
	# 	print r

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

def boundary_bin(vm, dist, sort_idx):
	# Function to bin the boundary distance profile
	# vm is an array of volume for each particle in a zone
	# dist is an array of distance of each particle to the closest "boundary particle"
	# min and max are the upper and lower bounds of the bin

	bin_vol_temp = [[] for _ in range(len(bins)+1)]
	bin_dist = [[] for _ in range(len(bins)+1)]
	bin_den = [[] for _ in range(len(bins)+1)]
	cnt = [0 for _ in range(len(bins)+1)]

	# sort_idx = [len(bins)-1. if max(sort_idx) >= len(bins) else a for a in sort_idx]

	for i in range(0,len(dist)):
		bin_dist[sort_idx[i]].append(dist[i])

		# bin_vol_temp[sort_idx[i].append([0. if math.isnan(t) else t for t in vm[i]])

		bin_vol_temp[sort_idx[i]].append(vm[i])
		cnt[sort_idx[i]] += 1

		# if not isnan(vm[i]):
		# 	bin_vol_temp[sort_idx[i]].append(vm[i])
		# 	cnt[sort_idx[i]] += 1
		# else:
		# 	bin_vol_temp.append(0.0)

	del cnt[0]
	del bin_dist[0]
	del bin_vol_temp[0]
	
	bin_den = [((1./sum(v_val))*c)/(np.int(numpart_vol)/vol_cone) for (v_val,c) in zip(bin_vol_temp,cnt)] 
	# np.set_nan_to_num(bin_den)
	bin_den = [0. if math.isnan(v) else v for v in bin_den]	
	
	return bin_den, bin_dist, cnt

def bin_zone(x_den_val,y_den_val,z_den_val, x_vol_val,y_vol_val,z_vol_val, x_pix_vol_val, y_pix_vol_val, zn_rad, dcon, zn_rad_pix, min_bin, max_bin):
	# This function returns in xyz centers (volume averaged and density), zone radius, density contrast, 
	# and zone ID for particles in a specific range of zone radii

	x_zone_stk = []
	y_zone_stk = []
	z_zone_stk = []
	x_vol_zone_stk = []
	y_vol_zone_stk = []
	z_vol_zone_stk = []
	x_pix_zone_stk = []
	y_pix_zone_stk = []
	zn_rad_stk = []
	zn_rad_pix_stk = []
	zone_dencon_stk = []
	zn = []


	# Loop over all zones and retain xyz and radius for those that have given size
	for i in range(0,len(zn_rad)):
		if zn_rad[i] > min_bin and zn_rad[i] <= max_bin:	
			x_zone_stk.append(x_den_val[i])
			y_zone_stk.append(y_den_val[i])
			z_zone_stk.append(z_den_val[i])
			x_vol_zone_stk.append(x_vol_val[i])
			y_vol_zone_stk.append(y_vol_val[i])
			z_vol_zone_stk.append(z_vol_val[i])
			x_pix_zone_stk.append(x_pix_vol_val[i])
			y_pix_zone_stk.append(y_pix_vol_val[i])
			zn_rad_stk.append(zn_rad[i])
			zone_dencon_stk.append(dcon[i])
			zn.append(zone_nonzero[i])
			zn_rad_pix_stk.append(zn_rad_pix[i])

	return x_zone_stk, y_zone_stk, z_zone_stk, x_vol_zone_stk, y_vol_zone_stk, z_vol_zone_stk, x_pix_zone_stk, y_pix_zone_stk, zn_rad_stk, zone_dencon_stk, zn, zn_rad_pix_stk

def adj_particles(same_zone_adj_bn,zn):
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
			if zone_nonzero[b] != zn:
				# xyz as array of arrays for each adjacency 
				x_non_zone_adj_temp_tot.append(x[b])
				y_non_zone_adj_temp_tot.append(y[b])
				z_non_zone_adj_temp_tot.append(z[b])

				# Get volume for each adjacency
				adj_cell_vol_temp_tot.append(vol[b])

				# Get ID of cell thats on a boundary
				cell_on_boundary_temp_tot.append(same_zone_id_bn[a])

			if zone_nonzero[b] != zn and z[b] <= slice_max_bn[a] and z[b] >= slice_min_bn[a]:
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
				cell_on_boundary_temp.append(same_zone_id_bn[a])
				
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
	tree_boundary = cKDTree(boundary_pts)

	x_part = []
	y_part = []
	z_part = []

	zn_idx = np.where(np.array(void_idx_cut)==zn)[0]
	# Find particles within rad_val of zone radius that are not in zone
	idx = tree.query_ball_point([x_vol_cut[zn_idx],y_vol_cut[zn_idx],z_vol_cut[zn_idx]],rad_val)

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
		cls_dist.append(tree_boundary.query([x_same_zone_bn[i],y_same_zone_bn[i],z_same_zone_bn[i]])[0])
		cls_idx.append(tree_boundary.query([x_same_zone_bn[i],y_same_zone_bn[i],z_same_zone_bn[i]])[1])

	# Find closest distance for each particle not in a zone to the boundary particle
	for i in range(0,len(x_part)):
		cls_dist_non_zn.append(tree_boundary.query([x_part[i],y_part[i],z_part[i]])[0])
		cls_idx_non_zn.append(tree_boundary.query([x_part[i],y_part[i],z_part[i]])[1])

	# Calculate density for each cell in the zone
	# den_same_zone_bn = [(1./volume) for volume in vol_same_zone_bn[0]]
	# vol_same_zone_bn = [(volume) for volume in vol_same_zone_bn[0]]

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

	sort_zn_idx = np.searchsorted(bins,cls_dist)
	sort_non_zn_idx = np.searchsorted(bins,cls_dist_non_zn)

	# Find den, dist, cnt for particles in zone

	# Density, distance, and num counts of for each bin.  Bins are normalized to the effective radius of each zone
	den_bins, dist_bins, ncnt_bins = boundary_bin(vol_same_zone_bn, cls_dist, sort_zn_idx)

	# Make arrays of den, dist, num counts of for bins
	# if den_temp != np.nan:
	# 	den_bins.append(den_temp)
	# else:
	# 	den_bins.append(0)

	# if dist_temp != np.nan:
	# 	dist_bins.append(dist_temp)
	# else:
	# 	dist_bins.append(0)
	# if ncnt_temp != np.nan:
	# 	ncnt_bins.append(ncnt_temp)
	# else:
	# 	ncnt_bins.append(0)

	# Find den, dist, cnt for particle not in zone
	
	# Density, distance, and num counts of for each bin.  Bins are normalized to the effective radius of each zone
	den_bins_non_zn, dist_bins_non_zn, ncnt_bins_non_zn = boundary_bin(vol_non_zn_part, cls_dist_non_zn, sort_non_zn_idx)

	# Make arrays of den, dist, num counts of for bins
	# if den_temp2 != np.nan:
	# 	den_bins_non_zn.append(den_temp2)
	# else:
	# 	den_bins_non_zn.append(0)

	# if dist_temp2 != np.nan:
	# 	dist_bins_non_zn.append(dist_temp2)
	# else:
	# 	dist_bins.append(0)
	# if ncnt_temp2 != np.nan:
	# 	ncnt_bins_non_zn.append(ncnt_temp2)
	# else:
	# 	ncnt_bins_non_zn.append(0)

	return den_bins, dist_bins, ncnt_bins, den_bins_non_zn, dist_bins_non_zn, ncnt_bins_non_zn
	

### GET XYZ OF HALOS IN PIXEL VALUES ###

def read_halo_pix(filename):

	f = open(filename,'r')

	x_halo_pix = []
	y_halo_pix = []
	z_halo_pix = []
	mass_halo_pix = []

	rows = f.readlines()
	for row in rows:
		data = row.split()
		x_halo_pix.append(data[0])
		y_halo_pix.append(data[1])
		z_halo_pix.append(data[2])
		mass_halo_pix.append(data[3])

	x_halo_pix = np.array(x_halo_pix).astype(np.float)
	y_halo_pix = np.array(y_halo_pix).astype(np.float)
	z_halo_pix = np.array(z_halo_pix).astype(np.float)
	mass_halo_pix = np.array(mass_halo_pix).astype(np.float)

	return x_halo_pix,y_halo_pix,z_halo_pix,mass_halo_pix


x_halo_pix_042,y_halo_pix_042,z_halo_pix_042,mass_halo_pix_042 = read_halo_pix('../0.042LightCone_halo.dat_LOS500.txt')
x_halo_pix_130,y_halo_pix_130,z_halo_pix_130,mass_halo_pix_130 = read_halo_pix('../0.130LightCone_halo.dat_LOS500.txt')
x_halo_pix_221,y_halo_pix_221,z_halo_pix_221,mass_halo_pix_221 = read_halo_pix('../0.221LightCone_halo.dat_LOS500.txt')
x_halo_pix_317,y_halo_pix_317,z_halo_pix_317,mass_halo_pix_317 = read_halo_pix('../0.317LightCone_halo.dat_LOS500.txt')
x_halo_pix_418,y_halo_pix_418,z_halo_pix_418,mass_halo_pix_418 = read_halo_pix('../0.418LightCone_halo.dat_LOS500.txt')
x_halo_pix_525,y_halo_pix_525,z_halo_pix_525,mass_halo_pix_525 = read_halo_pix('../0.525LightCone_halo.dat_LOS500.txt')

x_halo_pix_upto_z_525 = hstack((x_halo_pix_042,x_halo_pix_130,x_halo_pix_221,x_halo_pix_317,x_halo_pix_418,x_halo_pix_525))
y_halo_pix_upto_z_525 = hstack((y_halo_pix_042,y_halo_pix_130,y_halo_pix_221,y_halo_pix_317,y_halo_pix_418,y_halo_pix_525))
z_halo_pix_upto_z_525 = hstack((z_halo_pix_042,z_halo_pix_130,z_halo_pix_221,z_halo_pix_317,z_halo_pix_418,z_halo_pix_525))
mass_halo_pix_upto_z_525 = hstack((mass_halo_pix_042,mass_halo_pix_130,mass_halo_pix_221,mass_halo_pix_317,mass_halo_pix_418,mass_halo_pix_525))


### GET XYZ OF TOP N MOST MASSIVE HALOS ###################################

# Get indicies of most massive halos
massive_halo_idx = np.argpartition(mass_halo_pix_upto_z_525, -500)[-500:]

# Coordinates of most massive halos
x_halo_massive = [x_halo_pix_upto_z_525[valx]/5. for valx in massive_halo_idx]
y_halo_massive = [y_halo_pix_upto_z_525[valy]/5. for valy in massive_halo_idx]


# Mass of most massive halos
mass_halo_massive = [mass_halo_pix_upto_z_525[mass] for mass in massive_halo_idx]

# Radii for halos is sent to be 1 Mpc at the center of the box ie whatever 1 Mpc at the middle
# of the box is, is coverted to pixels and used as the radius for all halos
rad_halo_massive = np.ones(len(x_halo_massive))*15#(pix_count/(2*D_box_317*tan(radians(5.))))

#############################################################################


#############################################################################

# x_halo_pix = mpc_to_pix(x_halo_0_221,z_halo_0_221)
# y_halo_pix = mpc_to_pix(y_halo_0_221,z_halo_0_221)

####################################################################################################################

### CONVERT KAPPA PIXEL ARRAYS TO MPC ARRAYS ##########


# x_pix_mpc = kappa_pix_to_mpc(x_pix,D_upper)
# y_pix_mpc = kappa_pix_to_mpc(y_pix,D_upper)

#################################################

# Array of effective radius of each cell
r_eff = []
for v in vol:
	r = (v*(3./(4*pi)))**(1./3.)
	r_eff.append(r)


# Find index of cell with largest radius
max_ind = np.int(find_nearest(r_eff, max(r_eff)))

# Find index of cell with second largest radius
sec_lrg_ind = np.int(find_nearest(r_eff, second_largest(r_eff)))

# Find index of cell with smallest radius
min_ind = np.int(find_nearest(r_eff, min(r_eff)))

# Find index of cell with arbitrary radius
arb_ind = np.int(find_nearest(np.asarray(r_eff), 7.))

# Find index of max zone radius from vol.zone.txt file
max_rad_idx = np.int(find_nearest(x_denmin, x[max_ind]))

### FIND ALL PARTICLES THAT BELONG TO ZONE #####################################################

same_zone_id = []

# Loop over all zones and get index for each particle in that zone
for (i,t) in enumerate(zone_nonzero):
	if t == zone_nonzero[arb_ind]:
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

for val in same_zone_id:
	x_same_zone.append(x[val])

	y_same_zone.append(y[val])

	z_same_zone.append(z[val])

	r_eff_same_zone.append(r_eff[val])

	# Gets array of indices of adjacent particles to each particle in a specific zone
	same_zone_adj.append(adj_dict[val])

### CREATE SLICE IN THE Y DIMENSION ########################################################

# Indices of slices in the y direction
yslice_gal_loc = np.where(np.logical_and(array(y)>=-6., array(y)<=0.))[0]
xslice = array(x)[yslice_gal_loc]
zslice = array(z)[yslice_gal_loc]

# Indices of void centers that fall into this slice
yslice_zone_loc = np.where(np.logical_and(array(y_vol_cut)>=-6., array(y_vol_cut)<=0.))[0]
yslice_zone_rad = array(zone_rad_cut)[yslice_zone_loc]

xslice_zone = array(x_vol_cut)[yslice_zone_loc]
zslice_zone = array(z_vol_cut)[yslice_zone_loc]

# Find locations most massive zone in this slice
yslice_zone_lg = np.argpartition(yslice_zone_rad, -5)[-5:]



### CREATE SLICE IN THE Z DIMENSION ########################################################

x_slice = []
y_slice = []
slice_idx = []

x_slice_zone = []
y_slice_zone = []
slice_zone_idx = []
r_eff_zone_slice = []

# mult = 1. #for the slice of zone
mult2 = 1. #for the slice of the boundary particles

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

arb_ind_void = np.where(void_idx==zone_nonzero[arb_ind])[0][0]
denminrad = np.int(find_nearest(x_slice_zone, x_denmin[arb_ind_void]))

# same_zone_id_bn = np.where(zone_nonzero==zone_nonzero[arb_ind])[0]
# cell_on_boundary_tot_2303, cell_on_boundary_2303, x_non_zone_adj_2303, y_non_zone_adj_2303, z_non_zone_adj_2303, adj_cell_vol_2303 = adj_particles(same_zone_adj,arb_ind_void)
# x_vol_avg_2303, y_vol_avg_2303, z_vol_avg_2303 = vol_avg_center(x[cell_on_boundary_tot_2303],y[cell_on_boundary_tot_2303],z[cell_on_boundary_tot_2303], x_non_zone_adj_2303,y_non_zone_adj_2303,z_non_zone_adj_2303, vol[cell_on_boundary_tot_2303],adj_cell_vol_2303)


#################################################################################

### CREATE TREE FOR X,Y,Z COORDINATES FOR ALL HALOS #########################

# Create tree for all particles
halos = zip(np.array(x).ravel(), np.array(y).ravel(), np.array(z).ravel()) #makes x,y,z single arrays
tree = cKDTree(halos)

# Create tree for xy coordinates given pixel values for kappa maps
pix_val = zip(x_pix.ravel(),y_pix.ravel())
pix_tree = cKDTree(pix_val)

# Create kappa tree
# kval = np.array(kappa_dict.values()).ravel()
# kappa_tree = cKDTree(kval)
# raise()
#############################################################################

### FIND RADIUS AT WHICH THE BACKGROUND NUMBER DENSITY IS REACHED PER VOID ######################

zone_rad_sph_den = -99.*np.ones(len(x_vol_cut))
numden_zone_rad = -99.*np.ones(len(x_vol_cut))

# Create a dictionary of distances from zone center to each particle 
dist1 = {}
dist1 = [tree.query([x_vol_cut[i],y_vol_cut[i],z_vol_cut[i]], k=int(numpart_vol))[0] for i in xrange(len(x_vol_cut))]

for i in xrange(len(x_vol_cut)):
	print i
	for d in dist1[i]:
		# Find how many particles are within radius d for each zone
		part_loc = tree.query_ball_point([x_vol_cut[i], y_vol_cut[i], z_vol_cut[i]], d)

		# Get number den at this particular radius
		# numden = len(part_loc)/(((4.*pi)/3.)*d**3.)
		numden = len(part_loc)/np.sum(array(vol)[part_loc])

		# If the numden is greater than or equal to the total numden, add radius to radius array
		if numden >= 0.7*tot_numden:
			numden_zone_rad[i] = numden
			zone_rad_sph_den[i] = d
			break

	# part_loc = [tree.query_ball_point([x_vol_cut[i], y_vol_cut[i], z_vol_cut[i]], d) for d in dist1[i]]
	# numden = [len(part_loc[j])/(((4.*pi)/3.)*array(dist1[i])**3.) for j in xrange(len(part_loc))]
	# den_idx = [np.where(array(numden[h])>=0.7*tot_numden)[0] for h in xrange(len(numden))]

	# for k in xrange(len(numden)):
	# 	if len(den_idx[k]) > 0:
	# 		numden_zone_rad[i] = array(numden[k])[den_idx[k][0]]
	# 		zone_rad_sph_den[i] = dist1[i][den_idx[k][0]]
	# 	else:
	# 		numden_zone_rad[i] = -99.
	# 		zone_rad_sph_den[i] = -99.

zero_idx = np.where(zone_rad_sph_den==-99.)[0]
zone_rad_sph_den[zero_idx] = np.array(zone_rad)[zero_idx]

np.save('zone_rad_sph_0.7den_upto_z_525_cut', (zone_rad_sph_den,numden_zone_rad))
# zone_rad_sph_den, numden_zone_rad = np.load('zone_rad_sph_0.2den_upto_z_525_cut.npy')


###################################################################################################

#### CREATE PIXEL VALUES FOR XY COORDINATES OF VOID CENTERS ######################

# Conversion factors between pix/arcmin and deg/pix for KiDS mocks using 10 deg opening angle
pix_to_arcmin = 12.90833
deg_to_pix = 0.00129115558

### MPC TO PIXEL CONVERSION FACTOR ###

# Find x distance across sky for each z location and convert it to a 
# pixel scale by dividing the num of pixels (7745) by each distance across sky
# pix_factor = [pix_count/(2*valz*tan(radians(5.))) for valz in z_vol]
pix_factor = [pix_count/(2*valz*tan(radians(5.))) for valz in z_vol_cut]
# pix_factor_denmin = [pix_count/(2*valz*tan(radians(5.))) for valz in z_denmin_cut]

# Volume centers of void to pixel values
# x_pix_vol = [(arctan(valx/valz) + 5.*(pi/180.))*(pix_count/10.)*(180./pi) for (valx,valz) in zip(x_vol,z_vol)]
# y_pix_vol = [(arctan(valy/valz) + 5.*(pi/180.))*(pix_count/10.)*(180./pi) for (valy,valz) in zip(y_vol,z_vol)]
x_pix_vol = [(valx*factor)+pix_count/2. for valx,factor in zip(x_vol,pix_factor)]
y_pix_vol = [(valy*factor)+pix_count/2. for valy,factor in zip(y_vol,pix_factor)]
x_pix_vol_cut = [(valx*factor)+pix_count/2. for valx,factor in zip(x_vol_cut,pix_factor)]
y_pix_vol_cut = [(valy*factor)+pix_count/2. for valy,factor in zip(y_vol_cut,pix_factor)]
xpix_rand = random.uniform(0,pix_count,len(x_vol_cut))
ypix_rand = random.uniform(0,pix_count,len(x_vol_cut))
xpix_rand2 = random.uniform(0,pix_count,len(x_vol_cut))
ypix_rand2 = random.uniform(0,pix_count,len(x_vol_cut))
xpix_rand3 = random.uniform(0,pix_count,len(x_vol_cut))
ypix_rand3 = random.uniform(0,pix_count,len(x_vol_cut))
xpix_rand4 = random.uniform(0,pix_count,len(x_vol_cut))
ypix_rand4 = random.uniform(0,pix_count,len(x_vol_cut))

# x_pix_vol = [(LC.theta/2.+degrees(arctan((valx-(DA_upper/2.))/D_box))/deg_to_pix) + 7745/2. for valx in x_vol]
# y_pix_vol = [(LC.theta/2.+degrees(arctan((valy-(DA_upper/2.))/D_box))/deg_to_pix) + 7745/2. for valy in y_vol]
# x_pix_vol = [valp*valx for valp,valx in zip(pix_factor,x_vol)]
# y_pix_vol = [valp*valy for valp,valy in zip(pix_factor,y_vol)]

# Denmin centers of void to pixel values
# x_pix_denmin_cut = [(valx*factor)+pix_count/2. for valx,factor in zip(x_denmin_cut,pix_factor_denmin)]
# y_pix_denmin_cut = [(valy*factor)+pix_count/2. for valy,factor in zip(y_denmin_cut,pix_factor_denmin)]
# x_pix_denmin = [(LC.theta/2.+degrees(arctan((valx-(DA_upper/2.))/D_upper))/deg_to_pix) + 7745/2. for valx in x_denmin]
# y_pix_denmin = [(LC.theta/2.+degrees(arctan((valy-(DA_upper/2.))/D_upper))/deg_to_pix) + 7745/2. for valy in y_denmin]
# x_pix_denmin = [valp*valx for valp,valx in zip(pix_factor,x_denmin)]
# y_pix_denmin = [valp*valy for valp,valy in zip(pix_factor,y_denmin)]

### Convert zone radii from Mpc to pix ###

# zone_rad_pix = [((arctan((valx+rad)/valz) + 5.*(pi/180.))*(pix_count/10.)*(180./pi)) - ((arctan(valx/valz) + 5.*(pi/180.))*(pix_count/10.)*(180./pi)) for (rad,valx,valz) in zip(zone_rad,x_vol,z_vol)]
# zone_rad_pixy = [((arctan((valy+rad)/valz) + 5.*(pi/180.))*(pix_count/10.)*(180./pi)) - ((arctan(valy/valz) + 5.*(pi/180.))*(pix_count/10.)*(180./pi)) for (rad,valy,valz) in zip(zone_rad,y_vol,z_vol)]
# zone_rad_pix = [rad*factor for rad,factor in zip(zone_rad_sph_den,pix_factor)]
zone_rad_pix_cut = [rad*factor for rad,factor in zip(zone_rad_cut,pix_factor)]
zone_rad_rand_pix = [20*factor for factor in pix_factor]
# zone_rad_cut = 20.*np.ones(len(z_vol_cut))


### GET XY OF TOP N LARGEST VOIDS IN GIVEN REDSHIFT RANGE ###################################

# Index of voids in z = 0.317 bin
idx_317 = [j for j in xrange(len(zone_rad_cut)) if z_vol_cut[j]>=D_low_317 and z_vol_cut[j]<=D_upper_317]
pix_factor_317 = [pix_count/(2*valz*tan(radians(5.))) for valz in array(z_vol_cut)[idx_317]]

# XY coordinates for voids in z = 0.317 bin
x_vol_317 = array(x_vol_cut)[idx_317]
y_vol_317 = array(y_vol_cut)[idx_317]

# Most massive radii location for voids in z = 0.317
zone_lg_317 = np.argpartition(array(zone_rad_cut)[idx_317], -50)[-50:]

# XY locations of most massive radii voids in z = 0.317
x_vol_lg_317 = x_vol_317[zone_lg_317]
y_vol_lg_317 = y_vol_317[zone_lg_317]

x_vol_lg_pix_317 = [(valx*factor)+pix_count/2. for valx,factor in zip(x_vol_lg_317,pix_factor_317)]
y_vol_lg_pix_317 = [(valy*factor)+pix_count/2. for valy,factor in zip(y_vol_lg_317,pix_factor_317)]



#############################################################################

### CREATE SPHERICAL DENSITY PROFILE FOR A SINGLE ZONE ###############################

# This will take any void and build shells around it up to 2*R_v
# and find the number density per shell using the volume of the shell.

R_shell = np.linspace(0.001, 2*zone_rad[arb_ind_void], 20) #shells from ~0 to 2*R_v in units of Mpc/h
V_shell = ((4.*pi)/3.)*R_shell**3. #volume of each shell in units of Mpc**3 



count = []
count_void = []
nden = []


for i in R_shell:
	# Find number of halos in each concetric sphere of radius given by array R
	count_void.append(len(tree.query_ball_point([x_denmin[arb_ind_void], y_denmin[arb_ind_void], z_denmin[arb_ind_void]], i)))

for i in range(0,len(R_shell)):

	# This gives me a number density in each shell
	# looks for number of particles within a volume given by input radius
	if i==0:
		count_temp = len(tree.query_ball_point([x_denmin[arb_ind_void], y_denmin[arb_ind_void], z_denmin[arb_ind_void]], R_shell[i]))
		nden_temp = count_temp/V_shell[i]
	else:		
		count_temp1 = len(tree.query_ball_point([x_denmin[arb_ind_void], y_denmin[arb_ind_void], z_denmin[arb_ind_void]], R_shell[i]))
		count_temp2 = len(tree.query_ball_point([x_denmin[arb_ind_void], y_denmin[arb_ind_void], z_denmin[arb_ind_void]], R_shell[i-1]))
		count_temp = count_temp1-count_temp2
 		nden_temp = count_temp/(V_shell[i]-V_shell[i-1])

 	count.append(count_temp)
 	nden.append(nden_temp)



##########################################################################################################################################
###																																	   ###		
###		GET SPHERICAL PROFILE OF ALL ZONES BINNED BY EFFECTIVE RADIUS OF EACH ZONE 													   ###
###																																	   ###		
##########################################################################################################################################

# Split zones into bins from 0 to maximum radius of each zone
# zone_bins = np.linspace(0,20,2)
zone_bins = np.linspace(0,max(zone_rad_cut),2)
# zone_bins = np.linspace(0,max(zone_rad_sph_den),2)

# Range of spherical shells is from 0 to 2*R_eff of each zone in bin
R_mult = 4
num_shell = 20
R_stk = np.linspace(0.01,R_mult,num_shell)

# Create dictionary to store values for each output from the binning procedure
x_den_stk = {}
y_den_stk = {}
z_den_stk = {}
x_vol_stk = {}
y_vol_stk = {}
z_vol_stk = {}
x_pix_vol_stk = {}
y_pix_vol_stk = {}
zone_rad_stk = {}
zone_rad_pix_stk = {}
zone_dencon_stk = {}
zn_stk = {}
num_zn_stk = {}
avg_kappa_err_weight = {}
avg_kappa_err_nonNaN = {}
avg_delta_err = {}
delta_per_void = {}
kappa_per_void = {}
w = {}
weight = {}
k_err1_weight = {}
k_err1_nonNaN = {}



# For random locations
avg_kappa_sph2 = {}
kappa_per_void2 = {}
avg_kappa_err2 = {}
avg_kappa_sph3 = {}
kappa_per_void3 = {}
avg_kappa_err3 = {}
avg_kappa_sph4 = {}
kappa_per_void4 = {}
avg_kappa_err4 = {}
k_err12 = {}
k_err13 = {}
k_err14 = {}

for i in range(0,len(zone_bins)-1):
	# Loops over all the bins and puts xyz (den and vol), zone radii, zone dencon, num of zones in bin, and zone ID into dictionary
	# Note that values of the dictionary are referred to as the upper value of the bin
	# eg there will be no '0' entry, but will be a 'max(zone_radius)' entry
	x_den_stk[i],y_den_stk[i],z_den_stk[i],x_vol_stk[i],y_vol_stk[i],z_vol_stk[i],x_pix_vol_stk[i],y_pix_vol_stk[i],zone_rad_stk[i],zone_dencon_stk[i],zn_stk[i],zone_rad_pix_stk[i] = bin_zone(x_denmin_cut,y_denmin_cut,z_denmin_cut, x_vol_cut,y_vol_cut,z_vol_cut, x_pix_vol_cut, y_pix_vol_cut, zone_rad_cut, dencon, zone_rad_pix_cut, zone_bins[i], zone_bins[i+1])
	num_zn_stk[i] = len(zone_rad_stk[i])


# Create dictionaries for average counts, num den, and R_stk for each bins
avg_count_sph = {}
avg_den_sph = {}
avg_kappa_sph_weight = {}
avg_kappa_sph_nonNaN = {}

num_not_nan = np.zeros(len(R_stk))
num_not_nan2 = np.zeros(len(R_stk))
num_not_nan3 = np.zeros(len(R_stk))
num_not_nan4 = np.zeros(len(R_stk))

new_weight = np.zeros(len(R_stk))
# Get avg counts and nden for each shell in each bin
for i in range(0,len(zone_bins)-1):
	if zone_rad_stk[i] != []:
		
		
		# avg_count_sph[i], avg_den_sph[i], delta_per_void[i] = spherical_stk(zone_rad_stk[i],x_vol_stk[i],y_vol_stk[i],z_vol_stk[i],num_zn_stk[i])
		
		kappa_per_void[i] = sph_kappa_stk(zone_rad_pix_stk[i],x_pix_vol_stk[i],y_pix_vol_stk[i],num_zn_stk[i])
		# avg_kappa_sph[i] = [nanmean([(array(kappa_per_void[i])[k,j]) for k in xrange(len(kappa_per_void[i]))]) for j in xrange(len(R_stk))]
		
		for g in xrange(len(kappa_per_void[i])):
			for j,t in enumerate(kappa_per_void[i][g]):
				if np.isfinite(t):
					num_not_nan[j] += 1.
				
		# for h in xrange(len(w[i])):
		# 	for j,t in enumerate(w[i][h]):
		# 		new_weight[j] = new_weight[j]+t

		# print 'num_not_nan', num_not_nan
		# print 'new_weight', new_weight

		# # Create weights for each zone
		# weight[i] = (array(zone_rad_pix_stk[i])**2)/max(zone_rad_pix_stk[i])**2
		# wsum = sum(weight[i])
		
		# # Weighted normalistation
		# avg_kappa_sph_weight[i] = [nansum([weight[i][k]*array(kappa_per_void[i][k][j]) for k in xrange(len(kappa_per_void[i]))])/wsum for j in xrange(len(R_stk))]
		
		# Num_not_NaN normalisation
		avg_kappa_sph_nonNaN[i] = [nansum([array(kappa_per_void[i][k][j]) for k in xrange(len(kappa_per_void[i]))])/num_not_nan[j] for j in xrange(len(R_stk))]


		# COVARIANCE AND X^2
		cov_kappa, cov_norm_kappa = covariance(kappa_per_void[i],kappa_per_void[i])

		chi_sq_kappa = chi_sq(cov_kappa,avg_kappa_sph_nonNaN[i])


		# # FOR RANDOM LOCATIONS
		# kappa_per_void2[i] = sph_kappa_stk(zone_rad_pix_stk[i],xpix_rand2,ypix_rand2,num_zn_stk[i])
		
		# for g in xrange(len(kappa_per_void2[i])):
		# 	for j,t in enumerate(kappa_per_void2[i][g]):
		# 		if np.isfinite(t):
		# 			num_not_nan2[j] += 1.

		# # Weighted normalistation			
		# # avg_kappa_sph2[i] = [nansum([weight[i][k]*array(kappa_per_void2[i][k][j]) for k in xrange(len(kappa_per_void2[i]))])/wsum for j in xrange(len(R_stk))]
		
		# # Num_not_NaN normalisation
		# avg_kappa_sph2[i] = [nansum([array(kappa_per_void2[i][k][j]) for k in xrange(len(kappa_per_void2[i]))])/num_not_nan2[j] for j in xrange(len(R_stk))]


		# kappa_per_void3[i] = sph_kappa_stk(zone_rad_pix_stk[i],xpix_rand3,ypix_rand3,num_zn_stk[i])
		
		# for g in xrange(len(kappa_per_void3[i])):
		# 	for j,t in enumerate(kappa_per_void3[i][g]):
		# 		if np.isfinite(t):
		# 			num_not_nan3[j] += 1.
		
		# # Weighted normalistation
		# # avg_kappa_sph3[i] = [nansum([weight[i][k]*array(kappa_per_void3[i][k][j]) for k in xrange(len(kappa_per_void3[i]))])/wsum for j in xrange(len(R_stk))]
		
		# # Num_not_NaN normalisation
		# avg_kappa_sph3[i] = [nansum([array(kappa_per_void3[i][k][j]) for k in xrange(len(kappa_per_void3[i]))])/num_not_nan3[j] for j in xrange(len(R_stk))]


		# kappa_per_void4[i] = sph_kappa_stk(zone_rad_pix_stk[i],xpix_rand4,ypix_rand4,num_zn_stk[i])
		
		# for g in xrange(len(kappa_per_void4[i])):
		# 	for j,t in enumerate(kappa_per_void4[i][g]):
		# 		if np.isfinite(t):
		# 			num_not_nan4[j] += 1.
		
		# # Weighted normalistation			
		# # avg_kappa_sph4[i] = [nanmean([(array(kappa_per_void4[i])[k,j]) for k in xrange(len(kappa_per_void4[i]))]) for j in xrange(len(R_stk))]
		
		# # Num_not_NaN normalisation
		# avg_kappa_sph4[i] = [nansum([array(kappa_per_void4[i][k][j]) for k in xrange(len(kappa_per_void4[i]))])/num_not_nan4[j] for j in xrange(len(R_stk))]

		
		# ERROR ON MEAN OF KAPPA AND DELTA 
		# for h,t in enumerate(weight[i]):
		# 	avg_kappa_err[i] = [sqrt(nansum(t*array(kappa_per_void[i][h])[j]**2)/sum(weight[i]) - (nansum(t*array(kappa_per_void[i][h])[j])/sum(weight[i]))**2) for j in xrange(len(R_stk))]
		# avg_kappa_err[i] = [sqrt([nansum(weight[i][k]*(array(kappa_per_void[i][k])[j])**2)/wsum - (nansum([weight[i][k]*array(kappa_per_void[i][k])[j]))**2)/wsum for k in xrange(len(kappa_per_void[i])-1)]) for j in xrange(len(R_stk))]
		
		# # Weighted normalistation
		# k_err1_weight[i] = array([nansum([weight[i][k]*array(kappa_per_void[i][k][j])**2 for k in xrange(len(kappa_per_void[i]))])/wsum for j in xrange(len(R_stk))])
		# avg_kappa_err_weight[i] = sqrt(k_err1_weight[i]-array(avg_kappa_sph_weight[i])**2)/wsum

		# Num_not_NaN normalisation		
		k_err1_nonNaN[i] = array([nansum([array(kappa_per_void[i][k][j])**2 for k in xrange(len(kappa_per_void[i]))])/(num_not_nan[j])  for j in xrange(len(R_stk))])
		avg_kappa_err_nonNaN[i] = sqrt(k_err1_nonNaN[i]-array(avg_kappa_sph_nonNaN[i])**2)/sqrt(array(num_not_nan))

		# avg_delta_err[i] = [sqrt(mean([abs(array(delta_per_void[i])[k,j]-avg_den_sph[i][j])**2 for k in xrange(len(delta_per_void[i]))]))/sqrt(len(delta_per_void[i])) for j in xrange(len(R_stk))]

		# Weighted normalistation
		# k_err12[i] = array([nansum([weight[i][k]*array(kappa_per_void2[i][k][j])**2 for k in xrange(len(kappa_per_void2[i]))])/wsum for j in xrange(len(R_stk))])
		# avg_kappa_err2[i] = sqrt(k_err12[i]-array(avg_kappa_sph2[i])**2)/wsum

		# # Num_not_NaN normalisation
		# k_err12[i] = array([nansum([array(kappa_per_void2[i][k][j])**2 for k in xrange(len(kappa_per_void2[i]))])/num_not_nan2[j]  for j in xrange(len(R_stk))])
		# avg_kappa_err2[i] = sqrt(k_err12[i]-array(avg_kappa_sph2[i])**2)/sqrt(array(num_not_nan2))

		# Weighted normalistation
		# k_err13[i] = array([nansum([weight[i][k]*array(kappa_per_void3[i][k][j])**2 for k in xrange(len(kappa_per_void3[i]))])/wsum for j in xrange(len(R_stk))])
		# avg_kappa_err3[i] = sqrt(k_err13[i]-array(avg_kappa_sph3[i])**2)/wsum

		# # Num_not_NaN normalisation
		# k_err13[i] = array([nansum([array(kappa_per_void3[i][k][j])**2 for k in xrange(len(kappa_per_void3[i]))])/num_not_nan3[j]  for j in xrange(len(R_stk))])
		# avg_kappa_err3[i] = sqrt(k_err13[i]-array(avg_kappa_sph3[i])**2)/sqrt(array(num_not_nan3))

		# Weighted normalistation
		# k_err14[i] = array([nansum([weight[i][k]*array(kappa_per_void4[i][k][j])**2 for k in xrange(len(kappa_per_void4[i]))])/wsum for j in xrange(len(R_stk))])
		# avg_kappa_err4[i] = sqrt(k_err14[i]-array(avg_kappa_sph4[i])**2)/wsum

		# # Num_not_NaN normalisation
		# k_err14[i] = array([nansum([array(kappa_per_void4[i][k][j])**2 for k in xrange(len(kappa_per_void4[i]))])/num_not_nan4[j]  for j in xrange(len(R_stk))])
		# avg_kappa_err4[i] = sqrt(k_err14[i]-array(avg_kappa_sph4[i])**2)/sqrt(array(num_not_nan4))
		
		# avg_kappa_err3[i] = [sqrt(nanmean([abs(array(kappa_per_void3[i])[k,j]-avg_kappa_sph3[i][j])**2 for k in xrange(len(kappa_per_void3[i]))]))/sqrt(len(kappa_per_void3[i])) for j in xrange(len(R_stk))]
		# avg_kappa_err4[i] = [sqrt(nanmean([abs(array(kappa_per_void4[i])[k,j]-avg_kappa_sph4[i][j])**2 for k in xrange(len(kappa_per_void4[i]))]))/sqrt(len(kappa_per_void4[i])) for j in xrange(len(R_stk))]
		# avg_delta_err[i] = [sqrt(mean([abs(array(delta_per_void[i])[k,j]-avg_den_sph[i][j])**2 for k in xrange(len(delta_per_void[i]))]))/sqrt(len(delta_per_void[i])) for j in xrange(len(R_stk))]
		
# avg_kappa_sph_all, avg_kappa_err_all = sph_kappa_stk(zone_rad_pix_cut,x_pix_vol_cut,y_pix_vol_cut,len(zone_rad_pix_cut))

		# np.save('sph_kappa_Mocks_%d' % zone_bins[i+1], (avg_kappa_sph[i], avg_kappa_err[i]))

# KAPPA PROFILE FOR HALOS
# num_not_nan_halo = np.zeros(len(R_stk))

# kappa_per_halo = sph_kappa_stk(rad_halo_massive,x_halo_massive,y_halo_massive,len(x_halo_massive))

# for g in xrange(len(kappa_per_halo)):
# 	for j,t in enumerate(kappa_per_halo[g]):
# 		if np.isfinite(t):
# 			num_not_nan_halo[j] += 1.


# avg_kappa_sph_halo = [nansum([array(kappa_per_halo[k][j]) for k in xrange(len(kappa_per_halo))])/num_not_nan_halo[j] for j in xrange(len(R_stk))]

# k_err1_halo = array([nansum([array(kappa_per_halo[k][j])**2 for k in xrange(len(kappa_per_halo))])/(num_not_nan_halo[j])  for j in xrange(len(R_stk))])
# avg_kappa_err_halo = sqrt(k_err1_halo-array(avg_kappa_sph_halo)**2)/sqrt(array(num_not_nan_halo))


# Load spherical avg counts and avg nden
# for i in range(0,len(zone_bins)-1):
	# if zone_bins[i+1] < 10:
		# avg_kappa_sph[i], avg_kappa_err[i] = np.load('sph_kappa_Mocks_%d.npy' % zone_bins[i+1])
		# avg_count_sph[i], avg_den_sph[i] = np.load('sph_count_nden_Mocks_0_221_noncube_0%d.npy' % zone_bins[i+1])	
	# else:
# 		avg_count_sph[i], avg_den_sph[i] = np.load('sph_count_nden_Mocks_0_221_noncube_%d.npy' % zone_bins[i+1])

################################################################################################################################################
###																																			 ###	
###				LOOP OVER ALL ZONES TO CREATE STACKED BOUNDARY PROFILE 																		 ###
###																																			 ###
################################################################################################################################################


# Bins and bin mids to be used for each zone.  These will be normalized to the effective zone radius
# bins = np.linspace(0,3.0,20)
bins = np.arange(0,2*max(zone_rad_cut),5)


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

zncnt = 0
# for i in range(0,len(zone_bins)-1):
# 	print 'num of zones in %d' % i, len(zn_stk[i])
# 	t_begin2 = time.time()
# 	print zncnt
# 	for zn in zn_stk[i]:

# 		# Loop over all zones and get index for each particle in that zone
# 		# zn = zn_stk[8][0]
# 		same_zone_id_bn = np.where(zone_nonzero == zn)[0]

# 		tot_zone_vol_bn = 0
# 		vol_same_zone_bn = []
# 		slice_max_bn = []
# 		slice_min_bn = []

# 		mult = 1.5

# 		for n in same_zone_id_bn:
# 			tot_zone_vol_bn += vol[n]
# 			vol_same_zone_bn.append(vol[n])
# 			slice_max_bn.append((mult*r_eff[n]+z[n]))
# 			slice_min_bn.append((z[n]-mult*r_eff[n]))
		
# 		r_eff_zone_tot_bn = (tot_zone_vol*(3./(4*pi)))**(1./3.)

# 		x_same_zone_bn = []
# 		y_same_zone_bn = []
# 		z_same_zone_bn = []
# 		r_eff_same_zone_bn = []
# 		same_zone_adj_bn = []


# 		for valx in same_zone_id_bn:
# 			x_same_zone_bn.append(x[valx])

# 		for valy in same_zone_id_bn:
# 			y_same_zone_bn.append(y[valy])

# 		for valz in same_zone_id_bn:
# 			z_same_zone_bn.append(z[valz])

# 		for valrad in same_zone_id_bn:
# 			r_eff_same_zone_bn.append(r_eff[valrad])

# 		for valadj in same_zone_id_bn:
# 			# Gets array of indices of adjacent particles to each particle in a specific zone
# 			same_zone_adj_bn.append(adj_dict[valadj])



# 		#### GET ADJACENT PARTICLES THAT ARE NOT PART OF THE ZONE ###########################
# 		cell_on_boundary_tot, cell_on_boundary, x_non_zone_adj_arr_tot, y_non_zone_adj_arr_tot, z_non_zone_adj_arr_tot, adj_cell_vol_tot = adj_particles(same_zone_adj_bn,zn)

# 		# Volume weighted average of boundary
# 		x_vol_avg, y_vol_avg, z_vol_avg = vol_avg_center(array(x)[cell_on_boundary_tot],array(y)[cell_on_boundary_tot],array(z)[cell_on_boundary_tot], x_non_zone_adj_arr_tot,y_non_zone_adj_arr_tot,z_non_zone_adj_arr_tot, array(vol)[cell_on_boundary_tot],adj_cell_vol_tot)
# 		# x_vol_avg_plot, y_vol_avg_plot, z_vol_avg_plot = vol_avg_center(x[cell_on_boundary],y[cell_on_boundary],z[cell_on_boundary], x_non_zone_adj_arr_slice,y_non_zone_adj_arr_slice,z_non_zone_adj_arr_slice, vol[cell_on_boundary],adj_cell_vol)


# 		### CREATE BOUNDARY DISTANCE PROFILE FOR A SINGLE ZONE AND APPEND TO STACKED VALUE ###############################
# 		den_bins, dist_bins, ncnt_bins, den_bins_non_zn, dist_bins_non_zn, ncnt_bins_non_zn = boundary_stk(x_vol_avg,y_vol_avg,z_vol_avg, x_same_zone_bn,y_same_zone_bn,z_same_zone_bn, vol_same_zone_bn, int(zn), 2*max(zone_rad_cut))


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
# 	zncnt += 1			
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

# 	# Save den_bins_all for whichever bin 
# 	# np.save('den_bins_all_69_test', (den_bins_all))
# 	# np.save('den_bins_all_Mocks_v2_%d' % zone_bins[i+1], (den_bins_all))
# 	# db_all = np.load('den_bins_all_69_test.npy')
# 	t_end2 = time.time()
# 	print 'time of %d bin: \t%g minutes' % (i, ((t_end2-t_begin2)/60.))


# den_bins_all = np.load('den_bins_all_35_45.npy')

# Load filesfor boundary stacked bins
# for i in range(0,len(zone_bins)-1):
# 	if zone_bins[i+1] < 10:
# 		den_bins_all_dict[i] = np.load('den_bins_all_Mocks_v2_0%d.npy' % zone_bins[i+1])
# 	else:
# 		den_bins_all_dict[i] = np.load('den_bins_all_Mocks_v2_%d.npy' % zone_bins[i+1])

### PRINT STATEMENTS ##############################################################################

# print 'x,y,z', x[arb_ind], y[arb_ind], z[arb_ind]
# print 'x,y,z of den min of largest zone', x_denmin[np.int(zone_nonzero[arb_ind])], y_denmin[np.int(zone_nonzero[arb_ind])], z_denmin[np.int(zone_nonzero[arb_ind])]
print 'radius of largest cell in zone slice', max(r_eff_same_zone)
# print 'vol',vol[arb_ind]
print 'radius of density min cell of zone', r_eff_zone_slice[denminrad]
print 'zone_id', zone_nonzero[arb_ind]
print ''

print 'Num of same zone IDs',  len(same_zone_id)
print 'Particles in slice with same zone', len(x_slice_zone)
print ''

print 'zone radius from file', zone_rad[arb_ind_void]
print 'zone radius from adding cell volumes', r_eff_zone_tot
print 'max zone radius', max(zone_rad)
print ''

# print 'num of boundary cells', len(cell_on_boundary)

t_end = time.time()
print 'time of code: \t%g minutes' % ((t_end-t_begin)/60.)

###################################################################################################

### CREATE PLOTS ###

###################################################################################################

# fig1 = plt.figure(figsize=(10,8))
# fig2 = plt.figure(figsize=(10,8))
# fig3 = plt.figure(figsize=(10,8))
# fig4 = plt.figure(figsize=(10,8))
# fig5 = plt.figure(figsize=(10,8))
# fig6 = plt.figure(figsize=(10,8))
# fig7 = plt.figure(figsize=(10,8))
fig8 = plt.figure(figsize=(10,8))
# fig9 = plt.figure(figsize=(10,8))
# fig10 = plt.figure(figsize=(10,8))
# fig11 = plt.figure(figsize=(10,8))
# fig12 = plt.figure(figsize=(10,8))

# Create ability to cycle through linestyles
lines = ["-","--","-.",":"]
colors = ['k','m','c','g', 'y']
colorcycler = cycle(colors)
linecycler = cycle(lines)


# Create scatter plot of all halos in slice and all halos representing void centers
# ax1 = fig1.add_subplot(111)
# ax1.set_xlim(x_vol[arb_ind_void]-(2.*zone_rad[arb_ind_void]),x_vol[arb_ind_void]+(2.*zone_rad[arb_ind_void]))
# ax1.set_ylim(y_vol[arb_ind_void]-(2.*zone_rad[arb_ind_void]),y_vol[arb_ind_void]+(2.*zone_rad[arb_ind_void]))
# ax1.set_xlabel(r'$\mathrm{x}$')
# ax1.set_ylabel(r'$\mathrm{y}$')
# ax1.scatter(x_slice, y_slice, color='red', marker='o')
# ax1.scatter(x_slice_zone, y_slice_zone, color='green', marker='s')
# # ax1.scatter(x[arb_ind], y[arb_ind], color='black', marker='d', s=4)
# circ2=plt.Circle((x_denmin[arb_ind_void],y_denmin[arb_ind_void]),zone_rad[arb_ind_void], fill=None)
# for i in range(0,len(slice_zone_idx)):
# 	circ=plt.Circle((x_slice_zone[i],y_slice_zone[i]),r_eff_zone_slice[i]*0.5, color='b', alpha=0.5)
# 	ax1.add_artist(circ)
# # circ=plt.Circle((x[arb_ind],y[arb_ind]),r_eff[arb_ind], color='black', alpha=0.5)
# ax1.scatter(x_denmin[arb_ind_void],y_denmin[arb_ind_void], color='black', marker='d', s=100)
# ax1.scatter(x_vol[arb_ind_void],y_vol[arb_ind_void], color='black', marker='+', s=100)
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
# ax2.axvline(zone_rad[zone_nonzero[arb_ind]], linewidth=2, linestyle='--', color='black')
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
# ax3.set_ylim(0,2)
# ax3.set_xlabel(r'$\mathrm{r/R_{zone}}$')
# ax3.set_ylabel(r'$\mathrm{\delta(r)+1}$')
# for i in range(0,len(zone_bins)-1):
# 	if zone_rad_stk[i] != []:
# 		clr = next(colorcycler)
# 		ax3.plot(R_stk, np.array(avg_den_sph[i]), linewidth=3, label='%2d' % zone_bins[i+1], linestyle = next(linecycler), color = clr)
# 		ax3.errorbar(R_stk, np.array(avg_den_sph[i]), yerr=np.array(avg_delta_err[i]), fmt='x', color = clr)
# ax3.spines['top'].set_linewidth(2.3)
# ax3.spines['left'].set_linewidth(2.3)
# ax3.spines['right'].set_linewidth(2.3)
# ax3.spines['bottom'].set_linewidth(2.3)
# for tick in ax3.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax3.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# ax3.legend(loc='best', fancybox = True, shadow = True, ncol=2)
# fig3.savefig('KiDS_Mocks_upto_z_525_spherical_prof_stk_bin', format='pdf')


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
# ax5.set_xlim(x_vol[arb_ind_void]-50,x_vol[arb_ind_void]+50)
# ax5.set_ylim(y_vol[arb_ind_void]-50,y_vol[arb_ind_void]+50)
# ax5.set_xlabel(r'$\mathrm{x}$')
# ax5.set_ylabel(r'$\mathrm{y}$')
# ax5.scatter(x_slice, y_slice, color='red', marker='o')
# ax5.scatter(x_slice_zone, y_slice_zone, color='green', marker='s')
# circ2=plt.Circle((x_denmin[arb_ind_void],y_denmin[arb_ind_void]),zone_rad[arb_ind_void], fill=None)
# # circ=plt.Circle((x[arb_ind],y[arb_ind]),r_eff[arb_ind], color='black', alpha=0.5)
# ax5.scatter(x_non_zone_adj_2303, y_non_zone_adj_2303, color='blue', marker='*', s=100)
# ax5.scatter(x_vol_avg_2303, y_vol_avg_2303, color='purple', marker='^', s=100)
# ax5.scatter(x_denmin[arb_ind_void],y_denmin[arb_ind_void], color='black', marker='d', s=100)
# ax5.scatter(x_vol[arb_ind_void],y_vol[arb_ind_void], color='black', marker='+', s=100)
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
# ax6 = fig6.add_subplot(111)
# ax6.set_xlim(-50,50)
# # ax6.set_ylim(0,2)
# ax6.set_xlabel(r'$\mathrm{{\it D} \hspace{0.5cm} [Mpc]}$')
# ax6.set_ylabel(r'$\mathrm{\delta({\it D})+1}$') 
# # ax6.plot(bins_mid_full, den_bins_all, linewidth=3, linestyle = next(linecycler))
# for i in range(0,len(zone_bins)-1):
# 	ax6.plot(bins_mid_full, den_bins_all_dict[i], linewidth=3, label='%2d' % zone_bins[i+1], linestyle = next(linecycler))
# ax6.spines['top'].set_linewidth(2.3)
# ax6.spines['left'].set_linewidth(2.3)
# ax6.spines['right'].set_linewidth(2.3)
# ax6.spines['bottom'].set_linewidth(2.3)
# for tick in ax6.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax6.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# ax6.legend(loc='best', fancybox = True, shadow = True)
# fig6.savefig('KiDS_Mocks_0_221_boundary_prof_stk_bin', format='pdf')

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

ax8 = fig8.add_subplot(111)
# ax8.set_xlim(0,33)
ax8.set_ylim(-0.0003,0.0003)
ax8.set_xlabel(r'$\mathrm{r/R_{zone}}$')
ax8.set_ylabel(r'$\mathrm{\kappa}$')
for i in range(0,len(zone_bins)-1):
	if zone_rad_stk[i] != []:
		# clr = next(colorcycler)
		# ax8.plot(R_stk, np.array(avg_kappa_sph_weight[i]), linewidth=3, label='%2d Weighted Norm' % zone_bins[i+1], linestyle = next(linecycler), color = clr)
		# ax8.errorbar(R_stk, np.array(avg_kappa_sph_weight[i]), yerr=np.array(avg_kappa_err_weight[i]), fmt='x', color = clr)
		clr = next(colorcycler)
		ax8.plot(R_stk, np.array(avg_kappa_sph_nonNaN[i]), linewidth=3, label='%2d' % zone_bins[i+1], linestyle = next(linecycler), color = clr)
		ax8.errorbar(R_stk, np.array(avg_kappa_sph_nonNaN[i]), yerr=np.array(avg_kappa_err_nonNaN[i]), fmt='x', color = clr)
		
		# Randoms
		# clr = next(colorcycler)
		# ax8.plot(R_stk, np.array(avg_kappa_sph2[i]), linewidth=3, label='%2d' % zone_bins[i+1], linestyle = next(linecycler), color = clr)
		# ax8.errorbar(R_stk, np.array(avg_kappa_sph2[i]), yerr=np.array(avg_kappa_err2[i]), fmt='x', color = clr)
		# clr = next(colorcycler)
		# ax8.plot(R_stk, np.array(avg_kappa_sph3[i]), linewidth=3, label='%2d' % zone_bins[i+1], linestyle = next(linecycler), color = clr)
		# ax8.errorbar(R_stk, np.array(avg_kappa_sph3[i]), yerr=np.array(avg_kappa_err3[i]), fmt='x', color = clr)
		# clr = next(colorcycler)
		# ax8.plot(R_stk, np.array(avg_kappa_sph4[i]), linewidth=3, label='%2d' % zone_bins[i+1], linestyle = next(linecycler), color = clr)
		# ax8.errorbar(R_stk, np.array(avg_kappa_sph4[i]), yerr=np.array(avg_kappa_err4[i]), fmt='x', color = clr)
# ax8.plot(R_stk, np.array(avg_kappa_sph_halo), linewidth=3)
# ax8.errorbar(R_stk, np.array(avg_kappa_sph_halo), yerr=np.array(avg_kappa_err_halo), fmt='x')
ax8.spines['top'].set_linewidth(2.3)
ax8.spines['left'].set_linewidth(2.3)
ax8.spines['right'].set_linewidth(2.3)
ax8.spines['bottom'].set_linewidth(2.3)
# ax8.set_xscale('log')
# ax8.set_yscale('log')
for tick in ax8.xaxis.get_major_ticks():
    tick.label.set_fontsize(27)
for tick in ax8.yaxis.get_major_ticks():
    tick.label.set_fontsize(27)
ax8.legend(loc='best', fancybox = True, shadow = True)#, ncol = 2)
# fig8.savefig('KiDS_Mocks_upto_z_525_kappa_sph_random_allzones3', format='png')

# Create kappa vs R_v for stacked halos
# ax9 = fig9.add_subplot(111)
# # ax9.set_xlim(0,33)
# # ax9.set_ylim(0,2)
# ax9.set_xlabel(r'$\mathrm{r/R_{zone}}$')
# ax9.set_ylabel(r'$\mathrm{\kappa}$')
# clr = next(colorcycler)
# ax9.plot(R_stk, np.array(avg_kappa_sph_halo), linewidth=3, linestyle = next(linecycler), color = clr)
# ax9.errorbar(R_stk, np.array(avg_kappa_sph_halo), yerr=np.array(avg_kappa_err_halo), fmt='x', color = clr)
# ax9.spines['top'].set_linewidth(2.3)
# ax9.spines['left'].set_linewidth(2.3)
# ax9.spines['right'].set_linewidth(2.3)
# ax9.spines['bottom'].set_linewidth(2.3)
# ax9.set_yscale('log')
# ax9.set_xscale('log')
# for tick in ax9.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax9.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# ax9.legend(loc='best', fancybox = True, shadow = True, ncol=2)
# fig9.savefig('KiDS_Mocks_upto_z_525_spherical_prof_stk_bin', format='pdf')

# Create scatter plot of galaxies in LC within y slice and some zones
# ax1 = fig10.add_subplot(111)
# ax1.set_xlabel(r'$\mathrm{x}$')
# ax1.set_ylabel(r'$\mathrm{z}$')
# ax1.set_xlim(-150,150)
# ax1.set_ylim(1000,1300)
# ax1.scatter(xslice, zslice, color='red', marker='o')
# for i in xrange(0,len(yslice_zone_loc)):
# 	circ=plt.Circle((xslice_zone[i],zslice_zone[i]),yslice_zone_rad[i], color='b', alpha=0.5)
# 	ax1.add_artist(circ)
# 	ax1.scatter(xslice_zone[i],zslice_zone[i], color='black', marker='d', s=100)
# ax1.add_artist(circ)
# ax1.spines['top'].set_linewidth(2.3)
# ax1.spines['left'].set_linewidth(2.3)
# ax1.spines['right'].set_linewidth(2.3)
# ax1.spines['bottom'].set_linewidth(2.3)
# for tick in ax1.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax1.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)   
# fig10.savefig('KiDS_Mocks_LC_yslice_zones', format='png')

# Create 2D histogram of yslice with void locations.
# ax1 = fig11.add_subplot(111)
# ax1.set_xlabel(r'$\mathrm{x}$')
# ax1.set_ylabel(r'$\mathrm{z}$')
# hst = ax1.hist2d(xslice, zslice, bins=25, norm=mclr.LogNorm(), range = [[-150,150],[1000,1300]])
# fig11.colorbar(hst,fraction=0.046, pad=0.04)
# ax1.scatter(xslice_zone,zslice_zone, color='k', marker='d', s=100)
# ax1.spines['top'].set_linewidth(2.3)
# ax1.spines['left'].set_linewidth(2.3)
# ax1.spines['right'].set_linewidth(2.3)
# ax1.spines['bottom'].set_linewidth(2.3)
# for tick in ax1.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax1.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)  
# fig11.savefig('KiDS_Mocks_yslice_hist2D', format='png')

# Create 2D histogram of yslice with void locations.
# ax1 = fig12.add_subplot(111)
# ax1.set_xlabel(r'$\mathrm{x_{pix}}$')
# ax1.set_ylabel(r'$\mathrm{y_{pix}}$')
# ax1.set_ylim(0,len(kappa))
# ax1.set_xlim(0,len(kappa))
# im = ax1.imshow(kappa, cmap='hot')
# # divider = make_axes_locatable(ax1)
# # cax = divider.append_axes("right", size="5%", pad=0.05)
# fig12.colorbar(im,fraction=0.046, pad=0.04)
# ax1.scatter(x_vol_lg_pix_317,y_vol_lg_pix_317, color='w', marker='x', s=100)
# ax1.spines['top'].set_linewidth(2.3)
# ax1.spines['left'].set_linewidth(2.3)
# ax1.spines['right'].set_linewidth(2.3)
# ax1.spines['bottom'].set_linewidth(2.3)
# for tick in ax1.xaxis.get_major_ticks():
#     tick.label.set_fontsize(27)
# for tick in ax1.yaxis.get_major_ticks():
#     tick.label.set_fontsize(27)  
# #fig12.savefig('KiDS_Mocks_kappa_voids_at_0_317', format='png')

plt.show()