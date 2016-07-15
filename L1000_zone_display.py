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

Lbox = 1000

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

		if r == (len(zone_rad_stk)/2):
			print "Halfway there!"
		elif r == (len(zone_rad_stk)-1):
			print 'Last one!'

		# Find index of halo representing min value of zone
		ind_x_stk = find_nearest(x,x_zone_stk[r])
		ind_y_stk = find_nearest(y,y_zone_stk[r])
		ind_z_stk = find_nearest(z,z_zone_stk[r])

		# # Rotate coordinates to make halo representing min den of zone in the center of the box
		# x_rot = coord_rot(x,x[ind_x_stk],Lbox)
		# y_rot = coord_rot(y,y[ind_y_stk],Lbox)
		# z_rot = coord_rot(z,z[ind_z_stk],Lbox)

		# halos_rot = zip(x_rot.ravel(), y_rot.ravel(), z_rot.ravel()) #makes x,y,z single arrays
		# tree_rot = spatial.cKDTree(halos_rot)


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
		avg_count = np.array(avg_count_temp)/numzone_stk
		avg_nden = np.array(avg_nden_temp)/numzone_stk

	return avg_count, avg_nden, R_stk

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

for n in same_zone_id:
	tot_zone_vol += vol[n]

r_eff_zone_tot = (tot_zone_vol*(3./(4*pi)))**(1./3.)

### GET COORDINATES OF PARTICLES THAT BELONG TO ZONE BELONG TO ZONE #####################################################

x_same_zone = []
y_same_zone = []
z_same_zone = []
r_eff_same_zone = []

for valx in same_zone_id:
	x_same_zone.append(x[valx])

for valy in same_zone_id:
	y_same_zone.append(y[valy])

for valz in same_zone_id:
	z_same_zone.append(z[valz])

for valrad in same_zone_id:
	r_eff_same_zone.append(r_eff[valrad])

### CREATE SLICE IN THE Z DIMENSION ########################################################

x_slice = []
y_slice = []
slice_idx = []

x_slice_zone = []
y_slice_zone = []
slice_zone_idx = []
r_eff_zone_slice = []

slice_max = (4.5*r_eff[arb_ind]+z[arb_ind])
slice_min = (z[arb_ind]-4.5*r_eff[arb_ind])


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
### CREATE TREE FOR X,Y,Z COORDINATES FOR ALL HALOS #########################

# Create tree with period boundary conditions
halos = zip(x.ravel(), y.ravel(), z.ravel()) #makes x,y,z single arrays
bounds = np.array([Lbox,Lbox,Lbox])
periodic_tree = PeriodicCKDTree(bounds, halos)

#############################################################################


### CREATE SPHERICAL DENSITY PROFILE FOR A SINGLE VOID ###############################

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

# Set up 'small', 'medium', 'large' void arrays
x_zone_stk_sm = []
y_zone_stk_sm = []
z_zone_stk_sm = []
x_vol_zone_stk_sm = []
y_vol_zone_stk_sm = []
z_vol_zone_stk_sm = []
zone_rad_stk_sm = []
zone_dencon_stk_sm = []

x_zone_stk_md = []
y_zone_stk_md = []
z_zone_stk_md = []
x_vol_zone_stk_md = []
y_vol_zone_stk_md = []
z_vol_zone_stk_md = []
zone_rad_stk_md = []
zone_dencon_stk_md = []

x_zone_stk_lg = []
y_zone_stk_lg = []
z_zone_stk_lg = []
x_vol_zone_stk_lg = []
y_vol_zone_stk_lg = []
z_vol_zone_stk_lg = []
zone_rad_stk_lg = []
zone_dencon_stk_lg = []


# Loop over all zones and retain xyz and radius for those that have given size
for i in range(0,len(ID)):
	if zone_rad[i] <= 35.:	
		x_zone_stk_sm.append(x_denmin[i])
		y_zone_stk_sm.append(y_denmin[i])
		z_zone_stk_sm.append(z_denmin[i])
		x_vol_zone_stk_sm.append(x_vol[i])
		y_vol_zone_stk_sm.append(y_vol[i])
		z_vol_zone_stk_sm.append(z_vol[i])
		zone_rad_stk_sm.append(zone_rad[i])
		zone_dencon_stk_sm.append(dencon[i])

	if zone_rad[i] > 35. and zone_rad[i] <= 50.:	
		x_zone_stk_md.append(x_denmin[i])
		y_zone_stk_md.append(y_denmin[i])
		z_zone_stk_md.append(z_denmin[i])
		x_vol_zone_stk_md.append(x_vol[i])
		y_vol_zone_stk_md.append(y_vol[i])
		z_vol_zone_stk_md.append(z_vol[i])
		zone_rad_stk_md.append(zone_rad[i])
		zone_dencon_stk_md.append(dencon[i])

	if zone_rad[i] > 50.:	
		x_zone_stk_lg.append(x_denmin[i])
		y_zone_stk_lg.append(y_denmin[i])
		z_zone_stk_lg.append(z_denmin[i])
		x_vol_zone_stk_lg.append(x_vol[i])
		y_vol_zone_stk_lg.append(y_vol[i])
		z_vol_zone_stk_lg.append(z_vol[i])
		zone_rad_stk_lg.append(zone_rad[i])
		zone_dencon_stk_lg.append(dencon[i])

numzone_stk_sm = len(zone_rad_stk_sm)
numzone_stk_md = len(zone_rad_stk_md)
numzone_stk_lg = len(zone_rad_stk_lg)

avg_count_temp_sm = []
avg_nden_temp_sm = []
avg_count_temp_md = []
avg_nden_temp_md = []
avg_count_temp_lg = []
avg_nden_temp_lg = []

print 'number of small zones', numzone_stk_sm
print 'number of medium zones', numzone_stk_md
print 'number of large zones', numzone_stk_lg
print ''


### XYZ of den min center
avg_count_sm, avg_nden_sm, R_stk = spherical_stk(zone_rad_stk_sm, x_zone_stk_sm, y_zone_stk_sm, z_zone_stk_sm, numzone_stk_sm)
avg_count_md, avg_nden_md, R_stk = spherical_stk(zone_rad_stk_md,x_zone_stk_md,y_zone_stk_md,z_zone_stk_md,numzone_stk_md)
avg_count_lg, avg_nden_lg, R_stk = spherical_stk(zone_rad_stk_lg,x_zone_stk_lg,y_zone_stk_lg,z_zone_stk_lg,numzone_stk_lg)

# Save files
np.save('sm_bin_count_nden', (avg_count_sm, avg_nden_sm))
np.save('md_bin_count_nden', (avg_count_md, avg_nden_md))
np.save('lg_bin_count_nden', (avg_count_lg, avg_nden_lg))

### XYZ of vol avg center
avg_count_sm_vol, avg_nden_sm_vol, R_stk = spherical_stk(zone_rad_stk_sm, x_vol_zone_stk_sm, y_vol_zone_stk_sm, z_vol_zone_stk_sm, numzone_stk_sm)
avg_count_md_vol, avg_nden_md_vol, R_stk = spherical_stk(zone_rad_stk_md, x_vol_zone_stk_md, y_vol_zone_stk_md, z_vol_zone_stk_md, numzone_stk_md)
avg_count_lg_vol, avg_nden_lg_vol, R_stk = spherical_stk(zone_rad_stk_lg, x_vol_zone_stk_lg, y_vol_zone_stk_lg, z_vol_zone_stk_lg, numzone_stk_lg)

np.save('sm_bin_count_nden_vol', (avg_count_sm_vol, avg_nden_sm_vol))
np.save('md_bin_count_nden_vol', (avg_count_md_vol, avg_nden_md_vol))
np.save('lg_bin_count_nden_vol', (avg_count_lg_vol, avg_nden_lg_vol))


### PRINT STATEMENTS ##############################################################################

print 'x,y,z', x[arb_ind], y[arb_ind], z[arb_ind]
print 'x,y,z of den min of largest zone', x_denmin[np.int(zone[arb_ind])], y_denmin[np.int(zone[arb_ind])], z_denmin[np.int(zone[arb_ind])]
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

t_end = time.time()
print 'time of code: \t%g minutes' % ((t_end-t_begin)/60.)

###################################################################################################

### CREATE PLOTS ###

###################################################################################################

fig1 = plt.figure(figsize=(12,10))
# fig2 = plt.figure(figsize=(12,10))
fig3 = plt.figure(figsize=(12,10))
fig4 = plt.figure(figsize=(12,10))

# Create scatter plot of all halos in slice and all halos representing void centers
ax1 = fig1.add_subplot(111)
ax1.set_xlim(200,500)
ax1.set_ylim(550,850)
ax1.set_xlabel(r'$\mathrm{x}$')
ax1.set_ylabel(r'$\mathrm{y}$')
ax1.scatter(x_slice, y_slice, color='red', marker='o')
ax1.scatter(x_slice_zone, y_slice_zone, color='green', marker='s')
# ax1.scatter(x[arb_ind], y[arb_ind], color='black', marker='d', s=4)
circ2=plt.Circle((x_denmin[np.int(zone[arb_ind])],y_denmin[np.int(zone[arb_ind])]),zone_rad[np.int(zone[arb_ind])], fill=None)
for i in range(0,len(slice_zone_idx)):
	circ=plt.Circle((x_slice_zone[i],y_slice_zone[i]),r_eff_zone_slice[i]*0.5, color='b', alpha=0.5)
	ax1.add_artist(circ)
# circ=plt.Circle((x[arb_ind],y[arb_ind]),r_eff[arb_ind], color='black', alpha=0.5)
ax1.scatter(x_denmin[np.int(zone[arb_ind])],y_denmin[np.int(zone[arb_ind])], color='black', marker='d', s=100)
ax1.scatter(x_vol[np.int(zone[arb_ind])],y_vol[np.int(zone[arb_ind])], color='black', marker='+', s=100)
ax1.add_artist(circ)
ax1.add_artist(circ2)
ax1.spines['top'].set_linewidth(2.3)
ax1.spines['left'].set_linewidth(2.3)
ax1.spines['right'].set_linewidth(2.3)
ax1.spines['bottom'].set_linewidth(2.3)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(27)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(27)   
# fig1.savefig('Zone_2131', format='pdf')

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
label = [r'$\mathrm{R_{zone}}<=35$',r'$35<\mathrm{R_{zone}}<=50$',r'$\mathrm{R_{zone}}>50$']
ax3 = fig3.add_subplot(111)
#ax3.set_xlim(0,33)
ax3.set_ylim(0,2)
ax3.set_xlabel(r'$\mathrm{r/R_{zone}}$')
ax3.set_ylabel(r'$\mathrm{\delta(r)+1}$')
ax3.plot(R_stk, (np.array(avg_nden_sm)/tot_numden), linewidth=3)
ax3.plot(R_stk, (np.array(avg_nden_md)/tot_numden), linewidth=3, linestyle='--')
ax3.plot(R_stk, (np.array(avg_nden_lg)/tot_numden), linewidth=3, linestyle='-.')
ax3.spines['top'].set_linewidth(2.3)
ax3.spines['left'].set_linewidth(2.3)
ax3.spines['right'].set_linewidth(2.3)
ax3.spines['bottom'].set_linewidth(2.3)
for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_fontsize(27)
for tick in ax3.yaxis.get_major_ticks():
    tick.label.set_fontsize(27)
ax3.legend(label, loc='best', fancybox = True, shadow = True)
# fig3.savefig('L1000_tot_zone_spherical_prof_multi_bin', format='pdf')


# Create density contrast vs R_v for stacked voids with vol avg center
label = [r'$\mathrm{R_{zone}}<=35$',r'$35<\mathrm{R_{zone}}<=50$',r'$\mathrm{R_{zone}}>50$']
ax4 = fig4.add_subplot(111)
#ax4.set_xlim(0,33)
ax4.set_ylim(0,2)
ax4.set_xlabel(r'$\mathrm{r/R_{zone}}$')
ax4.set_ylabel(r'$\mathrm{\delta(r)+1}$')
ax4.plot(R_stk, (np.array(avg_nden_sm_vol)/tot_numden), linewidth=3)
ax4.plot(R_stk, (np.array(avg_nden_md_vol)/tot_numden), linewidth=3, linestyle='--')
ax4.plot(R_stk, (np.array(avg_nden_lg_vol)/tot_numden), linewidth=3, linestyle='-.')
ax4.spines['top'].set_linewidth(2.3)
ax4.spines['left'].set_linewidth(2.3)
ax4.spines['right'].set_linewidth(2.3)
ax4.spines['bottom'].set_linewidth(2.3)
for tick in ax4.xaxis.get_major_ticks():
    tick.label.set_fontsize(27)
for tick in ax4.yaxis.get_major_ticks():
    tick.label.set_fontsize(27)
ax4.legend(label, loc='best', fancybox = True, shadow = True)
# fig4.savefig('L1000_tot_zone_spherical_prof_multi_bin_vol_center', format='pdf')


plt.show()