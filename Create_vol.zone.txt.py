import numpy as np
import sys
sys.path.append('/Users/Vasiliy/Desktop/PhD/Scripts/ZOBOV')
sys.path.append('/Users/Vasiliy/Desktop/PhD/Scripts')
from read_vol_zone_void import *
from Cosmology_class import *

Lbox = 252.5

LC = LC_cosmology()

Lbox = 252.5

# WMAP9+SN+BAO cosmology
LC.H0 = 68.98 #km s^{-1} Mpc^{-1} 
LC.omega_m = 0.2905
LC.omega_L = 0.7095
LC.z_box = 0.525
LC.Lbox_full = 505. #Mpc/h this is the size of the simulation box that the light cone was in, but only half of it is used so it's 252.5 Mpc/h
LC.theta = 10. #degrees on sky of each mock
LC.Lcube = 265.

# Get comoving distances of middle of box, lower end, and upper end as well as angular distances of lower and upper edge of cone
D_box, D_low, D_upper, DA_low, DA_upper = LC.comoving_dist()

### VOLUMES ###############################################################

# Volume of light cone
vol_cone = (1./3.)*(DA_low**2 + (DA_low*DA_upper) + DA_upper**2)*(LC.Lbox_full/2.)

numpart, numzones, zone = read_zone('KiDS_Mocks_0_525_noncube.zone') #zobov_poisson_test/PVBUF
numpart_vol, vol = read_vol('KiDS_Mocks_0_525_noncube.vol.txt', vol_cone)
x,y,z = read_AllHalo('KiDS_Mocks_gal_pos_0_525_v2.txt')
# x,y,z = read_inner('zobov_poisson_test/inner.pos')
VoidNum, FileVoidNum, CoreParticle, CoreDens, ZoneVol, ZoneNumPart, VoidNumZones, VoidVol, VoidNumPart, VoidDenCon, VoidProb = read_zobov('KiDS_Mocks_0_525_noncube.txt')

def coord_rot(x_arr,y_arr,z_arr, valx,valy,valz, Lbox):
	# Rotate box to put a particular coordinate in the middle
	# This requires subtracting the coordinates of the cell from all coordinates making it at (0,0,0)
	# Then adding half the box size to each coordinates
	# Then adding the box size and take the np.mod of box size of all coordinates to get the desired
	# point at the center
	coord_x_rot = np.mod(((x_arr-valx) + (Lbox+(Lbox/2.))),Lbox)
	coord_y_rot = np.mod(((y_arr-valy) + (Lbox+(Lbox/2.))),Lbox)
	coord_z_rot = np.mod(((z_arr-valz) + (Lbox+(Lbox/2.))),Lbox)
	return coord_x_rot, coord_y_rot, coord_z_rot

numx = [0 for _ in range(0,int(max(zone))+1)]
numy = [0 for _ in range(0,int(max(zone))+1)]
numz = [0 for _ in range(0,int(max(zone))+1)]
x_vol = []
y_vol = []
z_vol = []
x_den = []
y_den = []
z_den = []
dencon = []


dem = [0 for _ in range(0,int(max(zone))+1)]

# Get xyz of density min
for j in range(int(min(FileVoidNum)),int(max(FileVoidNum))+1): #
	if j in FileVoidNum:
		index = np.where(FileVoidNum==j)[0][0]
		x_den.append(x[CoreParticle[index]])
		y_den.append(y[CoreParticle[index]])
		z_den.append(z[CoreParticle[index]])
		dencon.append(VoidDenCon[index])


# x_den = [x[CoreParticle[j]] for j in range(0,int(max(FileVoidNum))+1)]
# y_den = [y[CoreParticle[j]] for j in range(0,int(max(FileVoidNum))+1)]
# z_den = [z[CoreParticle[j]] for j in range(0,int(max(FileVoidNum))+1)]

# Get location and vol of points that do not have a volume of 0
vol_nonzero = []
ID_vol_nonzero = []
zone_nonzero = []
g = 0
for i,t in enumerate(vol):
	if t > 1E-25:
		g+=1
		vol_nonzero.append(t)
		zone_nonzero.append(int(zone[i]))
		ID_vol_nonzero.append(i)

print 'num of particles with nonzero vol', g

# Get xyz of vol averaged
# Volume weighted average is obtained by this formula: sum(x_cell*vol_cell)/sum(vol_cell) for each zone


no_dup_zone_nonzero = rm_duplicate(zone_nonzero)
for i in no_dup_zone_nonzero: #range(0,int(max(zone))+1): 
	# Get indices of where 'zone' is 'i' where 'i' is from 0 to max(zone)
	idx = np.where(zone==i)[0]


	#rotate coordinates around the density min xyz of each zone
	# x_rot, y_rot, z_rot = coord_rot(x,y,z, x_den[i],y_den[i],z_den[i], Lbox)

	#get volume average of rotated coordinates for periodic box
	# numx[i] = sum([x_rot[val]*vol[val] for val in idx])
	# numy[i] = sum([y_rot[val]*vol[val] for val in idx])
	# numz[i] = sum([z_rot[val]*vol[val] for val in idx])

	# if any(idx) in ID_vol_nonzero:
	# 	print 'if index works', i
	#get volume averaged coordinates for non periodic box
	numx[i] = (sum([x[val]*vol[val] for val in idx]))
	numy[i] = (sum([y[val]*vol[val] for val in idx]))
	numz[i] = (sum([z[val]*vol[val] for val in idx]))
	

	dem[i] = (sum([vol[val] for val in idx]))
	

	# x_vol_temp = numx[i]/dem[i]
	# y_vol_temp = numy[i]/dem[i]
	# z_vol_temp = numz[i]/dem[i]
	# if dem[i] !=0.:
	x_vol.append(numx[i]/dem[i])
	y_vol.append(numy[i]/dem[i])
	z_vol.append(numz[i]/dem[i])


	# x_vol.append(np.mod(((x_vol_temp+x_den[i]) - (Lbox-(Lbox/2.))),Lbox))
	# y_vol.append(np.mod(((y_vol_temp+y_den[i]) - (Lbox-(Lbox/2.))),Lbox))
	# z_vol.append(np.mod(((z_vol_temp+z_den[i]) - (Lbox-(Lbox/2.))),Lbox))


r_zone = [(v*(3./(4*pi)))**(1./3.) for v in dem if v != 0]

# f = open('KiDS_Mocks_0_525_noncube.vol.zone.txt', 'w')
# f.writelines("1: ID,  2-4: xyz (volume-weighted centres), 5-7: xyz (centres of density minima), 8: void radius [Mpc/h], 9: rho_min / rho_bar -1\n")
# for i in range(0,len(x_vol)):
# 	f.write("{} {} {} {} {} {} {} {} {}\n".format(ID_vol_nonzero[i],x_vol[i],y_vol[i],z_vol[i],x_den[i],y_den[i],z_den[i],r_zone[i],dencon[i]))
# f.close()





