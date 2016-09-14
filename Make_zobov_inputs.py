import numpy as np
import sys
sys.path.append('/Users/Vasiliy/Desktop/PhD/Scripts/ZOBOV')
from read_vol_zone_void import *
from scipy.io import FortranFile
import random
from scipy.integrate import quad


numpart_0_525, x_gal_0_525,y_gal_0_525,z_gal_0_525, x_halo_0_525,y_halo_0_525,z_halo_0_525, m200c_0_525, r200c_0_525, Rs_0_525, c_0_525 = read_HOD('L505Mpc_HOD+0.525_v2.dat')
# x,y,z, = read_AllHalo('/Users/Vasiliy/Desktop/PhD/Scripts/ZOBOV/GR/AllHalo_1000Mpc_a1.0_M12.8.txt')
# numpart = len(x)

# WMAP9+SN+BAO cosmology
H0 = 68.98 #km s^{-1} Mpc^{-1} 
h = H0/100.
omega_m = 0.2905
omega_L = 0.7095
z_box = 0.525
c = 2.99792E5 #km/s
Lbox = 505. #Mpc/h this is the size of the simulation box that the light cone was in, but only half of it is used so it's 252.5 Mpc/h
theta = 10. #degrees on sky of each mock
D_H = c/H0 #Hubble distance in units of Mpc
Lcube = 265.

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

x_min = abs(min(x_gal_0_525))
y_min = abs(min(y_gal_0_525))
x_gal_0_525_shift = np.add(x_gal_0_525,x_min)
y_gal_0_525_shift = np.add(y_gal_0_525,y_min)


# Get comoving distances of middle of box, lower end, and upper end as well as angular distances of lower and upper edge of cone
D_box, D_low, D_upper, DA_low, DA_upper = comoving_dist(z_box,omega_m,omega_L,H0,h,c,theta,Lbox)

### VOLUMES ###############################################################

# Volume of light cone
vol_cone = (1./3.)*(DA_low**2 + (DA_low*DA_upper) + DA_upper**2)*(Lbox/2.)
numden_cone = numpart_0_525/vol_cone

# Volumes
diffx = (DA_upper-DA_low)/2. # This is the same for diffy!!

delta_cube_box = max(z_gal_0_525)-min(z_gal_0_525)-(Lbox/2.)

vol_prism_lg = (1./2.)*(Lbox/2.)*(max(x_gal_0_525)-(max(x_gal_0_525)-diffx))*Lcube
vol_prism_sm = (1./2.)*(Lbox/2.)*(max(y_gal_0_525)-(max(y_gal_0_525)-diffx))*((max(x_gal_0_525)-diffx)-(min(x_gal_0_525)+diffx))
vol_rect = Lcube*Lcube*(Lcube-(Lbox/2.))

# The number density is increased so that the buffer is much denser than the LC
numpart_prism_lg = 10*numden_cone*vol_prism_lg
numpart_prism_sm = 10*numden_cone*vol_prism_sm
numpart_rect = 10*numden_cone*vol_rect


################################################################################

### GENERATE RANDOM PARTICLES FOR EACH PRISM ##################################

# Right side of XZ plane
x_right_min = max(x_gal_0_525)-diffx
delta_z = Lbox/2.
delta_x_right = max(x_gal_0_525)-x_right_min
xrand_temp_right = np.random.uniform(x_right_min+1,max(x_gal_0_525)+1,10*int(numpart_prism_lg))
xrand_temp_right_new = np.random.uniform(x_right_min,max(x_gal_0_525)+1,10*int(numpart_prism_lg))
zrand_temp_xz = np.random.uniform(min(z_gal_0_525),(min(z_gal_0_525)+(Lbox/2.))+10,10*int(numpart_prism_lg))
# xrand_temp_right2 = [xrand_temp_right[i] for i in xrange(20*int(numpart_prism_lg)) if zrand_temp_xz[i] < (delta_z/delta_x_right)*xrand_temp_right[i] and zrand_temp_xz[i] > (delta_z/delta_x_right)*xrand_temp_right[i]-30.]
# zrand_temp_right2 = [zrand_temp_xz[i] for i in xrange(20*int(numpart_prism_lg)) if zrand_temp_xz[i] < (delta_z/delta_x_right)*xrand_temp_right[i] and zrand_temp_xz[i] > (delta_z/delta_x_right)*xrand_temp_right[i]-30.]
xrand_temp_right2 = [xrand_temp_right[i] for i in xrange(10*int(numpart_prism_lg)) if zrand_temp_xz[i] < (delta_z/delta_x_right)*xrand_temp_right[i]]
zrand_temp_right2 = [zrand_temp_xz[i] for i in xrange(10*int(numpart_prism_lg)) if zrand_temp_xz[i] < (delta_z/delta_x_right)*xrand_temp_right[i]]
xrand_right = xrand_temp_right2[0:int(numpart_prism_lg)]
# xrand_right = [s+2. for s in xrand_right1]
zrand_right_xz = zrand_temp_right2[0:int(numpart_prism_lg)]

# Left side of XZ plane
x_left_min = min(x_gal_0_525) + diffx
delta_x_left = min(x_gal_0_525) - x_left_min
xrand_temp_left = np.random.uniform(min(x_gal_0_525)-1,x_left_min-1,10*int(numpart_prism_lg))
# xrand_temp_left2 = [xrand_temp_left[i] for i in xrange(20*int(numpart_prism_lg)) if zrand_temp_xz[i] < (delta_z/delta_x_left)*xrand_temp_left[i] and zrand_temp_xz[i] > (delta_z/delta_x_left)*xrand_temp_left[i]-30.]
# zrand_temp_left2 = [zrand_temp_xz[i] for i in xrange(20*int(numpart_prism_lg)) if zrand_temp_xz[i] < (delta_z/delta_x_left)*xrand_temp_left[i] and zrand_temp_xz[i] > (delta_z/delta_x_left)*xrand_temp_left[i]-30.]
xrand_temp_left2 = [xrand_temp_left[i] for i in xrange(10*int(numpart_prism_lg)) if zrand_temp_xz[i] < (delta_z/delta_x_left)*xrand_temp_left[i]]
zrand_temp_left2 = [zrand_temp_xz[i] for i in xrange(10*int(numpart_prism_lg)) if zrand_temp_xz[i] < (delta_z/delta_x_left)*xrand_temp_left[i]]
xrand_left = xrand_temp_left2[0:int(numpart_prism_lg)]
# xrand_left= [s-2. for s in xrand_left1]
zrand_left_xz = zrand_temp_left2[0:int(numpart_prism_lg)]


yrand_right_xz = np.random.uniform(min(y_gal_0_525),max(y_gal_0_525),int(numpart_prism_lg))
yrand_left_xz = np.random.uniform(min(y_gal_0_525),max(y_gal_0_525),int(numpart_prism_lg))


print 'x'
print len(xrand_right)
print len(xrand_left)
print numpart_prism_lg
print

# Right side of YZ plane
y_right_min = max(y_gal_0_525) - diffx
delta_y_right = max(y_gal_0_525) - y_right_min
yrand_temp_right = np.random.uniform(y_right_min+1,max(y_gal_0_525)+1,10*int(numpart_prism_sm))
zrand_temp_yz = np.random.uniform(min(z_gal_0_525),(min(z_gal_0_525)+(Lbox/2.)),10*int(numpart_prism_sm))
# yrand_temp_right3 = [yrand_temp_right[i] for i in xrange(20*int(numpart_prism_sm)) if zrand_temp_yz[i] < (delta_z/delta_y_right)*yrand_temp_right[i] and zrand_temp_yz[i] > (delta_z/delta_y_right)*yrand_temp_right[i]-30.]
# zrand_temp_right3 = [zrand_temp_yz[i] for i in xrange(20*int(numpart_prism_sm)) if zrand_temp_yz[i] < (delta_z/delta_y_right)*yrand_temp_right[i] and zrand_temp_yz[i] > (delta_z/delta_y_right)*yrand_temp_right[i]-30.]
yrand_temp_right3 = [yrand_temp_right[i] for i in xrange(10*int(numpart_prism_sm)) if zrand_temp_yz[i] < (delta_z/delta_y_right)*yrand_temp_right[i]]
zrand_temp_right3 = [zrand_temp_yz[i] for i in xrange(10*int(numpart_prism_sm)) if zrand_temp_yz[i] < (delta_z/delta_y_right)*yrand_temp_right[i]]
yrand_right = yrand_temp_right3[0:int(numpart_prism_sm)]
# yrand_right = [s+2. for s in yrand_right1]
zrand_right_yz = zrand_temp_right3[0:int(numpart_prism_sm)]


# Left side of YZ plane
y_left_min = min(y_gal_0_525) + diffx
delta_y_left = min(y_gal_0_525) - y_left_min
yrand_temp_left = np.random.uniform(min(y_gal_0_525)-1,y_left_min-1,10*int(numpart_prism_sm))
# yrand_temp_left3 = [yrand_temp_left[i] for i in xrange(20*int(numpart_prism_sm)) if zrand_temp_yz[i] < (delta_z/delta_y_left)*yrand_temp_left[i] and zrand_temp_yz[i] > (delta_z/delta_y_left)*yrand_temp_left[i]-30.]
# zrand_temp_left3 = [zrand_temp_yz[i] for i in xrange(20*int(numpart_prism_sm)) if zrand_temp_yz[i] < (delta_z/delta_y_left)*yrand_temp_left[i] and zrand_temp_yz[i] > (delta_z/delta_y_left)*yrand_temp_left[i]-30.]
rand_temp_left3 = [yrand_temp_left[i] for i in xrange(10*int(numpart_prism_sm)) if zrand_temp_yz[i] < (delta_z/delta_y_left)*yrand_temp_left[i]]
zrand_temp_left3 = [zrand_temp_yz[i] for i in xrange(10*int(numpart_prism_sm)) if zrand_temp_yz[i] < (delta_z/delta_y_left)*yrand_temp_left[i]]
yrand_left = yrand_temp_left3[0:int(numpart_prism_sm)]
# yrand_left = [s-2. for s in yrand_left1]
zrand_left_yz = zrand_temp_left3[0:int(numpart_prism_sm)]


xrand_right_yz = np.random.uniform((min(x_gal_0_525)+diffx),(max(x_gal_0_525)-diffx),int(numpart_prism_sm))
xrand_left_yz = np.random.uniform((min(x_gal_0_525)+diffx),(max(x_gal_0_525)-diffx),int(numpart_prism_sm))

print'y'
print len(yrand_right)
print len(yrand_left)
print numpart_prism_sm

# Random for upper part of Z to make it from Lbox/2 --> Lcube
xrand_rect_upper = np.random.uniform(min(x_gal_0_525)-1,max(x_gal_0_525)+1,int(numpart_rect))
yrand_rect_upper = np.random.uniform(min(y_gal_0_525)-1,max(y_gal_0_525)+1,int(numpart_rect))
zrand_rect_upper = np.random.uniform(max(z_gal_0_525),max(z_gal_0_525),int(numpart_rect))


# Random for lower part of Z to make it from Lbox/2 --> Lcube
xrand_rect_lower = np.random.uniform(min(xrand_left)+diffx,max(xrand_right)-diffx,int(numpart_rect))
yrand_rect_lower = np.random.uniform(min(yrand_left)+diffx,max(yrand_right)-diffx,int(numpart_rect))
zrand_rect_lower = np.random.uniform(min(z_gal_0_525),min(z_gal_0_525),int(numpart_rect))


### APPEND RANDOM PARTICLES TO ALL GALAXIES TO TURN CONE INTO CUBE
### APPENDING WILL BE DONE IN THE FOLLOWING ORDER: RIGHT PRISM_XZ, LEFT PRISM_XZ, RIGHT PRISM_YZ, LEFT PRISM_YZ, TOP OF Z, BOTTOM OF Z
### THIS ORDER WILL BE KEPT PER XYZ COORDINATE	

# x_gal_0_525.append(xrand_right,xrand_left,xrand_yz,xrand_rect)
# y_gal_0_525.append(yrand_xz,yrand_right,yrand_left,yrand_rect)
# z_gal_0_525.append(zrand_right_xz,zrand_left_xz,zrand_right_yz,zrand_left_yz,zrand_rect)

# All particles --> light cone and buffer
x_full = np.hstack((x_gal_0_525,xrand_right,xrand_left,xrand_right_yz,xrand_left_yz,xrand_rect_upper,xrand_rect_lower))
y_full = np.hstack((y_gal_0_525,yrand_right_xz,yrand_left_xz,yrand_right,yrand_left,yrand_rect_upper,yrand_rect_lower))
z_full = np.hstack((z_gal_0_525,zrand_right_xz,zrand_left_xz,zrand_right_yz,zrand_left_yz,zrand_rect_upper,zrand_rect_lower))

# Buffer particles only
x_buf = np.hstack((xrand_right,xrand_left,xrand_right_yz,xrand_left_yz,xrand_rect_upper,xrand_rect_lower))
y_buf = np.hstack((yrand_right_xz,yrand_left_xz,yrand_right,yrand_left,yrand_rect_upper,yrand_rect_lower))
z_buf = np.hstack((zrand_right_xz,zrand_left_xz,zrand_right_yz,zrand_left_yz,zrand_rect_upper,zrand_rect_lower))



data_0_525 = np.array([x_gal_0_525,y_gal_0_525,z_gal_0_525])
data_0_525 = data_0_525.T
# np.insert(data_0_525,0,[numpart_0_525])
# np.savetxt('KiDS_Mocks_gal_pos_0_525_shift', data_0_525, fmt='%f')

# All particles
allpart_num = len(x_gal_0_525)+len(x_buf)
f = open('KiDS_Mocks_0_525_noncube_ultradense.pos','w')
f.write("{} {}\n".format(allpart_num,len(x_gal_0_525)))
for i in range(0,len(x_full)):
	f.write("{} {} {}\n".format(x_full[i],y_full[i],z_full[i]))
f.close


# LC galaxies only
f = open('KiDS_Mocks_0_525_LC_noncube_ultradense.txt','w')
for i in range(0,len(x_gal_0_525)):
	f.write("{} {} {}\n".format(x_gal_0_525[i],y_gal_0_525[i],z_gal_0_525[i]))
f.close

# Buffer Particles
f = open('KiDS_Mocks_0_525_buffer_noncube_ultradense.txt','w')
for i in range(0,len(x_buf)):
	f.write("{} {} {}\n".format(x_buf[i],y_buf[i],z_buf[i]))
f.close

### Write unformatted fortran file with the num of particles and their xyz
numpart_0_525 = [numpart_0_525]
# all_data = np.concatenate((numpart_0_525,x_gal_0_525,y_gal_0_525,z_gal_0_525), axis=0)

