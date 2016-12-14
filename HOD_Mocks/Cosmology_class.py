


#- IMPORTS -#

from scipy.integrate import quad
from numpy import radians, tan, sqrt
import scipy.stats as stat 
import numpy as np


#- GLOBALS -#

c = 2.99792E5 # km/s

#- LC COSMOLOGY CLASS  -#

class LC_cosmology():

	def __init__(self):
		## DEFAULT COSMOLOGY --> FLAT LCDM CFHTLenS+WMAP9 COSMOLOGY ##
		self.z_box = 0.0
		self.omega_m = 0.268
		self.omega_L = 0.732
		self.omega_k = 0.0
		self.omega_b = 0.0452
		self.H0 = 70.9 #km s^{-1} Mpc^{-1}
		self.theta = 0.0
		self.Lbox_full = 0.0
		self.w0 = -1
		self.sigma_8 = 0.812
		self.n_s = 0.976


	def get_h(self):
		return self.H0 * 0.01

	def get_D_H(self):
		global c
		return c / self.H0

	def comoving_dist_integrand(self, z):
	    return (1./sqrt(self.omega_m*(1+z)**3 + self.omega_L))

	def comoving_dist(self):

		### FINDING COMOVING DISTANCE TO LOWER, MIDDLE, AND UPPER LIGHTCONE IN Z 
		### AND FINDING UPPER AND LOWER ANGULAR DIAMETER DISTANCE FOR X AND Y

		# Find comoving distance to center of box and then to the z_max and z_min in order to find x_min(y_min)
		# and x_max(y_max) at z_min and z_max

		ans, err = quad(self.comoving_dist_integrand, 0, self.z_box)
		D_box = self.get_h() * self.get_D_H() * ans #This is the comoving distance to the 'middle' of the light cone in units of Mpc/h

		# Distance to 'lower' and 'upper' section of the light cone
		# Lbox is divided by 4 because the lightcone already is taking up Lbox/2
		D_low = D_box - (self.Lbox_full * 0.25) 
		D_upper = D_box + (self.Lbox_full * 0.25)

		# Angular distance of 'lower' and 'upper' section of lightcone
		DA_low = 2.0 * D_low * tan(radians(self.theta * 0.5))
		DA_upper = 2.0 * D_upper * tan(radians(self.theta * 0.5))

		return D_box, D_low, D_upper, DA_low, DA_upper




def dist(x_orig, y_orig, z_orig, x_pt, y_pt, z_pt):
	d = sqrt( (x_orig - x_pt)**2 + (y_orig - y_pt)**2 + (z_orig - z_pt)**2)
	return d

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

def rm_duplicate(seq):
	# Removes duplicates in array while preserving order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def extended_gcd(aa, bb):
    lastremainder, remainder = abs(aa), abs(bb)
    x, lastx, y, lasty = 0, 1, 1, 0
    while remainder:
        lastremainder, (quotient, remainder) = remainder, divmod(lastremainder, remainder)
        x, lastx = lastx - quotient*x, x
        y, lasty = lasty - quotient*y, y
    return lastremainder, lastx * (-1 if aa < 0 else 1), lasty * (-1 if bb < 0 else 1)
 
def modinv(a, m):
	## Get inverse mod ##
	g, x, y = extended_gcd(a, m)
	if g != 1:
		raise ValueError
	return (x % m)

def covariance(val_arr, val_arr2):
	# Calculate covariance matrix between two arrays

	# Create array that masks out NaN in data and get it's data values
	mask_arr = np.ma.array(val_arr, mask=np.isnan(val_arr))
	mask_arr2 = np.ma.array(val_arr2, mask=np.isnan(val_arr2))

	# Create array of counts for non masked ie non NaN values along the columns
	non_NaN_cnt = np.ma.count(mask_arr,axis=0)
	non_NaN_cnt2 = np.ma.count(mask_arr2,axis=0)

	# Get the mean value of the masked array.  This is equivalent of adding all the kappa values per void per radial bin and 
	# dividing by the number of non NaN values, for example
	avg_arr = np.mean(mask_arr,axis=0).data
	avg_arr2 = np.mean(mask_arr2,axis=0).data

	# CALCULATE THE COVARIANCE MATRIX FROM Cai et al. 2016 (RSD around voids) EQ. 26

	cov = np.empty([len(avg_arr), len(avg_arr2)])
	cov_norm = np.empty([len(avg_arr), len(avg_arr2)])


	for i in xrange(len(avg_arr)):
		for j in xrange(len(avg_arr2)):
			cov[j,i] = (1./(non_NaN_cnt[i]*non_NaN_cnt2[j]))*np.nansum((np.array(val_arr)[:,i]-avg_arr[i])*(np.array(val_arr2)[:,j]-avg_arr2[j]))

	# Normalize covariance across the diagonal
	for i in xrange(len(avg_arr)):
		for j in xrange(len(avg_arr2)):
			cov_norm[j,i] = cov[j,i]/np.sqrt(cov[j,j]*cov[i,i])		

	return cov, cov_norm

def chi_sq(cov, avg_arr):
	# This function returns the chi squared of a covariance matrix for a null test
	# The equation is from Cai et al. 2016 (RSD around voids) Eq. 27
	# Note that in this case \delta_i is the transpose of the avg array and \delta_j is the average array

	try:
		inverse_cov = np.linalg.inv(cov)
	except numpy.linalg.LinAlgError:
		print 'Covariance not invertible'

	chi_sq = np.matrix(avg_arr)*np.matrix(inverse_cov)*np.matrix(avg_arr).T

	# Get cumulative chi sq for each consecutive DOF ie radial bin
	cum_chi_sq = []
	for i in xrange(len(avg_arr)):
		chi_temp = np.matrix(avg_arr[0:i+1])*np.matrix(inverse_cov[0:i+1,0:i+1])*np.matrix(avg_arr[0:i+1]).T
		cum_chi_sq.append(chi_temp.A[0][0])

	return chi_sq.A[0][0], np.array(cum_chi_sq)


def chi_sq_to_sigma(chi_arr,dof):
	# Convert from chi^2 to sigma
	# Both chi_arr and dof need to be numpy arrays

	# Get confidence interval given a chi^2 and the dof
	conf_int = stat.chi2.cdf(chi_arr,dof)
	# Convert confidence interval to sigma value
	sigma = stat.norm.ppf(1.-(1.-conf_int)/2.)

	return sigma








