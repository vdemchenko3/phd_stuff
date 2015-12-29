import numpy as np
import decimal as d
from numpy import *
import matplotlib
matplotlib.use('Agg')
matplotlib.use('GTKAgg')
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import *
from decimal import * 
import Nbody40LCDM_vitest as nb40 

font = {'family' : 'serif', 'serif' : ['Times'], 'size' : '35'}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)

with localcontext() as c:
	c.prec = 112 #sets precision to 3 times the original
	
	#create function for diff eq
	def derivjL(yjL,t):
	    #break it up into dR/dt = v and dv/dt = acc eq
	    Rprime = yjL[1]
	    Vprime = -((G1*(MjL[j]))/yjL[0]**2) + omega_L*yjL[0]*Ho2**2
	    yjprime = np.array([Rprime, Vprime])
	    return yjprime

	def sheth(R, p, p_b):
	    delV = []
	    delM = []
	    rho_n = []
	    for n in range(0, len(R)):
		if n == 0:
		    delVi = (4./3)*pi*(R[n]**3)
		    delMi = (p[n]*(4./3)*pi*R[n]**3)
		    rho_ni = delMi/delVi
		else:
		    delVi = (4./3)*pi*(R[n]**3 - R[n-1]**3)
		    delMi = (p[n]*(4./3)*pi*R[n]**3) - (p[n-1]*(4./3)*pi*R[n-1]**3)
		    rho_ni = delMi/delVi
		
		delV.append(delVi)
		delM.append(delMi)
		rho_n.append(rho_ni)
	    
	    return (rho_n/p_b) -1.
	#set initial conditions
	tnewL = np.linspace(600., 13800., 1000)
	z = np.linspace(9., 0, 1000) #redshifts from zi to z0

	#initial shell radii (Mpc/h) normalized to average radius (20 Mpc/h in this case) and vpeculiar velocities (km s^-1)
	ri, vi = np.loadtxt('VoidProfiles/ProfilesV.35.0-45.0_sig40.2OL0.0_All.dat',dtype=float, unpack=True)
	#ri = np.linspace(1.08E-3, 2.3E-3, 20)
	#ri1 = 8.3E-4
	#initial density contrast for each shell
	di = np.loadtxt('VoidProfiles/Profiles.35.0-45.0_sig40.2OL0.0_All.dat',dtype=float, usecols=(1,))
	#di1 = -4.63E-3

	r1, d1 = np.loadtxt('VoidProfiles/a1.0/Profiles.35.0-45.0_sig40.2OL0.0_All.dat', dtype=float, unpack=True)
	r5, d5 = np.loadtxt('VoidProfiles/a0.5/Profiles.35.0-45.0_sig40.2OL0.0_All.dat', dtype=float, unpack=True)
	v1 = np.loadtxt('VoidProfiles/a1.0/ProfilesV.35.0-45.0_sig40.2OL0.0_All.dat', dtype=float, usecols=(1,))
	v5 = np.loadtxt('VoidProfiles/a0.5/ProfilesV.35.0-45.0_sig40.2OL0.0_All.dat', dtype=float, usecols=(1,))

	RfL = np.zeros(len(ri))
	R0L = np.zeros(len(ri))
	RmidL = np.zeros(len(ri))
	VfL = np.zeros(len(ri))
	V0L = np.zeros(len(ri))
	VmidL = np.zeros(len(ri))
	vpecfL = np.zeros(len(ri))
	vpec0L = np.zeros(len(ri))
	vpecmidL = np.zeros(len(ri))
	rho_i = np.zeros(1000)
	rho9L = np.zeros(len(ri))
	rho_midL = np.zeros(len(ri))
	rho0L = np.zeros(len(ri))
	dijL = np.zeros(1000)
	v0jL2 = np.zeros(len(ri))
	rho_iL = np.zeros(len(ri))
	MjL = np.zeros(len(ri))
	diavg = np.zeros(len(ri))
	dij = np.zeros(len(ri))
	vpecilin = np.zeros(len(ri))
	vpeci = np.zeros(len(ri))
	
	#cosmology parameters
	at =1./(1+z)
	#print at
	omega_L = 0.76 #LCDM vacuum energy density parameter
	omega_m = 0.24 #LCDM matter density parameter
	Ho1 = 1.0247E-4 #Mpc Myr^-1 Mpc^-1 normalized to h
	Ho = 100. #km s^-1 Mpc^-1
	Ho2 = 7.17E-5 #Mpc Myr^-1 Mpc^-1
	h = .7
	G1 = 4.524319E-21 #Msun^-1 Mpc^3 Myr^-2
	G = 4.302E-9 #Mpc Msun^-1 (km/s)^2
	theta = arctan((omega_m/omega_L)**(0.5)*at**(-3./2))
	tL = (2./(3*Ho2*omega_L**(0.5)))*log((1+cos(theta))/sin(theta)) #time in model for given redshifts
	func = interp1d(tL,theta, bounds_error=False, kind='quadratic')
	thetanew = func(tnewL)
	atL = (tan(thetanew)*(omega_L/omega_m)**(0.5))**(-2./3)
	E2 = omega_m*atL**(-3.) + omega_L
	omega_mz = (omega_m*atL**(-3.))/E2 #omega matter as a function of a
	omega_Lz = omega_L/(omega_L + omega_m*atL**(-3.))
	H2 = (omega_L + omega_m*atL**(-3.))*Ho2**2
	f = omega_mz**(0.55)

	#solving for linear growth factor
	def integrand(a):
	    return ((omega_m*a**(-1)+omega_L*a**2)**(-3./2))

	ans, err = quad(integrand, 0, atL[0])
	ans2, err2 = quad(integrand, 0, atL[-1])

	D = ans2/(((omega_m*atL[0]**(-3.) + omega_L)**(0.5))*ans) #linear theory growth factor

	rho_c = (3*H2)/(8*pi*G1) #Msun Mpc^-3
	rho_bL = omega_mz*rho_c #Msun Mpc^-3
	velcon = 3.085677581E19/3.15576E13 #Mpc Myr^-1 to km s^-1 conversion
	vi = vi/velcon #Mpc Myr^-1
	rix = ri
	ri = (ri*40*atL[0])/h #physical units non averaged
	r1 = r1*40/h
	r5 = r5*40/h
	  
	for j in range(0, len(ri)):
	    rho_iL[j] = (di[j]+1)*rho_bL[0]
	    #set up initial density equations
	    if j == 0:
		MjL[j] = (4./3)*pi*rho_iL[j]*ri[j]**3
	    else:
		MjL[j] = MjL[j-1] + (4./3)*pi*rho_iL[j]*(ri[j]**3 - ri[j-1]**3)

	    if j == 0:
		dij[j] = (di[j])*((4./3)*pi*ri[j]**3)
	    else:
		dij[j] = dij[j-1] + di[j]*((4./3)*pi*(ri[j]**3-ri[j-1]**3))
	    diavg[j] = dij[j]/((4./3)*pi*ri[j]**3)

	    vpecilin[j] = -(diavg[j]*f[0]*sqrt(H2[0])*ri[j])/3
	    #print 'lin vel', vpecilin[j]*velcon
	    
	    dijL.fill(diavg[j])
	    v0jL = (((omega_Lz[0]*(ri[j]**2)*(H2[0]))/(1+dijL))+(2*G1*(MjL[j]))/(ri[j]*(1+dijL)))**(0.5)
	    v0jL2[j] = v0jL[0]
	    vpeci[j] = v0jL2[j]-sqrt(H2[0])*ri[j]
	    #print 'econs vel', vpeci[j]*velcon
	   
	    #print v0jL[j]*velcon - (sqrt(H2[0])*ri[j])*velcon 
	    
	    #setting intial conditions for diff eq and calc diff eq
	    yjL = np.zeros([1000,2])
	    yjL[0,0] = ri[j]
	    yjL[0,1] = v0jL[j]#vpecilin[j] + sqrt(H2[0])*ri[j]#vi[j]+ sqrt(H2[0])*ri[j]# sqrt(H2[0])*ri[j]#
	    yinitjL = np.array([ri[j], v0jL[j]])#vpecilin[j] + sqrt(H2[0])*ri[j]])#vi[j]+ sqrt(H2[0])*ri[j]])#sqrt(H2[0])*ri[j]])#
	    accsoljL = odeint(derivjL, yinitjL, tnewL)
	    
	    #Numerical radius of void (Mpc)
	    RvjL = accsoljL[:,0]
	    RfL[j] = accsoljL[999,0]
	    R0L[j] = accsoljL[0,0]
	    RmidL[j] = accsoljL[435,0]
	    
	    #Numerical velocity of void at each radius (Mpc Myr^-1)
	    VjL = accsoljL[:,1]
	    VfL[j] = accsoljL[999,1]
	    V0L[j] = accsoljL[0,1]
	    VmidL[j] = accsoljL[435,1]

	    vpecfL[j] = VfL[j]-sqrt(H2[999])*RfL[j]
	    vpec0L[j] = V0L[j]-sqrt(H2[0])*R0L[j]
	    vpecmidL[j] = VmidL[j]-sqrt(H2[435])*RmidL[j]
		
	    #Numerically evolving denisty using numerical approximation (Msun Mpc^-3)
	    rho_tnumjL = (3.*(MjL[j]))/(4.*pi*accsoljL[:,0]**3)
	    rho9L[j] = rho_tnumjL[999]
	    rho0L[j] = rho_tnumjL[0]
	    rho_midL[j] = rho_tnumjL[435]
	    dtL = (rho_tnumjL/rho_bL)-1.
		        
	#create figures
	fig1 = plt.figure(figsize=(12,10))
	#fig2 = plt.figure(figsize=(12,10))
	#fig3 = plt.figure(figsize=(12,10))
	#fig4 = plt.figure(figsize=(12,10))
	#fig5 = plt.figure(figsize=(12,10))
	labelsheth = [r'$a(t)=1$', r'$a(t) = 0.5$', r'$a(t)=0.1$', r'$a(t)=1$', r'$a(t) = 0.5$']
	label = ['linear theory', 'energy con', 'intial condition']

	#run density contrast function
	dt_dis1L = sheth(RfL, rho9L,  rho_bL[999])
	dt_dis2L = sheth(RmidL, rho_midL, rho_bL[435])
	dt_dis3L = sheth(R0L,  rho0L, rho_bL[0])

	#percentage difference between simulation and LCDM
	v1diff = abs((vpecfL*velcon-v1)/((vpecfL*velcon+v1)/2))*100
	v2diff = abs((vpecmidL*velcon-v5)/((vpecmidL*velcon+v5)/2))*100

	#d1diff = abs((dt_dis1L-d1)/((dt_dis1L+d1)/2))*100
	#d2diff = abs((dt_dis2L-d5)/((dt_dis2L+d5)/2))*100
	d1diff = (dt_dis1L-d1)
	d2diff = (dt_dis2L-d5)

	#print average(d2diff)
	#print average(d1diff)

	#print average(v3diff[1:len(v3diff)])
	#print average(v2diff[1:len(v2diff)])
	#print average(v1diff[1:len(v1diff)])

	#plot numerical sheth 
	ax = fig1.add_subplot(111)
	#ax.set_title('LCDM Sheth plot')
	ax.set_xlim(0,160)
	#ax5.set_ylim(-1, 1.5)
	ax.set_xlabel(r'$\mathrm{R_v}$ $\mathrm{(Mpc/h)}$')
	ax.set_ylabel(r'$\mathrm{\delta}$')
	ax.plot(RfL/atL[999],dt_dis1L, linewidth = 6, linestyle=':', color='blue')
	ax.plot(RmidL/atL[435],dt_dis2L, linewidth = 6, linestyle=':', color='green')
	ax.plot(nb40.RfL/nb40.atL[999],nb40.dt_dis1L, nb40.RmidL/nb40.atL[435],nb40.dt_dis2L, linewidth = 4)
	ax.plot(r1,d1, linestyle='--', linewidth = 6, color='blue')
	ax.plot(r5,d5, linestyle='--', linewidth = 6, color='green')
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(27)
	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(27)
	ax.axhline(0,linestyle=':', color='black',linewidth=1.5)
	ax.spines['top'].set_linewidth(2.3)
	ax.spines['left'].set_linewidth(2.3)
	ax.spines['right'].set_linewidth(2.3)
	ax.spines['bottom'].set_linewidth(2.3)
	#box1 = ax.get_position()
	#ax.set_position([box1.x0, box.y0-0.04, box.width, box.height])
	#ax.legend(labelsheth, loc='best', fancybox = True, shadow = True)
	fig1.savefig('Nbody_LCDM_Sheth_Plot40_econs_vi_comparison', format='pdf')

	'''ax2 = fig2.add_subplot(111)
	#ax2.set_title('LCDM Velocity Profile')
	#ax2.set_xlim(8.1E-4,9.5E-4)
	#ax2.set_ylim(-1, 1.5)
	ax2.set_xlabel(r'$R_v$ $(Mpc)$', fontsize=26)
	ax2.set_ylabel(r'$v_{pec}$ $(km s^{-1})$', fontsize=31)
	ax2.plot(RfL/atL[999],vpecfL*velcon, RmidL/atL[435],vpecmidL*velcon, R0L/atL[0],vpec0L*velcon, linewidth = 4)
	ax2.plot(r1,v1, linestyle='--', linewidth = 6, color='blue')
	ax2.plot(r5,v5, linestyle='--', linewidth = 6, color='green')
	for tick in ax2.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax2.yaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	ax2.spines['top'].set_linewidth(2.3)
	ax2.spines['left'].set_linewidth(2.3)
	ax2.spines['right'].set_linewidth(2.3)
	ax2.spines['bottom'].set_linewidth(2.3)
	#box2 = ax2.get_position()
	#ax2.set_position([box2.x0, box2.y0-0.04, box2.width, box2.height])
	#ax2.legend(labelsheth, loc='best', fancybox = True, shadow = True)
	#fig2.savefig('Nbody_Velocity_Profile40', format='pdf')'''

	#ax3 = fig3.add_subplot(111)
	#ax3.plot(ri, di, ri, diavg, linewidth = 3)
	#fig3.savefig('Nbody LCDM Initial dencon Profile40', format='pdf')
	'''ax4 = fig4.add_subplot(111)
	ax4.set_xlabel('R Mpc')
	ax4.set_ylabel('Initial V km/s')
	ax4.plot(ri, vpecilin*velcon, ri,vpeci*velcon, ri, vi*velcon, linewidth = 3)
	ax4.legend(label, loc='best', fancybox = True, shadow = True)
	ax4.spines['top'].set_linewidth(2.3)
	ax4.spines['left'].set_linewidth(2.3)
	ax4.spines['right'].set_linewidth(2.3)
	ax4.spines['bottom'].set_linewidth(2.3)
	fig4.savefig('Nbody Initial Velocity Profile40', format='pdf')

	ax5 = fig5.add_subplot(111)
	ax5.set_xlabel('R Mpc')
	ax5.set_ylabel('Initial V km/s')
	ax5.plot(ri, (vpecilin + sqrt(H2[0])*ri)*velcon, ri,v0jL2*velcon, ri,(vi+ sqrt(H2[0])*ri)*velcon, linewidth = 3)
	ax5.legend(label, loc='best', fancybox = True, shadow = True)
	ax5.spines['top'].set_linewidth(2.3)
	ax5.spines['left'].set_linewidth(2.3)
	ax5.spines['right'].set_linewidth(2.3)
	ax5.spines['bottom'].set_linewidth(2.3)
	fig5.savefig('Nbody Initial Tot Velocity Profile40', format='pdf')
	#ax5 = fig5.add_subplot(111)
	#ax5.plot(ri,((omega_L*(ri**2)*H2[0])/(1+diavg))**(0.5), ri, +((2*G1*(MjL))/(ri*(1+diavg)))**(0.5), linewidth = 3)'''

	plt.show()
