from math import isnan

import astropy.constants as c

import numpy as np

import os

from scipy.interpolate import CubicSpline

from mosfit.modules.engines.engine import Engine

CLASS_NAME = 'Fallback'


class Fallback(Engine):
	"""A tde engine.
    """

	def __init__(self,**kwargs):
		# call super version of init

		super(Fallback, self).__init__(**kwargs) 
		
		# load dmde info

		#------ DIRECTORY PARAMETERS -> need to change to variable names used in mosfit, then won't have to set any variables here

		# It is assumed that there are different files for each beta (such as 2.500.dat for beta = 2.5)
		# The first row is energy, the second is dmde. This could be changed so that
		# each beta has a different subdirectory

		# for now just use astrocrash dmdes (converted from astrocrash dmdts)

		dmdedir = os.path.dirname(__file__)[:-15]+'models/tde/data/' #'../../models/tde/data/'
	   
		#dmdedir = '/Users/brennamockler/Dropbox (Personal)/Research/smooth+rebin/mpoly_5-3_4-3_1e6/gkernel35/'

		#--------- GET SIMULATION BETAS -----------------

		# hardcode in the simulation betas for gamma = 4-3 for now
		self._sim_beta = [0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 1.000, 1.100,  1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.850, 1.900, 2.000, 2.500, 3.000, 3.500, 4.000]

		
		#-- CREATE INTERPOLATION FUNCTIONS; FIND SLOPES & YINTERs -------

		# these three lists, in addition to 'sim_beta', are the lists that will hold dmde info to be accessed after init is run
		self._beta_slope = []
		self._beta_yinter = []
		self._energy = []

	   # need to pad with extra zeros for dmde files from astrocrash 
		e_lo, dmde_lo = np.loadtxt(dmdedir+'{:.3f}'.format(self._sim_beta[0])+'.dat') # format requires 3 digits after decimal point
		#e_lo, dmde_lo = np.loadtxt(dmdedir+'dmde'+str(self._sim_beta[0])+'.dat')
		for i in range(1,len(self._sim_beta)): # bc calculating slope and yintercepts BETWEEN each simulation beta
			self._energy.append(e_lo) # save to access later in process function
	 		# dmde.append(dmde_lo) # save to access later in process function --> don't need, can just use interpolations but might not be exact for betas = simulation betas
			
			e_hi, dmde_hi= np.loadtxt(dmdedir+'{:.3f}'.format(self._sim_beta[i])+'.dat') #astrocrash format

			# smoothed flash file format
		 	
		 	# Interpolate  e array so that we can create same energy steps for lo and hi arrays.
	 		# since using e_lo array, only need to interpolate hi arrays.
		 	# (using e_lo array bc it is w/in the energy range of e_hi array)

			# note that x array for CubicSpline needs to be monotonically increasing
			funchi = CubicSpline(e_hi, dmde_hi)
			
			#funchi = CubicSpline(np.flipud(e_hi), np.flipud(dmde_hi)) 
		 	
		 	# get dmde_hi at values of e_lo so I can interpolate in beta
			dmde_hi_new = funchi(e_lo)
			#dmde_hi_new = np.flipud(funchi(np.flipud(e_lo)))

			# get slope for linear interpolation (in beta)
			self._beta_slope.append((dmde_hi_new - dmde_lo)/(self._sim_beta[i]-self._sim_beta[i-1]))
			
			# get y intercept for linear interpolation (in beta)
			yinterlo = dmde_lo - self._beta_slope[-1]*self._sim_beta[i-1]
			yinterhi = dmde_hi_new - self._beta_slope[-1]*self._sim_beta[i]

			self._beta_yinter.append((yinterlo+yinterhi)/2.0) # take average of yinterlo and yinterhi to get y intercept used in calculation (note that James just uses yinterlo)

			e_lo, dmde_lo = e_hi, dmde_hi

	def process(self, **kwargs):
	   
		beta_interp=True
		beta_outside_range=False

	   # change this so I get variables from mosfit
		G = c.G.cgs.value # 6.67259e-8 cm3 g-1 s-2
		Msolar = c.M_sun.cgs.value #1.989e33 grams
		Mhbase = 1.0e6*Msolar # this is the generic size of bh used in astrocrash sim
		Mstarbase = Msolar

		self._beta = kwargs['beta']
	   
		if 'dense_times' in kwargs:
			self._times = kwargs['dense_times']
		else:
			self._times = kwargs['rest_times']

	   # Check that beta chosen is within range of simulation betas
		if self._beta<self._sim_beta[0]:
			beta_outside_range=True
			interp_index_low=0
			print ('beta below simulation range: '+str(self._sim_beta[0])+'-'+str(self._sim_beta[-1]))
			print ('choose beta within range')
			beta_interp=False

		if self._beta>self._sim_beta[-1]:
			beta_outside_range=True
			interp_index_high=len(self._sim_beta)-1
			print ('beta above simulation range: '+str(self._sim_beta[0])+'-'+str(self._sim_beta[-1]))
			print ('choose beta within range')
			beta_interp=False

	   # find simulation betas to interpolate between
		for i in range(len(self._sim_beta)):
			if self._beta==self._sim_beta[i]: # don't need to interpolate, already have dmde and t for this beta
				beta_interp=False
				if i == len(self._beta_slope): # chosen beta value == highest sim beta value
					interp_index_low = i-1		# interpolations only calculated between values, therefore need to use lower interpolation for this to work
				else: interp_index_low = i  # so that conversion from dmde --> dmdt works (uses e_lo for conversion)

				#print ('exists simulation beta equal to user beta, no beta interpolation necessary, calculating dmdt...')
				break
			if self._beta<self._sim_beta[i]: 
				interp_index_high=i
				interp_index_low=i-1
				break


		if beta_outside_range == False:
			#----------- LINEAR BETA INTERPOLATION --------------

			# get new dmde
			#print (len(self._beta_yinter),len(self._beta_slope))
			#print (interp_index_low, self._sim_beta[interp_index_low])
			#print (self._beta)
			dmde = self._beta_yinter[interp_index_low] + self._beta_slope[interp_index_low]*self._beta


			#----------- CONVERT dm/de --> dm/dt --------------

		
		

	   		#if beta_interp == True:

			# should check that at simulation betas this interpolation gives the simulation dmdes back
			#if beta_interp == False: # files haven't been loaded yet
			#   e_lo, dmdenew = np.loadtxt(dmdedir+'dmde'+sim_beta_str[interp_index_low]+'.dat')

			# only convert dm/de --> dm/dt for mass that is bound to BH (energy < 0)
			ebound = np.array(self._energy[interp_index_low][self._energy[interp_index_low]<0]) # cuts off part of array with positive e (unbound)
			dmdebound = np.array(dmde[self._energy[interp_index_low]<0])

			# calculate de/dt, time and dm/dt arrays
			dedt = (1.0/3.0)*(-2.0*ebound)**(5.0/2.0)/(2.0*np.pi*G*Mhbase)  # in erg/s

			time = (2.0*np.pi*G*Mhbase)*(-2.0*ebound)**(-3.0/2.0)   # in seconds
			time = time/(24*3600) # time in days
			
			dmdt = dmdebound*dedt 

			# ----------- EXTRAPOLATE dm/dt TO EARLY TIMES -------------
			# new dmdt(t[0]) should == min(old dmdt)
			# use power law to fit : dmdt = b*t^xi

			# calculate floor dmdt and t to extrapolate down to
			dfloor = np.min(dmdt)

			if dmdt[0] >= dfloor*1.01: # not within 1% of floor, extrapolate

				ipeak = np.argmax(dmdt) # index of peak

				p = 0.1 # fraction of pre-peak dmdt to use for extrapolation to early times
				start = 5 # will cut off some part of original dmdt array, this # might change

				index1 = int(ipeak*p)

				while (index1 < 8):  # p should not be larger than 0.3
					p += 0.1
					index1 = int(ipeak*p)
					if p == 0.3: break


				while (index1-start < 5): # ensure extrapolation will include at least 5 pts 
					start -=1
					if start == 0: break

				if p*2 < 0.5 : index2 = int(ipeak*(p*2)) # ensure extrap. won't extend more than halfway to peak
				else: index2 = int(ipeak*0.5)

				t1 = time[start:index1]
				d1 = dmdt[start:index1]

				t2 = time[index2 - (index1 - start):index2]
				d2 = dmdt[index2 - (index1 - start):index2]

				# exponent for power law fit
				print (start, index1, index2)
				xi = np.log(d1/d2)/np.log(t1/t2)
				xiavg = np.mean(xi)

				# multiplicative factor for power law fit
				b1 = d1/(t1**xi)
				if t1[-1] < t2[0]: # if arrays don't overlap take mean of all values
					b2 = d2/(t2**xi)
					bavg = np.mean(np.array([b1,b2])) 
				else: bavg = np.mean(b1)

				logtfloor = np.log10(dfloor/bavg)/xiavg # log(new start time)

				textp = np.logspace(logtfloor, np.log10(time[start+index1/2]), num = 75) # ending extrapolation here will help make it a smoother transition
				dextp = bavg*textp**xiavg

				time = np.concatenate((textp,time[start+index1/2 + 1:]))
				dmdt = np.concatenate((dextp,dmdt[start+index1/2 + 1:]))


			# ----------- SCALE dm/dt TO BH & STAR SIZE --------------

			# bh mass for dmdt's in astrocrash is 1e6 solar masses 
			# dmdt ~ Mh^(-1/2)
			self._bhmass = kwargs['bhmass']*Msolar # right now kwargs bhmass is in solar masses, want in cgs
			# star mass for dmdts in astrocrash is 1 solar mass
			self._starmass = kwargs['starmass']*Msolar
			
			dmdt = dmdt * np.sqrt(Mhbase/self._bhmass) * (self._starmass/Mstarbase)**2.0
			time = time * np.sqrt(self._bhmass/Mhbase) * (Mstarbase/self._starmass)
			
			# this assumes t is increasing
			timeinterp = CubicSpline(time, dmdt)
			
			# this assumes t is decreasing 
			#timeinterp = CubicSpline(np.flipud(time), np.flipud(dmdt)) 

			# this assumes t is increasing
			dmdtnew = timeinterp(self._times)
			# this assumes t is decreasing 
			#dmdtnew = np.flipud(timeinterp(self._times))

			# Can uncomment following line to save files for testing
			#np.savetxt('test/files/beta'+'{:.3f}'.format(self._beta)+'mbh'+'{:.0f}'.format(self._bhmass)+'.dat',(time,dmdt),fmt='%1.18e')
			
			# self._epsilon = kwargs['epsilon']
			luminosities = 0.1*dmdtnew*c.c.cgs.value*c.c.cgs.value

			return {'kappagamma': kwargs['kappa'], 'luminosities': luminosities}
