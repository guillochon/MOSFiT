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
		# rundir is the directory holding the original flash data.  
		# This is only used to get the number of runs (= # of diff beta values) 
		# & the different beta values from run_info.dat. 
		#rundir= '/Users/brennamockler/tdefit/dmdts/mpoly_5-3_4-3_1e6/'

		# directory that holds smoothed and rebinned dmdes (NOT original flash dmdes)
		# It is assumed that there are different files for each beta (such as dmde2.5.dat for beta = 2.5)
		# The first row is energy, the second is dmde. This could be changed so that
		# each beta has a different subdirectory

		# for now just use astrocrash dmdes (converted from astrocrash dmdts)
		#print (os.path.dirname(__file__)[:-15])
		dmdedir = os.path.dirname(__file__)[:-15]+'models/tde/data/' #'../../models/tde/data/'
	   
		#dmdedir = '/Users/brennamockler/Dropbox (Personal)/Research/smooth+rebin/mpoly_5-3_4-3_1e6/gkernel35/'

		#--------- GET SIMULATION BETAS -----------------

		# Following block could be updated to get beta's from file names instead.

		# assumes run_info.dat folder exists with number of betas as first row and then each beta as 
		# single entry on following rows

		# the dmdes are in '/Users/brennamockler/tdefit/dmdts/mpoly_5-3_(gamma)_1e6/(beta)/dmde.dat'
		# where gamma = 4-3 or 5-3 and beta is one of the betas in run_info.dat
	   
		# following code gets simulation betas if you have 'run_info.dat' file
		#with open('/Users/brennamockler/tdefit/dmdts/mpoly_5-3_4-3_1e6/'+'run_info.dat') as f:
    	#   info=f.readlines()
    	#   nruns=int(info.pop(0)) # number of runs for this gamma value, each run corresponds to different beta
    	#   sim_beta_str=[]
    	#   for i in range(nruns):
    		   # beta values of runs are in run_info.dat after the number of runs,
    	#	  sim_beta_str.append(info.pop(0).strip()) #.strip() gets rid of trailing '\n' 
			   
    	# use sim_beta for figuring out which beta to interpolate between
		#self._sim_beta=[float(b) for b in sim_beta_str] 

		# hardcode in the simulation betas for gamma = 4-3 for now
		self._sim_beta = [0.600,0.650,0.700,0.750,0.800,0.850,0.900,1.000,1.100,1.200,1.300,1.400,1.500,1.600,1.700,1.800,1.850,1.900,2.000,2.500,3.000,3.500,4.000]

		#self._str_sim_beta = ['0.600','0.650','0.700','0.750','0.800','0.850','0.900','1.000','1.100','1.200','1.300','1.400','1.500','1.600','1.700','1.800','1.850','1.900','2.000','2.500','3.000','3.500','4.000']
		#-- CREATE INTERPOLATION FUNCTIONS; FIND SLOPES & YINTERs -------

		# these three lists, in addition to 'sim_beta', are the lists that will hold dmde info to be accessed after init is run
		self._beta_slope = []
		self._beta_yinter = []
		self._energy = []

	   # need to pad with extra zeros for dmde files from astrocrash 
		e_lo, dmde_lo = np.loadtxt(dmdedir+'{:.3f}'.format(self._sim_beta[0])+'.dat') # format requires 3 digits after decimal point
		#e_lo, dmde_lo = np.loadtxt(dmdedir+str(self._str_sim_beta[0])+'.dat')
		#e_lo, dmde_lo = np.loadtxt(dmdedir+'dmde'+str(self._sim_beta[0])+'.dat')
		for i in range(1,len(self._sim_beta)): # bc calculating slope and yintercepts BETWEEN each simulation beta
			self._energy.append(e_lo) # save to access later in process function
	 		# dmde.append(dmde_lo) # save to access later in process function --> don't need, can just use interpolations but might not be exact for betas = simulation betas
			
			e_hi, dmde_hi= np.loadtxt(dmdedir+'{:.3f}'.format(self._sim_beta[i])+'.dat') #astrocrash format
			#e_hi, dmde_hi = np.loadtxt(dmdedir+str(self._str_sim_beta[i])+'.dat')

			# smoothed flash file format
			#e_hi, dmde_hi= np.loadtxt(dmdedir+'dmde'+str(self._sim_beta[i])+'.dat')
		 	
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

			#self._mintime1 = 0
			#self._maxtime1 = 15

		    # Should I deallocate or delete unused variables at the end of __init__ ?

	def process(self, **kwargs):
	   
		beta_interp=True
		beta_outside_range=False

	   # change this so I get variables from mosfit
		G = c.G.cgs.value # 6.67259e-8 cm3 g-1 s-2
		Msolar = c.M_sun.cgs.value #1.989e33 grams
		Mhbase = 1.0e6*Msolar # this is the generic size of bh used in astrocrash sim
		

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
				interp_index_low = i # so that conversion from dmde --> dmdt works (uses e_lo for conversion)
				print ('exists simulation beta equal to user beta, no beta interpolation necessary, calculating dmdt...')
				break
			if self._beta<self._sim_beta[i]: 
				interp_index_high=i
				interp_index_low=i-1
				break


	   
		#----------- LINEAR BETA INTERPOLATION --------------

		# get new dmde
		dmde = self._beta_yinter[interp_index_low] + self._beta_slope[interp_index_low]*self._beta


		#----------- CONVERT dm/de --> dm/dt --------------

		
		if beta_outside_range == False:

	   		#if beta_interp == True:

			# should check that at simulation betas this interpolation gives the simulation dmdes back
			#if beta_interp == False: # files haven't been loaded yet
			#   e_lo, dmdenew = np.loadtxt(dmdedir+'dmde'+sim_beta_str[interp_index_low]+'.dat')

			# only convert dm/de --> dm/dt for mass that is bound to BH (energy < 0)
			ebound = np.array(self._energy[interp_index_low][self._energy[interp_index_low]<0]) # cuts off part of array with positive e (unbound)
			dmdebound = np.array(dmde[self._energy[interp_index_low]<0])
			#print ('beta = '+str(self._beta)+'; bhmass = '+str(kwargs['bhmass']))
			#print (min(ebound),max(ebound))
			#print (min(dmdebound),max(dmdebound))

			# calculate de/dt, time and dm/dt arrays
			dedt = (1.0/3.0)*(-2.0*ebound)**(5.0/2.0)/(2.0*np.pi*G*Mhbase)  # in erg/s

			
			#print (min(dedt),max(dedt))

			time = (2.0*np.pi*G*Mhbase)*(-2.0*ebound)**(-3.0/2.0)   # in seconds
			time = time/(24*3600) # time in days
			
			dmdt = dmdebound*dedt 

			#----------- SCALE dm/dt TO BH SIZE --------------

			# bh size for dmdt's in astrocrash is 1e6 solar masses 
			# dmdt ~ Mh^(-1/2)
			self._bhmass = kwargs['bhmass']*Msolar # right now kwargs bhmass is in solar masses, want in cgs

			
			dmdt = dmdt*np.sqrt(Mhbase/self._bhmass)
			time = time*np.sqrt(self._bhmass/Mhbase)
			
			# print statemetns for testing
			#print (min(dmdt),max(dmdt))
			#print (min(time),max(time))
			#print (max(time),1.*10**(self._maxtime1))
			#if max(time)<10.**(self._maxtime1): 
			#	self._maxtime1 = np.log(max(time))
			#	print ('log of min of maxtimes is: '+str(self._maxtime1))	
			
			#if min(time)>self._mintime1: 	
			#	self._mintime1 = min(time)
			#	print ('max of mintimes is: '+str(self._mintime1))		
    		# interpolate dmdt so that it has values at times = self._time
			
			# this assumes t is increasing
			timeinterp = CubicSpline(time, dmdt)
			
			# this assumes t is decreasing 
			#timeinterp = CubicSpline(np.flipud(time), np.flipud(dmdt)) 

			# this assumes t is increasing
			dmdtnew = timeinterp(self._times)

			#{:.3f}'.format(self._sim_beta[0])
			np.savetxt('test/files/beta'+'{:.3f}'.format(self._beta)+'mbh'+'{:.0f}'.format(self._bhmass)+'.dat',(self._times,dmdtnew),fmt='%1.18e')
			# this assumes t is decreasing 
			#dmdtnew = np.flipud(timeinterp(self._times))

			luminosities = 0.1*dmdtnew*c.c.cgs.value

    		#return {'luminosities': luminosities, 'times': time}
			return {'kappagamma': kwargs['kappa'], 'luminosities': luminosities}
