
# 2/9/17
# has early time extrapolation
# and does gamma interpolation
 

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

        self._gammas = ['4-3','5-3']

        # dictionaries with gamma's as keys.    
        self._beta_slope = {self._gammas[0]:[], self._gammas[1]:[]}
        self._beta_yinter = {self._gammas[0]:[], self._gammas[1]:[]}
        self._energy = {self._gammas[0]:[], self._gammas[1]:[]}
        self._sim_beta = {self._gammas[0]:[], self._gammas[1]:[]}

        for g in self._gammas:
            
            dmdedir = os.path.dirname(__file__)[:-15] + 'models/tde/data/' + g + '/' #'../../models/tde/data/'
           
            #dmdedir = '/Users/brennamockler/Dropbox (Personal)/Research/smooth+rebin/mpoly_5-3_4-3_1e6/gkernel35/'

            #--------- GET SIMULATION BETAS -----------------

            # hardcode in the simulation betas for gamma = 4-3 for now
            sim_beta_files = os.listdir(dmdedir)
            
            #sim_beta_new = {g:[float(b[:-4]) for b in sim_beta_files]}
            self._sim_beta[g].extend([float(b[:-4]) for b in sim_beta_files]) #{self._sim_beta.items() + sim_beta_new.items()} #[0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 1.000, 1.100,  1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.850, 1.900, 2.000, 2.500, 3.000, 3.500, 4.000]

            
            #-- CREATE INTERPOLATION FUNCTIONS; FIND SLOPES & YINTERs -------

            # these three lists, in addition to 'sim_beta', are the lists that will hold dmde info to be accessed after init is run
 

           # need to pad with extra zeros for dmde files from astrocrash 
            #e_lo, dmde_lo = np.loadtxt(dmdedir+'{:.3f}'.format(self._sim_beta[0])+'.dat') # format requires 3 digits after decimal point
            e_lo, dmde_lo = np.loadtxt(dmdedir+sim_beta_files[0])
            #e_lo, dmde_lo = np.loadtxt(dmdedir+'dmde'+str(self._sim_beta[0])+'.dat')
            for i in range(1,len(self._sim_beta[g])): # bc calculating slope and yintercepts BETWEEN each simulation beta
        
                e_hi, dmde_hi= np.loadtxt(dmdedir+sim_beta_files[i]) #np.loadtxt(dmdedir+'{:.3f}'.format(self._sim_beta[i])+'.dat') #astrocrash format


                # these two if statements needed bc of how interpolation is done
                # used for ~ 1/3 of data, higher beta have higher spread in e, and this 'mostly' maps out to larger e ranges in sims
                # e is monotonically increasing, with most negative (most bound) first
                if min(e_lo) < min(e_hi): 
                    #print ('min(e_lo) < min(e_hi)','gamma:', g, 'beta high:', self._sim_beta[g][i])
                    dmde_lo = dmde_lo[e_lo > min(e_hi)] # match new e_lo array sliced below
                    e_lo = e_lo[e_lo > min(e_hi)] # cut off first e values up to where e_lo within range of e_hi

                if max(e_lo) > max(e_hi):
                    #print ('max(e_lo) > max(e_hi)','gamma:', g, 'beta high:', self._sim_beta[g][i])
                    dmde_lo = dmde_lo[e_lo < max(e_hi)] # match new e_lo array sliced below
                    e_lo = e_lo[e_lo < max(e_hi)] # cut off last e values up to where e_lo within range of e_hi

                self._energy[g].append(e_lo) # save to access later in process function
                # dmde.append(dmde_lo) # save to access later in process function --> don't need, can just use interpolations but might not be exact for betas = simulation betas

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
                self._beta_slope[g].append((dmde_hi_new - dmde_lo)/(self._sim_beta[g][i]-self._sim_beta[g][i-1]))
                
                # get y intercept for linear interpolation (in beta)
                yinterlo = dmde_lo - self._beta_slope[g][-1]*self._sim_beta[g][i-1]
                yinterhi = dmde_hi_new - self._beta_slope[g][-1]*self._sim_beta[g][i]

                self._beta_yinter[g].append((yinterlo+yinterhi)/2.0) # take average of yinterlo and yinterhi to get y intercept used in calculation (note that James just uses yinterlo)

                e_lo, dmde_lo = e_hi, dmde_hi

    def process(self, **kwargs):
       
        beta_interp=True
        beta_outside_range=False

       # change this so I get variables from mosfit
        G = c.G.cgs.value # 6.67259e-8 cm3 g-1 s-2
        Msolar = c.M_sun.cgs.value #1.989e33 grams
        Mhbase = 1.0e6*Msolar # this is the generic size of bh used in astrocrash sim
        Mstarbase = Msolar

        # this is not beta, but rather a way to map beta_4-3 --> beta_5-3
        # b = 0 --> min disruption, b = 1 --> full disruption, b = 2 --> max beta of sims
        self._b = kwargs['b'] # change beta to this in parameters.json and tde.json

        if 0 <= self._b < 1 : 
            # 0.6 + (1.12 - 0.6)*b --> 0.6 is min disruption beta43, 1.12 is full disruption beta43
            beta43 = 0.6 + 0.52*self._b
            # 0.5 + (0.67 - 0.5)*b --> 0.5 is min disruption beta53, 0.67 is full disruption beta53
            beta53 = 0.5 + 0.17*self._b 

            self._beta = {'4-3': beta43, '5-3': beta53}

        elif 1 <= self._b <= 2:
            beta43 = 1.12 + 2.88*(self._b - 1)
            beta53 = 0.67 + 1.83*(self._b - 1)
            self._beta = {'4-3': beta43, '5-3': beta53}

        else: 
            print ('b outside range, bmin = 0; bmax = 2; b =', self._b)
            beta_outside_range = True
        #self._beta = kwargs['beta']
       
        

        # GET GAMMA VALUE

        gamma_interp = False

        # are first two statements necessary?
        if kwargs['starmass'] <= 0.3 or kwargs['starmass'] >= 22 : gammas = [self._gammas[1]] # gamma = ['5-3']
        elif 1 <= kwargs['starmass'] <= 15 : gammas = [self._gammas[0]] # gamma = ['4-3']
        elif 0.3 < kwargs['starmass'] < 1:  # region going from gamma = 5/3 to gamma = 4/3 as mass increases
            gamma_interp = True
            gammas = self._gammas
            # gfrac should == 0 for 4/3; == 1 for 5/3
            gfrac = (kwargs['starmass'] - 1.)/(0.3 - 1.)
        elif 15 < kwargs['starmass'] < 22 : # region going from gamma = 4/3 to gamma = 5/3 as mass increases
            gamma_interp = True
            gammas = self._gammas
            # gfrac should == 0 for 4/3; == 1 for 5/3
            gfrac =  (kwargs['starmass'] - 15.)/(22. - 15.)

        if 'dense_times' in kwargs:
            self._times = kwargs['dense_times']
        else:
            self._times = kwargs['rest_times']


        timedict = {} # will hold time arrays for each g in gammas 
        dmdtdict = {} # will hold dmdt arrays for each g in gammas

        for g in gammas:          
            # find simulation betas to interpolate between
            for i in range(len(self._sim_beta[g])):
                if self._beta[g]==self._sim_beta[g][i]: # don't need to interpolate, already have dmde and t for this beta
                    beta_interp=False
                    if i == len(self._beta_slope[g]): # chosen beta value == highest sim beta value
                        interp_index_low = i-1      # interpolations only calculated between values, therefore need to use lower interpolation for this to work
                    else: interp_index_low = i  # so that conversion from dmde --> dmdt works (uses e_lo for conversion)

                    #print ('exists simulation beta equal to user beta, no beta interpolation necessary, calculating dmdt...')
                    break
                if self._beta[g]<self._sim_beta[g][i]: 
                    interp_index_high=i
                    interp_index_low=i-1
                    break


            if beta_outside_range == False:
                #----------- LINEAR BETA INTERPOLATION --------------

                # get new dmde
                dmde = self._beta_yinter[g][interp_index_low] + self._beta_slope[g][interp_index_low]*self._beta[g]

                # quick fix for one neg. dmde value --> shouldn't be needed bc of lines ~80 - ~90 that cut e_lo array so that this shouldn't happend
                if len(dmde[dmde<0]) > 0:
                    if (dmde[0] < 0) and (len(dmde[dmde<0]) == 1):
                        dmde = dmde[1:]
                        self._energy[g][interp_index_low] = self._energy[g][interp_index_low][1:]
                        print ('negative first value of dmde, beta =',self._beta[g])
                    else: print ('more than the first value of (the beta interpolated) dmde is negative')

                #----------- CONVERT dm/de --> dm/dt --------------
 

                #if beta_interp == True:

                # should check that at simulation betas this interpolation gives the simulation dmdes back
                #if beta_interp == False: # files haven't been loaded yet
                #   e_lo, dmdenew = np.loadtxt(dmdedir+'dmde'+sim_beta_str[interp_index_low]+'.dat')

                # only convert dm/de --> dm/dt for mass that is bound to BH (energy < 0)
                ebound = np.array(self._energy[g][interp_index_low][self._energy[g][interp_index_low]<0]) # cuts off part of array with positive e (unbound)
                dmdebound = np.array(dmde[self._energy[g][interp_index_low]<0])


                if min(dmdebound)<0: 

                    print ('beta, negative dmdebound', self._beta[g], dmdebound[dmdebound<0])
                    
                # calculate de/dt, time and dm/dt arrays
                dedt = (1.0/3.0)*(-2.0*ebound)**(5.0/2.0)/(2.0*np.pi*G*Mhbase)  # in erg/s

                time = (2.0*np.pi*G*Mhbase)*(-2.0*ebound)**(-3.0/2.0)   # in seconds
                time = time/(24*3600) # time in days

                dmdt = dmdebound*dedt 
                #if len(dmdt[dmdt<0])>0: print ('dmdt',dmdt)

                # ----------- EXTRAPOLATE dm/dt TO EARLY TIMES -------------
                # new dmdt(t[0]) should == min(old dmdt)
                # use power law to fit : dmdt = b*t^xi

                # calculate floor dmdt and t to extrapolate down to this value for early times
                dfloor = np.min(dmdt)

                if dmdt[0] >= dfloor*1.01: # not within 1% of floor, extrapolate

                    ipeak = np.argmax(dmdt) # index of peak

                    p = 0.1 # fraction of pre-peak dmdt to use for extrapolation to early times
                    start = 5 # will cut off some part of original dmdt array, this # might change

                    index1 = int(ipeak*p)

                    while (index1 < 8):  # p should not be larger than 0.3
                        p += 0.1
                        index1 = int(ipeak*p)
                        if p >= 0.3: 
                            #print ('enter')
                            break


                    while (index1-start < 5): # ensure extrapolation will include at least 5 pts 
                        start -=1
                        if start == 0: break


                    if p*2 < 0.5 : index2 = int(ipeak*(p*2)) # ensure extrap. won't extend more than halfway to peak
                    else: index2 = int(ipeak*0.5)

                    #print ('ipeak, p, start, index1, index2', ipeak, p, start, index1, index2)
                    
                    t1 = time[start:index1]
                    d1 = dmdt[start:index1]

                    t2 = time[index2 - (index1 - start):index2]
                    d2 = dmdt[index2 - (index1 - start):index2]

                    # exponent for power law fit
                    #print (start, index1, index2)
                    xi = np.log(d1/d2)/np.log(t1/t2)
                    xiavg = np.mean(xi)

                    # multiplicative factor for power law fit
                    b1 = d1/(t1**xiavg)
                    if t1[-1] < t2[0]: # if arrays don't overlap take mean of all values
                        b2 = d2/(t2**xiavg)
                        bavg = np.mean(np.array([b1,b2])) # np.mean flattens the array, so this works
                    else: bavg = np.mean(b1)

                    logtfloor = np.log10(dfloor/bavg)/xiavg # log(new start time)

                    textp = np.logspace(logtfloor, np.log10(time[start+int(index1/2)]), num = 75) # ending extrapolation here will help make it a smoother transition
                    dextp = bavg*textp**xiavg

                    time = np.concatenate((textp,time[start+int(index1/2) + 1:]))
                    dmdt = np.concatenate((dextp,dmdt[start+int(index1/2) + 1:]))

                timedict[g] = time
                dmdtdict[g] = dmdt

            # ---------------- GAMMA INTERPOLATION -------------------

        if gamma_interp == True:

            

            # find indexes of dmdt peaks
            ipeak43 = np.argmax(dmdtdict['4-3']) 
            ipeak53 = np.argmax(dmdtdict['5-3'])

            tpeak43 = timedict['4-3'][ipeak43]
            tpeak53 = timedict['5-3'][ipeak53]

            
            # shift 'higher' time array so it matches lower one for interpolation
            # Note that we expect tpeak53 < tpeak43
            if tpeak53 < tpeak43: 
                glo = '4-3'
                ghi = '5-3'
                timedict['4-3'] = timedict['4-3'] + (tpeak53 - tpeak43)*gfrac # assuming gfrac = 0 for gamma = 4/3, gfrac = 1 for gamma = 5/3
                timedict['5-3'] = timedict['5-3'] - (tpeak53 - tpeak43)*(1-gfrac)

            else: 
                glo = '5-3'
                ghi = '4-3'
                timedict['4-3'] = timedict['4-3'] - (tpeak43 - tpeak53)*gfrac # assuming gfrac = 0 for gamma = 4/3, gfrac = 1 for gamma = 5/3
                timedict['4-3'] = timedict['4-3'] + (tpeak43 - tpeak53)*(1-gfrac)
                # following line not necessary, but if this would be weird so print something for now
                if tpeak53 == tpeak43: print ('(time of peak for gamma = 4/3) = (time of peak for gamma = 5/3)')

            # gamma = 5/3 should be more spread out in time, but just in case:
            if timedict[glo][0] > timedict[ghi][0]: # gamma hi data more spread out at early times
                ginterp = ghi
                gnointerp = glo
                if timedict[glo][-1] > timedict[ghi][-1]: # gamma lo data more spread out at late times
                    # cut off part of glo array with time > max(ghi time array)
                    dmdtdict[glo] = dmdtdict[glo][timedict[glo] < timedict[ghi][-1]]
                    timedict[glo] = timedict[glo][timedict[glo] < timedict[ghi][-1]]
                    #print ('cuts made at late time for gamma interpolation')

            if timedict[ghi][0] > timedict[glo][0]: # gamma lo data more spread out at early times
                ginterp = glo
                gnointerp = ghi
                if timedict[glo][-1] < timedict[ghi][-1]: # gamma hi data more spread out at late times
                    # cut off part of ghi array with time > max(glo time array)
                    dmdtdict[ghi] = dmdtdict[ghi][timedict[ghi] < timedict[glo][-1]]
                    timedict[ghi] = timedict[ghi][timedict[ghi] < timedict[glo][-1]]     


            # now ginterp should be set to the gamma with a time array with a larger range (in both early and late times)
            # than the other gamma.

            # Interpolate dmdt array corresponding to larger time range gamma to match times of smaller time range gamma

            timeinterpfunc = CubicSpline(timedict[ginterp], dmdtdict[ginterp])
            dmdtinterp = timeinterpfunc(timedict[gnointerp])
            time = timedict[gnointerp]

            # recall gfrac = 0 --> gamma = 4/3, gfrac = 1 --> gamma 5/3
            if ginterp == '5-3':
                dmdt = dmdtdict['4-3'] + (dmdtinterp - dmdtdict['4-3'])*gfrac
            else: # ginterp == '4-3'
                dmdt = dmdtdict['5-3'] + (dmdtinterp - dmdtdict['5-3'])*(1. - gfrac)

        # ----------- SCALE dm/dt TO BH & STAR SIZE --------------

        # bh mass for dmdt's in astrocrash is 1e6 solar masses 
        # dmdt ~ Mh^(-1/2)
        self._bhmass = kwargs['bhmass']*Msolar # right now kwargs bhmass is in solar masses, want in cgs
        # star mass for dmdts in astrocrash is 1 solar mass
        self._starmass = kwargs['starmass']*Msolar
        
        dmdt = dmdt * np.sqrt(Mhbase/self._bhmass) * (self._starmass/Mstarbase)**2.0
        time = time * np.sqrt(self._bhmass/Mhbase) * (Mstarbase/self._starmass)
        
        # this assumes t is increasing
        timeinterpfunc = CubicSpline(time, dmdt)
        
        # this assumes t is decreasing 
        #timeinterp = CubicSpline(np.flipud(time), np.flipud(dmdt)) 

        # this assumes t is increasing
        dmdtnew = timeinterpfunc(self._times)
        dmdtnew[dmdtnew < 0] = 0 # don't allow negative dmdts. At some point maybe fit to time of peak somewhere in here?

        #if min(self._times) < min(time):
           # dmdtnew = dmdtnew[self._]

        # this assumes t is decreasing 
        #dmdtnew = np.flipud(timeinterp(self._times))

        # Can uncomment following line to save files for testing
        #np.savetxt('test/files/beta'+'{:.3f}'.format(self._beta)+'mbh'+'{:.0f}'.format(self._bhmass)+'.dat',(time,dmdt),fmt='%1.18e')
        
        self._efficiency = kwargs['efficiency']
        luminosities = self._efficiency*dmdtnew*c.c.cgs.value*c.c.cgs.value

        return {'kappagamma': kwargs['kappa'], 'luminosities': luminosities}
