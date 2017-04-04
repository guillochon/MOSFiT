
# 2/9/17
# has early time extrapolation
# and does gamma interpolation


from math import isnan

import astropy.constants as c

import numpy as np

import os

from scipy.interpolate import CubicSpline

from scipy.interpolate import interp1d

from mosfit.modules.engines.engine import Engine

CLASS_NAME = 'Fallback'


class Fallback(Engine):
    """A tde engine.
    """

    def __init__(self,**kwargs):
        # call super version of init

        super(Fallback, self).__init__(**kwargs)


        G = c.G.cgs.value # 6.67259e-8 cm3 g-1 s-2
        Msolar = c.M_sun.cgs.value #1.989e33 grams
        Rsolar = c.R_sun.cgs.value
        Mhbase = 1.0e6*Msolar # this is the generic size of bh used in astrocrash sim
        Mstarbase = Msolar
        Rstarbase = Rsolar

        self.TESTING = False
        ##### FOR TESTING ######
        if self.TESTING == True:
            self.testnum = 0
            filestodelete = os.listdir('test_dir/test_fallback/pregammainterp/g5-3')
            for f in filestodelete:
                os.remove('test_dir/test_fallback/pregammainterp/g5-3/'+f)
            filestodelete = os.listdir('test_dir/test_fallback/pregammainterp/g4-3')
            for f in filestodelete:
                os.remove('test_dir/test_fallback/pregammainterp/g4-3/'+f)
            filestodelete = os.listdir('test_dir/test_fallback/postgammainterp')
            for f in filestodelete:
                os.remove('test_dir/test_fallback/postgammainterp/' + f)
            filestodelete = os.listdir('test_dir/test_fallback/endfallback')
            for f in filestodelete:
                os.remove('test_dir/test_fallback/endfallback/' + f)   
            filestodelete = os.listdir('test_dir/test_viscous/endviscous')
            for f in filestodelete:
                os.remove('test_dir/test_viscous/endviscous/' + f) 
             
             
        #########################
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
        self._sim_beta = {self._gammas[0]:[], self._gammas[1]:[]}
        self._mapped_time = {self._gammas[0]:[], self._gammas[1]:[]}
        self._premaptime = {self._gammas[0]:[], self._gammas[1]:[]} # for converting back from mapped time to actual times and doing interpolation in actual time
        self._premapdmdt = {self._gammas[0]:[], self._gammas[1]:[]} 

        for g in self._gammas:
            
            dmdedir = os.path.dirname(__file__)[:-15] + 'models/tde/data/' + g + '/' #'../../models/tde/data/'


            #--------- GET SIMULATION BETAS -----------------

            sim_beta_files = os.listdir(dmdedir)

            self._sim_beta[g].extend([float(b[:-4]) for b in sim_beta_files]) #{self._sim_beta.items() + sim_beta_new.items()} #[0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 1.000, 1.100,  1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.850, 1.900, 2.000, 2.500, 3.000, 3.500, 4.000]


            #-- CREATE INTERPOLATION FUNCTIONS; FIND SLOPES & YINTERs -------
            
            time = {}
            dmdt = {}
            ipeak = {}
            mapped_time = {}
            # get dmdt and t for the lowest beta value
            e, d = np.loadtxt(dmdedir+sim_beta_files[0]) # energy and dmde in cgs units
            # only convert dm/de --> dm/dt for mass that is bound to BH (energy < 0)
            ebound = e[e<0]# cuts off part of array with positive e (unbound)
            dmdebound = d[e<0]

            if min(dmdebound)<0: # shouldn't happen, just a check 

                print ('beta, gamma, negative dmde bound:', self._sim_beta[g], g, dmdebound[dmdebound<0])

            # calculate de/dt, time and dm/dt arrays
            dedt = np.log10((1.0/3.0)*(-2.0*ebound)**(5.0/2.0)/(2.0*np.pi*G*Mhbase))  # in log(erg/s)
            time['lo'] = np.log10((2.0*np.pi*G*Mhbase)*(-2.0*ebound)**(-3.0/2.0))   # in log(seconds)
            dmdt['lo'] = np.log10(dmdebound*dedt) # in log(g/s) 

            ipeak['lo'] = np.argmax(dmdt['lo'])
            # split time['lo'] & dmdt['lo'] into pre-peak and post-peak array
            time['lo'] = np.array([time['lo'][:ipeak['lo']], time['lo'][ipeak['lo']:]]) # peak is in second array       
            dmdt['lo'] = np.array([dmdt['lo'][:ipeak['lo']], dmdt['lo'][ipeak['lo']:]]) # peak is in second array
            
            self._premaptime[g].append(np.copy(time['lo'])) # will contain time arrays (split into pre peak and post peak times) for each beta value
            self._premapdmdt[g].append(np.copy(dmdt['lo'])) # will contain time arrays (split into pre peak and post peak dmdts) for each beta value
            for i in range(1,len(self._sim_beta[g])): # indexing like this bc calculating slope and yintercepts BETWEEN each simulation beta

                e, d = np.loadtxt(dmdedir+sim_beta_files[i]) #np.loadtxt(dmdedir+'{:.3f}'.format(self._sim_beta[i])+'.dat') #astrocrash format
                # only convert dm/de --> dm/dt for mass that is bound to BH (energy < 0)
                ebound = e[e<0]# cuts off part of array with positive e (unbound)
                dmdebound = d[e<0]

                if min(dmdebound)<0: # shouldn't happen, just a check 
                    print ('beta, gamma, negative dmde bound:', self._sim_beta[g], g, dmdebound[dmdebound<0])
                
                # calculate de/dt, time and dm/dt arrays
                dedt = (1.0/3.0)*(-2.0*ebound)**(5.0/2.0)/(2.0*np.pi*G*Mhbase)  # in log(erg/s)
                time['hi'] = np.log10((2.0*np.pi*G*Mhbase)*(-2.0*ebound)**(-3.0/2.0))   # in log(seconds)
                dmdt['hi'] = np.log10(dmdebound*dedt) # in log(g/s) 
               
                ipeak['hi'] = np.argmax(dmdt['hi'])
                
                # split time_hi and dmdt_hi into pre-peak and post-peak array
                time['hi'] = np.array([time['hi'][:ipeak['hi']], time['hi'][ipeak['hi']:]]) # peak is in second array       
                dmdt['hi'] = np.array([dmdt['hi'][:ipeak['hi']], dmdt['hi'][ipeak['hi']:]]) # peak is in second array
                self._premapdmdt[g].append(np.copy(dmdt['hi'])) # will contain time arrays (split into pre peak and post peak dmdts) for each beta value
                self._premaptime[g].append(np.copy(time['hi'])) # will contain time arrays (split into pre peak and post peak times) for each beta value

                #print ('length of time[hi] and dmdt[hi] arrays in init:', len(dmdt['hi'][0]), len(time['hi'][0]), len(dmdt['hi'][1]), len(time['hi'][1]))
                

                mapped_time['hi'] = []
                mapped_time['lo'] = []

                self._beta_slope[g].append([])
                self._beta_yinter[g].append([])
                self._mapped_time[g].append([])
                for j in [0,1]: # once before peak, once after peak
                    # choose more densely sampled curve to map times from times to 0-1. 
                    # less densely sampled curve will be interpolated to match
                    if len(time['lo'][j]) < len(time['hi'][j]): # hi array more densely sampled 
                        interp = 'lo'
                        nointerp = 'hi'
                    else: 
                        interp = 'hi'
                        nointerp = 'lo' # will also catch case where they have the same lengths
                    # map times from more densely sampled curves (bost pre & post peak, might be from diff. dmdts) 
                    # to 0 - 1

                    mapped_time[nointerp].append( 1./(time[nointerp][j][-1] - time[nointerp][j][0]) * (time[nointerp][j] - time[nointerp][j][0]) )
                    mapped_time[interp].append( 1./(time[interp][j][-1] - time[interp][j][0]) * (time[interp][j] - time[interp][j][0]) ) 
                    
                    # make sure bounds are the same for interp and nointerp  before interpolation
                    #(they should be 0 and 1 from above, but could be slightly off due to rounding errors in python)
                    mapped_time[interp][j][0] = 0
                    mapped_time[interp][j][-1] = 1
                    mapped_time[nointerp][j][0] = 0
                    mapped_time[nointerp][j][-1] = 1
                    
                    func = interp1d(mapped_time[interp][j], dmdt[interp][j])
                    
                    dmdtinterp = func(mapped_time[nointerp][j])
                   

                    if interp == 'hi': slope = (dmdtinterp - dmdt['lo'][j])/(self._sim_beta[g][i]-self._sim_beta[g][i-1])
                    else: slope = (dmdt['hi'][j] - dmdtinterp)/(self._sim_beta[g][i]-self._sim_beta[g][i-1]) # interp == 'lo'
                    self._beta_slope[g][-1].append(slope)

                    yinter1 = dmdt[nointerp][j] - self._beta_slope[g][-1][j]*self._sim_beta[g][i-1]
                    yinter2 = dmdtinterp - self._beta_slope[g][-1][j]*self._sim_beta[g][i]
                    self._beta_yinter[g][-1].append((yinter1+yinter2)/2.0)
                    self._mapped_time[g][-1].append(np.array(mapped_time[nointerp][j])) # for passing to process
           
                time['lo'], dmdt['lo'] = np.copy(time['hi']), np.copy(dmdt['hi']) 


    def process(self, **kwargs):

        beta_interp=True
        beta_outside_range=False

       # change this so I get variables from mosfit
        G = c.G.cgs.value # 6.67259e-8 cm3 g-1 s-2
        Msolar = c.M_sun.cgs.value #1.989e33 grams
        Rsolar = c.R_sun.cgs.value
        Mhbase = 1.0e6*Msolar # this is the generic size of bh used in astrocrash sim
        Mstarbase = Msolar
        Rstarbase = Rsolar

        # this is not beta, but rather a way to map beta_4-3 --> beta_5-3
        # b = 0 --> min disruption, b = 1 --> full disruption, b = 2 --> max beta of sims
        self._b = kwargs['b'] # change beta to this in parameters.json and tde.json

        if 0 <= self._b < 1 :
            # 0.6 + (1.85 - 0.6)*b --> 0.6 is min disruption beta43, 1.85 is full disruption beta43
            beta43 = 0.6 + 1.25*self._b
            # 0.5 + (0.9 - 0.5)*b --> 0.5 is min disruption beta53, 0.9 is full disruption beta53
            beta53 = 0.5 + 0.4*self._b

            self._beta = {'4-3': beta43, '5-3': beta53}

        elif 1 <= self._b <= 2:
            beta43 = 1.85 + 2.15*(self._b - 1)
            beta53 = 0.9 + 1.6*(self._b - 1)
            self._beta = {'4-3': beta43, '5-3': beta53}

        else:
            print ('b outside range, bmin = 0; bmax = 2; b =', self._b)
            beta_outside_range = True


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


        timedict = {} # will hold time arrays for each g in gammas
        dmdtdict = {} # will hold dmdt arrays for each g in gammas

        for g in gammas:
            # find simulation betas to interpolate between
            for i in range(len(self._sim_beta[g])):
                if self._beta[g] == self._sim_beta[g][i]: # don't need to interpolate, already have dmdt and t for this beta
                    beta_interp = False
                    interp_index_low = i
                    break

                if self._beta[g] < self._sim_beta[g][i]:
                    interp_index_high = i
                    interp_index_low = i-1
                    beta_interp = True
                    break


            if beta_outside_range == False and beta_interp == True:
                #----------- LINEAR BETA INTERPOLATION --------------

                # get new dmdts  (2 arrays, before and after peak (peak in 2nd array))
                # use interp_index_low bc of how slope and yintercept are saved (slope[0] corresponds to between beta[0] and beta[1] etc.)
                dmdt = np.array([self._beta_yinter[g][interp_index_low][0] + self._beta_slope[g][interp_index_low][0]*self._beta[g], 
                        self._beta_yinter[g][interp_index_low][1] + self._beta_slope[g][interp_index_low][1]*self._beta[g]])

                # map mapped_times back to actual times, requires interpolation in time
                # first for pre peak times

                time = []
                for i in [0,1]:
                    # interp_index_low indexes beta

                    time_betalo = (self._mapped_time[g][interp_index_low][i] * 
                                (self._premaptime[g][interp_index_low][i][-1] - self._premaptime[g][interp_index_low][i][0]) + 
                                self._premaptime[g][interp_index_low][i][0]) # mapped time between beta low and beta high
                    time_betahi = (self._mapped_time[g][interp_index_low][i] *
                                (self._premaptime[g][interp_index_high][i][-1] - self._premaptime[g][interp_index_high][i][0]) +
                                self._premaptime[g][interp_index_high][i][0])

                    time.append(time_betalo + (time_betahi - time_betalo)*(self._beta[g] - 
                            self._sim_beta[g][interp_index_low])/(self._sim_beta[g][interp_index_high] - self._sim_beta[g][interp_index_low]))
                time = np.array(time)
                
                # ----------- EXTRAPOLATE dm/dt TO EARLY TIMES -------------
                # new dmdt(t[0]) should == min(old dmdt)
                # use power law to fit : dmdt = b*t^xi

                # calculate floor dmdt and t to extrapolate down to this value for early times
                '''
                dfloor = np.min(dmdt) 
                
                if dmdt[0] >= dfloor*1.01: # not within 1% of floor, extrapolate

                    ipeak = np.argmax(dmdt) # index of peak

                    prepeakfunc = CubicSpline(time[:ipeak], dmdt[:ipeak])
                    prepeaktimes = np.logspace(np.log10(time[0]),np.log10(time[ipeak-1]),1000)
                    prepeakdmdt = prepeakfunc(prepeaktimes)

                    p = 0.1 # fraction of pre-peak dmdt to use for extrapolation to early times
                    start = 5 # will cut off some part of original dmdt array, this # might change

                    index1 = int(len(prepeakdmdt)*p) #int(ipeak*p)

                    while (index1 < 8):  # p should not be larger than 0.3
                        p += 0.1
                        index1 = int(len(prepeakdmdt)*p) #int(ipeak*p)
                        if p >= 0.3:
                            #print ('enter')
                            break


                    while (index1-start < 5): # ensure extrapolation will include at least 5 pts
                        start -=1
                        if start == 0: break


                    if p*2 < 0.5 : index2 = int(len(prepeakdmdt)*p*2) #int(ipeak*(p*2)) # ensure extrap. won't extend more than halfway to peak
                    else: index2 = int(len(prepeakdmdt)*0.5) #int(ipeak*0.5)

                    #print ('ipeak, p, start, index1, index2', ipeak, p, start, index1, index2)

                    t1 = prepeaktimes[start:index1]
                    d1 = prepeakdmdt[start:index1]

                    t2 = prepeaktimes[index2 - (index1 - start):index2]
                    d2 = prepeakdmdt[index2 - (index1 - start):index2]

                    #print ('index1:',index1)
                    #print ('index2:', index2)
                    #print ('p:', p)
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

                    indexext = len(time[time<prepeaktimes[index1]])
                    textp = np.logspace(logtfloor, np.log10(time[start+int(indexext)]), num = 75) # ending extrapolation here will help make it a smoother transition
                    dextp = bavg*textp**xiavg

                    time = np.concatenate((textp,time[start+int(indexext) + 1:]))
                    dmdt = np.concatenate((dextp,dmdt[start+int(indexext) + 1:]))
                '''
                timedict[g] = time
                dmdtdict[g] = dmdt

            elif beta_outside_range == False and beta_interp == False: 
                if (len(self._premaptime[g][interp_index_low][0]) != len(self._premapdmdt[g][interp_index_low][0]) or
                    len(self._premaptime[g][interp_index_low][1]) != len(self._premapdmdt[g][interp_index_low][1]) ): 
                    print ('length premaptime and premapdmdt in process:',
                        len(self._premaptime[g][interp_index_low][0]), len(self._premapdmdt[g][interp_index_low][0]),
                        len(self._premaptime[g][interp_index_low][1]),len(self._premapdmdt[g][interp_index_low][1]) )
                timedict[g] = np.copy(self._premaptime[g][interp_index_low])
                dmdtdict[g] = np.copy(self._premapdmdt[g][interp_index_low])

        # ----------------TESTING ----------------
        
        '''
        if gamma_interp == True:

            #print ('beta interp =', beta_interp, len(np.append(timedict['4-3'][0], timedict['4-3'][1])), len(np.append(dmdtdict['4-3'][0], dmdtdict['4-3'][1])))
            np.savetxt('test_dir/test_fallback/pregammainterp/g4-3/time+dmdt'+'{:03d}'.format(self.testnum)+'g'+gammas[0]+'b'+str(self._b)+'.txt',
            (np.append(timedict['4-3'][0],timedict['4-3'][1]), np.append(dmdtdict['4-3'][0], dmdtdict['4-3'][1])))
            np.savetxt('test_dir/test_fallback/pregammainterp/g5-3/time+dmdt'+'{:03d}'.format(self.testnum)+'g'+gammas[1]+'b'+str(self._b)+'.txt',
            (np.append(timedict['5-3'][0],timedict['5-3'][1]), np.append(dmdtdict['5-3'][0], dmdtdict['5-3'][1])))
        '''
        # ----------------------------------------

        # ---------------- GAMMA INTERPOLATION -------------------

        if gamma_interp == True:

            mapped_time = {'4-3': [], '5-3': []}

            time = []
            dmdt = []
            for j in [0,1]: # once before peak, once after peak
                # choose more densely sampled curve to map times from times to 0-1. 
                # less densely sampled curve will be interpolated to match
                if len(timedict['4-3'][j]) < len(timedict['5-3'][j]): # gamma = 5/3 array more densely sampled 
                    interp = '4-3'
                    nointerp = '5-3'
                else: 
                    interp = '5-3'
                    nointerp = '4-3' # will also catch case where they have the same lengths
                # map times from more densely sampled curves (bost pre & post peak, might be from diff. dmdts) 
                # to 0 - 1

                mapped_time[nointerp].append( 1./(timedict[nointerp][j][-1] - timedict[nointerp][j][0]) * 
                                            (timedict[nointerp][j] - timedict[nointerp][j][0]) )
                mapped_time[interp].append( 1./(timedict[interp][j][-1] - timedict[interp][j][0]) * 
                                        (timedict[interp][j] - timedict[interp][j][0]) ) 
                # make sure bounds are the same for interp and nointerp  before interpolation
                #(they should be 0 and 1 from above, but could be slightly off due to rounding errors in python)
                mapped_time[interp][j][0] = 0
                mapped_time[interp][j][-1] = 1
                mapped_time[nointerp][j][0] = 0
                mapped_time[nointerp][j][-1] = 1

                func = interp1d(mapped_time[interp][j], dmdtdict[interp][j])
                dmdtdict[interp][j] = func(mapped_time[nointerp][j])

                # recall gfrac = 0 --> gamma = 4/3, gfrac = 1 --> gamma 5/3
                if interp == '5-3': # then mapped_time = mapped_time[nointerp] = mapped_time['4-3']
                    time53 = mapped_time['4-3'][j] * (timedict['5-3'][j][-1] - timedict['5-3'][j][0]) + timedict['5-3'][j][0]
                    time.extend(10**(timedict['4-3'][j] + (time53 - timedict['4-3'][j])*gfrac))
                else: # interp == '4-3'
                    time43 = mapped_time['5-3'][j] * (timedict['4-3'][j][-1] - timedict['4-3'][j][0]) + timedict['4-3'][j][0]
                    time.extend(10**(time43 + (timedict['5-3'][j] - time43) * gfrac)) # convert back from logspace before adding to time array


                # recall gfrac = 0 --> gamma = 4/3, gfrac = 1 --> gamma 5/3 
                dmdt.extend(10**(dmdtdict['4-3'][j] + (dmdtdict['5-3'][j] - dmdtdict['4-3'][j])*gfrac)) # convert back from logspace before adding to time array
              
        else: # gamma_interp == False:
            # in this case, g will still be g from loop over gammas, 
            # but there was only one gamma (no interpolation), so g is the correct gamma
            # note that timedict[g] is a list not an array
            time = np.concatenate((timedict[g][0], timedict[g][1])) # no longer need a prepeak and postpeak array
            time = 10**time
            dmdt = np.concatenate((dmdtdict[g][0], dmdtdict[g][1]))
            dmdt = 10**dmdt

        time = np.array(time)
        dmdt = np.array(dmdt)
        # ----------------TESTING ----------------
        if self.TESTING == True:
            if gamma_interp == True:
                np.savetxt('test_dir/test_fallback/postgammainterp/time+dmdt'+'{:03d}'.format(self.testnum)+'gfrac'+str(gfrac)+'b'+str(self._b)+'.txt',
                (time, dmdt))
            else: 
                np.savetxt('test_dir/test_fallback/postgammainterp/time+dmdt'+'{:03d}'.format(self.testnum)+'g'+str(g)+'b'+str(self._b)+'.txt',
                (time, dmdt))
        
        # ----------- SCALE dm/dt TO BH & STAR MASS & STAR RADIUS --------------

        if 'dense_times' in kwargs:
            self._times = kwargs['dense_times'] # time in days
        else:
            print ('in fallback, dense_times NOT in kwargs')
            self._times = kwargs['rest_times']

        # bh mass for dmdt's in astrocrash is 1e6 solar masses
        # dmdt ~ Mh^(-1/2)
        self._bhmass = kwargs['bhmass']*Msolar # right now kwargs bhmass is in solar masses, want in cgs
        # star mass for dmdts in astrocrash is 1 solar mass
        self._starmass = kwargs['starmass']*Msolar

        self._Rstar = kwargs['Rstar']*Rsolar

        dmdt = dmdt * np.sqrt(Mhbase/self._bhmass) * (self._starmass/Mstarbase)**2.0 * (Rstarbase/self._Rstar)**1.5
        # tpeak ~ Mh^(1/2) * Mstar^(-1)
        time = time * np.sqrt(self._bhmass/Mhbase) * (Mstarbase/self._starmass) * (self._Rstar/Rstarbase)**1.5
        tnew = time/(3600 * 24) # time is now in days to match self._times


        # try aligning first fallback time of simulation
        # (whatever first time is after early t extrapolation) with parameter texplosion
        self.rest_t_explosion = kwargs['resttexplosion'] # resttexplosion in days (very close
        # to texplosion, using this bc it's what's used in transform.py)
       
        tnew = tnew - (tnew[0] - self.rest_t_explosion)

        # ----------------TESTING ----------------
        #if gamma_interp == True:
        #    np.savetxt('viscoustests/noearlytimeextrap/precutfallback/times+lums'+'{:03d}'.format(self.testnum)+'g'+str(gfrac)+'b'+str(self._b)+'.txt',
        #    ((tnew-self.rest_t_explosion), kwargs['efficiency']*dmdt*c.c.cgs.value*c.c.cgs.value )) # set time = 0 when explosion goes off
        #else: np.savetxt('viscoustests/noearlytimeextrap/precutfallback/times+lums'+'{:03d}'.format(self.testnum)+'g'+str(gammas[0])+'b'+str(self._b)+'.txt',
        # ((tnew-self.rest_t_explosion), kwargs['efficiency']*dmdt*c.c.cgs.value*c.c.cgs.value ))
        # ----------------------------------------

        #timeinterpfunc = CubicSpline(tnew, dmdt)
        #print ('length tnew, length dmdt :',len(tnew), len(dmdt))
        timeinterpfunc = interp1d(tnew, dmdt)

        lengthpretimes = len(np.where(self._times < tnew[0])[0])
        lengthposttimes = len(np.where(self._times > tnew[-1])[0])

        # this removes all extrapolation by setting dmdtnew = 0 outside of bounds of tnew
        dmdt1 = np.zeros(lengthpretimes)
        dmdt3 = np.zeros(lengthposttimes)
        dmdt2 = timeinterpfunc(self._times[lengthpretimes:len(self._times)-lengthposttimes])
        dmdtnew = np.append(dmdt1,dmdt2)
        dmdtnew = np.append(dmdtnew, dmdt3)

        dmdtnew[dmdtnew < 0] = 0 # set floor for dmdt. At some point maybe fit to time of peak somewhere in here?

        self._efficiency = kwargs['efficiency']
        luminosities = self._efficiency*dmdtnew*c.c.cgs.value*c.c.cgs.value # expected in cgs so ergs/s

        # ----------------TESTING ----------------
        if self.TESTING == True:
            np.savetxt('test_dir/test_fallback/endfallback/time+dmdt'+'{:03d}'.format(self.testnum)+'.txt',
                        (self._times, dmdtnew)) # set time = 0 when explosion goes off
            self.testnum += 1
        
        # ----------------------------------------

        return {'dense_luminosities': luminosities}
