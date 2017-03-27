import numexpr as ne
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from mosfit.constants import C_CGS, FOUR_PI, KM_CGS, M_SUN_CGS, DAY_CGS
from mosfit.modules.transforms.transform import Transform

CLASS_NAME = 'Viscous'


class Viscous(Transform):
    """Photon diffusion transform.
    """
    logsteps = True
    N_INT_TIMES = 5 * 10000
    testnum = 0

    def process(self, **kwargs):
        #print ('running viscous.py')
        self.set_times_lums(**kwargs)
        self._kappa = kwargs['kappa']
        #self._kappa_gamma = kwargs['kappagamma']
        #self._v_ejecta = kwargs['vejecta']
        #self._m_ejecta = kwargs['mejecta']
        ipeak = np.argmax(self._dense_luminosities)
        lum_peak = self._dense_luminosities[ipeak]
        tpeak = self._dense_times_since_exp[ipeak]
        Tvisc = kwargs['Tviscous'] * tpeak


        new_lum = []
        evaled = False
        lum_cache = {}
        lum_func = interp1d(self._dense_times_since_exp, self._dense_luminosities) #, fill_value = 'extrapolate')
        sparse_luminosities = lum_func(self._times_since_exp)
        timesteps = self.N_INT_TIMES
        addlums = False
        nummatch = int(0.02*len(self._dense_times_since_exp)) # this is kinda random right now...
        if nummatch < 10 : nummatch = 10 # have to match at least 10 pts
        min_te = min(self._dense_times_since_exp)
        #print (self._dense_times_since_expmin_te)
        lumzero = True

        # ----------------TESTING ----------------
        #np.savetxt('viscoustests/noearlytimeextrap/previscous/densetimes+lums_runnum'+'{:03d}'.format(self.testnum)+'.txt', 
        #    (self._dense_times_since_exp, self._dense_luminosities) )
        #self.testnum += 1
        # ----------------------------------------
        #print ('length self._dense_times_since_exp =', len(self._dense_times_since_exp))
        #print ('length self._times_since_exp =', len(self._times_since_exp))
        for j,te in enumerate(self._dense_times_since_exp) : #enumerate(self._times_since_exp):
            if te <= 0.0: # self._times_since_exp should have resttexplosion at t= 0
                            # so the luminosity at t<0 should be zero
                new_lum.append(0.0)
                continue
            if lumzero == True:
                if  lum_func(te) <= 0:
                    new_lum.append(0.0)
                    continue
                else: te_zero = te # this is first te with nonzero lum value

            if te in lum_cache:
                new_lum.append(lum_cache[te])   
                continue

            lumzero = False # at least one nonzero luminosity term

            if te/Tvisc > timesteps*0.1 and timesteps < 1e5 : # might want to change first part of if statement
                #if te > tpeak: timesteps = timesteps*10
                timesteps = timesteps*10
                #else: print ('te/Tvisc > timesteps*0.1 and timesteps < 1e5 but te < tpeak')

            
            if self.logsteps == True: 
                tb = max(1e-4, min_te)
                int_times = np.logspace(np.log10(tb), np.log10(te), num = timesteps)
                if int_times[0] < self._dense_times_since_exp[0] : int_times[0] = self._dense_times_since_exp[0]
                if (int_times[-1] > self._dense_times_since_exp[-1]) : int_times[-1] = self._dense_times_since_exp[-1]
                #print (tb, te, int_times[-1])
            else:
                tb = max(0.0, min_te) # start at time = 0 if min time < 0, else start at min time
                int_times = np.linspace(tb, te, timesteps)
                dt = int_times[1] - int_times[0] # all the spacings are the same bc of linspace

  
            int_lums = lum_func(int_times)


            if not evaled:
                int_arg = ne.evaluate('exp((-te + int_times)/Tvisc) * int_lums')
                evaled = True
            else:
                int_arg = ne.re_evaluate()

            int_arg[np.isnan(int_arg)] = 0.0 # could also make cuts on luminosity here
            #print (int_arg)]
            
            #print (int_arg, np.shape(int_arg), dt, int_arg[0])
            #print (np.trapz(int_arg,dt))
            if self.logsteps == True: lum_val = np.trapz(int_arg, int_times)/Tvisc          
            else: lum_val = np.trapz(int_arg, dx = dt)/Tvisc
            
            #lum_cache[te] = lum_val
            new_lum.append(lum_val)

            # The following if statement tests whether the viscously delayed luminosities have converged to 
            # the old ones and if they have then it switches back to the old ones (doesn't do further integration)
            '''if te > tpeak and len(new_lum) > nummatch:
                if (np.abs(new_lum[-nummatch:]-sparse_luminosities[j-nummatch:j]/(
                    sparse_luminosities[j-nummatch:j] + 1.)) < .01).all() == True: # if the last 'nummatch' new 
                    # luminosities match the old ones --> viscously delayed lums have converged back to old light curve, 
                    # then just use old light curve for rest of points. Also + 1. in the denominator prevents divisions by zero
                    addlums = True
                    break
            '''
        
        if addlums == True:
            #notdenselums = lum_func(self._times_since_exp[j+1+i])
            #new_lum.extend(self._dense_luminosities[j+1:])
            new_lum.extend(sparse_luminosities[j+1:])
            # add additional lums to dictionary lum_cache here as well:
            # NEXT LINE CURRENTLY WRONG, INDEXES OF self._dense_luminosities DIFF FROM INDEXES OF sparse_luminosities
            for i in range(len(self._dense_luminosities[j+1:])):
                lum_cache[self._times_since_exp[j+1+i]] = sparse_luminosities[j+1+i] #lum_func(self._dense_luminosities[j+1+i])

        # ----------------TESTING ----------------
        
        #np.savetxt('viscoustests/noearlytimeextrap/postviscous/times+lums_runnum'+'{:03d}'.format(self.testnum)+'_Tvis'+str(kwargs['Tviscous'])+'.txt', 
        #    (self._dense_times_since_exp,new_lum) )
        #self.testnum += 1
        '''
        np.savetxt('viscoustests/noearlytimeextrap/postviscous/times+lums_runnum'+'{:03d}'.format(self.testnum)+'_Tvis'+str(kwargs['Tviscous'])+'.txt', 
            (self._times_since_exp,new_lum) )
        self.testnum += 1
        '''
        # ----------------------------------------
        
        new_func = interp1d(self._dense_times_since_exp, new_lum)
        new_lum = new_func(self._times_since_exp)
        for i,t in enumerate(self._times_since_exp):
            lum_cache[t] = new_lum[i]
        
        return {'luminosities': new_lum}
