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

    N_INT_TIMES = 1000
    #DIFF_CONST = 2.0 * M_SUN_CGS / (13.7 * C_CGS * KM_CGS)
    #TRAP_CONST = 3.0 * M_SUN_CGS / (FOUR_PI * KM_CGS**2)

    def process(self, **kwargs):
        #print ('running viscous.py')
        self.set_times_lums(**kwargs)
        self._kappa = kwargs['kappa']
        self._kappa_gamma = kwargs['kappagamma']
        self._m_ejecta = kwargs['mejecta']
        self._v_ejecta = kwargs['vejecta']
        ipeak = np.argmax(self._dense_luminosities)
        lum_peak = self._dense_luminosities[ipeak]
        tpeak = self._dense_times_since_exp[ipeak]
        Tvisc = kwargs['Tviscous'] * tpeak
        #self._tau_diff = np.sqrt(self.DIFF_CONST * self._kappa *
                                 #self._m_ejecta / self._v_ejecta) / DAY_CGS
        #self._trap_coeff = (self.TRAP_CONST * self._kappa_gamma *
                            #self._m_ejecta / (self._v_ejecta**2)) / DAY_CGS**2
        #td2, A = self._tau_diff**2, self._trap_coeff

        new_lum = []
        evaled = False
        lum_cache = {}
        lum_func = interp1d(self._dense_times_since_exp, self._dense_luminosities)
        sparse_luminosities = lum_func(self._times_since_exp)
        timesteps = self.N_INT_TIMES
        addlums = False
        nummatch = int(0.02*len(self._dense_times_since_exp)) # this is kinda random right now...
        if nummatch < 10 : nummatch = 10 # have to match at least 10 pts
        min_te = min(self._dense_times_since_exp)
        #j = 0
        lumzero = True
        #print (self._dense_luminosities)
        #print (self._dense_times_since_exp)
        for j,te in enumerate(self._times_since_exp):
            if te <= 0.0:
                new_lum.append(0.0)
                continue
            if te in lum_cache:
                new_lum.append(lum_cache[te])   
                continue
            if lumzero == True and lum_func(te) <= 0:
                new_lum.append(0.0)
                continue

            lumzero = False # at least one nonzero luminosity term

            if te/Tvisc > timesteps*0.1 and timesteps < 1e5 : # might want to change first part of if statement
                #if te > tpeak: timesteps = timesteps*10
                timesteps = timesteps*10
                #else: print ('te/Tvisc > timesteps*0.1 and timesteps < 1e5 but te < tpeak')

            tb = max(0.0, min_te) # start at time = 0 if min time < 0, else start at min time
            int_times = np.linspace(tb, te, timesteps)
            dt = int_times[1] - int_times[0] # all the spacings are the same bc of linspace

            int_lums = lum_func(int_times)

            #print (int_lums)
            #int_lums = np.interp(int_times, self._dense_times_since_exp,
                                 #self._dense_luminosities)

            if not evaled:
                int_arg = ne.evaluate('exp((-te + int_times)/Tvisc) * int_lums')
                evaled = True
            else:
                int_arg = ne.re_evaluate()

            int_arg[np.isnan(int_arg)] = 0.0 # could also make cuts on luminosity here
            #print (int_arg)]
            
            #print (int_arg, np.shape(int_arg), dt, int_arg[0])
            #print (np.trapz(int_arg,dt))
            lum_val = np.trapz(int_arg, dx = dt)
            lum_cache[te] = lum_val
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
            for i in range(len(self._dense_luminosities[j+1:])):
                lum_cache[self._times_since_exp[j+1+i]] = sparse_luminosities[j+1+i] #lum_func(self._dense_luminosities[j+1+i])


        return {'luminosities': new_lum}
