from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS, C_CGS
from mosfit.modules.photospheres.photosphere import Photosphere

from scipy.interpolate import interp1d

class tde_photosphere(Photosphere):
    """Photosphere that expands/recedes as a power law of Mdot (or equivalently L (proportional to Mdot) ).
    """

    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    RAD_CONST = KM_CGS * DAY_CGS

    def process(self, **kwargs):
        kwargs = self.prepare_input('luminosities', **kwargs)
        self._times = np.array(kwargs['rest_times']) #np.array(kwargs['dense_times']) #kwargs['rest_times']

        #self._kappagamma = kwargs['kappagamma']
        self._Mh = kwargs['bhmass']
        self._Mstar = kwargs['starmass']
        self._l = kwargs['lphoto']
        self._Rph_0 = 10.0**(kwargs['Rph0']) # parameter is varied in logspace, kwargs['Rph_0'] = log10(Rph0)
        self._luminosities = np.array(kwargs['luminosities'])
        #self._beta = kwargs['beta'] # getting beta at this point in process is more complicated than expected bc
        # it can be a beta for a 4/3 - 5/3 combination. Can easily get 'b' -- scaled constant that is linearly related to beta
        # but beta itself is not well defined. -- what does this mean exactly? beta = rt/rp
        Rsolar = c.R_sun.cgs.value
        self._Rstar = kwargs['Rstar']*Rsolar


        # Assume solar metallicity for now
        kappa_t = 0.2*(1 + 0.74) # thompson opacity using solar metallicity
        tpeak = self._times[np.argmax(self._luminosities)]
        #print (np.where(self._luminosities <= 0),len(self._luminosities))
        ilumzero = len(np.where(self._luminosities <= 0)[0]) # index of first mass accretion
        # semi-major axis of material that accretes at time = tpeak --> shouldn't T be tpeak - tdisruption?
        a_p =(c.G.cgs.value * self._Mh * M_SUN_CGS * ((tpeak -
             self._times[ilumzero]) * DAY_CGS / np.pi)**2)**(1. / 3.)
        Ledd = (4 * np.pi * c.G.cgs.value * self._Mh * M_SUN_CGS *
                C_CGS / kappa_t)


        # semi-major axis of material that accretes at self._times, only calculate for times after first mass accretion
        a_t = (c.G.cgs.value * self._Mh * M_SUN_CGS * ((self._times[ilumzero:] -
             self._times[ilumzero]) * DAY_CGS / np.pi)**2)**(1. / 3.)
        
        #rp = (self._Mh/self._Mstar)**(1./3.) * self._Rstar/self._beta
        rphotmax = 2 * a_p #2*rp + 2*a_p

        #r_isco = 6 * c.G.cgs.value * self._Mh * M_SUN_CGS / (C_CGS * C_CGS) # Risco in cgs
        rphotmin = r_isco #2*rp #r_isco

        rphot = np.ones(ilumzero)*rphotmin # set rphot to minimum before mass starts accreting (when
        # the luminosity is zero)

        rphot = np.concatenate((rphot, (self._Rph_0 * a_t * self._luminosities[ilumzero:]/ Ledd)**self._l))
        rphot[rphot < rphotmin] = rphotmin
        rphot[rphot > rphotmax] = rphotmax

        Tphot = (self._luminosities / (rphot**2 * self.STEF_CONST))**0.25

        return {'radiusphot': rphot, 'temperaturephot': Tphot} # return sparse luminosities here
