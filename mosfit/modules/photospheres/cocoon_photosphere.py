"""Definitions for the `CocoonPhotosphere` class."""
import numpy as np
from astrocats.catalog.source import SOURCE
from astropy import constants as c

from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, C_CGS, M_SUN_CGS
from mosfit.modules.photospheres.photosphere import Photosphere


# Important: Only define one ``Module`` class per file.


class CocoonPhotosphere(Photosphere):
    """
    Piro and Kollmeier 2018
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2018ApJ...855..103P'}
    ]

    DIFF_CONST = M_SUN_CGS / (FOUR_PI * C_CGS * KM_CGS)
    STEF_CONST = (FOUR_PI * c.sigma_sb).cgs.value
    RAD_CONST = KM_CGS * DAY_CGS
    C_KMS = C_CGS / KM_CGS

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._times = kwargs[self.key('rest_times')]
        self._luminosities = kwargs[self.key('luminosities')]
        self._v_ejecta = kwargs[self.key('vejecta')]
        self._m_ejecta = kwargs[self.key('mejecta')]
        self._kappa = kwargs[self.key('kappa')]
        self._shocked_fraction = kwargs[self.key('shock_frac')]

        m_shocked = self._m_ejecta * self._shocked_fraction

        self._tau_diff = np.sqrt(self.DIFF_CONST * self._kappa *
                                 m_shocked / self._v_ejecta) / DAY_CGS

        t_thin = (C_KMS / self._v_ejecta)**0.5 * self._tau_diff


        rphot = []
        Tphot = []
        for li, lum in enumerate(self._luminosities):

            ts = self._times[li] - self._rest_t_explosion

            vphot = self._v_ejecta * (ts/t_thin)**(-2./(s+3))
            
            radius = RAD_CONST * vphot * max(ts, 0.0)
            
            if lum == 0.0:
                temperature = 0.0
            else:
                temperature = (lum / (self.STEF_CONST * radius**2)) ** 0.25

            rphot.append(radius)

            Tphot.append(temperature)

        return {self.key('radiusphot'): rphot,
                self.key('temperaturephot'): Tphot
        }
