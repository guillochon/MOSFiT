"""Definitions for the `Shock` class."""
from math import isnan
from astrocats.catalog.source import SOURCE

import numpy as np

from mosfit.constants import C_CGS, DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class Shock(Engine):
    """Cooling emission from shock-heated cocoon.

    Follows Piro and Kollmeier 2018.

    Shock heating can be turned off in bns model by setting shock_frac=0
    or cos_theta_cocoon=1

    Uses softmax (tanh) function to turn off cocoon for t > t_thin (no material left to cool)
    """

    DIFF_CONST = M_SUN_CGS / (FOUR_PI * C_CGS * KM_CGS)
    C_KMS = C_CGS / KM_CGS

    _REFERENCES = [
        {SOURCE.BIBCODE: '2018ApJ...855..103P'}
    ]

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._kappa = kwargs[self.key('kappa')]
        self._m_ejecta = kwargs[self.key('mejecta')]
        self._v_ejecta = kwargs[self.key('vejecta')]
        self.radius_star = kwargs[self.key('radius_star')]  # AU to cm

        # shocked fraction of the ejecta - set equal to 1 by default
                     
        self._shocked_fraction = kwargs[self.key('shock_frac')]
        # Shocked ejecta power law density profile
        self._s = kwargs[self.key('s')]
        # Shock breakout timescale in seconds
        # Radius where material is shocked by ejecta:
        # radius is a free parameter

        m_shocked = self._m_ejecta * self._shocked_fraction

        self._tau_diff = np.sqrt(self.DIFF_CONST * self._kappa *
                                 m_shocked / self._v_ejecta) / DAY_CGS

        t_thin = (self.C_KMS / self._v_ejecta)**0.5 * self._tau_diff

        L0 = (1/2) * (m_shocked * M_SUN_CGS *
                      self._v_ejecta * KM_CGS * self.radius_star / (self._tau_diff * DAY_CGS)**2)

        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]

        # tanh function added by MN to turn off cocoon emission smoothly
        # once all layers are optically thin (have radiated their energy)
        luminosities = [L0 * (t/self._tau_diff)**-(4/(self._s+2)) * (1 +
                                                                     np.tanh(t_thin-t))/2 for t in ts]

        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        return {self.dense_key('luminosities'): luminosities}
