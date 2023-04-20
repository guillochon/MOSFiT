"""Definitions for the `NSBHConstraints` class."""
import astropy.constants as c
import numpy as np

from mosfit.constants import C_CGS, M_SUN_CGS, KM_CGS, G_CGS
from mosfit.modules.constraints.constraint import Constraint

# G_CGS = c.G.cgs.value


class NSBHConstraints(Constraint):
    """NSBH constraints.

    1. M1 > Mtov
    2. M2 < Mtov
    3. M2 > 0.8 Msun
    4. 9 < R_ns < 16
    5. Causality relation relating Mtov and radius

    Free parameters are Mchirp and q. Thus for large q, M1 can be larger or M2
    smaller than allowed by NS EoS. Constraint penalises masses outside range

    Realistic EoS prevents extreme values of radius
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(NSBHConstraints, self).__init__(**kwargs)

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0

        # Mass of BH
        self._Mbh = kwargs[self.key('M1')]
        # Mass of NS
        self._Mns = kwargs[self.key('M2')]
        self._m_tov = kwargs[self.key('Mtov')]
        self._Rns = kwargs[self.key('radius_ns')]

        # Soft max/min, proportional to diff^2 and scaled to -100 for 0.1 Msun
        # 1
        if self._Mbh < self._m_tov:
            self._score_modifier -= (100. * (self._m_tov-self._Mbh))**2
        
        # 2
        if self._Mns > self._m_tov:
            self._score_modifier -= (100. * (self._Mns-self._m_tov))**2

        # 3
        if self._Mns < 0.8:
            self._score_modifier -= (100. * (0.8-self._Mns))**2

        # 4
        if self._Rns > 16:
            self._score_modifier -= (20. * (self._Rns-16))**2

        if self._Rns < 9:
            self._score_modifier -= (20. * (9-self._Rns))**2


        # 5
        Mcaus = 1/2.82 * C_CGS**2 * self._Rns * KM_CGS / G_CGS / M_SUN_CGS

        if self._m_tov > Mcaus:
            self._score_modifier -= (100. * (self._m_tov-Mcaus))**2

        return {self.key('score_modifier'): self._score_modifier}
