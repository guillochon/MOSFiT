"""Definitions for the `TDEConstraints` class."""
import astropy.constants as c

import numpy as np
from mosfit.constants import C_CGS, M_SUN_CGS, DAY_CGS
from mosfit.modules.constraints.constraint import Constraint


class TDEConstraints(Constraint):
    """TDE constraints.

    1. rp > rs --> the pericenter radius must be greater than the Schwarzschild
    radius or the bh will swallow the star whole (no disruption or flare)

    2. rphot < max photosphere. max photosphere size determined either by 
    apocenter of orbit of debris (old default) or by a relativistic wind 
    launched at tfallback (current default)
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(TDEConstraints, self).__init__(**kwargs)

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0
        self._rp = kwargs[self.key('rp')]  # already in cgs
        self._bhmass = kwargs['bhmass']
        self._Rs = (2* c.G.cgs.value * self._bhmass * M_SUN_CGS /
                    (C_CGS * C_CGS))

        # Pericenter radius is getting close to Schwarzschild radius
        # if (self._Rg / self._rp > 0.1):
        #       soft limit
        #    self._score_modifier -= 1000.0**(10.0*(self._Rg / self._rp - 0.9))

        self._score_modifier -= 10.0 * (self._Rs / self._rp) ** 3

        # constraints on the maximum photosphere size
        self._rphotmaxwind = kwargs['rphotmaxwind']
        self._vphotmaxwind = kwargs['vphotmaxwind']
        self._times = np.array(kwargs['rest_times'])
        self._rest_t_explosion = kwargs['resttexplosion']
        self._radius_phot = np.array(kwargs['radiusphot'])
        self._rphotmin = np.array(kwargs['rphotmin'])

        # semi-major axis of material that accretes at self._times,
        # only calculate for times after first mass accretion
        a_t = (c.G.cgs.value * self._bhmass * M_SUN_CGS * ((
            self._times - self._rest_t_explosion) * DAY_CGS / np.pi)**2)**(
                1. / 3.)
        a_t[self._times < self._rest_t_explosion] = 0.0

        if self._rphotmaxwind:
            # assume wind is launched from circularization radius at first light
            # assume rphotmax at early times is apocenter of debris, then at later times is wind radius
            # adding allows for smooth transition from minimum radius = circularization radius to minimum radius = apocenter of debris to minimum radius = wind radius
            self._rphotmax = 2*self._rp + self._vphotmaxwind * C_CGS * (
            self._times - self._rest_t_explosion) * DAY_CGS + 2 * a_t #self._rp + 2 * a_t 
            self._rphotmax[self._times < self._rest_t_explosion] = 0.0
        else:
            self._rphotmax = self._rp + 2 * a_t  # rphotmax set to apocenter of debris stream

        # if radius_phot = rphotmax, score_modifier == -10
        self._score_modifier -= 10.0 * (np.max(self._radius_phot/self._rphotmax)) ** 2


        return {self.key('score_modifier'): self._score_modifier}