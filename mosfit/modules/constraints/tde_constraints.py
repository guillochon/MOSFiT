"""Definitions for the `TDEConstraints` class."""
from mosfit.constants import M_SUN_CGS, C_CGS
from mosfit.modules.constraints.constraint import Constraint
import astropy.constants as c


class TDEConstraints(Constraint):
    """TDE constraints.

    1. rp > rs --> the pericenter radius must be less than the Schwarzschild
    radius or the bh will swallow the star whole (no disruption or flare)
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(TDEConstraints, self).__init__(**kwargs)
        self._max_modification = 3.0

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0
        self._rp = kwargs[self.key('rp')]  # already in cgs
        self._bhmass = kwargs['bhmass']
        self._Rs = c.G.cgs.value*self._bhmass*M_SUN_CGS/(C_CGS*C_CGS)

        # Pericenter radius is getting close to Schwarzschild radius
        if (self._Rs/self._rp > 0.1):
            # soft limit
            self._score_modifier += - (
                self._Rs/self._rp * self._max_modification)/(
                self._Rs/self._rp + self._max_modification)

        return {self.key('score_modifier'): self._score_modifier}
