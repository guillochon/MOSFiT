"""Definitions for the `TDEConstraints` class."""
import astropy.constants as c

from mosfit.constants import C_CGS, M_SUN_CGS
from mosfit.modules.constraints.constraint import Constraint


class TDEConstraints(Constraint):
    """TDE constraints.

    1. rp > rs --> the pericenter radius must be greater than the Schwarzschild
    radius or the bh will swallow the star whole (no disruption or flare)
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(TDEConstraints, self).__init__(**kwargs)

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0
        self._rp = kwargs[self.key('rp')]  # already in cgs
        self._bhmass = kwargs['bhmass']
        self._Rs = (2 * c.G.cgs.value * self._bhmass * M_SUN_CGS /
                    (C_CGS * C_CGS))

        # Pericenter radius is getting close to Schwarzschild radius
        self._score_modifier -= 10.0 * (self._Rs / self._rp) ** 3

        return {self.key('score_modifier'): self._score_modifier}
