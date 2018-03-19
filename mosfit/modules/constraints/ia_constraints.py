"""Definitions for the `IaConstraints` class."""
from mosfit.constants import KM_CGS, M_P_CGS, M_SUN_CGS, MEV_CGS
from mosfit.modules.constraints.constraint import Constraint


# Important: Only define one ``Module`` class per file.


class IaConstraints(Constraint):
    """Ia constraints.

    1. Kinetic energy cannot excede nuclear burning release
    """

    _REFERENCES = []

    def __init__(self, **kwargs):
        """Initialize module."""
        super(IaConstraints, self).__init__(**kwargs)
        self._wants_dense = True

        self._excess_constant = -(
            56.0 / 4.0 * 2.4249 - 53.9037) / M_P_CGS * MEV_CGS

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0
        self._mejecta = kwargs[self.key('mejecta')] * M_SUN_CGS
        self._vejecta = kwargs[self.key('vejecta')] * KM_CGS
        self._fnickel = kwargs[self.key('fnickel')]

        # Maximum energy from burning, assumes pure Helium to Nickel.
        self._Emax = self._excess_constant * self._mejecta * self._fnickel

        # Ejecta kinetic energy, assuming <v_ej> at R/2.
        self._Ek = 0.5 * self._mejecta * (self._vejecta / 2.0) ** 2

        # Make sure kinetic energy < burning energy.
        if self._Ek > self._Emax:
            self._score_modifier -= ((
                self._Ek - self._Emax) / (2 * (0.1 * self._Emax))) ** 2

        return {self.key('score_modifier'): self._score_modifier}
