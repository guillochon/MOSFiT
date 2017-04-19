"""Definitions for the `MagnetarConstraints` class."""
import numpy as np
from mosfit.constants import DAY_CGS, KM_CGS, M_SUN_CGS
from mosfit.modules.constraints.constraint import Constraint


# Important: Only define one ``Module`` class per file.


class MagnetarConstraints(Constraint):
    """Magnetar constraints.

    Kinetic energy cannot excede magnetar rotational energy
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(MagnetarConstraints, self).__init__(**kwargs)
        self._wants_dense = True

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0
        self._Pspin = kwargs[self.key('Pspin')]
        self._Mns = kwargs[self.key('Mns')]
        self._mejecta = kwargs[self.key('mejecta')] * M_SUN_CGS
        self._vejecta = kwargs[self.key('vejecta')] * KM_CGS
        self._times = kwargs[self.key('dense_times')]
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._lums = kwargs[self.key('dense_luminosities')]
        self._neutrino_energy = kwargs[self.key('neutrino_energy')]

        # Magnetar rotational energy
        self._Ep = 2.6e52 * (self._Mns / 1.4) ** (3. /
                                                  2.) * self._Pspin ** (-2)

        # Ejecta kinetic energy
        self._Ek = 0.5 * self._mejecta * self._vejecta ** 2

        # Construct array of rest-frame times since explosion
        norm_times = self._times - self._rest_t_explosion

        # Shift array to get delta_t between observations
        shift_times = norm_times[:-1]
        shift_times = np.insert(shift_times, 0, 0.0)

        norm_times[norm_times < 0] = 0.0
        shift_times[shift_times < 0] = 0.0

        L_arr = np.array(self._lums)

        # integrate bolometric light curve to find radiative losses
        E_rad = sum(L_arr * (norm_times - shift_times) * DAY_CGS)

        # Kinetic energy < magnetar energy - radiative loss + neutrinos (10^51)
        if (self._Ek > self._Ep - E_rad + self._neutrino_energy):
            self._score_modifier += -(
                self._Ek - (self._Ep - E_rad + self._neutrino_energy)) ** 2 / (
                    2 * self._neutrino_energy ** 2)

        return {self.key('score_modifier'): self._score_modifier}
