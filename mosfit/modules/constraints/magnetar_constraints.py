"""Definitions for the `MagnetarConstraints` class."""
import numpy as np
from mosfit.constants import DAY_CGS, KM_CGS, LIKELIHOOD_FLOOR, M_SUN_CGS
from mosfit.modules.constraints.constraint import Constraint


# Important: Only define one ``Module`` class per file.


class MagnetarConstraints(Constraint):
    """Magnetar constraints.

    Kinetic energy cannot excede magnetar rotational energy
    """

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0
        self._Pspin = kwargs['Pspin']
        self._Mns = kwargs['Mns']
        self._mejecta = kwargs['mejecta'] * M_SUN_CGS
        self._vejecta = kwargs['vejecta'] * KM_CGS
        self._times = kwargs['all_times']
        self._t_explosion = kwargs['texplosion']
        self._lums = kwargs['luminosities']
        self._redshift = kwargs['redshift']

        # Magnetar rotational energy
        self._Ep = 2.6e52 * (self._Mns / 1.4) ** (3. /
                                                  2.) * self._Pspin ** (-2)

        # Ejecta kinetic energy
        self._Ek = 0.5 * self._mejecta * self._vejecta**2

        # Construct array of rest-frame times since explosion
        norm_times = (self._times - self._t_explosion) / (1.0 + self._redshift)

        # Shift array to get delta_t between observations
        shift_times = norm_times[:-1]
        shift_times = np.insert(shift_times, 0, 0.0)

        norm_times[norm_times < 0] = 0.0
        shift_times[shift_times < 0] = 0.0

        L_arr = np.array(self._lums)

        # integrate bolometric light curve to find radiative losses
        E_rad = sum(L_arr * (norm_times - shift_times) * DAY_CGS)

        # Kinetic energy < magnetar energy - radiative loss + neutrinos (10^51)
        if (self._Ek > self._Ep - E_rad + 1.e51):
            self._score_modifier += LIKELIHOOD_FLOOR

        return {'score_modifier': self._score_modifier}
