"""Definitions for the `MagnetarConstraints` class."""
import numpy as np
from scipy import interpolate

from mosfit.constants import KM_CGS, LIKELIHOOD_FLOOR, M_SUN_CGS, DAY_CGS
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

        self._Ep = 2.6e52 * (self._Mns / 1.4) ** (3. / 2.) * self._Pspin ** (-2)

        # print('Ep',self._Pspin,self._Mns,self._Ep)

        self._Ek = 0.5 * self._mejecta * self._vejecta**2

        # print('Ek',self._mejecta/M_SUN_CGS,self._vejecta/KM_CGS,self._Ek)


        norm_times = (self._times - self._t_explosion) / (1.0 + self._redshift)

        shift_times = norm_times[:-1]
        shift_times = np.insert(shift_times,0,0.)

        norm_times[norm_times<0] = 0
        shift_times[shift_times<0] = 0

        L_arr = np.array(self._lums)

        # integrate bolometric light curve
        E_rad = sum(L_arr*(norm_times-shift_times)*DAY_CGS)

        # Kinetic energy < magnetar energy - radiative losses + neutrinos
        if (self._Ek > self._Ep - E_rad + 1.e51):
            self._score_modifier += LIKELIHOOD_FLOOR

        return {'score_modifier': self._score_modifier}
