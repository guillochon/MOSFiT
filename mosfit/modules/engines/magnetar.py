"""Definitions for the `Magnetar` class."""
from math import isnan

import numpy as np

from mosfit.constants import DAY_CGS
from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class Magnetar(Engine):
    """Magnetar spin-down engine."""

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs['dense_times']
        self._Pspin = kwargs['Pspin']
        self._Bfield = kwargs['Bfield']
        self._Mns = kwargs['Mns']
        self._thetaPB = kwargs['thetaPB']
        self._rest_t_explosion = kwargs['resttexplosion']

        Ep = 2.6e52 * (self._Mns / 1.4) ** (3. / 2.) * self._Pspin ** (-2)
        # ^ E_rot = 1/2 I (2pi/P)^2, unit = erg

        tp = 1.3e5 * self._Bfield ** (-2) * self._Pspin ** 2 * (
            self._Mns / 1.4) ** (3. / 2.) * (np.sin(self._thetaPB)) ** (-2)
        # ^ tau_spindown = P/(dP/dt), unit = s
        # Magnetic dipole: power = 2/(3c^3)*(R^3 Bsin(theta))^2 * (2pi/P)^4
        # Set equal to -d/dt(E_rot) to derive tau

        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]

        luminosities = [2 * Ep / tp / (
            1. + 2 * t * DAY_CGS / tp) ** 2 for t in ts]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]
        # ^ From Ostriker and Gunn 1971 eq 4

        # Add on to any existing luminosity
        luminosities = self.add_to_existing_lums(luminosities)

        return {self.dense_key('luminosities'): luminosities}
