"""Definitions for the `ExpPow` class."""
from math import isnan

import numpy as np

from mosfit.modules.engines.engine import Engine


# Important: Only define one `Module` class per file.


class ExpPow(Engine):
    """A simple analytical engine."""

    def process(self, **kwargs):
        """Process module."""
        if 'dense_times' in kwargs:
            self._times = kwargs['dense_times']
        else:
            self._times = kwargs['rest_times']
        self._alpha = kwargs['alpha']
        self._beta = kwargs['beta']
        self._t_peak = kwargs['tpeak']
        self._lum_scale = kwargs['lumscale']
        self._rest_t_explosion = kwargs['resttexplosion']

        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]

        luminosities = [
            self._lum_scale * (1.0 - np.exp(-t / self._t_peak))
            ** self._alpha * (t / self._t_peak) ** (-self._beta) for t in ts
        ]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        # Add on to any existing luminosity
        luminosities = self.add_to_existing_lums(luminosities)

        return {'kappagamma': kwargs['kappa'], 'luminosities': luminosities}
