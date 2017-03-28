"""Definitions for the `NickelCobalt` class."""
from math import isnan

import numpy as np

from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class NickelCobalt(Engine):
    """Nickel/Cobalt decay engine."""

    NI56_LUM = 6.45e43
    CO56_LUM = 1.45e43
    NI56_LIFE = 8.8
    CO56_LIFE = 111.3

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs['dense_times']
        self._mnickel = kwargs['fnickel'] * kwargs['mejecta']
        self._rest_t_explosion = kwargs['resttexplosion']

        # From 1994ApJS...92..527N
        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]
        luminosities = [
            self._mnickel * (self.NI56_LUM * np.exp(-t / self.NI56_LIFE) +
                             self.CO56_LUM * np.exp(-t / self.CO56_LIFE))
            for t in ts
        ]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        # Add on to any existing luminosity
        old_luminosities = kwargs.get('dense_luminosities', None)
        if old_luminosities is not None:
            luminosities = [
                x + y for x, y in zip(old_luminosities, luminosities)
            ]

        # Add on to any existing luminosity
        luminosities = self.add_to_existing_lums(luminosities)

        return {'dense_luminosities': luminosities}
