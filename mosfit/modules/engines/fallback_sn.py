"""Definitions for the `Fallback` class."""
from math import isnan

import numpy as np

from mosfit.constants import DAY_CGS
from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class Fallback(Engine):
    """Black hole accretion engine."""

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        self._L0 = kwargs[self.key('L_init')]
        # self._t0 = kwargs[self.key('t_fb')] * DAY_CGS
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]


        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]

        luminosities = [self._L0 * (t * DAY_CGS) ** (-5./3.) for t in ts]

        # luminosities = [self._L0 * (t * DAY_CGS / self._t0) ** (-5./3.) for t in ts]
        # ^ From Ostriker and Gunn 1971 eq 4
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        return {self.dense_key('luminosities'): luminosities}
