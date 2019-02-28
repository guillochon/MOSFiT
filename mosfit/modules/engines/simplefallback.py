"""Definitions for the `Simplefallback` class."""
from math import isnan

import numpy as np
from astrocats.catalog.source import SOURCE

from mosfit.constants import DAY_CGS
from mosfit.modules.engines.engine import Engine

# Important: Only define one ``Module`` class per file.


class Simplefallback(Engine):
    """
    Simple fallback energy input.

    Flat energy at first and then proportional to t**(-5/3).
    """

    _REFERENCES = [{SOURCE.BIBCODE: '2013ApJ...772...30D'}]

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        self._L0 = kwargs[self.key('Lat1sec')]
        self._ton1 = kwargs[self.key('ton')]
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]

        ts = [
            np.inf if self._rest_t_explosion > x else
            (x - self._rest_t_explosion) for x in self._times
        ]

        luminosities = [
            self._L0 / (t * DAY_CGS)**(5. / 3.) if t - self._ton1 > 0 else
            self._L0 / (self._ton1 * DAY_CGS)**(5. / 3.) for t in ts
        ]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        return {self.dense_key('luminosities'): luminosities}
