"""Definitions for the `RProcess` class."""
from math import isnan

import numpy as np
from astrocats.catalog.source import SOURCE

from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class RProcess(Engine):
    """r-process decay engine.

    input luminosity adapted from Metzger 2016: 2016arXiv161009381M

    For 'red' kilonovae, use kappa ~ 10.
    For 'blue' kilonovae, use kappa ~1.
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2016arXiv161009381M'}
    ]

    M_sun = 1.99e33

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs['dense_times']
        self._mass = kwargs['mejecta'] * self.M_sun
        self._rest_texplosion = kwargs['resttexplosion']

        ts = [
            np.inf
            if self._rest_texplosion > x else (x - self._rest_texplosion)
            for x in self._times
        ]
        luminosities = [
            self._mass * 4.0e18 * (0.5 - (1. / np.pi) * np.arctan(
                (t * 86400. - 1.3) / 0.11)) ** 1.3 * 0.36 *
            (np.exp(-0.56 * t) + np.log(1.0 + 2.0 * 0.17 * (t) ** 0.74) /
             (2.0 * 0.17 * (t) ** 0.74)) for t in ts
        ]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        return {self.dense_key('luminosities'): luminosities}
