"""Definitions for the `RProcess` class."""
from math import isnan

import numpy as np
from astrocats.catalog.source import SOURCE
from scipy.interpolate import RegularGridInterpolator

from mosfit.modules.engines.engine import Engine

# Important: Only define one ``Module`` class per file.


class RProcess(Engine):
    """r-process decay engine.

    input luminosity adapted from Metzger 2016: 2016arXiv161009381M

    For 'red' kilonovae, use kappa ~ 10.
    For 'blue' kilonovae, use kappa ~1.
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2017LRR....20....3M'},
        {SOURCE.BIBCODE: '2017arXiv170708132V'}
    ]

    M_sun = 1.99e33
    c = 3.e5

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Engine, self).__init__(**kwargs)
        self._wants_dense = True
        barnes_v = np.asarray([0.1, 0.2, 0.3])
        barnes_M = np.asarray([1.e-3, 5.e-3, 1.e-2, 5.e-2])
        barnes_a = np.asarray([[2.01, 4.52, 8.16], [0.81, 1.9, 3.2], [0.56, 1.31, 2.19], [.27, .55, .95]])
        barnes_b = np.asarray([[0.28, 0.62, 1.19], [0.19, 0.28, 0.45], [0.17, 0.21, 0.31], [0.10, 0.13, 0.15]])
        barnes_d = np.asarray([[1.12, 1.39, 1.52], [0.86, 1.21, 1.39], [0.74, 1.13, 1.32], [0.6, 0.9, 1.13]])

        self.therm_func_a = RegularGridInterpolator((barnes_M, barnes_v), barnes_a,bounds_error=False,fill_value=None)
        self.therm_func_b = RegularGridInterpolator((barnes_M, barnes_v), barnes_b,bounds_error=False,fill_value=None)
        self.therm_func_d = RegularGridInterpolator((barnes_M, barnes_v), barnes_d,bounds_error=False,fill_value=None)



    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        self._mass = kwargs[self.key('mejecta')] * self.M_sun
        self._rest_texplosion = kwargs[self.key('resttexplosion')]
        self._vejecta = kwargs[self.key('vejecta')]
        self._a = self.therm_func_a([self._mass/self.M_sun,self._vejecta/self.c])[0]
        self._b = self.therm_func_b([self._mass/self.M_sun,self._vejecta/self.c])[0]
        self._d = self.therm_func_d([self._mass/self.M_sun,self._vejecta/self.c])[0]

        ts = [
            np.inf
            if self._rest_texplosion > x else (x - self._rest_texplosion)
            for x in self._times
        ]
        luminosities = [
            self._mass * 4.0e18 * (0.5 - (1. / np.pi) * np.arctan(
                (t * 86400. - 1.3) / 0.11)) ** 1.3 * 0.36 *
            (np.exp(-self._a * t) +
             np.log(1.0 + 2.0 * self._b *
                    (t) ** self._d) /
             (2.0 * self._b * (t) **
              self._d)) for t in ts
        ]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]


        return {self.dense_key('luminosities'): luminosities}
