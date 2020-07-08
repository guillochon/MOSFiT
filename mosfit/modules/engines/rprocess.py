"""Definitions for the `RProcess` class."""
from math import isnan

import numpy as np
from astrocats.catalog.source import SOURCE
from mosfit.constants import C_CGS, DAY_CGS, IPI, KM_CGS, M_SUN_CGS
from mosfit.modules.engines.engine import Engine
from scipy.interpolate import RegularGridInterpolator


# Important: Only define one ``Module`` class per file.


class RProcess(Engine):
    """r-process decay engine.

    input luminosity adapted from Metzger 2016: 2017LRR....20....3M
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2013ApJ...775...18B'},
        {SOURCE.BIBCODE: '2017LRR....20....3M'},
        {SOURCE.BIBCODE: '2017arXiv170708132V'}
    ]

    ckm = C_CGS / KM_CGS

    def __init__(self, **kwargs):
        """Initialize module."""
        super(RProcess, self).__init__(**kwargs)
        self._wants_dense = True
        barnes_v = np.asarray([0.1, 0.2, 0.3, 0.4])
        barnes_M = np.asarray([1.e-3, 5.e-3, 1.e-2, 5.e-2, 1.e-1])
        barnes_a = np.asarray([[2.01, 4.52, 8.16, 16.3], [0.81, 1.9, 3.2, 5.0],
                              [0.56, 1.31, 2.19, 3.0], [.27, .55, .95, 2.0],
                              [0.20, 0.39, 0.65, 0.9]])
        barnes_b = np.asarray([[0.28, 0.62, 1.19, 2.4], [0.19, 0.28, 0.45, 0.65],
                              [0.17, 0.21, 0.31, 0.45], [0.10, 0.13, 0.15, 0.17],
                              [0.06, 0.11, 0.12, 0.12]])
        barnes_d = np.asarray([[1.12, 1.39, 1.52, 1.65], [0.86, 1.21, 1.39, 1.5],
                              [0.74, 1.13, 1.32, 1.4], [0.6, 0.9, 1.13, 1.25],
                              [0.63, 0.79, 1.04, 1.5]])

        self.therm_func_a = RegularGridInterpolator(
            (barnes_M, barnes_v), barnes_a, bounds_error=False, fill_value=None)
        self.therm_func_b = RegularGridInterpolator(
            (barnes_M, barnes_v), barnes_b, bounds_error=False, fill_value=None)
        self.therm_func_d = RegularGridInterpolator(
            (barnes_M, barnes_v), barnes_d, bounds_error=False, fill_value=None)

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        self._mass = kwargs[self.key('mejecta')] * M_SUN_CGS
        self._rest_texplosion = kwargs[self.key('resttexplosion')]
        self._vejecta = kwargs[self.key('vejecta')]
        self._a = self.therm_func_a(
            [self._mass / M_SUN_CGS, self._vejecta / self.ckm])[0]
        self._bx2 = 2.0 * self.therm_func_b(
            [self._mass / M_SUN_CGS, self._vejecta / self.ckm])[0]
        self._d = self.therm_func_d(
            [self._mass / M_SUN_CGS, self._vejecta / self.ckm])[0]

        ts = [
            np.inf
            if self._rest_texplosion > x else (x - self._rest_texplosion)
            for x in self._times
        ]

        self._lscale = self._mass * 4.0e18 * 0.36
        luminosities = [
            self._lscale * (0.5 - IPI * np.arctan(
                (t * DAY_CGS - 1.3) / 0.11)) ** 1.3 *
            (np.exp(-self._a * t) + np.log1p(
                self._bx2 * t ** self._d) / (self._bx2 * t ** self._d))
            for t in ts
        ]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        return {self.dense_key('luminosities'): luminosities}
