"""Definitions for the `CSM` class."""
from math import isnan

import numpy as np
from astrocats.catalog.source import SOURCE
from scipy import interpolate

from mosfit.constants import AU_CGS, DAY_CGS, M_SUN_CGS
from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class CSM(Engine):
    """CSM energy injection.

    input luminosity calculation based on
    http://adsabs.harvard.edu/abs/2012ApJ...746..121C
    with coefficients from
    http://adsabs.harvard.edu/abs/1982ApJ...258..790C

    There are two major changes in the input luminosity from Chatzopoulos,
    Wheeler & Vinko (2012):
    1. ti is set to a small number, rather than the time it takes the
    ejecta to reach the csm shell
    2. you can fit/choose an efficiency factor between KE and luminosity
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2012ApJ...746..121C'},
        {SOURCE.BIBCODE: '2013ApJ...773...76C'}
    ]

    def process(self, **kwargs):
        """Process module."""
        self._s = kwargs[self.key('s')]
        self._delta = kwargs[self.key('delta')]  # [0,3)
        self._n = kwargs[self.key('n')]  # [6,10]
        self._kappa = kwargs[self.key('kappa')]
        self._R0 = kwargs[self.key('r0')] * AU_CGS  # AU to cm
        self._mejecta = kwargs[self.key('mejecta')] * M_SUN_CGS  # Msol to grms
        self._mcsm = kwargs[self.key('mcsm')] * M_SUN_CGS
        self._rho = kwargs[self.key('rho')]
        self._vph = kwargs[self.key('vejecta')] * 1.e5
        self._Esn = 3. * self._vph**2 * self._mejecta / 10.
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._efficiency = kwargs[self.key('efficiency')]
        self._times = kwargs[self.key('dense_times')]

        # g**n is scaling parameter for ejecta density profile
        self._g_n = (1.0 / (4.0 * np.pi * (self._n - self._delta)) * (
            2.0 * (5.0 - self._delta) * (self._n - 5.0) * self._Esn)**(
                (self._n - 3.) / 2.0) / (
                    (3.0 - self._delta) * (self._n - 3.0) * self._mejecta)**(
                        (self._n - 5.0) / 2.0))

        self._ti = 1.0  # set ti to small number

        if self._s == 0:
            ns = [6, 7, 8, 9, 10, 12, 14]
            Bfs = [1.256, 1.181, 1.154, 1.140, 1.131, 1.121, 1.116]
            Brs = [0.906, 0.935, 0.950, 0.960, 0.966, 0.974, 0.979]
            As = [2.4, 1.2, 0.71, 0.47, 0.33, 0.19, 0.12]

        else:  # s == 2
            ns = [6, 7, 8, 9, 10, 12, 14]
            Bfs = [1.377, 1.299, 1.267, 1.250, 1.239, 1.226, 1.218]
            Brs = [0.958, 0.970, 0.976, 0.981, 0.984, 0.987, 0.990]
            As = [0.62, 0.27, 0.15, 0.096, 0.067, 0.038, 0.025]

        Bf_func = interpolate.interp1d(ns, Bfs)
        Br_func = interpolate.interp1d(ns, Brs)
        A_func = interpolate.interp1d(ns, As)

        self._Bf = Bf_func(self._n)
        self._Br = Br_func(self._n)
        self._A = A_func(self._n)

        # scaling constant for CSM density profile.
        self._q = self._rho * self._R0**self._s

        # outer radius of CSM shell.
        self._Rcsm = (
            ((3.0 - self._s) /
             (4.0 * np.pi * self._q) * self._mcsm + self._R0 ** (
                 3.0 - self._s)) ** (1.0 / (3.0 - self._s)))

        # radius of photosphere (should be within CSM).
        self._Rph = abs(
            (-2.0 * (1.0 - self._s) /
             (3.0 * self._kappa * self._q) + self._Rcsm**(1.0 - self._s)) **
            (1.0 /
             (1.0 - self._s)))

        # mass of the optically thick CSM (tau > 2/3).
        self._Mcsm_th = np.abs(4.0 * np.pi * self._q / (3.0 - self._s) * (
            self._Rph**(3.0 - self._s) - self._R0 **
            (3.0 - self._s)))

        # time at which shock breaks out of optically thick CSM - forward shock
        # power input then terminates.
        self._t_FS = (
            abs((3.0 - self._s) * self._q**(
                (3.0 - self._n) / (self._n - self._s)) * (
                    self._A * self._g_n) ** ((self._s - 3.0) / (
                        self._n - self._s)) /
                (4.0 * np.pi * self._Bf**(3.0 - self._s)))**(
                    (self._n - self._s) / (
                        (self._n - 3.0) * (3.0 - self._s))) * (
                            self._Mcsm_th) ** (
                                (self._n - self._s) / (
                                    (self._n - 3.0) * (3.0 - self._s))))

        # time at which reverse shock sweeps up all ejecta - reverse shock
        # power input then terminates.
        self._t_RS = (
            self._vph / (self._Br * (self._A * self._g_n / self._q) ** (
                1.0 / (self._n - self._s))) *
            (1.0 - (3.0 - self._n) * self._mejecta /
             (4.0 * np.pi * self._vph **
              (3.0 - self._n) * self._g_n))**(1.0 / (3.0 - self._n))) ** (
                  (self._n - self._s) / (self._s - 3.0))

        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]

        luminosities = [
            self._efficiency * (2.0 * np.pi / (
                self._n - self._s) ** 3 * self._g_n ** (
                (5.0 - self._s) / (self._n - self._s)) * self._q**(
                    (self._n - 5.0) /
                    (self._n - self._s)) * (self._n - 3.0)**2 *
                (self._n - 5.0) * self._Bf**(5.0 - self._s) * self._A**(
                (5.0 - self._s) /
                (self._n - self._s)) * (t * DAY_CGS + self._ti)**(
                    (2.0 * self._n + 6.0 * self._s - self._n * self._s - 15.) /
                    (self._n - self._s)) * (
                        (self._t_FS - t * DAY_CGS) > 0) + 2.0 * np.pi *
                (self._A * self._g_n / self._q)**(
                (5.0 - self._n) /
                (self._n - self._s)) * self._Br**(5.0 - self._n) * self._g_n *
                ((3.0 - self._s) /
                 (self._n - self._s))**3 * (t * DAY_CGS + self._ti)**(
                (2.0 * self._n + 6.0 * self._s - self._n * self._s - 15.0) /
                (self._n - self._s)) * ((self._t_RS - t * DAY_CGS) > 0))
            for t in ts
        ]

        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        return {self.dense_key('luminosities'): luminosities,
                self.key('mcsmth'): self._Mcsm_th / M_SUN_CGS}
