"""Definitions for the `CSMConstraints` class."""
import numpy as np
from scipy import interpolate

from mosfit.constants import KM_CGS, LIKELIHOOD_FLOOR, M_SUN_CGS
from mosfit.modules.constraints.constraint import Constraint


# Important: Only define one ``Module`` class per file.


class CSMConstraints(Constraint):
    """CSM constraints.

    1. R0 <= Rph <= Rcsm. The photospheric radius is within the CSM
    2. td < ts. The diffusion time is less than the shock crossing time.
    """

    def process(self, **kwargs):
        """Process module. Add constraints below."""
        self._score_modifier = 0.0

        self._n = kwargs[self.key('n')]
        self._delta = kwargs[self.key('delta')]
        self._mejecta = kwargs[self.key('mejecta')] * M_SUN_CGS
        self._vejecta = kwargs[self.key('vejecta')] * KM_CGS
        self._kappa = kwargs[self.key('kappa')]
        self._rho = kwargs[self.key('rho')]
        self._r0 = kwargs[self.key('r0')] * 1.496e13  # AU to cm
        self._s = kwargs[self.key('s')]
        self._mcsm = kwargs[self.key('mcsm')] * M_SUN_CGS
        self._Esn = 3. * self._vejecta ** 2 * self._mejecta / 10.

        if self._s == 0:
            ns = [6, 7, 8, 9, 10, 12, 14]
            Bfs = [1.256, 1.181, 1.154, 1.140, 1.131, 1.121, 1.116]
            As = [2.4, 1.2, 0.71, 0.47, 0.33, 0.19, 0.12]

        else:  # s == 2
            ns = [6, 7, 8, 9, 10, 12, 14]
            Bfs = [1.377, 1.299, 1.267, 1.250, 1.239, 1.226, 1.218]
            As = [0.62, 0.27, 0.15, 0.096, 0.067, 0.038, 0.025]

        Bf_func = interpolate.interp1d(ns, Bfs)
        A_func = interpolate.interp1d(ns, As)

        beta_f = Bf_func(self._n)
        A = A_func(self._n)

        self._gn = (1.0 / (4.0 * np.pi * (self._n - self._delta)) * (
            2.0 * (5.0 - self._delta) * (self._n - 5.0) * self._Esn) ** (
                (self._n - 3.) / 2.0) / (
                    (3.0 - self._delta) * (self._n - 3.0) * self._mejecta) ** (
                        (self._n - 5.0) / 2.0)
        )  # g ** n is scaling parameter for ejecta density profile
        self._q = self._rho * self._r0 ** self._s
        self._R_csm = ((3. - self._s) / (4. * np.pi * self._q) *
                       self._mcsm + self._r0 ** (3. - self._s)) ** (
                           1. / (3. - self._s))
        self._R_ph = (self._R_csm ** (1. - self._s) - 2. * (1. - self._s) /
                      (3. * self._kappa * self._q)) ** (1. / (1. - self._s))
        # mass of the optically thick CSM (tau > 2/3)
        self._Mcsm_th = 4.0 * np.pi * self._q / (3.0 - self._s) * (
            self._R_ph ** (3.0 - self._s) - self._r0 ** (3.0 - self._s))
        self._ts = (
            (self._R_csm - self._r0) / beta_f / (A * self._gn / self._q) ** (
                1. / (self._n - self._s))) ** ((self._n - self._s) /
                                               (self._n - 3))
        self._td = np.sqrt(2. * self._kappa * self._Mcsm_th /
                           (self._vejecta * 13.7 * 3.e10))
        # Constraint 1: R0<= Rph <= Rcsm
        if (self._R_csm < self._R_ph) | (self._r0 > self._R_ph):
            self._score_modifier += LIKELIHOOD_FLOOR

        # Constraint 2: td < ts
        if (self._ts < self._td):
            self._score_modifier += LIKELIHOOD_FLOOR
        return {self.key('score_modifier'): self._score_modifier}
