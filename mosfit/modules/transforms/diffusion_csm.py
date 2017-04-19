"""Definitions for the `DiffusionCSM` class."""
from collections import OrderedDict

import numexpr as ne
import numpy as np
from mosfit.constants import C_CGS, DAY_CGS, M_SUN_CGS, AU_CGS
from mosfit.modules.transforms.transform import Transform


# Important: Only define one ``Module`` class per file.


class DiffusionCSM(Transform):
    """Photon diffusion transform for CSM model."""

    N_INT_TIMES = 1000
    MIN_EXP_ARG = 50.0

    def process(self, **kwargs):
        """Process module."""
        super(DiffusionCSM, self).process(**kwargs)
        self._kappa = kwargs[self.key('kappa')]
        self._mass = kwargs[self.key('mcsm')] * M_SUN_CGS
        self._R0 = kwargs[self.key('r0')] * AU_CGS  # AU to cm
        self._s = kwargs[self.key('s')]
        self._rho = kwargs[self.key('rho')]
        # scaling constant for CSM density profile
        self._q = self._rho * self._R0 ** self._s
        # outer radius of CSM shell
        self._Rcsm = (
            (3.0 - self._s) / (4.0 * np.pi * self._q) * self._mass +
            self._R0 ** (3.0 - self._s)) ** (1.0 / (3.0 - self._s))
        # radius of photosphere (should be within CSM)
        self._Rph = abs(
            (-2.0 * (1.0 - self._s) / (3.0 * self._kappa * self._q) +
             self._Rcsm ** (1.0 - self._s)) ** (1.0 / (1.0 - self._s)))
        self._tau_diff = (
            self._kappa * self._mass) / (13.8 * C_CGS * self._Rph) / DAY_CGS

        tbarg = self.MIN_EXP_ARG * self._tau_diff ** 2
        new_lum = np.zeros_like(self._times_to_process)
        evaled = False
        lum_cache = OrderedDict()
        min_te = min(self._dense_times_since_exp)
        for ti, te in enumerate(self._times_to_process):
            if te <= 0.0:
                continue
            if te in lum_cache:
                new_lum[ti] = lum_cache[te]
                continue
            te2 = te ** 2
            tb = max(np.sqrt(max(te2 - tbarg, 0.0)), min_te)
            int_times = np.linspace(tb, te, self.N_INT_TIMES)
            dt = int_times[1] - int_times[0]
            td = self._tau_diff

            int_lums = np.interp(int_times, self._dense_times_since_exp,
                                 self._dense_luminosities)

            if not evaled:
                int_arg = ne.evaluate('int_lums * int_times / td**2 * '
                                      'exp((int_times - te) / td)')
                evaled = True
            else:
                int_arg = ne.re_evaluate()

            int_arg[np.isnan(int_arg)] = 0.0
            lum_val = np.trapz(int_arg, dx=dt)
            lum_cache[te] = lum_val
            new_lum[ti] = lum_val
        return {self.dense_key('luminosities'): new_lum}
