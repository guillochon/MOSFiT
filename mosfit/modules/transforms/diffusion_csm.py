"""Definitions for the `DiffusionCSM` class."""
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d

from mosfit.constants import C_CGS, DAY_CGS, M_SUN_CGS, AU_CGS
from mosfit.modules.transforms.transform import Transform


# Important: Only define one ``Module`` class per file.


class DiffusionCSM(Transform):
    """Photon diffusion transform for CSM model."""

    N_INT_TIMES = 3000
    MIN_LOG_SPACING = -3


    def process(self, **kwargs):
        """Process module."""
        Transform.process(self, **kwargs)
        self._kappa = kwargs[self.key('kappa')]
        self._mass = kwargs[self.key('mcsm')] * M_SUN_CGS
        self._R0 = kwargs[self.key('r0')] * AU_CGS  # AU to cm
        self._s = kwargs[self.key('s')]
        self._rho = kwargs[self.key('rho')]
        self._mejecta = kwargs[self.key('mejecta')] * M_SUN_CGS  # Msol to grms

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

        # mass of the optically thick CSM (tau > 2/3).
        self._Mcsm_th = np.abs(4.0 * np.pi * self._q / (3.0 - self._s) * (
            self._Rph**(3.0 - self._s) - self._R0 **
            (3.0 - self._s)))
        beta =  4. * np.pi ** 3. / 9.
        td2 = self._tau_diff**2
        td = self._tau_diff
        t0 = self._kappa * (self._Mcsm_th) \
                / (beta * C_CGS * self._Rph) / DAY_CGS
        new_lums = np.zeros_like(self._times_to_process)
        if len(self._dense_times_since_exp) < 2:
            return {self.dense_key('luminosities'): new_lums}
        min_te = min(self._dense_times_since_exp)
        tb = max(0.0, min_te)
        linterp = interp1d(
            self._dense_times_since_exp, self._dense_luminosities, copy=False,
            assume_sorted=True, bounds_error=False, fill_value=0.0)

        uniq_times = np.unique(self._times_to_process[
            (self._times_to_process >= tb) & (
                self._times_to_process <= self._dense_times_since_exp[-1])])
        lu = len(uniq_times)

        num = int(round(self.N_INT_TIMES / 2.0))
        lsp = np.logspace(
            np.log10(t0 /
                     self._dense_times_since_exp[-1]) +
            self.MIN_LOG_SPACING, 0, num)
        xm = np.unique(np.concatenate((lsp, 1 - lsp)))

        int_times = np.clip(
            tb + (uniq_times.reshape(lu, 1) - tb) * xm, tb,
            self._dense_times_since_exp[-1])
        int_times = tb + (uniq_times.reshape(lu, 1) - tb) * xm
        int_tes = int_times[:, -1]

        int_lums = linterp(int_times)  # noqa: F841
        int_args = int_lums * np.exp((int_times) / t0)
        int_args[np.isnan(int_args)] = 0.0

        uniq_lums = np.trapz(int_args, int_times)
        uniq_lums*= np.exp(-int_tes/t0)/t0
        new_lums = uniq_lums[np.searchsorted(uniq_times,
                                             self._times_to_process)]

        return {self.dense_key('luminosities'): new_lums}
