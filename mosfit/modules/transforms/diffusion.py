"""Definitions for the `Diffusion` class."""
import numpy as np
from mosfit.constants import C_CGS, DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.transforms.transform import Transform
from scipy.interpolate import interp1d


# Important: Only define one ``Module`` class per file.


class Diffusion(Transform):
    """Photon diffusion transform."""

    N_INT_TIMES = 1000
    DIFF_CONST = 2.0 * M_SUN_CGS / (13.7 * C_CGS * KM_CGS)
    TRAP_CONST = 3.0 * M_SUN_CGS / (FOUR_PI * KM_CGS ** 2)

    def process(self, **kwargs):
        """Process module."""
        super(Diffusion, self).process(**kwargs)
        self._kappa = kwargs[self.key('kappa')]
        self._kappa_gamma = kwargs[self.key('kappagamma')]
        self._m_ejecta = kwargs[self.key('mejecta')]
        self._v_ejecta = kwargs[self.key('vejecta')]

        self._tau_diff = np.sqrt(self.DIFF_CONST * self._kappa *
                                 self._m_ejecta / self._v_ejecta) / DAY_CGS
        self._trap_coeff = (
            self.TRAP_CONST * self._kappa_gamma * self._m_ejecta /
            (self._v_ejecta ** 2)) / DAY_CGS ** 2
        td2, A = self._tau_diff ** 2, self._trap_coeff  # noqa: F841

        new_lums = np.zeros_like(self._times_to_process)
        min_te = min(self._dense_times_since_exp)
        tb = max(0.0, min_te)
        linterp = interp1d(self._dense_times_since_exp,
                           self._dense_luminosities, copy=False,
                           assume_sorted=True)

        uniq_times = np.unique(self._times_to_process[
            self._times_to_process > tb])
        lu = len(uniq_times)

        xm = np.linspace(0, 1, self.N_INT_TIMES)
        int_times = tb + (uniq_times.reshape(lu, 1) - tb) * xm

        dts = int_times[:, 1] - int_times[:, 0]
        int_te2s = int_times[:, -1] ** 2
        int_lums = linterp(int_times)  # noqa: F841
        int_args = int_lums * int_times * np.exp(
            (int_times**2 - int_te2s.reshape(lu, 1)) / td2)
        int_args[np.isnan(int_args)] = 0.0

        uniq_lums = np.sum(int_args[:, 2:-1], axis=1) + 0.5 * (
            int_args[:, 0] + int_args[:, -1])
        uniq_lums *= -2.0 * np.expm1(-A / int_te2s) * dts / td2

        new_lums = uniq_lums[np.searchsorted(uniq_times,
                                             self._times_to_process)]

        return {self.dense_key('luminosities'): new_lums}
