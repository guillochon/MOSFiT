"""Definitions for the `Viscous` class."""
import numpy as np
from scipy.interpolate import interp1d

from mosfit.modules.transforms.transform import Transform

CLASS_NAME = 'Viscous'


class Viscous(Transform):
    """Viscous delay transform."""

    N_INT_TIMES = 1000

    def process(self, **kwargs):
        """Process module."""
        super(Viscous, self).process(**kwargs)

        tvisc = kwargs['Tviscous']

        new_lums = np.zeros_like(self._times_to_process)
        if len(self._dense_times_since_exp) < 2:
            return {self.dense_key('luminosities'): new_lums}
        min_te = min(self._dense_times_since_exp)
        tb = max(0.0, min_te)
        linterp = interp1d(
            self._dense_times_since_exp, self._dense_luminosities, copy=False,
            assume_sorted=True)

        uniq_times = np.unique(self._times_to_process[
            (self._times_to_process >= tb) & (
                self._times_to_process <= self._dense_times_since_exp[-1])])
        lu = len(uniq_times)

        num = int(self.N_INT_TIMES/2.0)
        expstart = np.log10(tvisc) - 3
        if expstart > -1:
            expstart = -1
        xm = np.unique(np.concatenate(
            (np.array([0]), np.logspace(expstart, 0, num),
             1 - np.logspace(expstart, 0, num))))

        int_times = np.clip(tb + (uniq_times.reshape(lu, 1) - tb) * xm, tb,
                            self._dense_times_since_exp[-1])

        int_tes = int_times[:, -1]
        int_lums = linterp(int_times)

        int_args = int_lums * np.exp(
            (int_times - int_tes.reshape(lu, 1)) / tvisc)
        int_args[np.isnan(int_args)] = 0.0

        uniq_lums = np.trapz(int_args, int_times)/tvisc
        new_lums = uniq_lums[np.searchsorted(uniq_times,
                                             self._times_to_process)]

        return {self.dense_key('luminosities'): new_lums}
