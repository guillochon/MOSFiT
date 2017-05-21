"""Definitions for the `Viscous` class."""
import numpy as np
from scipy.interpolate import interp1d

from mosfit.modules.transforms.transform import Transform

CLASS_NAME = 'Viscous'


class Viscous(Transform):
    """Viscous delay transform."""

    logsteps = True
    N_INT_TIMES = 1000  # 1e5

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

        if self.logsteps:
            num = int(self.N_INT_TIMES/2.0)
            xm = np.unique(np.concatenate(
                (np.logspace(-7, 0, num),
                 1 - np.logspace(-7, 0, num))))
            # print(xm)
        else:
            xm = np.linspace(0, 1, self.N_INT_TIMES)

        int_times = np.clip(tb + (uniq_times.reshape(lu, 1) - tb) * xm, tb,
                            self._dense_times_since_exp[-1])

        int_tes = int_times[:, -1]
        int_lums = linterp(int_times)  # noqa: F841

        if self.logsteps:
            int_args = int_lums * np.exp(
                (int_times - int_tes.reshape(lu, 1)) / tvisc)
        else:
            # int_args = int_lums * int_times * np.exp(
            #    (int_times - int_tes.reshape(lu, 1)) / tvisc)
            int_args = int_lums * np.exp(
                (int_times - int_tes.reshape(lu, 1)) / tvisc)
        int_args[np.isnan(int_args)] = 0.0

        if self.logsteps:
            uniq_lums = np.trapz(int_args, int_times)/tvisc
            # new_lums = uniq_lums
            # new_lums = np.trapz(int_args, int_times)/tvisc
            # if len(new_lums)!= len()
            # new_lums = uniq_lums

        else:
            dts = int_times[:, 1] - int_times[:, 0]
            uniq_lums = np.sum(int_args[:, 2:-1], axis=1) + 0.5 * (
                int_args[:, 0] + int_args[:, -1])
            uniq_lums *= 2.0 * dts / tvisc
        new_lums = uniq_lums[np.searchsorted(uniq_times,
                                                 self._times_to_process)]

        return {self.dense_key('luminosities'): new_lums}
