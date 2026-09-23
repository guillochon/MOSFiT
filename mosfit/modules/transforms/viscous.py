"""Definitions for the `Viscous` class."""
import numpy as np

from mosfit.modules.transforms.transform import Transform

CLASS_NAME = 'Viscous'


class Viscous(Transform):
    """Viscous delay transform."""

    N_INT_TIMES = 1000
    MIN_LOG_SPACING = -3

    def process(self, **kwargs):
        """Process module."""
        Transform.process(self, **kwargs)

        tvisc = float(kwargs['Tviscous'])

        times = np.asarray(self._times_to_process, dtype=float)
        new_lums = np.zeros_like(times)
        if len(self._dense_times_since_exp) < 2:
            return {self.dense_key('luminosities'): new_lums}
        dense_t = np.asarray(self._dense_times_since_exp, dtype=float)
        dense_l = np.asarray(self._dense_luminosities, dtype=float)
        min_te = float(np.min(dense_t))
        tb = max(0.0, min_te)
        t_end = float(dense_t[-1])

        mask = (times >= tb) & (times <= t_end)
        uniq_times = np.unique(times[mask])
        if uniq_times.size == 0:
            return {self.dense_key('luminosities'): new_lums}

        # Imported here so that `numba` is only required by runs that
        # actually use this model.
        from mosfit.modules.transforms._viscous_kernels import (
            viscous_exp_filter)

        uniq_lums = viscous_exp_filter(
            dense_t, dense_l, uniq_times, tvisc, tb, t_end)
        new_lums = uniq_lums[np.searchsorted(uniq_times, times)]

        return {self.dense_key('luminosities'): new_lums}
