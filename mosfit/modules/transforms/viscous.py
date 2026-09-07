"""Definitions for the `Viscous` class."""
import numpy as np
from numba import njit

from mosfit.modules.transforms.transform import Transform

CLASS_NAME = 'Viscous'


@njit(cache=True)
def _viscous_step(I0, L0, L1, t0, t1, tau):
    """Advance the viscous integral over a linear ``L`` segment."""
    dt = t1 - t0
    if dt <= 0.0 or tau <= 0.0:
        return I0
    alpha = dt / tau
    if alpha > 700.0:
        return L1
    e1 = np.exp(-alpha)
    one_m_e = -np.expm1(-alpha)
    if alpha < 1.0e-8:
        w = alpha * (0.5 - alpha / 6.0 + alpha * alpha / 24.0)
    else:
        w = (alpha - one_m_e) / alpha
    return e1 * I0 + L0 * one_m_e + (L1 - L0) * w


@njit(cache=True)
def viscous_exp_filter(dense_t, dense_l, out_t, tau, tb, t_end):
    """Piecewise-linear exact viscous delay at ``out_t``.

    ``L_out(t) = (1/τ) ∫_{tb}^{t} L(s) exp((s-t)/τ) ds`` with ``L`` the linear
    interpolant of ``(dense_t, dense_l)``.
    """
    n = dense_t.shape[0]
    I_knot = np.zeros(n)
    i0 = 0
    while i0 < n and dense_t[i0] < tb:
        i0 += 1
    if i0 < n:
        if dense_t[i0] > tb:
            L_tb = np.interp(tb, dense_t, dense_l)
            I_knot[i0] = _viscous_step(
                0.0, L_tb, dense_l[i0], tb, dense_t[i0], tau)
        for j in range(i0 + 1, n):
            if dense_t[j] > t_end:
                break
            I_knot[j] = _viscous_step(
                I_knot[j - 1], dense_l[j - 1], dense_l[j],
                dense_t[j - 1], dense_t[j], tau)

    m = out_t.shape[0]
    out = np.empty(m)
    for k in range(m):
        t = out_t[k]
        if t <= tb:
            out[k] = 0.0
            continue
        if t > t_end:
            t = t_end
        j = np.searchsorted(dense_t, t)
        if j < n and dense_t[j] == t:
            out[k] = I_knot[j]
            continue
        if j == 0:
            L_tb = np.interp(tb, dense_t, dense_l)
            Lt = np.interp(t, dense_t, dense_l)
            out[k] = _viscous_step(0.0, L_tb, Lt, tb, t, tau)
            continue
        t0 = dense_t[j - 1]
        L0 = dense_l[j - 1]
        if t0 < tb:
            L_tb = np.interp(tb, dense_t, dense_l)
            Lt = np.interp(t, dense_t, dense_l)
            out[k] = _viscous_step(0.0, L_tb, Lt, tb, t, tau)
        else:
            Lt = np.interp(t, dense_t, dense_l)
            out[k] = _viscous_step(I_knot[j - 1], L0, Lt, t0, t, tau)
    return out


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

        uniq_lums = viscous_exp_filter(
            dense_t, dense_l, uniq_times, tvisc, tb, t_end)
        new_lums = uniq_lums[np.searchsorted(uniq_times, times)]

        return {self.dense_key('luminosities'): new_lums}
