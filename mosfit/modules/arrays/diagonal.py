"""Definitions for the `Diagonal` class."""

import numpy as np

from mosfit.modules.arrays.array import Array
from mosfit.utils import flux_density_unit


# Important: Only define one ``Module`` class per file.


def _none_to_nan(values):
    """Convert a sequence that may contain ``None`` to float with NaN."""
    arr = np.asarray(values, dtype=object).ravel()
    out = np.empty(arr.shape[0], dtype=float)
    for i, val in enumerate(arr):
        out[i] = np.nan if val is None else float(val)
    return out


class Diagonal(Array):
    """Calculate the diagonal/residuals for a model kernel."""

    MIN_COV_TERM = 1.0e-30

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Diagonal, self).__init__(**kwargs)
        self._observation_types = np.array([])

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)
        self._model_observations = np.copy(kwargs['model_observations'])
        self._model_observations = self._model_observations[self._observed]

        ret = {}

        allowed_otypes = [
            'countrate', 'magnitude', 'fluxdensity', 'magcount',
            'luminosity']

        if np.any([x not in allowed_otypes for x in self._o_types]):
            print([x for x in self._o_types if x not in allowed_otypes])
            raise ValueError('Unrecognized observation type.')

        # Calculate (model - obs) residuals with the same definition as the
        # original per-row zip, using masked arrays.
        x = np.asarray(self._model_observations, dtype=float)
        u = np.asarray(self._upper_limits, dtype=bool)
        t = np.asarray(self._o_types)
        y = self._mags_f
        lum = self._lums_f
        ct = self._cts_f
        fd = self._fds_f

        is_cr = (t == 'countrate') | (t == 'magcount')
        is_mag = t == 'magnitude'
        is_lum = t == 'luminosity'
        is_fd = t == 'fluxdensity'
        if np.any(~(is_cr | is_mag | is_lum | is_fd)):
            raise ValueError('Null residual.')

        obs = np.where(is_cr, ct, np.where(
            is_mag, y, np.where(is_lum, lum, fd)))
        obs_ok = np.isfinite(obs)
        finite_x = np.isfinite(x)
        brighter = np.where(is_fd, x > obs, x < obs)
        use_resid = ((~u) & obs_ok) | (finite_x & obs_ok & brighter)
        residuals = np.where(use_resid, np.abs(x - obs), 0.0)

        # Observational errors to be put in diagonal of error matrix.
        el = self._e_l_mags_f
        eu = self._e_u_mags_f
        lm_el = self._e_l_lums_f
        lm_eu = self._e_u_lums_f
        ctel = self._e_l_cts_f
        cteu = self._e_u_cts_f
        fdel = self._e_l_fds_f
        fdeu = self._e_u_fds_f

        mag_err = np.where(np.isnan(y) | (x > y), el, eu)
        lum_err = np.where(np.isnan(lum) | (x > lum), lm_el, lm_eu)
        cr_err = np.where(np.isfinite(ct) & (x > ct), ctel, cteu)
        fd_err = np.where(np.isfinite(fd) & (x > fd), fdel, fdeu)
        diag = np.where(is_cr, cr_err, np.where(
            is_mag, mag_err, np.where(is_lum, lum_err, fd_err)))
        diag = np.nan_to_num(diag, nan=0.0) ** 2

        preset = kwargs.get('emulator_preset_systematic_mag')
        if preset is not None:
            preset = np.asarray(preset, dtype=np.float64)
            if preset.shape[0] == len(self._observed):
                preset = preset[self._observed]
            if preset.shape == diag.shape:
                mag_like = np.isin(
                    self._o_types, ('magnitude', 'magcount'))
                pv = np.nan_to_num(preset, nan=0.0, posinf=0.0, neginf=0.0)
                pv = np.maximum(pv, 0.0)
                diag = diag + mag_like * (pv ** 2)

        ret['kdiagonal'] = diag
        ret['kresiduals'] = residuals

        return ret

    def preprocess(self, **kwargs):
        """Construct arrays of observations based on data keys."""
        otypes = np.array(kwargs.get('observation_types', []))
        if np.array_equiv(
                otypes, self._observation_types) and self._preprocessed:
            return
        self._observation_types = otypes
        self._mags = np.array(kwargs.get('magnitudes', []))
        self._lums = np.array(kwargs.get('observed_luminosities', []))
        self._fds = np.array(kwargs.get('fluxdensities', []))
        self._cts = np.array(kwargs.get('countrates', []))
        self._e_u_mags = kwargs.get('e_upper_magnitudes', [])
        self._e_l_mags = kwargs.get('e_lower_magnitudes', [])
        self._e_mags = kwargs.get('e_magnitudes', [])
        self._e_u_lums = kwargs.get('e_upper_observed_luminosities', [])
        self._e_l_lums = kwargs.get('e_lower_observed_luminosities', [])
        self._e_lums_sym = kwargs.get('e_observed_luminosities', [])
        self._e_u_fds = kwargs.get('e_upper_fluxdensities', [])
        self._e_l_fds = kwargs.get('e_lower_fluxdensities', [])
        self._e_fds = kwargs.get('e_fluxdensities', [])
        self._u_fds = kwargs.get('u_fluxdensities', [])
        self._e_u_cts = kwargs.get('e_upper_countrates', [])
        self._e_l_cts = kwargs.get('e_lower_countrates', [])
        self._e_cts = kwargs.get('e_countrates', [])
        self._u_cts = kwargs.get('u_countrates', [])
        self._upper_limits = np.array(kwargs.get('upperlimits', []),
                                      dtype=bool)
        self._observed = np.array(kwargs.get('observed', []), dtype=bool)
        self._o_types = self._observation_types[self._observed]

        default_ul = kwargs['default_upper_limit_error']
        default_none = kwargs['default_no_error_bar_error']

        def _fill_maglike(sym, sided):
            e = _none_to_nan(sym)
            side = _none_to_nan(sided)
            both = np.isnan(e) & np.isnan(side)
            out = np.where(np.isnan(side), e, side)
            out = np.where(both & self._upper_limits, default_ul, out)
            out = np.where(both & ~self._upper_limits, default_none, out)
            return out

        # Magnitudes first
        # Note: Upper limits (censored data) currently treated as a
        # half-Gaussian, this is very approximate and can be improved upon.
        self._e_u_mags = _fill_maglike(self._e_mags, self._e_u_mags)
        self._e_l_mags = _fill_maglike(self._e_mags, self._e_l_mags)
        self._e_u_lums = _fill_maglike(self._e_lums_sym, self._e_u_lums)
        self._e_l_lums = _fill_maglike(self._e_lums_sym, self._e_l_lums)

        # Ignore upperlimits for countrate if magnitude is present.
        self._upper_limits[self._observation_types[
            self._observed] == 'magcount'] = False

        cts = _none_to_nan(self._cts)
        e_cts = _none_to_nan(self._e_cts)
        e_u_cts = _none_to_nan(self._e_u_cts)
        e_l_cts = _none_to_nan(self._e_l_cts)
        both_u = np.isnan(e_cts) & np.isnan(e_u_cts)
        both_l = np.isnan(e_cts) & np.isnan(e_l_cts)
        self._e_u_cts = np.where(both_u, cts, np.where(np.isnan(e_u_cts),
                                                       e_cts, e_u_cts))
        self._e_l_cts = np.where(both_l, cts, np.where(np.isnan(e_l_cts),
                                                       e_cts, e_l_cts))

        fds = _none_to_nan(self._fds)
        e_fds = _none_to_nan(self._e_fds)
        e_u_fds = _none_to_nan(self._e_u_fds)
        e_l_fds = _none_to_nan(self._e_l_fds)
        both_uf = np.isnan(e_fds) & np.isnan(e_u_fds)
        both_lf = np.isnan(e_fds) & np.isnan(e_l_fds)
        self._e_u_fds = np.where(both_uf, fds, np.where(np.isnan(e_u_fds),
                                                        e_fds, e_u_fds))
        self._e_l_fds = np.where(
            self._upper_limits, 0.0,
            np.where(both_lf, fds, np.where(np.isnan(e_l_fds), e_fds, e_l_fds)))

        u_fds = np.asarray(self._u_fds, dtype=object)
        units = np.array([
            flux_density_unit(u) if u is not None else 1.0 for u in u_fds],
            dtype=float)
        units[units == 0.0] = 1.0
        self._fds = np.where(np.isfinite(fds), fds / units, np.nan)
        self._e_u_fds = np.where(
            np.isfinite(self._e_u_fds), self._e_u_fds / units, np.nan)
        self._e_l_fds = np.where(
            np.isfinite(self._e_l_fds), self._e_l_fds / units, np.nan)

        self._mags_f = _none_to_nan(self._mags)
        self._lums_f = _none_to_nan(self._lums)
        self._cts_f = np.asarray(cts, dtype=float)
        self._fds_f = np.asarray(self._fds, dtype=float)
        self._e_u_mags_f = np.asarray(self._e_u_mags, dtype=float)
        self._e_l_mags_f = np.asarray(self._e_l_mags, dtype=float)
        self._e_u_lums_f = np.asarray(self._e_u_lums, dtype=float)
        self._e_l_lums_f = np.asarray(self._e_l_lums, dtype=float)
        self._e_u_cts_f = np.asarray(self._e_u_cts, dtype=float)
        self._e_l_cts_f = np.asarray(self._e_l_cts, dtype=float)
        self._e_u_fds_f = np.asarray(self._e_u_fds, dtype=float)
        self._e_l_fds_f = np.asarray(self._e_l_fds, dtype=float)

        self._preprocessed = True
