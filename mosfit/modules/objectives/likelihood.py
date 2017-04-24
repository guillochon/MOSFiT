"""Definitions for the `Likelihood` class."""
from collections import OrderedDict
from math import isnan

import numpy as np
import scipy
from mosfit.constants import ANG_CGS, C_CGS, LIKELIHOOD_FLOOR
from mosfit.modules.module import Module
from mosfit.utils import flux_density_unit
from six import string_types


# Important: Only define one ``Module`` class per file.


class Likelihood(Module):
    """Calculate the maximum likelihood score for a model."""

    MIN_COV_TERM = 1.0e-30

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)
        self._model_observations = kwargs['model_observations']
        self._observation_types = kwargs['observation_types']
        self._fractions = kwargs['fractions']
        ret = {'value': LIKELIHOOD_FLOOR, 'kmat': None}
        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return ret
        for oi, obs in enumerate(self._model_observations):
            if not self._upper_limits[oi] and (isnan(obs) or
                                               not np.isfinite(obs)):
                return ret
        self._score_modifier = kwargs.get(self.key('score_modifier'), 0.0)
        self._codeltatime = kwargs.get(self.key('codeltatime'), -1)
        self._codeltalambda = kwargs.get(self.key('codeltalambda'), -1)

        # Get band variances
        self._variance = kwargs.get(self.key('variance'), 0.0)

        self._band_v_vars = OrderedDict()
        for key in kwargs:
            if key.startswith('variance-band-'):
                self._band_v_vars[key.split('-')[-1]] = kwargs[key]

        if self._variance_bands:
            self._o_variance_bands = [
                self._variance_bands[i] for i in self._all_band_indices]

            self._band_vs = np.array([
                self._band_v_vars.get(i, self._variance) if
                isinstance(i, string_types)
                else (i[0] * self._band_v_vars.get(i[1][0], self._variance) +
                      (1.0 - i[0]) * self._band_v_vars.get(
                          i[1][0], self._variance))
                for i in self._o_variance_bands])
        else:
            self._band_vs = np.full(
                len(self._all_band_indices), self._variance)

        self._o_band_vs = self._band_vs[self._observed]

        self._model_observations[self._cmask] = -2.5 * np.log10(
            self._model_observations[self._cmask])

        # Calculate (model - obs) residuals.
        residuals = np.array([
            (abs(x - ct) if not u or (x < ct and not isnan(x)) else 0.0)
            if t == 'countrate' and ct is not None
            else
            (abs(x - y) if not u or (x < y and not isnan(x)) else 0.0)
            if t == 'magnitude' and y is not None
            else
            (abs(x - fd) if not u or (x > fd and not isnan(x)) else 0.0)
            if t == 'fluxdensity' and fd is not None else None
            for x, y, ct, fd, u, o, t in zip(
                self._model_observations, self._mags, self._cts, self._fds,
                self._upper_limits, self._observed, self._observation_types)
            if o
        ])

        if np.any(residuals == None):  # noqa: E711
            raise ValueError('Null residual.')

        # Observational errors to be put in diagonal of error matrix.
        diag = np.array([
            ((ctel if x > ct else cteu) ** 2)
            if t == 'countrate' and ct is not None else
            ((el if x > y else eu) ** 2)
            if t == 'magnitude' and y is not None else
            ((fdel if x < fd else fdeu) ** 2)
            if t == 'fluxdensity' and fd is not None else None
            for x, y, eu, el, fd, fdeu, fdel, ct, ctel, cteu, o, t in zip(
                self._model_observations, self._mags,
                self._e_u_mags, self._e_l_mags, self._fds, self._e_u_fds,
                self._e_l_fds, self._cts, self._e_l_cts, self._e_u_cts,
                self._observed, self._observation_types) if o
        ])

        if np.any(diag == None):  # noqa: E711
            raise ValueError('Null error.')

        is_diag = False
        if self._codeltatime >= 0 or self._codeltalambda >= 0:
            kmat = np.outer(self._o_band_vs, self._o_band_vs)
        else:
            # Shortcut when matrix is diagonal.
            is_diag = True
            value = -0.5 * np.sum(
                residuals ** 2 / (self._o_band_vs ** 2 + diag) +
                np.log(self._o_band_vs ** 2 + diag))

        if not is_diag:
            kn = len(self._o_times)

            if self._codeltatime >= 0:
                kmat *= np.exp(self._dtmat / self._codeltatime ** 2)

            if self._codeltalambda >= 0:
                kmat *= np.exp(self._dlmat / self._codeltalambda ** 2)

            # Add observed errors to diagonal
            for i in range(kn):
                kmat[i, i] += diag[i]

            ret['kmat'] = kmat

            # full_size = np.count_nonzero(kmat)

            # Remove small covariance terms
            # min_cov = self.MIN_COV_TERM * np.max(kmat)
            # kmat[kmat <= min_cov] = 0.0

            # print("Sparse frac: {:.2%}".format(
            #     float(full_size - np.count_nonzero(kmat)) / full_size))

            try:
                chol_kmat = scipy.linalg.cholesky(kmat, check_finite=False)

                value = -np.linalg.slogdet(chol_kmat)[-1]
                value -= 0.5 * (
                    np.matmul(residuals.T, scipy.linalg.cho_solve(
                        (chol_kmat, False), residuals, check_finite=False)))
            except Exception:
                value = -0.5 * (
                    np.matmul(np.matmul(residuals.T, scipy.linalg.inv(kmat)),
                              residuals) + np.log(scipy.linalg.det(kmat)))

        score = self._score_modifier + value
        if isnan(score) or not np.isfinite(score):
            return ret
        ret['value'] = max(LIKELIHOOD_FLOOR, score)
        return ret

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        self._average_wavelengths = requests.get('average_wavelengths', [])
        self._variance_bands = requests.get('variance_bands', [])

    def preprocess(self, **kwargs):
        """Construct arrays of observations based on data keys."""
        if self._preprocessed:
            return
        self._times = np.array(kwargs.get('times', []))
        self._mags = kwargs.get('magnitudes', [])
        self._fds = kwargs.get('fluxdensities', [])
        self._cts = np.array(kwargs.get('countrates', []))
        self._freqs = kwargs.get('frequencies', [])
        self._e_u_mags = kwargs.get('e_upper_magnitudes', [])
        self._e_l_mags = kwargs.get('e_lower_magnitudes', [])
        self._e_mags = kwargs.get('e_magnitudes', [])
        self._e_u_fds = kwargs.get('e_upper_fluxdensities', [])
        self._e_l_fds = kwargs.get('e_lower_fluxdensities', [])
        self._e_fds = kwargs.get('e_fluxdensities', [])
        self._u_fds = kwargs.get('u_fluxdensities', [])
        self._u_freqs = kwargs.get('u_frequencies', [])
        self._e_u_cts = kwargs.get('e_upper_countrates', [])
        self._e_l_cts = kwargs.get('e_lower_countrates', [])
        self._e_cts = kwargs.get('e_countrates', [])
        self._u_cts = kwargs.get('u_countrates', [])
        self._upper_limits = kwargs.get('upperlimits', [])
        self._observed = np.array(kwargs.get('observed', []))
        self._all_band_indices = kwargs.get('all_band_indices', [])
        self._are_bands = np.array(self._all_band_indices) >= 0
        self._all_band_avgs = np.array([
            self._average_wavelengths[bi] if bi >= 0 else
            C_CGS / self._freqs[i] / ANG_CGS for i, bi in
            enumerate(self._all_band_indices)])
        self._n_obs = len(self._observed)

        # Magnitudes first
        # Note: Upper limits (censored data) currently treated as a
        # half-Gaussian, this is very approximate and can be improved upon.
        self._e_u_mags = [
            kwargs['default_upper_limit_error']
            if (e is None and eu is None and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e is None and eu is None) else (e if eu is None else eu))
            for i, (e, eu) in enumerate(zip(self._e_mags, self._e_u_mags))
        ]
        self._e_l_mags = [
            kwargs['default_upper_limit_error']
            if (e is None and el is None and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e is None and el is None) else (e if el is None else el))
            for i, (e, el) in enumerate(zip(self._e_mags, self._e_l_mags))
        ]

        # Now counts
        self._cmask = np.array([x is not None for x in self._cts])
        self._cts[self._cmask] = -2.5 * np.log10(self._cts[self._cmask]
                                                 .astype(np.float64))
        self._e_u_cts = [
            kwargs['default_upper_limit_error']
            if (e is None and eu is None and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e is None and eu is None) else
             2.5 * (np.log10(c + (e if eu is None else eu)) - np.log10(c)))
            for i, (c, e, eu) in enumerate(zip(
                self._cts, self._e_cts, self._e_u_cts))
        ]
        self._e_l_cts = [
            kwargs['default_upper_limit_error']
            if (e is None and el is None and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e is None and el is None) else
             2.5 * (np.log10(c) - np.log10(c - (e if el is None else el))))
            for i, (c, e, el) in enumerate(zip(
                self._cts, self._e_cts, self._e_l_cts))
        ]

        # Now flux densities
        self._e_u_fds = [
            v if (e is None and eu is None and self._upper_limits[i]) else
            (v if (e is None and eu is None) else (e if eu is None else eu))
            for i, (e, eu, v) in enumerate(
                zip(self._e_fds, self._e_u_fds, self._fds))
        ]
        self._e_l_fds = [
            0.0 if self._upper_limits[i] else (
                v if (e is None and el is None) else (e if el is None else el))
            for i, (e, el, v) in enumerate(
                zip(self._e_fds, self._e_l_fds, self._fds))
        ]
        self._fds = [
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._fds, self._u_fds)
        ]
        self._e_u_fds = [
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._e_u_fds, self._u_fds)
        ]
        self._e_l_fds = [
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._e_l_fds, self._u_fds)
        ]

        # Time deltas (radial distance) for covariance matrix.
        self._o_times = self._times[self._observed]
        # Wavelength deltas (radial distance) for covariance matrix.
        self._o_waves = self._all_band_avgs[self._observed]

        self._dtmat = -0.5 * (
            np.repeat(self._o_times[:, np.newaxis], self._n_obs, 1) -
            np.repeat(self._o_times[np.newaxis, :], self._n_obs, 0)) ** 2

        self._dlmat = -0.5 * (
            np.repeat(self._o_waves[:, np.newaxis], self._n_obs, 1) -
            np.repeat(self._o_waves[np.newaxis, :], self._n_obs, 0)) ** 2

        self._preprocessed = True
