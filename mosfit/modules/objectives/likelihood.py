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

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Likelihood, self).__init__(**kwargs)
        self._preprocessed = False

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)
        self._model_observations = kwargs['model_observations']
        self._fractions = kwargs['fractions']
        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return {'value': LIKELIHOOD_FLOOR}
        for oi, obs in enumerate(self._model_observations):
            if not self._upper_limits[oi] and (isnan(obs) or
                                               not np.isfinite(obs)):
                return {'value': LIKELIHOOD_FLOOR}
        self._score_modifier = kwargs.get('score_modifier', 0.0)

        # Calculate (model - obs) residuals.
        residuals = np.array([
            (x - y if not u or (x < y and not isnan(x)) else 0.0) if a else
            (x - fd if not u or (x > fd and not isnan(x)) else 0.0)
            for x, y, fd, u, o, a in zip(
                self._model_observations, self._mags, self._fds,
                self._upper_limits, self._observed, self._are_mags) if o
        ])

        # Observational errors to be put in diagonal of error matrix.
        diag = [
            ((el if x > y else eu) ** 2) if a else
            ((fdel if x < fd else fdeu) ** 2)
            for x, y, eu, el, fd, fdeu, fdel, o, a in zip(
                self._model_observations, self._mags,
                self._e_u_mags, self._e_l_mags, self._fds, self._e_u_fds,
                self._e_l_fds, self._observed, self._are_mags) if o
        ]

        is_diag = False
        if (kwargs.get('codeltatime', -1) >= 0 or
                kwargs.get('codeltalambda', -1) >= 0):
            kmat = np.array([
                [vi * vj for vi in self._o_band_vs] for vj in self._o_band_vs
            ])
        else:
            # Shortcut when matrix is diagonal.
            is_diag = True
            value = -0.5 * np.sum(
                residuals ** 2 / (self._o_band_vs ** 2 + diag) +
                np.log(self._o_band_vs ** 2 + diag))

        if not is_diag:
            kn = len(self._o_times)

            if kwargs.get('codeltatime', -1) >= 0:
                kmat *= np.array([
                    [1.0 if i == j else np.exp(
                        -0.5 * ((ti - tj) / kwargs['codeltatime']) ** 2) for
                     i, ti in enumerate(self._o_times)] for
                    j, tj in enumerate(self._o_times)
                ])

            if kwargs.get('codeltalambda', -1) >= 0:
                kmat *= np.array([
                    [1.0 if i == j else np.exp(
                        -0.5 * ((li - lj) / kwargs['codeltalambda']) ** 2) for
                     i, li in enumerate(self._o_waves)] for
                    j, lj in enumerate(self._o_waves)
                ])

            # Add observed errors to diagonal
            for i in range(kn):
                kmat[i, i] += diag[i]

            # full_size = np.count_nonzero(kmat)

            # Remove small covariance terms
            min_cov = self.MIN_COV_TERM * np.max(kmat)
            kmat[kmat <= min_cov] = 0.0

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
            return {'value': LIKELIHOOD_FLOOR}
        return {'value': max(LIKELIHOOD_FLOOR, score)}

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
        self._freqs = kwargs.get('frequencies', [])
        self._e_u_mags = kwargs.get('e_upper_magnitudes', [])
        self._e_l_mags = kwargs.get('e_lower_magnitudes', [])
        self._e_mags = kwargs.get('e_magnitudes', [])
        self._e_u_fds = kwargs.get('e_upper_fluxdensities', [])
        self._e_l_fds = kwargs.get('e_lower_fluxdensities', [])
        self._e_fds = kwargs.get('e_fluxdensities', [])
        self._u_fds = kwargs.get('u_fluxdensities', [])
        self._u_freqs = kwargs.get('u_frequencies', [])
        self._upper_limits = kwargs.get('upperlimits', [])
        self._observed = np.array(kwargs['observed'])
        self._all_band_indices = kwargs.get('all_band_indices', [])
        self._are_mags = np.array(self._all_band_indices) >= 0
        self._are_fds = np.array(self._all_band_indices) < 0
        self._all_band_avgs = np.array([
            self._average_wavelengths[bi] if bi >= 0 else
            C_CGS / self._freqs[i] / ANG_CGS for i, bi in
            enumerate(self._all_band_indices)])

        # Magnitudes first
        self._e_u_mags = [
            kwargs['default_upper_limit_error']
            if (e == '' and eu == '' and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e == '' and eu == '') else (e if eu == '' else eu))
            for i, (e, eu) in enumerate(zip(self._e_mags, self._e_u_mags))
        ]
        self._e_l_mags = [
            0.0 if self._upper_limits[i] else
            (kwargs['default_no_error_bar_error']
             if (e == '' and el == '') else (e if el == '' else el))
            for i, (e, el) in enumerate(zip(self._e_mags, self._e_l_mags))
        ]

        # Now flux densities
        self._e_u_fds = [
            v if (e == '' and eu == '' and self._upper_limits[i]) else
            (v if (e == '' and eu == '') else (e if eu == '' else eu))
            for i, (e, eu, v) in enumerate(
                zip(self._e_fds, self._e_u_fds, self._fds))
        ]
        self._e_l_fds = [
            0.0 if self._upper_limits[i] else (v if (e == '' and el == '') else
                                               (e if el == '' else el))
            for i, (e, el, v) in enumerate(
                zip(self._e_fds, self._e_l_fds, self._fds))
        ]
        self._fds = [
            x / flux_density_unit(y) if x != '' else ''
            for x, y in zip(self._fds, self._u_fds)
        ]
        self._e_u_fds = [
            x / flux_density_unit(y) if x != '' else ''
            for x, y in zip(self._e_u_fds, self._u_fds)
        ]
        self._e_l_fds = [
            x / flux_density_unit(y) if x != '' else ''
            for x, y in zip(self._e_l_fds, self._u_fds)
        ]

        # Time deltas (radial distance) for covariance matrix.
        self._o_times = self._times[self._observed]
        # Wavelength deltas (radial distance) for covariance matrix.
        self._o_waves = self._all_band_avgs[self._observed]

        # Get band variances
        self._variance = kwargs.get('variance', 0.0)

        band_vs = OrderedDict()
        for key in kwargs:
            if key.startswith('variance-band-'):
                band_vs[key.split('-')[-1]] = kwargs[key]

        if len(self._variance_bands):
            var_bands = [
                self._variance_bands[i] for i in self._all_band_indices]

            self._band_vs = np.array([
                band_vs.get(i, self._variance) if isinstance(i, string_types)
                else (i[0] * band_vs.get(i[1][0], self._variance) +
                      (1.0 - i[0]) * band_vs.get(i[1][0], self._variance))
                for i in var_bands])
        else:
            self._band_vs = np.full(
                len(self._all_band_indices), self._variance)

        self._o_band_vs = self._band_vs[self._observed]

        self._preprocessed = True
