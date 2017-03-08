"""Definitions for the `Likelihood` class."""
from collections import OrderedDict
from math import isnan

import numpy as np
import scipy
from mosfit.constants import LIKELIHOOD_FLOOR
from mosfit.modules.module import Module
from mosfit.utils import flux_density_unit


# Important: Only define one ``Module`` class per file.


class Likelihood(Module):
    """Calculate the maximum likelihood score for a model."""

    MIN_COV_TERM = 1.0e-10

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Likelihood, self).__init__(**kwargs)
        self._preprocessed = False

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)
        self._model_observations = kwargs['model_observations']
        self._all_bands = kwargs['all_bands']
        self._all_band_indices = kwargs['all_band_indices']
        self._fractions = kwargs['fractions']
        self._are_mags = np.array(self._all_band_indices) >= 0
        self._are_fds = np.array(self._all_band_indices) < 0
        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return {'value': LIKELIHOOD_FLOOR}
        for oi, obs in enumerate(self._model_observations):
            if not self._upper_limits[oi] and (isnan(obs) or
                                               not np.isfinite(obs)):
                return {'value': LIKELIHOOD_FLOOR}
        self._score_modifier = kwargs.get('score_modifier', 1.0)
        self._variance = kwargs.get('variance', 0.0)
        self._variance2 = self._variance ** 2

        band_vs = OrderedDict()
        for key in kwargs:
            if key.startswith('variance-band-'):
                band_vs[key.split('-')[-1]] = kwargs[key]

        self._band_vs = [
            band_vs.get(i, self._variance) for i in self._all_bands]

        # No covariance
        # sum_members = [
        #     (x - y if not u or (x < y and not isnan(x)) else 0.0) ** 2 / (
        #         (el if x > y else eu) ** 2 + v2) +
        #     np.log(v2 + 0.5 * (el ** 2 + eu ** 2))
        #     for x, y, eu, el, u, v2 in zip(self._model_observations, [
        #         i
        #         for i, o, a in zip(self._mags, self._observed, self._are_mags)
        #         if o and a
        #     ], [
        #         i for i, o, a in zip(self._e_u_mags, self._observed,
        #                              self._are_mags) if o and a
        #     ], [
        #         i for i, o, a in zip(self._e_l_mags, self._observed,
        #                              self._are_mags) if o and a
        #     ], [
        #         i for i, o, a in zip(self._upper_limits, self._observed,
        #                              self._are_mags) if o and a
        #     ], [
        #         band_v2s.get(i, self._variance2) for i, o, a in zip(
        #             self._all_bands, self._observed,
        #             self._are_mags) if o and a
        #     ])
        # ]
        # value = -0.5 * np.sum(sum_members)
        #
        # sum_members = [
        #     (x - y if not u or (x < y and not isnan(x)) else 0.0) ** 2 / (
        #         (el if x > y else eu) ** 2 + self._variance2) +
        #     np.log(self._variance2 + 0.5 * (el ** 2 + eu ** 2))
        #     for x, y, eu, el, u in zip(self._model_observations, [
        #         i for i, o, a in zip(self._fds, self._observed, self._are_fds)
        #         if o and a
        #     ], [
        #         i for i, o, a in zip(self._e_u_fds, self._observed,
        #                              self._are_fds) if o and a
        #     ], [
        #         i for i, o, a in zip(self._e_l_fds, self._observed,
        #                              self._are_fds) if o and a
        #     ], [
        #         i for i, o, a in zip(self._upper_limits, self._observed,
        #                              self._are_fds) if o and a
        #     ])
        # ]
        # value += -0.5 * np.sum(sum_members)

        # With covariance
        residuals = np.array([
            (x - y if not u or (x < y and not isnan(x)) else 0.0)
            for x, y, eu, el, u in zip(self._model_observations, [
                i
                for i, o, a in zip(self._mags, self._observed, self._are_mags)
                if o and a
            ], [
                i for i, o, a in zip(self._e_u_mags, self._observed,
                                     self._are_mags) if o and a
            ], [
                i for i, o, a in zip(self._e_l_mags, self._observed,
                                     self._are_mags) if o and a
            ], [
                i for i, o, a in zip(self._upper_limits, self._observed,
                                     self._are_mags) if o and a
            ])
        ])

        diag = [
            (el if x > y else eu) ** 2
            for x, y, eu, el in zip(self._model_observations, [
                i
                for i, o, a in zip(self._mags, self._observed, self._are_mags)
                if o and a
            ], [
                i for i, o, a in zip(self._e_u_mags, self._observed,
                                     self._are_mags) if o and a
            ], [
                i for i, o, a in zip(self._e_l_mags, self._observed,
                                     self._are_mags) if o and a
            ])
        ]

        # Time deltas (radial distance) for covariance matrix.
        self._o_m_times = [
            i for i, o, a in zip(self._times, self._observed,
                                 self._are_mags) if o and a
        ]
        kmat = np.array([
            [vi * vj * np.exp(
                -0.5 * ((ti - tj) / kwargs['cowidth']) ** 2) for ti, vi in
             zip(self._o_m_times, self._band_vs)] for tj, vj in
            zip(self._o_m_times, self._band_vs)
        ])

        # full_size = np.count_nonzero(kmat)

        # Remove small covariance terms
        min_cov = self.MIN_COV_TERM * np.max(kmat)
        kmat[kmat <= min_cov] = 0.0

        # print("Sparse frac: {:.2%}".format(
        #     float(full_size - np.count_nonzero(kmat)) / full_size))

        for i in range(len(kmat)):
            kmat[i, i] += diag[i]

        # ovalue = -0.5 * (
        #     np.matmul(np.matmul(residuals.T, scipy.linalg.inv(kmat)),
        #               residuals) + np.log(scipy.linalg.det(kmat)))

        chol_kmat = scipy.linalg.cholesky(kmat)

        logdet = np.linalg.slogdet(chol_kmat)[-1]
        value = -0.5 * (
            np.matmul(residuals.T, scipy.linalg.cho_solve(
                (chol_kmat, False), residuals))) - logdet

        # print(value, ovalue)

        if isnan(value) or isnan(self._score_modifier + value):
            return {'value': LIKELIHOOD_FLOOR}
        return {'value': max(LIKELIHOOD_FLOOR, value + self._score_modifier)}

    def preprocess(self, **kwargs):
        """Construct arrays of observations based on data keys."""
        if self._preprocessed:
            return
        self._times = kwargs.get('times', [])
        self._mags = kwargs.get('magnitudes', [])
        self._fds = kwargs.get('fluxdensities', [])
        self._e_u_mags = kwargs.get('e_upper_magnitudes', [])
        self._e_l_mags = kwargs.get('e_lower_magnitudes', [])
        self._e_mags = kwargs.get('e_magnitudes', [])
        self._e_u_fds = kwargs.get('e_upper_fluxdensities', [])
        self._e_l_fds = kwargs.get('e_lower_fluxdensities', [])
        self._e_fds = kwargs.get('e_fluxdensities', [])
        self._u_fds = kwargs.get('u_fluxdensities', [])
        self._u_freqs = kwargs.get('u_frequencies', [])
        self._upper_limits = kwargs.get('upperlimits', [])
        self._observed = kwargs['observed']

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
        self._preprocessed = True
