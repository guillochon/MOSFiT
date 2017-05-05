"""Definitions for the `Likelihood` class."""
from math import isnan

import numpy as np
import scipy

from mosfit.constants import LIKELIHOOD_FLOOR
from mosfit.modules.module import Module
from mosfit.utils import flux_density_unit


# Important: Only define one ``Module`` class per file.


class Likelihood(Module):
    """Calculate the maximum likelihood score for a model."""

    MIN_COV_TERM = 1.0e-30

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)
        self._score_modifier = kwargs.get(self.key('score_modifier'), 0.0)
        self._fractions = kwargs['fractions']
        self._model_observations = kwargs['model_observations']
        self._upper_limits = np.array(kwargs.get('upperlimits', []),
                                      dtype=bool)

        ret = {'value': LIKELIHOOD_FLOOR}

        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return ret
        for oi, obs in enumerate(self._model_observations):
            if not self._upper_limits[oi] and (isnan(obs) or
                                               not np.isfinite(obs)):
                return ret
        value = kwargs.get('value', LIKELIHOOD_FLOOR)
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
        if 'kmat' in kwargs:
            kmat = kwargs['kmat']

            kn = len(diag)

            # Add observed errors to diagonal
            for i in range(kn):
                kmat[i, i] += diag[i]

            # full_size = np.count_nonzero(kmat)

            # Remove small covariance terms
            # min_cov = self.MIN_COV_TERM * np.max(kmat)
            # kmat[kmat <= min_cov] = 0.0

            # print("Sparse frac: {:.2%}".format(
            #     float(full_size - np.count_nonzero(kmat)) / full_size))

            # try:
            #     import skcuda.linalg as skla
            #     import pycuda.gpuarray as gpuarray
            #
            #     chol_kmat = scipy.linalg.cholesky(kmat, check_finite=False)
            #     chol_kmat_gpu = gpuarray.to_gpu(
            #         np.asarray(chol_kmat, np.float64))
            #     value = -np.log(skla.det(chol_kmat_gpu, lib='cusolver'))
            #     res_gpu = gpuarray.to_gpu(np.asarray(residuals.reshape(
            #         len(residuals), 1), np.float64))
            #     # Right now cho_solve not working with cusolver lib.
            #     cho_mat_gpu = gpuarray.to_gpu(
            #         np.asarray(scipy.linalg.cho_solve(
            #             (chol_kmat, False), residuals,
            #             check_finite=False), np.float64))
            #     value -= (0.5 * (
            #         skla.mdot(skla.transpose(res_gpu), cho_mat_gpu))).get()
            #     # value -= 0.5 * (
            #     #     skla.mdot(skla.transpose(res_gpu), skla.cho_solve(
            #     #         chol_kmat_gpu, res_gpu, lib='cusolver')))
            # except ImportError:
            try:
                chol_kmat = scipy.linalg.cholesky(kmat, check_finite=False)

                value = -np.linalg.slogdet(chol_kmat)[-1]
                value -= 0.5 * (
                    np.matmul(residuals.T, scipy.linalg.cho_solve(
                        (chol_kmat, False), residuals,
                        check_finite=False)))
            except Exception:
                value = -0.5 * (
                    np.matmul(np.matmul(residuals.T,
                                        scipy.linalg.inv(kmat)),
                              residuals) + np.log(scipy.linalg.det(kmat)))
        else:
            # Shortcut when matrix is diagonal.
            self._o_band_vs = kwargs['obandvs']
            value = -0.5 * np.sum(
                residuals ** 2 / (self._o_band_vs ** 2 + diag) +
                np.log(self._o_band_vs ** 2 + diag))

        score = self._score_modifier + value
        if isnan(score) or not np.isfinite(score):
            return ret
        ret['value'] = max(LIKELIHOOD_FLOOR, score)
        return ret

    def preprocess(self, **kwargs):
        """Construct arrays of observations based on data keys."""
        self._mags = np.array(kwargs.get('magnitudes', []))
        self._fds = np.array(kwargs.get('fluxdensities', []))
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
        self._upper_limits = np.array(kwargs.get('upperlimits', []),
                                      dtype=bool)
        self._observation_types = np.array(kwargs['observation_types'])
        self._observed = np.array(kwargs.get('observed', []), dtype=bool)

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
        self._cmask = self._observation_types == 'countrate'
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
        self._fds = np.array([
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._fds, self._u_fds)
        ])
        self._e_u_fds = [
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._e_u_fds, self._u_fds)
        ]
        self._e_l_fds = [
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._e_l_fds, self._u_fds)
        ]
