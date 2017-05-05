"""Definitions for the `Likelihood` class."""
from math import isnan

import numpy as np
import scipy

from mosfit.constants import LIKELIHOOD_FLOOR
from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


class Likelihood(Module):
    """Calculate the maximum likelihood score for a model."""

    MIN_COV_TERM = 1.0e-30

    def process(self, **kwargs):
        """Process module."""
        self._score_modifier = kwargs.get(self.key('score_modifier'), 0.0)
        ret = {'value': LIKELIHOOD_FLOOR}
        value = kwargs.get('value', LIKELIHOOD_FLOOR)
        if 'kmat' in kwargs:
            kmat = kwargs['kmat']
            diag = kwargs['diag']
            residuals = kwargs['residuals']

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

        score = self._score_modifier + value
        if isnan(score) or not np.isfinite(score):
            return ret
        ret['value'] = max(LIKELIHOOD_FLOOR, score)
        return ret
