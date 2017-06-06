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
        ret = {'value': LIKELIHOOD_FLOOR}

        self._fractions = kwargs.get('fractions', [])
        if not len(self._fractions):
            return ret

        self._model_observations = kwargs['model_observations']
        self._score_modifier = kwargs.get(self.key('score_modifier'), 0.0)
        self._upper_limits = np.array(kwargs.get('upperlimits', []),
                                      dtype=bool)

        value = ret['value']

        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return ret
        for oi, obs in enumerate(self._model_observations):
            if not self._upper_limits[oi] and (isnan(obs) or
                                               not np.isfinite(obs)):
                return ret

        diag = kwargs.get('kdiagonal', None)
        residuals = kwargs.get('kresiduals', None)

        if diag is None or residuals is None:
            return ret

        if kwargs.get('kmat', None) is not None:
            kmat = kwargs['kmat']

            # Add observed errors to diagonal
            kmat[np.diag_indices_from(kmat)] += diag

            # full_size = np.count_nonzero(kmat)

            # Remove small covariance terms
            # min_cov = self.MIN_COV_TERM * np.max(kmat)
            # kmat[kmat <= min_cov] = 0.0

            # print("Sparse frac: {:.2%}".format(
            #     float(full_size - np.count_nonzero(kmat)) / full_size))

            condn = np.linalg.cond(kmat)
            if condn > 1.0e10:
                return ret

            use_cpu = True
            if self._model._fitter._cuda:
                use_cpu = False
                try:
                    import pycuda.gpuarray as gpuarray
                    import skcuda.linalg as skla

                    kmat_gpu = gpuarray.to_gpu(kmat)
                    # kmat will now contain the cholesky decomp.
                    skla.cholesky(kmat_gpu, lib='cusolver')
                    value = -np.log(skla.det(kmat_gpu, lib='cusolver'))
                    res_gpu = gpuarray.to_gpu(residuals.reshape(
                        len(residuals), 1))
                    cho_mat_gpu = res_gpu.copy()
                    skla.cho_solve(kmat_gpu, cho_mat_gpu, lib='cusolver')
                    value -= (0.5 * (
                        skla.mdot(skla.transpose(res_gpu),
                                  cho_mat_gpu)).get())[0][0]
                except ImportError:
                    use_cpu = True

            if use_cpu:
                try:
                    chol_kmat = scipy.linalg.cholesky(kmat, check_finite=False)

                    value = -np.linalg.slogdet(chol_kmat)[-1]
                    value -= 0.5 * (
                        np.matmul(residuals.T, scipy.linalg.cho_solve(
                            (chol_kmat, False), residuals,
                            check_finite=False)))
                except Exception:
                    try:
                        value = -0.5 * (
                            np.matmul(
                                np.matmul(
                                    residuals.T, scipy.linalg.inv(kmat)),
                                residuals) + np.log(scipy.linalg.det(kmat)))
                    except scipy.linalg.LinAlgError:
                        return ret

            ret['kdiagonal'] = diag
            ret['kresiduals'] = residuals
        elif 'kfmat' in kwargs:
            raise RuntimeError('Should not have kfmat in likelihood!')
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
