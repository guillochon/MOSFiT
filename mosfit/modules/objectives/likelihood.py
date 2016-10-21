from math import isnan

import numpy as np
from mosfit.constants import LIKELIHOOD_FLOOR
from mosfit.modules.module import Module

CLASS_NAME = 'Likelihood'


class Likelihood(Module):
    """Calculate the maximum likelihood score for a model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._preprocessed = False

    def process(self, **kwargs):
        self.preprocess(**kwargs)
        self._model_mags = kwargs['model_magnitudes']
        self._fractions = kwargs['fractions']
        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return {'value': LIKELIHOOD_FLOOR}
        for mi, mag in enumerate(self._model_mags):
            if (not self._upper_limits[mi] and
                    (isnan(mag) or not np.isfinite(mag))):
                return {'value': LIKELIHOOD_FLOOR}
        self._variance2 = kwargs['variance']**2

        sum_members = [
            (x - y if not u or (x < y and not isnan(x)) else 0.0)**2 / (
                (el if x > y else eu)**2 + self._variance2) +
            np.log(self._variance2 + 0.5 * ((0.0 if u else el)**2 + eu**2))
            for x, y, eu, el, u in zip(self._model_mags, self._mags,
                                       self._e_u_mags, self._e_l_mags,
                                       self._upper_limits)
        ]
        value = -0.5 * np.sum(sum_members)
        if isnan(value):
            return {'value': LIKELIHOOD_FLOOR}
        # if min(x) < 0.0:
        #     value = value + self._n_mags * np.sum([y for y in x if y < 0.0])
        # if max(x) > 1.0:
        #     value = value + self._n_mags * np.sum(
        #         [1.0 - y for y in x if y > 1.0])
        return {'value': value}

    def preprocess(self, **kwargs):
        if self._preprocessed:
            return
        self._times = kwargs['times']
        self._mags = kwargs['magnitudes']
        self._e_u_mags = kwargs['e_upper_magnitudes']
        self._e_l_mags = kwargs['e_lower_magnitudes']
        self._e_mags = kwargs['e_magnitudes']
        self._upper_limits = kwargs['upperlimits']
        self._e_u_mags = [
            kwargs['default_upper_limit_error']
            if (e == '' and eu == '' and self._upper_limits[i]) else
            (e if eu == '' else e)
            for i, (e, eu) in enumerate(zip(self._e_mags, self._e_u_mags))
        ]
        self._e_l_mags = [
            0.0 if self._upper_limits[i] else (e if el == '' else e)
            for i, (e, el) in enumerate(zip(self._e_mags, self._e_l_mags))
        ]
        self._n_mags = len(self._mags)
        self._preprocessed = True
