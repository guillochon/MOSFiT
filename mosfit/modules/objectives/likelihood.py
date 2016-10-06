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

    def process(self, **kwargs):
        self._model_mags = kwargs['model_magnitudes']
        self._fractions = kwargs['fractions']
        if min(self._fractions) < 0.0 or max(self._fractions) > 1.0:
            return {'value': LIKELIHOOD_FLOOR}
        for mag in self._model_mags:
            if isnan(mag):
                return {'value': LIKELIHOOD_FLOOR}
        self._variance2 = kwargs['variance']**2
        self._mags = kwargs['magnitudes']
        self._e_mags = kwargs['e_magnitudes']
        self._n_mags = len(self._mags)

        value = -0.5 * np.sum(
            [(x - y)**2 /
             (z**2 + self._variance2) + np.log(self._variance2 + z**2)
             for x, y, z in zip(self._model_mags, self._mags, self._e_mags)])
        if isnan(value):
            return {'value': LIKELIHOOD_FLOOR}
        # if min(x) < 0.0:
        #     value = value + self._n_mags * np.sum([y for y in x if y < 0.0])
        # if max(x) > 1.0:
        #     value = value + self._n_mags * np.sum(
        #         [1.0 - y for y in x if y > 1.0])
        return {'value': value}
