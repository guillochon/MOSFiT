from math import isnan

import numpy as np

from ..module import Module

CLASS_NAME = 'Likelihood'


class Likelihood(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._model_mags = kwargs['model_magnitudes']
        # raise SystemExit
        for mag in self._model_mags:
            if isnan(mag):
                return {'value': -np.inf}
        self._variance2 = kwargs['variance']**2
        self._mags = kwargs['magnitudes']
        self._e_mags = kwargs['e_magnitudes']

        # print(self._variance2, self._mags[:5], self._e_mags[:5],
        #       self._model_mags[:5])

        ret = {'value': -0.5 * np.sum(
            [(x - y)**2 /
             (z**2 + self._variance2) + np.log(self._variance2 + z**2)
             for x, y, z in zip(self._model_mags, self._mags, self._e_mags)])}
        return ret
