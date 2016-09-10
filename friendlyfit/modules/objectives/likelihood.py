from math import isnan

import numpy as np

from ..module import Module

CLASS_NAME = 'Likelihood'


class Likelihood(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._model_mags = kwargs['model_magnitudes']
        for mag in self._model_mags:
            if isnan(mag):
                return {'value': -np.inf}
        self._mags = kwargs['magnitudes']
        self._e_mags = kwargs['e_magnitudes']

        # Chi2 for now
        ret = {'value': -0.5 * np.sum(
            [(x - y)**2 / z**2
             for x, y, z in zip(self._model_mags, self._mags, self._e_mags)])}
        print(ret)
        return ret
