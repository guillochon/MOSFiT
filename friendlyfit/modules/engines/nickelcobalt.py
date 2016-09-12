from math import isnan

import numpy as np

from ..module import Module

CLASS_NAME = 'NickelCobalt'


class NickelCobalt(Module):
    """Nickel/Cobalt decay engine
    """

    NI56_LUM = 6.45e43
    CO56_LUM = 1.45e43
    NI56_LIFE = 8.8
    CO56_LIFE = 111.3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._times = kwargs['times']
        self._mnickel = kwargs['mnickel']
        self._texplosion = kwargs['texplosion']

        # From 1994ApJS...92..527N
        ts = [np.inf if self._texplosion > x else (x - self._texplosion)
              for x in self._times]
        luminosities = [self._mnickel * (self.NI56_LUM * np.exp(
            -t / self.NI56_LIFE) + self.CO56_LUM * np.exp(-t / self.CO56_LIFE))
                        for t in ts]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        # print(max(luminosities))
        return {'luminosities': luminosities}
