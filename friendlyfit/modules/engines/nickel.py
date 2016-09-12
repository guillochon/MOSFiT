from math import isnan

import numpy as np

from ...constants import LOG2
from ..module import Module

CLASS_NAME = 'Nickel'


class Nickel(Module):
    """Nickel decay engine
    """

    HALF_LIFE = 6.075  # in days
    DECAY_ENER = 2.803809e-6  # in ergs
    NUM_NI56_SUN = 2.14125561e55  # num ni56 atoms in solar mass of ni56
    DECAY_CONST = LOG2 * NUM_NI56_SUN * DECAY_ENER / HALF_LIFE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._times = kwargs['times']
        self._mnickel = kwargs['mnickel']
        self._texplosion = kwargs['texplosion']

        decay_facs = [np.inf if self._texplosion > x else np.power(0.5, (
            (x - self._texplosion) / self.HALF_LIFE)) for x in self._times]
        current_mnickel = [0.0 if isnan(x) else self._mnickel * x
                           for x in decay_facs]
        print(self._mnickel, min(current_mnickel), max(current_mnickel))

        luminosities = [x * y * self.DECAY_CONST
                        for x, y in zip(decay_facs, current_mnickel)]
        print(max(luminosities))
        return {'luminosities': luminosities}
