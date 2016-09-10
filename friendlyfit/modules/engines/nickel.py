from math import isnan

import numpy as np

from ...constants import LOG2
from ..module import Module

CLASS_NAME = 'Nickel'


class Nickel(Module):
    """Nickel decay engine
    """

    HALF_LIFE = 6.075  # in days
    DECAY_ENER = 3.42166689e-6  # in ergs
    NUM_NI56_SUN = 2.14125561e55  # num ni56 atoms in solar mass of ni56
    DECAY_CONST = LOG2 * NUM_NI56_SUN * DECAY_ENER / HALF_LIFE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self.times = kwargs['times']
        self.mnickel = kwargs['mnickel']
        self.texplosion = kwargs['texplosion']

        decay_facs = [np.inf if self.texplosion > x else
                      ((x - self.texplosion) / self.HALF_LIFE)
                      for x in self.times]
        current_mnickel = [self.mnickel * np.power(0.5, x) for x in decay_facs]
        current_mnickel = [0.0 if isnan(x) else x for x in current_mnickel]
        # print(self.mnickel, min(current_mnickel), max(current_mnickel))

        luminosities = [y * np.power(2.0, -x) * self.DECAY_CONST
                        for x, y in zip(decay_facs, current_mnickel)]
        # print(max(luminosities))
        return {'luminosities': luminosities}
