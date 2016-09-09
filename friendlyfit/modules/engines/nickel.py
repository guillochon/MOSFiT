from math import log

from ..module import Module

CLASS_NAME = 'Nickel'


class Nickel(Module):
    """Nickel decay engine
    """

    HALF_LIFE = 6.075 * 86400.
    DECAY_ENER = 3.42166689e-6
    NUM_NI56_SUN = 2.14125561e55

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self.times = kwargs['times']
        self.mnickel = kwargs['mnickel']
        self.texplosion = kwargs['texplosion']

        decay_facs = [(x - self.texplosion) / self.HALF_LIFE
                      for x in self.times]
        current_mnickel = [self.mnickel * 0.5**x for x in decay_facs]

        return {'luminosities': [y * 2.0**x * log(2.0) / self.HALF_LIFE
                for x, y in zip(decay_facs, current_mnickel)]}
