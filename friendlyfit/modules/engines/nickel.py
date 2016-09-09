from math import log

from ..module import Module

CLASS_NAME = 'Nickel'


class Nickel(Module):
    """Nickel decay engine
    """

    HALF_LIFE = 6.075 * 86400.
    DECAY_ENER = 3.42166689e-6
    NUM_NI56_SUN = 2.14125561e55

    def __init__(self, times, mnickel, texplosion):
        self.times = times
        self.mnickel = mnickel
        self.texplosion = texplosion

    def luminosity(self):
        decay_facs = [(x - self.texplosion) / self.HALF_LIFE
                      for x in self.times]
        current_mnickel = [self.mnickel * 0.5**x for x in decay_facs]

        return [current_mnickel * 2.0**x * log(2.0) / self.HALF_LIFE
                for x in decay_facs]
