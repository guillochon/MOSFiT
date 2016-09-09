from math import exp, pi

from astropy import constants as c

from ..module import Module

CLASS_NAME = 'Blackbody'


class Blackbody(Module):
    """Blackbody spectral energy distribution
    """

    FLUX_CONST = (2.0 * c.h / (c.c**2) * pi).value
    X_CONST = (c.h / c.k_B).value

    def __init__(self, temperature=1.e4):
        self._temperature = temperature

    def seds(self, frequencies):
        a = [self.X_CONST * x / self._temperature for x in frequencies]
        return [x**3 * self.FLUX_CONST / (exp(a) - 1.0) for x in frequencies]
