from math import exp, pi

from astropy import constants as c

from ..module import Module

CLASS_NAME = 'Blackbody'


class Blackbody(Module):
    """Blackbody spectral energy distribution
    """

    FLUX_CONST = (2.0 * c.h / (c.c**2) * pi).value
    X_CONST = (c.h / c.k_B).value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def seds(self, **kwargs):
        self._temperature = kwargs['temperature']
        a = [self.X_CONST * x / self._temperature
             for x in kwargs['frequencies']]
        return [x**3 * self.FLUX_CONST / (exp(a) - 1.0)
                for x in kwargs['frequencies']]
