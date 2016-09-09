from math import pi, exp

from astropy import units as u


class Blackbody:
    """Blackbody spectral energy distribution
    """

    FLUX_CONST = (2.0*u.h/(u.c**2)*pi).value
    X_CONST = (u.h/u.kb).value

    def __init__(self, temperature=1.e4):
        self._temperature = temperature

    def seds(self, frequencies):
        a = [self.X_CONST * x / self._temperature for x in frequencies]
        return [x**3*self.FLUX_CONST/(exp(a) - 1.0) for x in frequencies]
