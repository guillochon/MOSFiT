from math import exp, pi, sqrt

import numpy as np
from astropy import constants as c
from astropy import units as u

from ..module import Module

CLASS_NAME = 'Blackbody'


class Blackbody(Module):
    """Blackbody spectral energy distribution
    """

    FLUX_CONST = (2.0 * c.h / (c.c**2) * pi).value
    X_CONST = (c.h / c.k_B).value
    C_CONST = (c.c / u.Angstrom).value
    STEF_CONST = (4.0 * pi * c.sigma_sb).value
    FOUR_PI = 4.0 * pi
    N_PTS = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._wavelengths = []

    def process(self, **kwargs):
        self._luminosities = kwargs['luminosities']
        self._temperature = kwargs['temperature']
        radii = [sqrt(x / (self.STEF_CONST * self._temperature**4))
                 for x in self._luminosities]
        a = [self.X_CONST * x / self._temperature for x in self._frequencies]
        return {'seds': [[
            self.FOUR_PI * z**2 * x**3 * self.FLUX_CONST / (exp(y) - 1.0)
            for x, y in zip(self._frequencies, a)
        ] for z in radii]}

    def handle_requests(self, **requests):
        wavelength_ranges = requests.get('wavelengths', [])
        if not wavelength_ranges:
            return
        for rng in wavelength_ranges:
            self._wavelengths.extend(
                list(np.arange(rng[0], rng[1], (rng[1] - rng[0]) /
                               self.N_PTS)))
        self._wavelengths = list(set(self._wavelengths))
        self._wavelengths.sort()
        self._frequencies = [self.C_CONST / x for x in self._wavelengths]
