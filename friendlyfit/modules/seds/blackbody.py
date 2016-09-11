from math import pi

import numpy as np
from astropy import constants as c
from astropy import units as u

from ...constants import FOUR_PI
from ..module import Module

CLASS_NAME = 'Blackbody'


class Blackbody(Module):
    """Blackbody spectral energy distribution
    """

    FLUX_CONST = FOUR_PI * (2.0 * c.h / (c.c**2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    C_CONST = (c.c / u.Angstrom).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    N_PTS = 16 + 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._band_wavelengths = []
        self._band_names = []

    def process(self, **kwargs):
        self._luminosities = kwargs['luminosities']
        self._temperature = kwargs['temperature']
        self._bands = kwargs['bands']
        zp1 = (1.0 + kwargs['redshift'])
        seds = []
        for li, lum in enumerate(self._luminosities):
            cur_band = self._bands[li]
            bi = self._band_names.index(cur_band)
            rest_freqs = [x * zp1 for x in self._band_frequencies[bi]]
            rest_freqs3 = [x**3 for x in rest_freqs]
            radii2 = lum / (self.STEF_CONST * self._temperature**4)
            a = [np.exp(self.X_CONST * x / self._temperature) - 1.0
                 for x in rest_freqs]
            sed = [
                (self.FLUX_CONST * radii2 * x / y)
                for x, y in zip(rest_freqs3, a)
            ]
            seds.append(sed)
        return {'bandwavelengths': self._band_wavelengths, 'seds': seds}

    def handle_requests(self, **requests):
        wavelength_ranges = requests.get('bandwavelengths', [])
        self._band_names.extend(requests.get('bandnames', []))
        if not wavelength_ranges:
            return
        for rng in wavelength_ranges:
            self._band_wavelengths.append(
                list(np.linspace(rng[0], rng[1], self.N_PTS)))
        self._band_frequencies = [[self.C_CONST / x for x in y]
                                  for y in self._band_wavelengths]
