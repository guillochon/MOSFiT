from math import exp, pi, sqrt

import numpy as np
from astropy import constants as c
from astropy import units as u

from ...constants import FOUR_PI
from ..module import Module

CLASS_NAME = 'Blackbody'


class Blackbody(Module):
    """Blackbody spectral energy distribution
    """

    FLUX_CONST = (2.0 * c.h / (c.c**2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    C_CONST = (c.c / u.Angstrom).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    N_PTS = 16 + 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._wavelengths = []
        self._bands = []

    def process(self, **kwargs):
        self._luminosities = kwargs['luminosities']
        self._temperature = kwargs['temperature']
        zp1 = (1.0 + kwargs['redshift'])
        seds = []
        for wi, waveset in enumerate(self._wavelengths):
            rest_freqs = [x * zp1 for x in self._frequencies[wi]]
            radii = [sqrt(x / (self.STEF_CONST * self._temperature**4))
                     for x in self._luminosities]
            a = [self.X_CONST * x / self._temperature for x in rest_freqs]
            sed = [
                [
                    (FOUR_PI * z**2 * x**3 * self.FLUX_CONST / (exp(y) - 1.0))
                    for x, y in zip(rest_freqs, a)
                ] for z in radii
            ]
            seds.append(sed)
        return {'bands': self._bands,
                'wavelengths': self._wavelengths,
                'seds': seds}

    def handle_requests(self, **requests):
        wavelength_ranges = requests.get('wavelengths', [])
        self._bands.extend(requests.get('bands', []))
        if not wavelength_ranges:
            return
        for rng in wavelength_ranges:
            self._wavelengths.append(
                list(np.arange(rng[0], rng[1], (rng[1] - rng[0]) /
                               self.N_PTS)))
        self._frequencies = [[self.C_CONST / x for x in y]
                             for y in self._wavelengths]
