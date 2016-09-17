import numpy as np
from astropy import constants as c
from astropy import units as u

from ..module import Module

CLASS_NAME = 'SED'


class SED(Module):
    """Template class for SED Modules.
    """

    C_CONST = (c.c / u.Angstrom).cgs.value
    N_PTS = 16 + 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._band_wavelengths = []
        self._filters = []

    def handle_requests(self, **requests):
        wavelength_ranges = requests.get('bandwavelengths', [])
        self._filters = requests.get('filters', [])
        if not wavelength_ranges:
            return
        for rng in wavelength_ranges:
            self._band_wavelengths.append(
                list(np.linspace(rng[0], rng[1], self.N_PTS)))
        self._band_frequencies = [[self.C_CONST / x for x in y]
                                  for y in self._band_wavelengths]

    def request(self, request):
        if request == 'filters':
            return self._filters
        return []
