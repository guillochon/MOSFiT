import numpy as np

from ..module import Module
from ...constants import C_CGS

CLASS_NAME = 'SED'


class SED(Module):
    """Blackbody spectral energy distribution
    """

    N_PTS = 16 + 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._band_wavelengths = []
        self._band_names = []

    def handle_requests(self, **requests):
        wavelength_ranges = requests.get('bandwavelengths', [])
        self._band_names.extend(requests.get('bandnames', []))
        if not wavelength_ranges:
            return
        for rng in wavelength_ranges:
            self._band_wavelengths.append(
                list(np.linspace(rng[0], rng[1], self.N_PTS)))
        self._band_frequencies = [[C_CGS / x for x in y]
                                  for y in self._band_wavelengths]
