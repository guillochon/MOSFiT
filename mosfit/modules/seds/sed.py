import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.modules.module import Module

CLASS_NAME = 'SED'


class SED(Module):
    """Template class for SED Modules.
    """

    C_CONST = (c.c / u.Angstrom).cgs.value
    N_PTS = 16 + 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sample_wavelengths = []
        self._filters = []

    def handle_requests(self, **requests):
        self._filters = requests.get('filters', [])
        self._sample_wavelengths = requests.get('samplewavelengths', [])
        if not self._sample_wavelengths:
            wave_ranges = requests.get('band_wave_ranges', [])
            if not wave_ranges:
                return
            for rng in wave_ranges:
                self._sample_wavelengths.append(
                    list(np.linspace(rng[0], rng[1], self.N_PTS)))
        self._sample_frequencies = [[self.C_CONST / x for x in y]
                                    for y in self._sample_wavelengths]

    def add_to_existing_seds(self, new_seds, **kwargs):
        old_seds = kwargs.get('seds', None)
        if old_seds is not None:
            new_seds = [(i + j for i, j in zip(x, y))
                        for x, y in zip(old_seds, list(new_seds))]
        return new_seds

    def request(self, request):
        if request == 'filters':
            return self._filters
        elif request == 'samplewavelengths':
            return self._sample_wavelengths
        return []
