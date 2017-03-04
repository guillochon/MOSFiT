"""Definitions for the `Cutoff` class."""
import numpy as np
from astropy import constants as c

from mosfit.modules.seds.sed import SED

# Important: Only define one ``Module`` class per file.


class Cutoff(SED):
    """Apply a cutoff in the UV to the SED."""

    def process(self, **kwargs):
        """Process module."""
        self._seds = kwargs['seds']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        for si, sed in enumerate(self._seds):
            bi = self._band_indices[si]
            if bi >= 0:
                wav_arr = self._sample_wavelengths[bi]
            else:
                wav_arr = [c.c.cgs.value / self._frequencies[si]]

            # Account for UV absorption
            norm = np.sum(sed)
            sed[wav_arr < 3000] *= (
                0.0003 * wav_arr[wav_arr < 3000] - 0.0445)

            sed[sed < 0.0] = 0.0

            sed = np.nan_to_num(sed)

            # Normalize SED so no energy is lost
            sed *= norm / np.sum(sed)

            self._seds[si] = sed

        return {'seds': self._seds}
