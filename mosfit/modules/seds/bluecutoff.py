"""Definitions for the `BlueCutoff` class."""
import numpy as np
from astropy import constants as c

from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class BlueCutoff(SED):
    """Apply a cutoff blueward of the specified wavelength."""

    def process(self, **kwargs):
        """Process module."""
        self._seds = kwargs['seds']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._cutoff_wavelength = kwargs['cutoff_wavelength']
        zp1 = 1.0 + kwargs['redshift']
        for si, sed in enumerate(self._seds):
            bi = self._band_indices[si]
            if bi >= 0:
                wav_arr = self._sample_wavelengths[bi] / zp1
            else:
                wav_arr = [c.c.cgs.value / (zp1 * self._frequencies[si])]

            norm = np.trapz(sed, x=wav_arr)

            # Account for UV absorption: 0% transmission at 0 A, 100% at cutoff
            # wavelength.
            indices = wav_arr < self._cutoff_wavelength
            sed[indices] *= (wav_arr[indices] / self._cutoff_wavelength)

            sed[sed < 0.0] = 0.0

            sed = np.nan_to_num(sed)

            # Normalize SED so no energy is lost
            sed *= norm / np.trapz(sed, x=wav_arr)

            self._seds[si] = sed

        return {'seds': self._seds}
