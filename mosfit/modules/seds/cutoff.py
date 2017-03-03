import numpy as np
from astropy import constants as c

from mosfit.modules.seds.sed import SED

CLASS_NAME = 'Cutoff'


class Cutoff(SED):
    """Apply a cutoff in the UV to the SED.
    """

    def process(self, **kwargs):
        self._seds = kwargs['seds']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        zp1 = 1.0 + kwargs['redshift']
        for si, sed in enumerate(self._seds):
            bi = self._band_indices[si]
            if bi >= 0:
                wav_arr = self._sample_wavelengths[bi] / zp1
            else:
                wav_arr = [c.c.cgs.value / self._frequencies[si]]

            norm = np.trapz(sed, x=wav_arr)

            # Account for UV absorption: 0% transmission at 0 A, 100% at 3000A
            sed[wav_arr < 3000] *= (3.333e-4 * wav_arr[wav_arr < 3000])

            # Normalize SED so no energy is lost
            sed *= norm / np.trapz(sed, x=wav_arr)

            self._seds[si] = sed


        return {'seds': self._seds}
