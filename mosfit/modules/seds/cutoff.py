import numpy as np
from mosfit.modules.seds.sed import SED

CLASS_NAME = 'Cutoff'


class Cutoff(SED):
    """Apply a cutoff in the UV to the SED.
    """

    def process(self, **kwargs):
        self._seds = kwargs['seds']
        self._band_indices = kwargs['all_band_indices']
        for si, sed in enumerate(self._seds):
            bi = self._band_indices[si]
            wav_arr = np.array(self._sample_wavelengths[bi])

            # Account for UV absorption
            norm = np.sum(sed)
            sed[wav_arr < 3500] *= (
                0.00038 * wav_arr[wav_arr < 3500] - 0.32636)

            sed[sed < 0.0] = 0.0

            # Normalize SED so no energy is lost
            sed *= norm / np.sum(sed)

            self._seds[si] = sed

        return {'seds': self._seds}
