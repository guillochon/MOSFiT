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
            wav_arr = self._sample_wavelengths[bi]

            # Account for UV absorption
            norm = np.sum(sed)
            sed[wav_arr < 3000] *= (
                0.0003 * wav_arr[wav_arr < 3000] - 0.0445)

            sed[sed < 0.0] = 0.0

            # Normalize SED so no energy is lost
            sed *= norm / np.sum(sed)

            self._seds[si] = sed

        return {'seds': self._seds}
