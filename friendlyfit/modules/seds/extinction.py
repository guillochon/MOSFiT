import numpy as np

from extinction import apply as eapp
from extinction import odonnell94

from .sed import SED

CLASS_NAME = 'Extinction'


class Extinction(SED):
    """Expanding/receding photosphere with a core+envelope
    morphology and a blackbody spectral energy
    distribution.
    """

    MW_RV = 3.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._ebv = kwargs['ebv']
        self._seds = kwargs['seds']
        self._bands = kwargs['bands']

        av = self.MW_RV * self._ebv

        for si in range(len(self._seds)):
            cur_band = self._bands[si]
            bi = self._filters.find_band_index(cur_band)
            eapp(
                odonnell94(
                    np.array(self._band_wavelengths[bi]), av, self.MW_RV),
                self._seds[si],
                inplace=True)

        return {'bandwavelengths': self._band_wavelengths, 'seds': self._seds}
