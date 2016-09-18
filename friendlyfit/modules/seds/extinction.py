import numpy as np

from extinction import apply as eapp
from extinction import odonnell94

from .sed import SED

CLASS_NAME = 'Extinction'


class Extinction(SED):
    """Adds extinction to SED from both host galaxy and MW.
    """

    MW_RV = 3.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._ebv = kwargs['ebv']
        self._seds = kwargs['seds']
        self._bands = kwargs['bands']
        self._nh_host = kwargs['nhhost']
        zp1 = 1.0 + kwargs['redshift']
        self._band_rest_wavelengths = [[y / zp1 for y in x]
                                       for x in self._band_wavelengths]

        av = self.MW_RV * self._ebv
        av_host = self._nh_host / 1.8e21

        for si in range(len(self._seds)):
            cur_band = self._bands[si]
            bi = self._filters.find_band_index(cur_band)
            eapp(
                odonnell94(
                    np.array(self._band_wavelengths[bi]), av, self.MW_RV),
                self._seds[si],
                inplace=True)
            eapp(
                odonnell94(
                    np.array(self._band_rest_wavelengths[bi]), av_host,
                    self.MW_RV),
                self._seds[si],
                inplace=True)

        return {'bandwavelengths': self._band_wavelengths, 'seds': self._seds}
