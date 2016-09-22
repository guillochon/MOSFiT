import numpy as np

from extinction import apply as eapp
from extinction import odonnell94

from mosfit.modules.seds.sed import SED

CLASS_NAME = 'Extinction'


class Extinction(SED):
    """Adds extinction to SED from both host galaxy and MW.
    """

    MW_RV = 3.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._preprocessed = False

    def process(self, **kwargs):
        self.preprocess(**kwargs)
        self._seds = kwargs['seds']
        self._nh_host = kwargs['nhhost']

        av_host = self._nh_host / 1.8e21

        for si in range(len(self._bands)):
            cur_band = self._bands[si]
            bi = self._filters.find_band_index(cur_band)
            # First extinct out LOS dust from MW
            eapp(self._mw_extinct[si], self._seds[si], inplace=True)
            # Then extinct out host gal (using rest wavelengths)
            eapp(
                odonnell94(
                    np.array(self._band_rest_wavelengths[bi]), av_host,
                    self.MW_RV),
                self._seds[si],
                inplace=True)

        return {'bandwavelengths': self._band_wavelengths, 'seds': self._seds}

    def preprocess(self, **kwargs):
        if not self._preprocessed:
            zp1 = 1.0 + kwargs['redshift']
            self._ebv = kwargs['ebv']
            self._bands = kwargs['bands']
            self._band_rest_wavelengths = [[y / zp1 for y in x]
                                           for x in self._band_wavelengths]
            self._av_mw = self.MW_RV * self._ebv
            self._mw_extinct = []
            for si in range(len(self._bands)):
                cur_band = self._bands[si]
                bi = self._filters.find_band_index(cur_band)
                # First extinct out LOS dust from MW
                self._mw_extinct.append(
                    odonnell94(
                        np.array(self._band_wavelengths[bi]), self._av_mw,
                        self.MW_RV))
        self._preprocessed = True
