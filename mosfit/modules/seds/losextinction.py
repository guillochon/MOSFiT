import numpy as np

from extinction import odonnell94, apply as eapp
from mosfit.modules.seds.sed import SED

CLASS_NAME = 'LOSExtinction'


class LOSExtinction(SED):
    """Adds extinction to SED from both host galaxy and MW.
    """

    MW_RV = 3.1

    def __init__(self, **kwargs):
        super(LOSExtinction, self).__init__(**kwargs)
        self._preprocessed = False

    def process(self, **kwargs):
        self.preprocess(**kwargs)
        zp1 = 1.0 + kwargs['redshift']
        self._seds = kwargs['seds']
        self._nh_host = kwargs['nhhost']
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._band_rest_wavelengths = np.array(
            [np.array(x) / zp1 for x in self._sample_wavelengths])
        self._mw_extinct = []
        for si, cur_band in enumerate(self._bands):
            bi = self._band_indices[si]
            # First extinct out LOS dust from MW
            self._mw_extinct.append(
                odonnell94(
                    np.array(self._sample_wavelengths[bi]), self._av_mw,
                    self.MW_RV))

        av_host = self._nh_host / 1.8e21

        for si, cur_band in enumerate(self._bands):
            bi = self._band_indices[si]
            # First extinct out LOS dust from MW
            eapp(self._mw_extinct[si], self._seds[si], inplace=True)
            # Then extinct out host gal (using rest wavelengths)
            eapp(
                odonnell94(self._band_rest_wavelengths[bi], av_host,
                                self.MW_RV),
                self._seds[si],
                inplace=True)

        return {
            'sample_wavelengths': self._sample_wavelengths,
            'seds': self._seds
        }

    def preprocess(self, **kwargs):
        if not self._preprocessed:
            self._ebv = kwargs['ebv']
            self._av_mw = self.MW_RV * self._ebv
        self._preprocessed = True
