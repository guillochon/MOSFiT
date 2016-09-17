from .sed import SED

CLASS_NAME = 'Line'


class Line(SED):
    """Line spectral energy distribution, modifies existing SED.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._seds = kwargs['seds']
        self._band_wavelengths = kwargs['bandwavelengths']
        self._luminosities = kwargs['luminosities']
        zp1 = 1.0 + kwargs['redshift']

        seds = []
        for li, lum in enumerate(self._luminosities):
            cur_band = self._bands[li]
            bi = self._filters.find_band_index(cur_band)
            rest_freqs = [x * zp1 for x in self._band_frequencies[bi]]

            # Dummy function for now, needs implementation
            sed = [0.0 for x in rest_freqs]
            seds.append(list(sed))

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'bandwavelengths': self._band_wavelengths, 'seds': seds}
