from mosfit.modules.seds.sed import SED

CLASS_NAME = 'Line'


class Line(SED):
    """Line spectral energy distribution, modifies existing SED.
    """

    def process(self, **kwargs):
        self._seds = kwargs['seds']
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._sample_wavelengths = kwargs['sample_wavelengths']
        self._luminosities = kwargs['luminosities']
        zp1 = 1.0 + kwargs['redshift']

        seds = []
        for li, lum in enumerate(self._luminosities):
            cur_band = self._bands[li]
            bi = self._band_indices[li]
            rest_freqs = [x * zp1 for x in self._sample_frequencies[bi]]

            # Dummy function for now, needs implementation
            sed = [0.0 for x in rest_freqs]
            seds.append(list(sed))

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths, 'seds': seds}
