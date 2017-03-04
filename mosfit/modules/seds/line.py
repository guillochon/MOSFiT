"""Definitions for the `Line` class."""
from mosfit.modules.seds.sed import SED

# Important: Only define one `Module` class per file.


class Line(SED):
    """Line spectral energy distribution, modifies existing SED."""

    def process(self, **kwargs):
        """Process module."""
        self._seds = kwargs['seds']
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._sample_wavelengths = kwargs['sample_wavelengths']
        self._luminosities = kwargs['luminosities']
        zp1 = 1.0 + kwargs['redshift']

        seds = []
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if bi >= 0:
                rest_freqs = self._sample_frequencies[bi] * zp1
            else:
                rest_freqs = [self._frequencies[li] * zp1]

            # Dummy function for now, needs implementation
            sed = [0.0 for x in rest_freqs]
            seds.append(list(sed))

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths, 'seds': seds}
