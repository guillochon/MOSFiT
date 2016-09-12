from ...modules.seds.sed import SED

CLASS_NAME = 'Blackbody'


class Blackbody(SED):
    """Blackbody spectral energy distribution
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._seds = kwargs['seds']
        self._band_wavelengths = kwargs['bandwavelengths']
        return {'bandwavelengths': self._band_wavelengths, 'seds': self._seds}
