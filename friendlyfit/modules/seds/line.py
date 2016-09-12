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
        return {'bandwavelengths': self._band_wavelengths, 'seds': self._seds}
