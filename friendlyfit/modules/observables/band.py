from ..module import Module

CLASS_NAME = 'Band'


class Band(Module):
    """Band-pass filter
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._seds = kwargs['seds']
        return {'seds': self._seds}

    def request(self, request):
        if request == 'wavelengths':
            return [1000., 10000.]
        return []
