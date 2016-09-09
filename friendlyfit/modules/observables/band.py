from ..module import Module

CLASS_NAME = 'Band'


class Band(Module):
    """Band-pass filter
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def luminosity(self, **kwargs):
        self.luminosities = kwargs['luminosities']
        return self.luminosities
