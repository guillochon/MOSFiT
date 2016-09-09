from ..module import Module

CLASS_NAME = 'Band'


class Band(Module):
    """Band-pass filter
    """

    def __init__(self, times, luminosities, band):
        self.luminosities = luminosities

    def luminosity(self):
        return self.luminosities
