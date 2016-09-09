class Band:
    """Blackbody spectral energy distribution
    """

    def __init__(self, times, luminosities, band):
        self.luminosities = luminosities

    def luminosity(self):
        return self.luminosities
