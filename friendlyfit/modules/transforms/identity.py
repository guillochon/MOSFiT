class Identity:
    """Identity transform (no change to input).
    """

    def __init__(self, times, luminosities):
        self._luminosities = luminosities

    def luminosities(self):
        return self._luminosities
