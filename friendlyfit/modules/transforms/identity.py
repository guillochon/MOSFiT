from ..module import Module

CLASS_NAME = 'Identity'


class Identity(Module):
    """Identity transform (no change to input).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def luminosities(self, **kwargs):
        self._luminosities = kwargs['luminosities']
        return self._luminosities
