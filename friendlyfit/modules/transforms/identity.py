from ..module import Module

CLASS_NAME = 'Identity'


class Identity(Module):
    """Identity transform (no change to input).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._luminosities = kwargs['luminosities']
        return {'luminosities': self._luminosities}
