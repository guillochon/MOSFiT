from ..module import Module

CLASS_NAME = 'Diffusion'


class Diffusion(Module):
    """Identity transform (no change to input).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._times = kwargs['times']
        self._luminosities = kwargs['luminosities']
        return {'luminosities': self._luminosities}
