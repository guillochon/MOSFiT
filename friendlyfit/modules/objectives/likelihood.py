import numpy as np

from ..module import Module

CLASS_NAME = 'Likelihood'


class Likelihood(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        ret = {}

        ret['value'] = np.random.rand()
        return ret
