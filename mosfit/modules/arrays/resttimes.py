import numpy as np
from mosfit.modules.module import Module

CLASS_NAME = 'RestTimes'


class RestTimes(Module):
    """This class converts the observed times to rest-frame times.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._times = kwargs['times']
        self._t_explosion = kwargs['texplosion']

        outputs = {}
        outputs['resttimes'] = [
            x / (1.0 + kwargs['redshift']) for x in self._times
        ]
        outputs['resttexplosion'] = self._t_explosion / (
            1.0 + kwargs['redshift'])
        return outputs
