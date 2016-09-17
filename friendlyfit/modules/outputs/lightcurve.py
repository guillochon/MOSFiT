import json
import os

from ..module import Module

CLASS_NAME = 'LightCurve'


class LightCurve(Module):
    """Output a light curve to disk.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_times = kwargs.get('ntimes', 0)

    def process(self, **kwargs):
        with open(os.path.join('products', 'lc.json'), 'w') as flast, open(
                os.path.join('products', self._event_name + '.json'),
                'w') as f:
            output = {}
            for key in ['magnitudes', 'e_magnitudes', 'model_magnitudes',
                        'bands', 'times']:
                output[key] = kwargs[key]
            json.dump(output, flast)
            json.dump(output, f)
        return {}
