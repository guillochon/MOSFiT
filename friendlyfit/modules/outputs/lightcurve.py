import json

from ..module import Module

CLASS_NAME = 'LightCurve'


class LightCurve(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_times = kwargs.get("ntimes", 0)

    def process(self, **kwargs):
        with open('lc.json', 'w') as f:
            output = {}
            for key in ['model_magnitudes', 'bands', 'times']:
                output[key] = kwargs[key]
            json.dump(output, f)
        return {}
