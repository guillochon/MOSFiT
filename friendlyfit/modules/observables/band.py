import csv
import os

from ..module import Module

CLASS_NAME = 'Band'


class Band(Module):
    """Band-pass filter
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._path = kwargs['path']
        with open(os.path.join('friendlyfit', 'modules', self._path),
                  'r') as f:
            rows = list(csv.reader(f, delimiter='\t', skipinitialspace=True))
            self._wavelengths, self._transmissions = list(
                map(list, zip(*rows)))

    def process(self, **kwargs):
        self._seds = kwargs['seds']
        return {'model_magnitudes': self._seds}

    def request(self, request):
        if request == 'wavelengths':
            return [1000., 10000.]
        return []
