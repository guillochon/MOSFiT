from mosfit.modules.module import Module
from astrocats.catalog.photometry import PHOTOMETRY
from astrocats.catalog.entry import Entry, ENTRY

CLASS_NAME = 'LightCurve'


class LightCurve(Module):
    """Output a light curve to disk.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_times = kwargs.get('ntimes', 0)

    def process(self, **kwargs):
        output = {}
        for key in [
                'magnitudes', 'e_magnitudes', 'model_magnitudes', 'all_bands',
                'all_systems', 'all_instruments', 'all_bandsets', 'all_times',
                'observed'
        ]:
            output[key.replace('all_', '')] = kwargs[key]

        entry = Entry(kwargs['name'])

        return output
