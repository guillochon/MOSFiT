"""Definitions for the `LightCurve` class."""
from collections import OrderedDict

from mosfit.modules.outputs.output import Output


# Important: Only define one ``Module`` class per file.


class LightCurve(Output):
    """Output a light curve to disk."""

    _lc_keys = [
        'magnitudes', 'e_magnitudes', 'model_observations',
        'all_bands', 'all_systems', 'all_instruments', 'all_bandsets',
        'all_times', 'all_frequencies', 'observed', 'all_band_indices',
        'observation_types'
    ]

    def __init__(self, **kwargs):
        """Initialize module."""
        super(LightCurve, self).__init__(**kwargs)
        self._dense_keys = self._lc_keys
        self._n_times = kwargs.get('ntimes', 0)

    def process(self, **kwargs):
        """Process module."""
        output = OrderedDict()
        for key in self._dense_keys:
            output[key.replace('all_', '')] = kwargs[key]

        return output
