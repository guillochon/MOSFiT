"""Definitions for the `LightCurve` class."""
from mosfit.modules.module import Module

# Important: Only define one `Module` class per file.


class LightCurve(Module):
    """Output a light curve to disk."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(LightCurve, self).__init__(**kwargs)
        self._n_times = kwargs.get('ntimes', 0)

    def process(self, **kwargs):
        """Process module."""
        output = {}
        for key in [
                'magnitudes', 'e_magnitudes', 'model_observations',
                'all_bands', 'all_systems', 'all_instruments', 'all_bandsets',
                'all_times', 'all_frequencies', 'observed', 'all_band_indices'
        ]:
            if key == 'all_band_indices':
                output['observation_types'] = [
                    'magnitude' if x >= 0 else 'fluxdensity'
                    for x in kwargs[key]
                ]
            else:
                output[key.replace('all_', '')] = kwargs[key]

        return output
