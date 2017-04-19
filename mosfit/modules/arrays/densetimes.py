"""Definitions for the `DenseTimes` class."""
from collections import OrderedDict

import numpy as np
from mosfit.modules.arrays.array import Array


# Important: Only define one ``Module`` class per file.


class DenseTimes(Array):
    """Generate an evenly-spaced array of times for use in calculations.

    This class ensures an even time-sampling between the time of explosion
    and the last datapoint, as many transients may lack regular candence data.
    """

    N_TIMES = 100
    L_T_MIN = -6  # in days

    def __init__(self, **kwargs):
        """Initialize module."""
        super(DenseTimes, self).__init__(**kwargs)
        self._n_times = kwargs[
            'n_times'] if 'n_times' in kwargs else self.N_TIMES

    def process(self, **kwargs):
        """Process module."""
        self._rest_times = kwargs['rest_times']
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]

        outputs = OrderedDict()
        max_times = max(self._rest_times)
        if max_times > self._rest_t_explosion:
            outputs['dense_times'] = np.unique(
                np.concatenate(([0.0], [
                    x + self._rest_t_explosion
                    for x in np.logspace(
                        self.L_T_MIN,
                        np.log10(max_times - self._rest_t_explosion),
                        num=self._n_times)
                ], self._rest_times)))
        else:
            outputs['dense_times'] = np.array(self._rest_times)
        outputs['dense_indices'] = np.searchsorted(
            outputs['dense_times'], self._rest_times)
        return outputs
