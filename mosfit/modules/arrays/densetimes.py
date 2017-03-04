"""Definitions for the `DenseTimes` class."""
import numpy as np
from mosfit.modules.module import Module

# Important: Only define one `Module` class per file.


class DenseTimes(Module):
    """This class ensures an even time-sampling between the time of explosion
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
        self._t_explosion = kwargs['texplosion']

        outputs = {}
        max_times = max(self._rest_times)
        if max_times > self._t_explosion:
            outputs['dense_times'] = list(
                sorted(
                    set([0.0] + [
                        x + self._t_explosion
                        for x in np.logspace(
                            self.L_T_MIN,
                            np.log10(max_times - self._t_explosion),
                            num=self._n_times)
                    ] + self._rest_times)))
        else:
            outputs['dense_times'] = self._rest_times
        return outputs
