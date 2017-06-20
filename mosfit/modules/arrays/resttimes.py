"""Definitions for the `RestTimes` class."""
from collections import OrderedDict

import numpy as np
from mosfit.modules.arrays.array import Array


# Important: Only define one ``Module`` class per file.


class RestTimes(Array):
    """This class converts the observed times to rest-frame times."""

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs['all_times']
        self._t_explosion = kwargs[self.key('texplosion')]

        outputs = OrderedDict()
        outputs['rest_times'] = np.array([
            x / (1.0 + kwargs['redshift']) for x in self._times
        ])
        outputs[self.key('resttexplosion')] = self._t_explosion / (
            1.0 + kwargs[self.key('redshift')])
        return outputs
