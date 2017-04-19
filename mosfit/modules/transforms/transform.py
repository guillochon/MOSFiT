"""Definitions for the `Transform` class."""
import numpy as np
from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


class Transform(Module):
    """Parent class for transforms."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Transform, self).__init__(**kwargs)
        self._wants_dense = True

    def process(self, **kwargs):
        """Set `dense_*` and `*_since_exp` times/luminosities keys."""
        self._times = kwargs['rest_times']
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        if 'dense_times' in kwargs:
            self._dense_times = kwargs['dense_times']
            self._dense_luminosities = kwargs[self.key('dense_luminosities')]
        elif min(self._times) > self._rest_t_explosion:
            self._dense_times = np.concatenate(
                ([self._rest_t_explosion], self._times))
            self._dense_luminosities = np.concatenate(
                ([0.0], kwargs[self.key('dense_luminosities')]))
        self._times_since_exp = self._times - self._rest_t_explosion
        self._dense_times_since_exp = (
            self._dense_times - self._rest_t_explosion)
        if self._provide_dense:
            self._times_to_process = self._dense_times_since_exp
        else:
            self._times_to_process = self._times_since_exp
