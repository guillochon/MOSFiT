"""Definitions for the `NickelCobalt` class."""
import numpy as np

from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class NickelCobalt(Engine):
    """Nickel/Cobalt decay engine."""

    NI56_LUM = 6.45e43
    CO56_LUM = 1.45e43
    NI56_LIFE = 8.8
    CO56_LIFE = 111.3

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        self._mnickel = kwargs[self.key('fnickel')] * kwargs[
            self.key('mejecta')]
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]

        # From 1994ApJS...92..527N
        ts = np.empty_like(self._times)
        t_inds = self._times >= self._rest_t_explosion
        ts[t_inds] = self._times[t_inds] - self._rest_t_explosion

        luminosities = np.zeros_like(self._times)
        luminosities[t_inds] = self._mnickel * (
            self.NI56_LUM * np.exp(-ts[t_inds] / self.NI56_LIFE) +
            self.CO56_LUM * np.exp(-ts[t_inds] / self.CO56_LIFE))

        luminosities[np.isnan(luminosities)] = 0.0

        return {self.dense_key('luminosities'): luminosities}
