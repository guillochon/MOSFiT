"""Definitions for the `LeddCap` class."""
import numpy as np

from mosfit.modules.utilities.utility import Utility


# Important: Only define one ``Module`` class per file.


class LeddCap(Utility):
    """Soft Eddington cap ``L L_cap / (L + L_cap)`` applied after engines sum."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(LeddCap, self).__init__(**kwargs)
        self._wants_dense = True

    def process(self, **kwargs):
        """Process module."""
        cap = float(kwargs['Leddlim']) * float(kwargs['Ledd'])
        out = {}
        if 'dense_luminosities' in kwargs:
            dense = np.asarray(kwargs['dense_luminosities'], dtype=float)
            capped = dense * cap / (dense + cap)
            capped = np.where(np.isnan(capped), 0.0, capped)
            out['dense_luminosities'] = capped
            if 'dense_indices' in kwargs:
                idx = np.asarray(kwargs['dense_indices'], dtype=int)
                out['luminosities'] = capped[idx]
        else:
            kwargs = self.prepare_input('luminosities', **kwargs)
            lums = np.asarray(kwargs['luminosities'], dtype=float)
            capped = lums * cap / (lums + cap)
            out['luminosities'] = np.where(np.isnan(capped), 0.0, capped)
        return out
