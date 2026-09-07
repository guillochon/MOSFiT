"""Definitions for the `LeddCap` class."""
import numpy as np

from mosfit.modules.utilities.utility import Utility


# Important: Only define one ``Module`` class per file.


class LeddCap(Utility):
    """Soft Eddington cap ``L L_cap / (L + L_cap)``.

    Honors ``replacements`` so the cap can be applied to ``acc_luminosities``
    before the viscous transform, leaving the prompt shock term uncapped.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(LeddCap, self).__init__(**kwargs)
        self._wants_dense = True

    def process(self, **kwargs):
        """Process module."""
        cap = float(kwargs['Leddlim']) * float(kwargs['Ledd'])
        lum_key = self.key('luminosities')
        dense_in = self.key('dense_luminosities')
        out = {}
        if dense_in in kwargs:
            dense = np.asarray(kwargs[dense_in], dtype=float)
            capped = dense * cap / (dense + cap)
            capped = np.where(np.isnan(capped), 0.0, capped)
            out[self.dense_key('luminosities')] = capped
            if dense_in not in out:
                out[dense_in] = capped
            if 'dense_indices' in kwargs:
                idx = np.asarray(kwargs['dense_indices'], dtype=int)
                out[lum_key] = np.take(capped, idx)
        else:
            kwargs = self.prepare_input(lum_key, **kwargs)
            lums = np.asarray(kwargs[lum_key], dtype=float)
            capped = lums * cap / (lums + cap)
            out[lum_key] = np.where(np.isnan(capped), 0.0, capped)
        return out
