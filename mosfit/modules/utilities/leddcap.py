"""Definitions for the `LeddCap` class."""
import numpy as np

from mosfit.modules.utilities.utility import Utility


# Important: Only define one ``Module`` class per file.


class LeddCap(Utility):
    """Super-Eddington radiative-efficiency law.

    ``L = L_cap · ṁ / (1 + ṁ)^p`` with ``ṁ = L / L_cap``. ``p = 1`` is the
    harmonic cap ``L L_cap / (L + L_cap)``. Honors ``replacements`` so
    accretion and shock can use different exponents.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(LeddCap, self).__init__(**kwargs)
        self._wants_dense = True

    def _apply_law(self, lum, cap, p):
        """Apply ``L_cap ṁ / (1 + ṁ)^p``; ``p = 1`` is the harmonic cap."""
        lum = np.asarray(lum, dtype=float)
        if p == 1.0:
            capped = lum * cap / (lum + cap)
        else:
            mdot = lum / cap
            capped = cap * mdot / np.power(1.0 + mdot, p)
        return np.where(np.isnan(capped), 0.0, capped)

    def process(self, **kwargs):
        """Process module."""
        cap = float(kwargs['Leddlim']) * float(kwargs['Ledd'])
        p = float(kwargs.get(self.key('eddslope'), 1.0))
        lum_key = self.key('luminosities')
        dense_in = self.key('dense_luminosities')
        out = {}
        if dense_in in kwargs:
            dense = np.asarray(kwargs[dense_in], dtype=float)
            capped = self._apply_law(dense, cap, p)
            out[self.dense_key('luminosities')] = capped
            if dense_in not in out:
                out[dense_in] = capped
            if 'dense_indices' in kwargs:
                idx = np.asarray(kwargs['dense_indices'], dtype=int)
                out[lum_key] = np.take(capped, idx)
        else:
            kwargs = self.prepare_input(lum_key, **kwargs)
            lums = np.asarray(kwargs[lum_key], dtype=float)
            capped = self._apply_law(lums, cap, p)
            out[lum_key] = capped
        if lum_key == 'acc_luminosities' and np.size(capped):
            ledd = float(kwargs['Ledd'])
            out['acc_edd_ratio_peak'] = (
                float(np.max(capped) / ledd) if ledd else 0.0)
        return out
