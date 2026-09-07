"""Definitions for the `DarkYear` class."""
import numpy as np

from mosfit.constants import C_CGS, DAY_CGS, M_SUN_CGS
from mosfit.modules.parameters.parameter import Parameter

import astropy.constants as c


# Important: Only define one ``Module`` class per file.


class DarkYear(Parameter):
    """Viscous time from the Guillochon & Ramirez-Ruiz (2015) dark-year map.

    Stream self-intersection sits near pericenter when ``rp/rg`` is small
    (massive holes, deep encounters) and near the most-bound apoapsis when
    ``rp/rg`` is large. The circularization period is blended between
    ``P(2 rp)`` and ``t_fallback``, then scaled by the thick-disk factor
    ``viscfac = α^{-1}(h/r)^{-2}`` (fiducial 100).
    """

    _REFERENCES = [
        {'bibcode': '2015ApJ...809..166G'}
    ]

    VISCFAC = 100.0
    RP_RG_BLEND = 20.0
    BLEND_INDEX = 6.0
    TVISC_MIN = 1.0e-3
    TCOVER_OVER_TPEAK = 100.0

    def process(self, **kwargs):
        """Return ``Tviscous`` in days from ``rp_over_rg`` and fallback times."""
        if self._name in kwargs:
            return {}

        rp_over_rg = float(kwargs['rp_over_rg'])
        tfallback = float(kwargs['tfallback'])
        tpeak = float(kwargs.get('tpeak', tfallback))
        viscfac = float(kwargs.get('viscfac', self.VISCFAC))
        mh = float(kwargs['bhmass'])

        rg_over_c = c.G.cgs.value * mh * M_SUN_CGS / (C_CGS ** 3)
        p_rp = 2.0 * np.pi * rg_over_c * (rp_over_rg ** 1.5) / DAY_CGS
        p_2rp = p_rp * (2.0 ** 1.5)
        weight = 1.0 / (
            1.0 + (rp_over_rg / self.RP_RG_BLEND) ** self.BLEND_INDEX)
        p_circ = weight * p_2rp + (1.0 - weight) * tfallback
        tvisc = viscfac * p_circ
        tvisc = min(tvisc, self.TCOVER_OVER_TPEAK * max(tpeak, tfallback))
        tvisc = max(tvisc, self.TVISC_MIN)
        return {self._name: tvisc}
