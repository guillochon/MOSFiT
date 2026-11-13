"""Definitions for the `DarkYear` class."""
import numpy as np

from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class DarkYear(Parameter):
    """Viscous time from the Guillochon & Ramirez-Ruiz (2015) dark-year map,
    log10(T_visc/t_pk) = tviscnorm + tviscslope*log10((rp/rg)/47), clipped to
    [-1.5, 2], plus a per-event offset in dex supplied by the caller."""

    _REFERENCES = [
        {'bibcode': '2015ApJ...809..166G'}
    ]

    RP_RG_REF = 47.0
    LOG_MIN = -1.5
    LOG_MAX = 2.0
    TVISC_MIN = 1.0e-3   # days

    def process(self, **kwargs):
        """Return ``Tviscous`` in days from the calibrated ``rp/rg`` map."""
        if self._name in kwargs:
            return {}
        rp_over_rg = float(kwargs['rp_over_rg'])
        tfallback = float(kwargs['tfallback'])
        # time of peak fallback since disruption: first return plus the rise from
        # first return (fallback's tpeak is absolute rest-frame time, offset by texplosion)
        t_pk = tfallback + (float(kwargs['tpeak']) - float(kwargs['resttexplosion']))
        t_pk = max(t_pk, tfallback)
        logratio = (float(kwargs.get('tviscnorm', 1.0)) +
                    float(kwargs.get('tviscslope', 2.1)) *
                    np.log10(rp_over_rg / self.RP_RG_REF))
        logratio = min(max(logratio, self.LOG_MIN), self.LOG_MAX)
        logratio += float(kwargs.get('tviscoffset', 0.0))
        tvisc = max(10.0 ** logratio * t_pk, self.TVISC_MIN)
        return {self._name: tvisc}
