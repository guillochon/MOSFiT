"""Definitions for the `Diagonal` class."""
from math import isnan

import numpy as np

from mosfit.modules.arrays.array import Array
from mosfit.utils import flux_density_unit


# Important: Only define one ``Module`` class per file.


class Diagonal(Array):
    """Calculate the diagonal/residuals for a model kernel."""

    MIN_COV_TERM = 1.0e-30

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Diagonal, self).__init__(**kwargs)
        self._observation_types = np.array([])

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)
        self._model_observations = kwargs['model_observations']
        self._model_observations[self._cmask] = -2.5 * np.log10(
            self._model_observations[self._cmask])
        self._model_observations = self._model_observations[self._observed]
        self._o_types = self._observation_types[self._observed]

        ret = {}

        # Calculate (model - obs) residuals.
        residuals = np.array([
            (abs(x - ct) if not u or (x < ct and not isnan(x)) else 0.0)
            if t == 'countrate' and ct is not None
            else
            (abs(x - y) if not u or (x < y and not isnan(x)) else 0.0)
            if t == 'magnitude' and y is not None
            else
            (abs(x - fd) if not u or (x > fd and not isnan(x)) else 0.0)
            if t == 'fluxdensity' and fd is not None else None
            for x, y, ct, fd, u, t in zip(
                self._model_observations, self._mags, self._cts, self._fds,
                self._upper_limits, self._o_types)
        ])

        if np.any(residuals == None):  # noqa: E711
            raise ValueError('Null residual.')

        # Observational errors to be put in diagonal of error matrix.
        diag = np.array([
            ((ctel if x > ct else cteu) ** 2)
            if t == 'countrate' and ct is not None else
            ((el if x > y else eu) ** 2)
            if t == 'magnitude' and y is not None else
            ((fdel if x < fd else fdeu) ** 2)
            if t == 'fluxdensity' and fd is not None else None
            for x, y, eu, el, fd, fdeu, fdel, ct, ctel, cteu, t in zip(
                self._model_observations, self._mags,
                self._e_u_mags, self._e_l_mags, self._fds, self._e_u_fds,
                self._e_l_fds, self._cts, self._e_l_cts, self._e_u_cts,
                self._o_types)
        ])

        if np.any(diag == None):  # noqa: E711
            raise ValueError('Null error.')

        ret['kdiagonal'] = diag
        ret['kresiduals'] = residuals

        return ret

    def preprocess(self, **kwargs):
        """Construct arrays of observations based on data keys."""
        otypes = np.array(kwargs.get('observation_types', []))
        if np.array_equiv(
                otypes, self._observation_types) and self._preprocessed:
            return
        self._observation_types = otypes
        self._mags = np.array(kwargs.get('magnitudes', []))
        self._fds = np.array(kwargs.get('fluxdensities', []))
        self._cts = np.array(kwargs.get('countrates', []))
        self._e_u_mags = kwargs.get('e_upper_magnitudes', [])
        self._e_l_mags = kwargs.get('e_lower_magnitudes', [])
        self._e_mags = kwargs.get('e_magnitudes', [])
        self._e_u_fds = kwargs.get('e_upper_fluxdensities', [])
        self._e_l_fds = kwargs.get('e_lower_fluxdensities', [])
        self._e_fds = kwargs.get('e_fluxdensities', [])
        self._u_fds = kwargs.get('u_fluxdensities', [])
        self._e_u_cts = kwargs.get('e_upper_countrates', [])
        self._e_l_cts = kwargs.get('e_lower_countrates', [])
        self._e_cts = kwargs.get('e_countrates', [])
        self._u_cts = kwargs.get('u_countrates', [])
        self._upper_limits = np.array(kwargs.get('upperlimits', []),
                                      dtype=bool)
        self._observed = np.array(kwargs.get('observed', []), dtype=bool)

        # Magnitudes first
        # Note: Upper limits (censored data) currently treated as a
        # half-Gaussian, this is very approximate and can be improved upon.
        self._e_u_mags = [
            kwargs['default_upper_limit_error']
            if (e is None and eu is None and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e is None and eu is None) else (e if eu is None else eu))
            for i, (e, eu) in enumerate(zip(self._e_mags, self._e_u_mags))
        ]
        self._e_l_mags = [
            kwargs['default_upper_limit_error']
            if (e is None and el is None and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e is None and el is None) else (e if el is None else el))
            for i, (e, el) in enumerate(zip(self._e_mags, self._e_l_mags))
        ]

        # Now counts
        cmask = self._observation_types[self._observed] == 'countrate'
        self._cts[cmask] = -2.5 * np.log10(self._cts[cmask].astype(np.float64))
        self._cmask = self._observation_types == 'countrate'
        self._e_u_cts = [
            kwargs['default_upper_limit_error']
            if (e is None and eu is None and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e is None and eu is None) else
             2.5 * (np.log10(c + (e if eu is None else eu)) - np.log10(c)))
            for i, (c, e, eu) in enumerate(zip(
                self._cts, self._e_cts, self._e_u_cts))
        ]
        self._e_l_cts = [
            kwargs['default_upper_limit_error']
            if (e is None and el is None and self._upper_limits[i]) else
            (kwargs['default_no_error_bar_error']
             if (e is None and el is None) else
             2.5 * (np.log10(c) - np.log10(c - (e if el is None else el))))
            for i, (c, e, el) in enumerate(zip(
                self._cts, self._e_cts, self._e_l_cts))
        ]

        # Now flux densities
        self._e_u_fds = [
            v if (e is None and eu is None and self._upper_limits[i]) else
            (v if (e is None and eu is None) else (e if eu is None else eu))
            for i, (e, eu, v) in enumerate(
                zip(self._e_fds, self._e_u_fds, self._fds))
        ]
        self._e_l_fds = [
            0.0 if self._upper_limits[i] else (
                v if (e is None and el is None) else (e if el is None else el))
            for i, (e, el, v) in enumerate(
                zip(self._e_fds, self._e_l_fds, self._fds))
        ]
        self._fds = np.array([
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._fds, self._u_fds)
        ])
        self._e_u_fds = [
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._e_u_fds, self._u_fds)
        ]
        self._e_l_fds = [
            x / flux_density_unit(y) if x is not None else None
            for x, y in zip(self._e_l_fds, self._u_fds)
        ]

        self._preprocessed = True
