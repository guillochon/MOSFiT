"""Definitions for the `Kernel` class."""
from collections import OrderedDict

import numpy as np
from six import string_types

from mosfit.constants import ANG_CGS, C_CGS
from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


class Kernel(Module):
    """Calculate the maximum likelihood score for a model."""

    MIN_COV_TERM = 1.0e-30

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Kernel, self).__init__(**kwargs)
        self._times = np.array([])
        self._full = kwargs.get('full', False)

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)
        ret = {}
        self._codeltatime = kwargs.get(self.key('codeltatime'), -1)
        self._codeltalambda = kwargs.get(self.key('codeltalambda'), -1)

        # Get band variances
        self._variance = kwargs.get(self.key('variance'), 0.0)

        self._band_v_vars = OrderedDict()
        for key in kwargs:
            if key.startswith('variance-band-'):
                self._band_v_vars[key.split('-')[-1]] = kwargs[key]

        if self._variance_bands:
            self._o_variance_bands = [
                self._variance_bands[i] for i in self._all_band_indices]

            self._band_vs = np.array([
                self._band_v_vars.get(i, self._variance) if
                isinstance(i, string_types)
                else (i[0] * self._band_v_vars.get(i[1][0], self._variance) +
                      (1.0 - i[0]) * self._band_v_vars.get(
                          i[1][0], self._variance))
                for i in self._o_variance_bands])
        else:
            self._band_vs = np.full(
                len(self._all_band_indices), self._variance)

        self._o_band_vs = self._band_vs[self._observed]

        if self._codeltatime >= 0 or self._codeltalambda >= 0:
            kmat = np.outer(self._o_band_vs, self._o_band_vs)

            if self._codeltatime >= 0:
                kmat *= np.exp(self._dt2mat / self._codeltatime ** 2)

            if self._codeltalambda >= 0:
                kmat *= np.exp(self._dl2mat / self._codeltalambda ** 2)

            ret['kmat'] = kmat
        else:
            ret['obandvs'] = self._o_band_vs

        return ret

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        self._average_wavelengths = requests.get('average_wavelengths', [])
        self._variance_bands = requests.get('variance_bands', [])

    def preprocess(self, **kwargs):
        """Construct kernel distance arrays."""
        new_times = np.array(kwargs.get('all_times', []), dtype=float)
        if np.array_equiv(new_times, self._times) and self._preprocessed:
            return
        self._times = new_times
        self._all_band_indices = kwargs.get('all_band_indices', [])
        self._are_bands = np.array(self._all_band_indices) >= 0
        self._all_band_avgs = np.array([
            self._average_wavelengths[bi] if bi >= 0 else
            C_CGS / self._freqs[i] / ANG_CGS for i, bi in
            enumerate(self._all_band_indices)])
        if self._full:
            self._observed = np.full(len(self._times), True)
        else:
            self._observed = np.array(kwargs.get('observed', []), dtype=bool)
        self._n_obs = len(self._observed)

        self._o_times = self._times[self._observed]
        self._o_waves = self._all_band_avgs[self._observed]

        # Time deltas (radial distance) for covariance matrix.
        self._dtmat = self._o_times[:, None] - self._o_times[None, :]
        self._dt2mat = -0.5 * self._dtmat ** 2

        # Wavelength deltas (radial distance) for covariance matrix.
        self._dlmat = self._o_waves[:, None] - self._o_waves[None, :]
        self._dl2mat = -0.5 * self._dlmat ** 2

        self._preprocessed = True
