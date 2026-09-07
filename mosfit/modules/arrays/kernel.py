"""Definitions for the `Kernel` class."""
from collections import OrderedDict

import numpy as np
from six import string_types

from mosfit.constants import ANG_CGS, BOL_BAND_INDEX, C_CGS
from mosfit.modules.arrays.array import Array

# Important: Only define one ``Module`` class per file.


class Kernel(Array):
    """Calculate the kernel for use in computing the likelihood score."""

    MIN_COV_TERM = 1.0e-30

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Kernel, self).__init__(**kwargs)
        self._times = np.array([])

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)

        ret = OrderedDict()

        # Get band variances
        self._variance = kwargs.get(self.key('variance'), 0.0)

        # Get array of real observations.
        t = np.asarray(self._o_otypes)
        n = t.shape[0]
        self._observations = np.empty(n, dtype=object)
        is_cr = (t == 'countrate') | (t == 'magcount')
        is_mag = t == 'magnitude'
        is_lum = t == 'luminosity'
        is_fd = t == 'fluxdensity'
        mags = np.asarray(self._mags, dtype=object)
        cts = np.asarray(self._cts, dtype=object)
        fds = np.asarray(self._fds, dtype=object)
        lums = np.asarray(self._lums, dtype=object)
        self._observations[is_cr] = cts[is_cr]
        self._observations[is_mag] = mags[is_mag]
        self._observations[is_lum] = lums[is_lum]
        self._observations[is_fd] = fds[is_fd]
        self._observations[~(is_cr | is_mag | is_lum | is_fd)] = None

        # Get array of model observations.
        self._model_observations = kwargs.get('model_observations', [])

        # Handle band-specific variances if that option is enabled.
        self._band_v_vars = OrderedDict()
        for key in kwargs:
            if key.startswith('variance-band-'):
                self._band_v_vars[key.split('-')[-1]] = kwargs[key]

        if self._variance_bands:
            self._o_variance_bands = [
                self._variance_bands[i] for i in self._all_band_indices
            ]

            self._band_vs = np.array([
                self._band_v_vars.get(i, self._variance) if isinstance(
                    i, string_types) else
                (i[0] * self._band_v_vars.get(i[1][0], self._variance) +
                 (1.0 - i[0]) * self._band_v_vars.get(i[1][0], self._variance))
                for i in self._o_variance_bands
            ])
        else:
            self._band_vs = np.full(
                len(self._all_band_indices), self._variance)

        # Compute relative errors for count-based observations.
        self._band_vs[self._count_inds] = (
            10.0**(self._band_vs[self._count_inds] / 2.5) -
            1.0) * self._model_observations[self._count_inds]

        self._o_band_vs = self._band_vs[self._observed]

        ret['abandvs'] = self._band_vs
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
        self._freqs = kwargs.get('all_frequencies', [])
        self._mags = np.array(kwargs.get('magnitudes', []))
        self._fds = np.array(kwargs.get('fluxdensities', []))
        self._cts = np.array(kwargs.get('countrates', []))
        self._lums = np.array(kwargs.get('observed_luminosities', []))
        self._u_freqs = kwargs.get('all_u_frequencies', [])
        ai = np.asarray(self._all_band_indices)
        freqs = np.asarray(self._freqs, dtype=float)
        self._waves = np.full(ai.shape[0], np.nan, dtype=float)
        in_band = ai >= 0
        if np.any(in_band):
            awaves = np.asarray(self._average_wavelengths, dtype=float)
            self._waves[in_band] = awaves[ai[in_band]]
        freq_rows = (ai < 0) & (ai != BOL_BAND_INDEX)
        if np.any(freq_rows):
            self._waves[freq_rows] = C_CGS / freqs[freq_rows] / ANG_CGS
        self._observed = np.array(kwargs.get('observed', []), dtype=bool)
        self._observation_types = kwargs.get('observation_types')
        self._n_obs = len(self._observed)
        self._count_inds = np.logical_and(
            self._observation_types != 'magnitude',
            self._observation_types != 'luminosity')

        self._o_times = self._times[self._observed]
        self._o_waves = self._waves[self._observed]
        self._o_otypes = self._observation_types[self._observed]

        self._preprocessed = True
