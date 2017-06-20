"""Definitions for the `AllTimes` class."""
from collections import OrderedDict

import numpy as np
from mosfit.modules.arrays.array import Array
from mosfit.utils import frequency_unit


# Important: Only define one ``Module`` class per file.


class AllTimes(Array):
    """Generate all times for which observations will be constructed.

    Create lists of observation times that associated with real observations
    and interpolations/extrapolations if such flags are passed to MOSFiT.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(AllTimes, self).__init__(**kwargs)
        self._obs_keys = [
            'times', 'bands', 'telescopes', 'systems', 'instruments', 'modes',
            'bandsets', 'frequencies', 'u_frequencies', 'observed']
        self._okeys = [i for i in self._obs_keys if i not in ['observed']]
        for key in self._obs_keys:
            setattr(self, '_' + key, [])

    def process(self, **kwargs):
        """Process module."""
        old_observations = tuple(
            getattr(self, '_' + x) for x in self._obs_keys)
        if (kwargs.get('root', 'output') == 'output' and
                'extra_times' in kwargs):
            obslist = (list(
                zip(*([kwargs.get(k) for k in self._okeys] +
                      [[True for x in range(len(kwargs['times']))]]))
            ) + list(
                zip(*([kwargs.get('extra_' + k) for k in self._okeys] +
                      [[False for x in range(len(kwargs['extra_times']))]]))))
            obslist.sort()

            self._all_observations = np.concatenate([
                np.atleast_2d(np.array(x, dtype=object))
                for x in obslist], axis=0).T
            for ki, key in enumerate(self._obs_keys):
                setattr(self, '_' + key, self._all_observations[ki])
        else:
            for key in list(
                    set(self._obs_keys) - set([
                        'frequencies', 'observed'])):
                setattr(self, '_' + key, kwargs[key])
            self._frequencies = np.array([
                x / frequency_unit(y) if x is not None else None
                for x, y in zip(kwargs['frequencies'], kwargs['u_frequencies'])
            ])
            self._observed = np.full_like(kwargs['times'], True, dtype=bool)
            self._all_observations = tuple(
                getattr(self, '_' + x) for x in self._obs_keys)

        outputs = OrderedDict(
            [('all_' + x, getattr(self, '_' + x))
             for x in list(set(self._obs_keys) - set(['observed']))])
        if any(not np.array_equal(x, y) for x, y in zip(
                old_observations, self._all_observations)):
            self._all_band_indices = np.array([
                (self._photometry.find_band_index(
                    b, telescope=t, instrument=i, mode=m, bandset=bs, system=s)
                 if f is None else -1)
                for ti, b, t, s, i, m, bs, f, uf, o
                in zip(*self._all_observations)
            ])
            self._observation_types = np.array([
                self._photometry._band_kinds[bi] if bi >= 0 else
                'fluxdensity' for bi in self._all_band_indices
            ], dtype=object)
        outputs['all_band_indices'] = self._all_band_indices
        outputs['observation_types'] = self._observation_types
        outputs['observed'] = np.array(self._observed, dtype=bool)
        return outputs

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        self._photometry = requests.get('photometry', None)
