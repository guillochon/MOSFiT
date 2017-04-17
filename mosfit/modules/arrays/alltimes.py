"""Definitions for the `AllTimes` class."""
from collections import OrderedDict

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
        self._bands = []
        self._systems = []
        self._instruments = []
        self._bandsets = []
        self._frequencies = []
        self._countrates = []

    def process(self, **kwargs):
        """Process module."""
        old_bands = (self._systems, self._instruments, self._bandsets,
                     self._bands, self._frequencies, self._countrates)
        if (kwargs.get('root', 'output') == 'output' and
                'extra_times' in kwargs):
            obs_keys = ['times', 'systems', 'instruments', 'bandsets',
                        'bands', 'frequencies', 'countrates']
            obslist = (list(
                zip(*([kwargs.get(k) for k in obs_keys] +
                      [[True for x in range(len(kwargs['times']))]]))
            ) + list(
                zip(*([kwargs.get('extra_' + k) for k in obs_keys] +
                      [[False for x in range(len(kwargs['extra_times']))]]))))
            obslist.sort()

            (self._times, self._systems, self._instruments, self._bandsets,
             self._bands, self._frequencies, self._countrates,
             self._observed) = zip(*obslist)
        else:
            self._times = kwargs['times']
            self._systems = kwargs['systems']
            self._instruments = kwargs['instruments']
            self._bandsets = kwargs['bandsets']
            self._bands = kwargs['bands']
            self._frequencies = [
                x / frequency_unit(y) if x != '' else ''
                for x, y in zip(kwargs['frequencies'], kwargs['u_frequencies'])
            ]
            self._countrates = kwargs['countrates']
            self._observed = [True for x in kwargs['times']]

        outputs = OrderedDict()
        outputs['all_times'] = self._times
        outputs['all_systems'] = self._systems
        outputs['all_instruments'] = self._instruments
        outputs['all_bandsets'] = self._bandsets
        outputs['all_bands'] = self._bands
        outputs['all_frequencies'] = self._frequencies
        outputs['all_countrates'] = self._countrates
        if old_bands != (self._systems, self._instruments, self._bandsets,
                         self._bands, self._frequencies, self._countrates):
            self._all_band_indices = [
                (self._photometry.find_band_index(
                    w, instrument=x, bandset=y, system=z) if a == '' else -1)
                for w, x, y, z, a in zip(
                    self._bands, self._instruments, self._bandsets,
                    self._systems, self._frequencies)
            ]
            self._all_band_kinds = [
                'mag' if x == '' else 'cts' for x in self._countrates
            ]
        outputs['all_band_indices'] = self._all_band_indices
        outputs['all_band_kinds'] = self._all_band_kinds
        outputs['observed'] = self._observed
        return outputs

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        self._photometry = requests.get('photometry', None)
