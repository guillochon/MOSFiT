"""Definitions for the `SED` class."""
import numpy as np
from astropy import constants as c
from astropy import units as u

from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


class SED(Module):
    """Template class for SED Modules.

    Modules that inherit from the SED class should produce a `seds` key, which
    contains a spectral energy distribution for each time. The units of the SED
    should be in ergs/steradian/cm^2/Hz/Angstrom.
    """

    C_OVER_ANG = (c.c / u.Angstrom).cgs.value
    N_PTS = 16 + 1

    def __init__(self, **kwargs):
        """Initialize module."""
        super(SED, self).__init__(**kwargs)
        self._sample_wavelengths = []

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        self._sample_wavelengths = requests.get('sample_wavelengths', [])
        if not self._sample_wavelengths:
            wave_ranges = requests.get('band_wave_ranges', [])
            if not wave_ranges:
                return
            max_len = 0
            for rng in wave_ranges:
                min_wav, max_wav = min(rng), max(rng)
                rngc = list(rng)
                rngc.remove(min_wav)
                rngc.remove(max_wav)
                self._sample_wavelengths.append(np.unique(np.concatenate(
                    (np.linspace(min_wav, max_wav,
                                 self.N_PTS - len(rngc)), np.array(rngc)))))
                llen = len(self._sample_wavelengths[-1])
                if llen > max_len:
                    max_len = llen
            for wi, wavs in enumerate(self._sample_wavelengths):
                if len(wavs) != max_len:
                    self._sample_wavelengths[wi] = np.unique(np.concatenate(
                        (wavs, (max(wavs) - min(wavs)) * 1.0 / np.exp(
                            np.arange(1, 1 + max_len - len(
                                wavs))) + min(wavs))))
                    if len(self._sample_wavelengths[wi]) != max_len:
                        raise RuntimeError(
                            'Could not construct wavelengths for bandpass.')

            self._sample_wavelengths = np.array(self._sample_wavelengths,
                                                dtype=float)
        self._sample_frequencies = self.C_OVER_ANG / self._sample_wavelengths

    def add_to_existing_seds(self, new_seds, **kwargs):
        """Add SED from module to existing ``seds`` key.

        Parameters
        ----------
        new_seds : array
            The new SEDs to add to the existing SEDs.

        Returns
        -------
        new_seds : array
            The result of summing the new and existing SEDs.
        """
        old_seds = kwargs.get('seds', None)
        if old_seds is not None:
            new_seds += old_seds
        return new_seds

    def send_request(self, request):
        """Send a request."""
        if request == 'sample_wavelengths':
            return self._sample_wavelengths
        return []
