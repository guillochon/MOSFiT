"""Definitions for the `SED` class."""
import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.modules.module import Module

# Important: Only define one ``Module`` class per file.


class SED(Module):
    """Template class for SED Modules."""

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
            for rng in wave_ranges:
                self._sample_wavelengths.append(
                    np.linspace(rng[0], rng[1], self.N_PTS))
            self._sample_wavelengths = np.array(self._sample_wavelengths)
        self._sample_frequencies = self.C_OVER_ANG / self._sample_wavelengths

    def add_to_existing_seds(self, new_seds, **kwargs):
        """Add SED from present module to existing ``seds`` key.

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
