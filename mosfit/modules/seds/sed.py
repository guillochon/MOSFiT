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
    should be in erg/s/Angstrom.
    """

    C_OVER_ANG = (c.c / u.Angstrom).cgs.value

    def __init__(self, **kwargs):
        """Initialize module."""
        super(SED, self).__init__(**kwargs)
        self._N_PTS = 24 + 1
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
                                 self._N_PTS - len(rngc)), np.array(rngc)))))
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

            # Note: Many of these will just be 0 - 1, but faster to have a
            # single type numpy array than a ragged list of lists.
            self._sample_wavelengths = np.array(self._sample_wavelengths,
                                                dtype=float)
        self._sample_frequencies = self.C_OVER_ANG / self._sample_wavelengths

    def sed_row_width(self):
        """Wavelength samples per SED row (bands are padded to this width)."""
        sw = self._sample_wavelengths
        if sw is None or len(sw) == 0:
            return 1
        if isinstance(sw, np.ndarray) and sw.dtype != object:
            if sw.ndim >= 2:
                return int(sw.shape[-1])
            if sw.ndim == 1:
                return int(sw.shape[0])
        widths = [len(np.asarray(row).ravel()) for row in sw]
        return int(max(widths)) if widths else 1

    def alloc_seds(self, n_rows):
        """Allocate a rectangular ``(n_obs, n_wav)`` SED array of zeros."""
        return np.zeros((int(n_rows), self.sed_row_width()), dtype=float)

    @staticmethod
    def as_rectangular_seds(seds, n_wav=None):
        """Coerce list / object-dtype SEDs to a 2-D float array."""
        if seds is None:
            return None
        arr = np.asarray(seds)
        if arr.dtype != object:
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                return arr.reshape(arr.shape[0], 1)
            return arr
        rows = [np.asarray(x, dtype=float).ravel()
                for x in np.asarray(seds, dtype=object)]
        if not rows:
            return np.zeros((0, 1 if n_wav is None else int(n_wav)),
                            dtype=float)
        width = int(n_wav) if n_wav is not None else int(
            max(r.size for r in rows))
        out = np.zeros((len(rows), width), dtype=float)
        for i, row in enumerate(rows):
            n = min(int(row.size), width)
            out[i, :n] = row[:n]
        return out

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
        new_seds = self.as_rectangular_seds(new_seds)
        old_seds = kwargs.get('seds', None)
        if old_seds is None:
            return new_seds
        old_seds = self.as_rectangular_seds(
            old_seds, n_wav=new_seds.shape[1])
        if old_seds.shape == new_seds.shape:
            return new_seds + old_seds
        n_rows = new_seds.shape[0]
        n_wav = max(old_seds.shape[1], new_seds.shape[1])
        out = np.zeros((n_rows, n_wav), dtype=float)
        out[:, :new_seds.shape[1]] = new_seds
        n_old = min(old_seds.shape[0], n_rows)
        out[:n_old, :old_seds.shape[1]] += old_seds[:n_old]
        return out

    def send_request(self, request):
        """Send a request."""
        if request == 'sample_wavelengths':
            return self._sample_wavelengths
        return []

    def set_data(self, band_sampling_points):
        """Set SED data."""
        self._N_PTS = band_sampling_points
        return True
