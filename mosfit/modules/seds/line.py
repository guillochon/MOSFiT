"""Definitions for the `Line` class."""
import numexpr as ne
import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.modules.seds.sed import SED
from mosfit.constants import SQRT_2_PI


# Important: Only define one ``Module`` class per file.


class Line(SED):
    """Line spectral energy distribution, modifies existing SED."""

    C_CONST = c.c.cgs.value

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        prt = self._printer
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._times = kwargs[self.key('rest_times')]
        self._seds = kwargs.get(self.key('seds'))
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._sample_wavelengths = kwargs['sample_wavelengths']
        self._frequencies = kwargs['all_frequencies']
        self._luminosities = kwargs[self.key('luminosities')]
        self._line_wavelength = kwargs[self.key('line_wavelength')]
        self._line_width = kwargs[self.key('line_width')]
        self._line_time = kwargs[self.key('line_time')]
        self._line_duration = kwargs[self.key('line_duration')]
        self._line_amplitude = kwargs[self.key('line_amplitude')]
        lw = self._line_wavelength  # noqa: F841
        ls = self._line_width
        cc = self.C_CONST

        # Some temp vars for speed.
        zp1 = 1.0 + kwargs[self.key('redshift')]
        czp1A = cc / (zp1 * u.Angstrom.cgs.scale)

        amps = self._line_amplitude * np.array([
            np.exp(-0.5 * (
                (x - self._rest_t_explosion - self._line_time) /
                self._line_duration) ** 2) for x in self._times])

        if self._seds is None:
            raise ValueError(prt.message('line_sed'))

        seds = [x * (1.0 - amps[xi]) for xi, x in enumerate(self._seds)]
        amps *= self._luminosities / (ls * SQRT_2_PI)
        amps_dict = {}
        evaled = False

        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if lum == 0.0:
                continue

            bind = czp1A / self._frequencies[li] if bi < 0 else bi

            if bind not in amps_dict:
                # Leave `rest_wavs` in Angstroms.
                if bi >= 0:
                    rest_wavs = self._sample_wavelengths[bi] / zp1
                else:
                    rest_wavs = np.array([bind])  # noqa: F841

                if not evaled:
                    amps_dict[bind] = ne.evaluate(
                        'exp(-0.5 * ((rest_wavs - lw) / ls) ** 2)')
                    evaled = True
                else:
                    amps_dict[bind] = ne.re_evaluate()

            seds[li] += amps[li] * amps_dict[bind]

            # seds[li][np.isnan(seds[li])] = 0.0

        return {'sample_wavelengths': self._sample_wavelengths,
                self.key('seds'): seds}
