"""Definitions for the `Line` class."""
import numexpr as ne
import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.modules.seds.sed import SED
from scipy.special import erf


# Important: Only define one ``Module`` class per file.


class Line(SED):
    """Line spectral energy distribution, modifies existing SED."""

    C_CONST = c.c.cgs.value

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._times = kwargs[self.key('rest_times')]
        self._seds = kwargs[self.key('seds')]
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
        lw = self._line_wavelength
        ls = self._line_width
        cc = self.C_CONST
        zp1 = 1.0 + kwargs[self.key('redshift')]
        amps = [
            self._line_amplitude * np.exp(-0.5 * (
                (x - self._rest_t_explosion - self._line_time) /
                self._line_duration) ** 2) for x in self._times]

        seds = self._seds
        evaled = False
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if lum == 0.0:
                if bi >= 0:
                    seds.append(np.zeros_like(self._sample_wavelengths[bi]))
                else:
                    seds.append([0.0])
                continue
            if bi >= 0:
                rest_wavs = (self._sample_wavelengths[bi] *
                             u.Angstrom.cgs.scale / zp1)
            else:
                rest_wavs = [cc / (self._frequencies[li] * zp1)]  # noqa: F841

            amp = lum * amps[li]

            if not evaled:
                sed = ne.evaluate(
                    'amp * exp(-0.5 * ((rest_wavs - lw) / ls) ** 2)')
                evaled = True
            else:
                sed = ne.re_evaluate()

            sed = np.nan_to_num(sed)

            norm = (lum + amp / zp1 * np.sqrt(np.pi / 2.0) * (
                1.0 + erf(lw / (np.sqrt(2.0) * ls)))) / lum

            seds[li] += sed
            seds[li] /= norm

        return {'sample_wavelengths': self._sample_wavelengths,
                self.key('seds'): seds}
