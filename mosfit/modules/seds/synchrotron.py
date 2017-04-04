"""Definitions for the `Synchrotron` class."""
from math import pi

import numpy as np
from astropy import constants as c
from astropy import units as u

from mosfit.constants import FOUR_PI
from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class Synchrotron(SED):
    """Synchrotron spectral energy distribution."""

    C_CONST = c.c.cgs.value
    FLUX_CONST = FOUR_PI * (2.0 * c.h / (c.c ** 2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    ANG_CGS = u.Angstrom.cgs.scale

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._luminosities = kwargs[self.key('luminosities')]
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._radius_source = kwargs[self.key(self.key('radiussource'))]
        self._nu_max = kwargs[self.key(self.key('numax'))]
        self._p = kwargs[self.key(self.key('p'))]
        self._f0 = kwargs[self.key(self.key('f0'))]
        cc = self.C_CONST
        ac = self.ANG_CGS
        zp1 = 1.0 + kwargs[self.key(self.key('redshift'))]
        seds = []
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if lum == 0.0:
                if bi >= 0:
                    seds.append(np.zeros_like(self._sample_frequencies[bi]))
                else:
                    seds.append([0.0])
                continue
            if bi >= 0:
                rest_freqs = self._sample_frequencies[bi] * zp1
            else:
                rest_freqs = [self._frequencies[li] * zp1]

            # Below is not scaled properly, just proof of concept
            fmax = self._f0 * self._radius_source ** 2 * self._nu_max ** 2.5
            sed = [
                self._f0 * self._radius_source ** 2 * (x / self._nu_max) **
                2.5 * ac / cc * x ** 2 if x < self._nu_max
                else fmax * (x / self._nu_max) ** (-(self._p - 1.0) / 2.0) *
                ac / cc * x ** 2 for x in rest_freqs
            ]

            sed = np.nan_to_num(sed)

            seds.append(sed)

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths,
                self.key('seds'): seds}
