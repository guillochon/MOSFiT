"""Definitions for the `Blackbody` class."""
from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.constants import FOUR_PI
from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class Blackbody(SED):
    """Blackbody spectral energy dist. for given temperature and radius."""

    C_CONST = c.c.cgs.value
    FLUX_CONST = FOUR_PI * (
        2.0 * c.h * c.c ** 2 * pi).cgs.value * u.Angstrom.cgs.scale
    X_CONST = (c.h * c.c / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value

    def process(self, **kwargs):
        """Process module."""
        lum_key = self.key('luminosities')
        kwargs = self.prepare_input(lum_key, **kwargs)
        self._luminosities = kwargs[lum_key]
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._radius_phot = kwargs[self.key('radiusphot')]
        self._temperature_phot = kwargs[self.key('temperaturephot')]
        xc = self.X_CONST  # noqa: F841
        fc = self.FLUX_CONST  # noqa: F841
        cc = self.C_CONST
        temperature_phot = self._temperature_phot

        # Some temp vars for speed.
        zp1 = 1.0 + kwargs[self.key('redshift')]
        Azp1 = u.Angstrom.cgs.scale / zp1
        czp1 = cc / zp1

        seds = []
        rest_wavs_dict = {}
        evaled = False

        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if lum == 0.0:
                seds.append(np.zeros(len(
                    self._sample_wavelengths[bi]) if bi >= 0 else 1))
                continue

            if bi >= 0:
                rest_wavs = rest_wavs_dict.setdefault(
                    bi, self._sample_wavelengths[bi] * Azp1)
            else:
                rest_wavs = np.array(  # noqa: F841
                    [czp1 / self._frequencies[li]])

            radius_phot = self._radius_phot[li]  # noqa: F841
            temperature_phot = self._temperature_phot[li]  # noqa: F841

            if not evaled:
                seds.append(ne.evaluate(
                    'fc * radius_phot**2 / rest_wavs**5 / '
                    'expm1(xc / rest_wavs / temperature_phot)'))
                evaled = True
            else:
                seds.append(ne.re_evaluate())

            seds[-1][np.isnan(seds[-1])] = 0.0

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths, 'seds': seds}
