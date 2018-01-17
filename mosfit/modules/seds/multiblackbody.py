"""Definitions for the `MultiBlackbody` class."""
from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS  # noqa: F401
from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class MultiBlackbody(SED):
    """Generalized multiple blackbody spectral energy distribution."""

    FLUX_CONST = FOUR_PI * (2.0 * c.h / (c.c ** 2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value

    def process(self, **kwargs):
        """Process module."""
        raise RuntimeError('`MultiBlackbody` is not yet functional.')
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._luminosities = kwargs[self.key('luminosities')]
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._areas = kwargs[self.key('areas')]
        self._temperature_phots = kwargs[self.key('temperaturephots')]
        xc = self.X_CONST  # noqa: F841
        fc = self.FLUX_CONST  # noqa: F841
        temperature_phot = self._temperature_phot
        zp1 = 1.0 + kwargs[self.key('redshift')]
        seds = []
        for li, lum in enumerate(self._luminosities):
            cur_band = self._bands[li]  # noqa: F841
            bi = self._band_indices[li]
            rest_freqs = [x * zp1  # noqa: F841
                          for x in self._sample_frequencies[bi]]
            wav_arr = np.array(self._sample_wavelengths[bi])  # noqa: F841
            radius_phot = self._radius_phot[li]  # noqa: F841
            temperature_phot = self._temperature_phot[li]  # noqa: F841

            if li == 0:
                sed = ne.evaluate(
                    'fc * radius_phot**2 * rest_freqs**3 / '
                    '(exp(xc * rest_freqs / temperature_phot) - 1.0)')
            else:
                sed = ne.re_evaluate()

            sed = np.nan_to_num(sed)

            seds.append(list(sed))

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths,
                self.key('seds'): seds}
