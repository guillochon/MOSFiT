from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c

from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.seds.sed import SED

# Important: Only define one `Module` class per file.


class Blackbody(SED):
    """Blackbody spectral energy distribution for given temperature and radius
    """

    C_CONST = c.c.cgs.value
    FLUX_CONST = FOUR_PI * (2.0 * c.h / (c.c**2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value

    def process(self, **kwargs):
        self._luminosities = kwargs['luminosities']
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._radius_phot = kwargs['radiusphot']
        self._temperature_phot = kwargs['temperaturephot']
        xc = self.X_CONST
        fc = self.FLUX_CONST
        cc = self.C_CONST
        temperature_phot = self._temperature_phot
        zp1 = 1.0 + kwargs['redshift']
        seds = []
        evaled = False
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
            radius_phot = self._radius_phot[li]
            temperature_phot = self._temperature_phot[li]

            if not evaled:
                sed = ne.evaluate(
                    'fc * radius_phot**2 * rest_freqs**3 / '
                    '(exp(xc * rest_freqs / temperature_phot) - 1.0)')
                evaled = True
            else:
                sed = ne.re_evaluate()

            sed = np.nan_to_num(sed)

            seds.append(sed)

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths, 'seds': seds}
