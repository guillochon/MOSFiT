from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c

from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.seds.sed import SED

CLASS_NAME = 'blackbody'


class blackbody(SED):
    """Expanding/receding photosphere with a core+envelope
    morphology and a blackbody spectral energy
    distribution.
    """

    FLUX_CONST = FOUR_PI * (2.0 * c.h / (c.c**2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._luminosities = kwargs['luminosities']
        self._bands = kwargs['obsbands']
        self._radius_phot = kwargs['radiusphot']
        self._temperature_phot = kwargs['temperaturephot']
        xc = self.X_CONST
        fc = self.FLUX_CONST
        temperature_phot = self._temperature_phot
        zp1 = 1.0 + kwargs['redshift']
        seds = []
        for li, lum in enumerate(self._luminosities):
            cur_band = self._bands[li]
            bi = self._filters.find_band_index(cur_band)
            rest_freqs = [x * zp1 for x in self._sample_frequencies[bi]]
            wav_arr = np.array(self._sample_wavelengths[bi])
            radius_phot = self._radius_phot[li]
            temperature_phot = self._temperature_phot[li]

            if li == 0:
                sed = ne.evaluate(
                    'fc * radius_phot**2 * rest_freqs**3 / '
                    '(exp(xc * rest_freqs / temperature_phot) - 1.0)')
            else:
                sed = ne.re_evaluate()

            sed = np.nan_to_num(sed)

            seds.append(list(sed))

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'samplewavelengths': self._sample_wavelengths, 'seds': seds}
