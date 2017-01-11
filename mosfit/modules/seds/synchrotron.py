from math import pi

import numpy as np
from astropy import constants as c

from mosfit.constants import FOUR_PI
from mosfit.modules.seds.sed import SED

CLASS_NAME = 'Synchrotron'


class Synchrotron(SED):
    """Synchrotron spectral energy distribution
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
        self._radius_source = kwargs['radiussource']
        self._nu_max = kwargs['numax']
        self._p = kwargs['p']
        self._b0 = kwargs['b0']
        zp1 = 1.0 + kwargs['redshift']
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
            fmax = self._b0**-0.5 * self._radius_source**2 * self._nu_max**2.5
            sed = [
                self._b0**-0.5 * self._radius_source**2 * (x / self._nu_max)
                **2.5 if x < self._nu_max else fmax * (x / self._nu_max)
                **(-(self._p - 1.0) / 2.0) for x in rest_freqs
            ]

            sed = np.nan_to_num(sed)

            seds.append(sed)

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths, 'seds': seds}
