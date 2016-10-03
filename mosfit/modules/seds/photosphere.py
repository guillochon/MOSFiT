from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS
from mosfit.modules.seds.sed import SED

CLASS_NAME = 'Photosphere'


class Photosphere(SED):
    """Expanding/recending photosphere with a blackbody spectral energy
    distribution.
    """

    FLUX_CONST = FOUR_PI * (2.0 * c.h / (c.c**2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    RAD_CONST = KM_CGS * DAY_CGS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._preprocessed = False

    def process(self, **kwargs):
        self.preprocess(**kwargs)
        self._t_explosion = kwargs['texplosion']
        self._times = kwargs['times']
        self._luminosities = kwargs['luminosities']
        self._temperature = kwargs['temperature']
        self._v_ejecta = kwargs['vejecta']
        self._radius2 = [(self.RAD_CONST * self._v_ejecta *
                          (x - self._t_explosion))**2 for x in self._times]
        self._rec_radius2 = [x / (self.STEF_CONST * self._temperature**4)
                             for x in self._luminosities]
        xc = self.X_CONST
        fc = self.FLUX_CONST
        zp1 = 1.0 + kwargs['redshift']
        seds = []
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            rest_freqs = [x * zp1 for x in self._sample_frequencies[bi]]

            # Radius is determined via expansion, unless this would make
            # temperature lower than temperature parameter.
            radius2 = self._radius2[li]
            rec_radius2 = self._rec_radius2[li]
            if radius2 < rec_radius2:
                temperature = (lum / (self.STEF_CONST * radius2))**0.25
            else:
                radius2 = rec_radius2
                temperature = self._temperature

            if li == 0:
                sed = ne.evaluate('fc * radius2 * rest_freqs**3 / '
                                  'exp(xc * rest_freqs / temperature) - 1.0')
            else:
                sed = ne.re_evaluate()
            seds.append(list(sed))

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'samplewavelengths': self._sample_wavelengths, 'seds': seds}

    def preprocess(self, **kwargs):
        if not self._preprocessed:
            self._bands = kwargs['bands']
            self._band_indices = list(
                map(self._filters.find_band_index, self._bands))
        self._preprocessed = True
