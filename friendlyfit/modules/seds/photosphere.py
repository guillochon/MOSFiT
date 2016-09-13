from math import pi

import numpy as np
import numexpr as ne
from astropy import constants as c

from ...constants import DAY_CGS, FOUR_PI, KM_CGS
from .sed import SED

CLASS_NAME = 'Photosphere'


class Photosphere(SED):
    """Expanding/recending photosphere with a blackbody spectral energy
    distribution.
    """

    FLUX_CONST = FOUR_PI * (2.0 * c.h / (c.c**2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._t_explosion = kwargs['texplosion']
        self._times = kwargs['times']
        self._luminosities = kwargs['luminosities']
        self._temperature = kwargs['temperature']
        self._bands = kwargs['bands']
        self._v_ejecta = kwargs['vejecta']
        xc = self.X_CONST
        fc = self.FLUX_CONST
        zp1 = 1.0 + kwargs['redshift']
        seds = []
        for li, lum in enumerate(self._luminosities):
            cur_band = self._bands[li]
            bi = self._band_names.index(cur_band)
            rest_freqs = [x * zp1 for x in self._band_frequencies[bi]]
            # rest_freqs3 = [x**3 for x in rest_freqs]

            # Radius is determined via expansion, unless this would make
            # temperature lower than temperature parameter.
            # radius = self._v_ejecta * KM_CGS * (
            #     self._times[li] - self._t_explosion) * DAY_CGS
            rec_radius = np.sqrt(lum /
                                 (self.STEF_CONST * self._temperature**4))
            # if radius < rec_radius:
            #     radius2 = radius**2
            #     temperature = (lum / (self.STEF_CONST * radius2))**0.25
            # else:
            radius2 = rec_radius**2
            temperature = self._temperature

            if li == 0:
                sed = ne.evaluate('fc * radius2 * rest_freqs**3 / '
                                  'exp(xc * rest_freqs / temperature) - 1.0')
            else:
                sed = ne.re_evaluate()
            # a = [np.exp(self.X_CONST * x / temperature) - 1.0
            #      for x in rest_freqs]
            # sed = [
            #     (self.FLUX_CONST * radius2 * x / y)
            #     for x, y in zip(rest_freqs3, a)
            # ]
            seds.append(sed)
        return {'bandwavelengths': self._band_wavelengths, 'seds': seds}
