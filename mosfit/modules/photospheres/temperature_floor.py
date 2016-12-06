from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.photospheres.photosphere import photosphere

CLASS_NAME = 'temperature_floor'


class temperature_floor(photosphere):
    """Expanding/receding photosphere with a dense core + low-mass power-law
    envelope
    """

    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    RAD_CONST = KM_CGS * DAY_CGS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._rest_t_explosion = kwargs['resttexplosion']
        self._times = kwargs['rest_times']
        self._luminosities = kwargs['luminosities']
        self._temperature = kwargs['temperature']
        self._v_ejecta = kwargs['vejecta']
        self._m_ejecta = kwargs['mejecta']
        self._kappa = kwargs['kappa']
        self._radius2 = [(self.RAD_CONST *
                          self._v_ejecta * (x - self._rest_t_explosion))**2
                         for x in self._times]
        self._rec_radius2 = [
            x / (self.STEF_CONST * self._temperature**4)
            for x in self._luminosities
        ]
        rphot = []
        Tphot = []
        for li, lum in enumerate(self._luminosities):

            radius2 = self._radius2[li]
            rec_radius2 = self._rec_radius2[li]
            if radius2 < rec_radius2:
                temperature = (lum / (self.STEF_CONST * radius2))**0.25
            else:
                radius2 = rec_radius2
                temperature = self._temperature

            rphot.append(np.sqrt(radius2))

            Tphot.append(temperature)

        return {'radiusphot': rphot, 'temperaturephot': Tphot}
