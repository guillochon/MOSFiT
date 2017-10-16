"""Definitions for the `TemperatureFloor` class."""
import numpy as np
from astrocats.catalog.source import SOURCE
from astropy import constants as c

from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS
from mosfit.modules.photospheres.photosphere import Photosphere


# Important: Only define one ``Module`` class per file.


class TemperatureFloor(Photosphere):
    """Photosphere with a minimum allowed temperature.

    Photosphere that expands and cools with ejecta then recedes at constant
    final temperature.
    """

    _REFERENCES = [
        {SOURCE.NAME: 'Nicholl et al. 2017'}
    ]

    STEF_CONST = (FOUR_PI * c.sigma_sb).cgs.value
    RAD_CONST = KM_CGS * DAY_CGS

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._times = kwargs[self.key('rest_times')]
        self._luminosities = kwargs[self.key('luminosities')]
        self._temperature = kwargs[self.key('temperature')]
        self._v_ejecta = kwargs[self.key('vejecta')]
        self._m_ejecta = kwargs[self.key('mejecta')]
        self._kappa = kwargs[self.key('kappa')]
        self._radius2 = [(self.RAD_CONST *
                          self._v_ejecta * (x - self._rest_t_explosion)) ** 2
                         for x in self._times]
        self._rec_radius2 = [
            x / (self.STEF_CONST * self._temperature ** 4)
            for x in self._luminosities
        ]
        rphot = []
        Tphot = []
        for li, lum in enumerate(self._luminosities):

            radius2 = self._radius2[li]
            rec_radius2 = self._rec_radius2[li]
            if radius2 < rec_radius2:
                temperature = (lum / (self.STEF_CONST * radius2)) ** 0.25
            else:
                radius2 = rec_radius2
                temperature = self._temperature

            rphot.append(np.sqrt(radius2))

            Tphot.append(temperature)

        return {self.key('radiusphot'): rphot,
                self.key('temperaturephot'): Tphot}
