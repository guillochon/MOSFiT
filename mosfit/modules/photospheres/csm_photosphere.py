"""Definitions for the `TemperatureFloor` class."""
import numpy as np
from astrocats.catalog.source import SOURCE
from astropy import constants as c

from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS, AU_CGS
from mosfit.modules.photospheres.photosphere import Photosphere


# Important: Only define one ``Module`` class per file.


class TemperatureFloor(Photosphere):
    """Photosphere with a minimum allowed temperature.

    Photosphere that expands and cools with ejecta then recedes at constant
    final temperature.
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2017arXiv170600825N'}
    ]

    STEF_CONST = (FOUR_PI * c.sigma_sb).cgs.value
    RAD_CONST = KM_CGS * DAY_CGS

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._s = kwargs[self.key('s')]
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._times = kwargs[self.key('rest_times')]
        self._mcsm = kwargs[self.key('mcsm')] * M_SUN_CGS
        self._R0 = kwargs[self.key('r0')] * AU_CGS  # AU to cm
        self._rho = kwargs[self.key('rho')]
        self._luminosities = kwargs[self.key('luminosities')]
        self._temperature = kwargs[self.key('temperature')]
        self._v_ejecta = kwargs[self.key('vejecta')]
        self._m_ejecta = kwargs[self.key('mejecta')]
        self._kappa = kwargs[self.key('kappa')]
        self._radius2 = [(self.RAD_CONST *
                          self._v_ejecta * max(
                              x - self._rest_t_explosion, 0.0)) ** 2
                         for x in self._times]
        self._rec_radius2 = [
            x / (self.STEF_CONST * self._temperature ** 4)
            for x in self._luminosities
        ]

        # scaling constant for CSM density profile.
        self._q = self._rho * self._R0**self._s

        # outer radius of CSM shell.
        self._Rcsm = (
            ((3.0 - self._s) /
             (4.0 * np.pi * self._q) * self._mcsm + self._R0 ** (
                 3.0 - self._s)) ** (1.0 / (3.0 - self._s)))

        # radius of photosphere (should be within CSM).
        self._Rph = abs(
            (-2.0 * (1.0 - self._s) /
             (3.0 * self._kappa * self._q) + self._Rcsm**(1.0 - self._s)) **
            (1.0 /
             (1.0 - self._s)))

        rphot = []
        Tphot = []
        for li, lum in enumerate(self._luminosities):

            if lum == 0.0:
                temperature = 0.0
            else:
                temperature = (lum / (self.STEF_CONST * self._Rph ** 2)) ** 0.25

            rphot.append(self._Rph)

            Tphot.append(temperature)

        return {self.key('radiusphot'): rphot,
                self.key('temperaturephot'): Tphot}
