"""Definitions for the `DenseCore` class."""
from math import pi

import numpy as np
from astropy import constants as c
from mosfit.constants import DAY_CGS, KM_CGS, M_SUN_CGS
from mosfit.modules.photospheres.photosphere import Photosphere


# Important: Only define one ``Module`` class per file.


class DenseCore(Photosphere):
    """Photosphere with a dense core and a low-mass envelope.

    Expanding/receding photosphere with a dense core + low-mass power-law
    envelope.
    """

    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    PL_ENV = 10.0

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
        self._times = kwargs[self.key('rest_times')]
        self._luminosities = kwargs[self.key('luminosities')]
        self._v_ejecta = kwargs[self.key('vejecta')]
        self._m_ejecta = kwargs[self.key('mejecta')]
        self._kappa = kwargs[self.key('kappa')]
        slope = self.PL_ENV
        peak = np.argmax(np.array(self._luminosities))
        rphot = []
        Tphot = []
        temperature_last = 1.e5
        for li, lum in enumerate(self._luminosities):

            # Radius is determined via expansion
            radius = self._v_ejecta * KM_CGS * (
                self._times[li] - self._rest_t_explosion) * DAY_CGS

            # Compute density in core
            rho_core = (3.0 * self._m_ejecta * M_SUN_CGS /
                        (4.0 * pi * radius ** 3))

            tau_core = self._kappa * rho_core * radius

            # Attach power-law envelope of negligible mass
            tau_e = self._kappa * rho_core * radius / (slope - 1.0)

            # Find location of photosphere in envelope/core
            if tau_e > (2.0 / 3.0):
                radius_phot = (
                    2.0 * (slope - 1.0) /
                    (3.0 * self._kappa * rho_core * radius ** slope)) ** (
                        1.0 / (1.0 - slope))
            else:
                radius_phot = slope * radius / (slope - 1.0) - 2.0 / (
                    3.0 * self._kappa * rho_core)

            # Compute temperature
            # Prevent weird behaviour as R_phot -> 0
            if tau_core > 1.:
                temperature_phot = (
                    lum / (radius_phot ** 2 * self.STEF_CONST)) ** 0.25
                if li > peak and temperature_phot > temperature_last:
                    temperature_phot = temperature_last
                    radius_phot = (
                        lum / (temperature_phot ** 4 * self.STEF_CONST)) ** 0.5
            else:
                temperature_phot = temperature_last
                radius_phot = (
                    lum / (temperature_phot ** 4 * self.STEF_CONST)) ** 0.5

            temperature_last = temperature_phot

            rphot.append(radius_phot)

            Tphot.append(temperature_phot)

        Tphot[0] = Tphot[1]

        return {self.key('radiusphot'): rphot,
                self.key('temperaturephot'): Tphot}
