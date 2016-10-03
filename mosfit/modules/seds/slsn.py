from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c

from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.seds.sed import SED

CLASS_NAME = 'slsn'


class slsn(SED):
    """Expanding/receding photosphere with a core+envelope
    morphology and a blackbody spectral energy
    distribution.
    """

    FLUX_CONST = FOUR_PI * (2.0 * c.h / (c.c**2) * pi).cgs.value
    X_CONST = (c.h / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    PL_ENV = 10.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._t_explosion = kwargs['texplosion']
        self._times = kwargs['times']
        self._luminosities = kwargs['luminosities']
        self._bands = kwargs['bands']
        self._v_ejecta = kwargs['vejecta']
        self._m_ejecta = kwargs['mejecta']
        self._kappa = kwargs['kappa']
        xc = self.X_CONST
        fc = self.FLUX_CONST
        slope = self.PL_ENV
        zp1 = 1.0 + kwargs['redshift']
        seds = []
        temperature_last = 2.e4
        for li, lum in enumerate(self._luminosities):
            cur_band = self._bands[li]
            bi = self._filters.find_band_index(cur_band)
            rest_freqs = [x * zp1 for x in self._sample_frequencies[bi]]
            # rest_freqs3 = [x**3 for x in rest_freqs]

            # rest_wls = [x / zp1 for x in self._sample_wavelengths[bi]]

            # slsn_absorb = rest_wls
            # slsn_absorb[slsn_absorb>3500] = 1
            # print(slsn_absorb)

            # Radius is determined via expansion
            radius = self._v_ejecta * KM_CGS * (
                self._times[li] - self._t_explosion) * DAY_CGS

            # Compute density in core
            rho_core = (3.0 * self._m_ejecta * M_SUN_CGS /
                        (4.0 * pi * radius**3))

            # Attach power-law envelope of negligible mass
            tau_e = self._kappa * rho_core * radius / (slope - 1.0)

            # Find location of photosphere in envelope/core
            if tau_e > 0.667:
                radius_phot = (2.0 * (slope - 1.0) /
                               (3.0 * self._kappa * rho_core * radius
                                **slope))**(1.0 / (1.0 - slope))
            else:
                radius_phot = slope * radius / (slope - 1.0) - 2.0 / (
                    3.0 * self._kappa * rho_core)

            # Compute temperature
            # Prevent weird behaviour as R_phot -> 0
            if radius_phot > 0.55*radius:
                temperature_phot = (lum / (radius_phot**2 *
                                self.STEF_CONST))**(0.25)
            else:
                temperature_phot = temperature_last
                radius_phot = (lum / (temperature_phot**4 *
                                self.STEF_CONST))**(0.5)

            temperature_last = temperature_phot


            if li == 0:
                sed = ne.evaluate('fc * radius_phot**2 * rest_freqs**3 / '
                                  'exp(xc * rest_freqs / temperature_phot) - 1.0')
            else:
                sed = ne.re_evaluate()

            seds.append(list(sed))

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'samplewavelengths': self._sample_wavelengths, 'seds': seds}
