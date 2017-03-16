"""Definitions for the `BlackbodyCutoff` class."""
from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from astropy import units as u

from mosfit.constants import FOUR_PI
from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class BlackbodyCutoff(SED):
    """Blackbody spectral energy dist. for given temperature and radius,
        with a linear absorption function bluewards of a cutoff wavelength."""

    C_CONST = c.c.cgs.value
    FLUX_CONST = FOUR_PI * (
        2.0 * c.h * c.c ** 2 * pi).cgs.value * u.Angstrom.cgs.scale
    X_CONST = (c.h * c.c / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    ANG_CGS = u.Angstrom.cgs.scale

    def process(self, **kwargs):
        """Process module."""
        self._luminosities = kwargs['luminosities']
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._radius_phot = kwargs['radiusphot']
        self._temperature_phot = kwargs['temperaturephot']
        self._cutoff_wavelength = kwargs['cutoff_wavelength']
        self._times = kwargs['all_times']
        xc = self.X_CONST
        fc = self.FLUX_CONST
        cc = self.C_CONST
        ac = self.ANG_CGS
        temperature_phot = self._temperature_phot
        zp1 = 1.0 + kwargs['redshift']
        seds = []
        norm_arr = {}
        for li, lum in enumerate(self._luminosities):
            radius_phot = self._radius_phot[li]
            temperature_phot = self._temperature_phot[li]
            bi = self._band_indices[li]
            time = self._times[li]
            if lum == 0.0:
                if bi >= 0:
                    seds.append(np.zeros_like(self._sample_wavelengths[bi]))
                else:
                    seds.append([0.0])
                continue
            if bi >= 0:
                rest_wavs = (self._sample_wavelengths[bi]
                             * ac / zp1)
            else:
                rest_wavs = np.array([cc / (self._frequencies[li] * zp1)])

            # Apply absorption to SED only bluewards of cutoff wavelength
            absorbed = rest_wavs < self._cutoff_wavelength * ac

            # Blackbody SED
            sed = fc * (radius_phot**2 / rest_wavs**5) / (np.exp(xc /
                        rest_wavs / temperature_phot) - 1.0)

            # Absorbed blackbody: 0% transmission at 0 Angstroms
            # 100% at >3000 Angstroms
            sed[absorbed] = fc * (radius_phot**2 / ac / self._cutoff_wavelength
                        / rest_wavs[absorbed]**4) / (np.exp(xc /
                        rest_wavs[absorbed] / temperature_phot) - 1.0)

            sed = np.nan_to_num(sed)

            # Renormalise to conserve energy
            # print(time)
            if time in norm_arr:
                norm = norm_arr[time]
                # print('array',norm_arr[time])
                # Check if normalisation exists for given T(t)
            else:
                # If not, calculate ratio of absorbed blackbody to total
                # luminosity using this (series expansion) integral
                # of the absorbed blackbody function
                # wavelength < cutoff:
                f_blue = fc / ac * radius_phot**2 * temperature_phot * sum(
                np.exp(-n * xc / (self._cutoff_wavelength * ac *
                temperature_phot)) * (n**2 * xc**2 + 2 * n * temperature_phot *
                xc * self._cutoff_wavelength * ac + 2 * temperature_phot**2 *
                (self._cutoff_wavelength * ac)**2) / (n**3 * xc**3 *
                (self._cutoff_wavelength * ac)**3) for n in range (1,11))

                # wavelength > cutoff:
                f_red = fc / ac * radius_phot**2 * sum(
                temperature_phot * (6 * temperature_phot**3 - np.exp(
                -n * xc / (temperature_phot * self._cutoff_wavelength * ac)) *
                (n**3 * xc**3 + 3 * n**2 * temperature_phot * xc**2 *
                self._cutoff_wavelength * ac + 6 * n * temperature_phot**2 * xc
                * (self._cutoff_wavelength * ac)**2 + 6 * temperature_phot**3 *
                (self._cutoff_wavelength * ac)**3) /
                (self._cutoff_wavelength * ac)**3) / (n**4 * xc**4)
                for n in range (1,11))

                norm = lum / (f_blue + f_red)
                norm_arr[time] = norm
                # print('calc',norm_arr[time])

            # Apply renormalisation
            sed *= norm

            seds.append(sed)

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths, 'seds': seds}
