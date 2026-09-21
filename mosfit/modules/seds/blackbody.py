"""Definitions for the `Blackbody` class."""
from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.constants import BOL_BAND_INDEX, FOUR_PI
from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class Blackbody(SED):
    """Blackbody spectral energy dist. for given temperature and radius."""

    C_CONST = c.c.cgs.value
    FLUX_CONST = FOUR_PI * (
        2.0 * c.h * c.c ** 2 * pi).cgs.value * u.Angstrom.cgs.scale
    X_CONST = (c.h * c.c / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value

    def process(self, **kwargs):
        """Process module."""
        lum_key = self.key('luminosities')
        kwargs = self.prepare_input(lum_key, **kwargs)
        self._luminosities = kwargs[lum_key]
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._radius_phot = kwargs[self.key('radiusphot')]
        self._temperature_phot = kwargs[self.key('temperaturephot')]
        self._times = np.array(kwargs['rest_times'])
        xc = self.X_CONST  # noqa: F841
        fc = self.FLUX_CONST  # noqa: F841
        cc = self.C_CONST
        temperature_phot = self._temperature_phot

        # Some temp vars for speed.
        zp1 = 1.0 + kwargs[self.key('redshift')]
        Azp1 = u.Angstrom.cgs.scale / zp1
        czp1 = cc / zp1

        luminosities = np.asarray(self._luminosities, dtype=float)
        band_indices = np.asarray(self._band_indices)
        frequencies = np.asarray(self._frequencies)
        radius_all = np.asarray(self._radius_phot, dtype=float)
        temperature_all = np.asarray(self._temperature_phot, dtype=float)
        n_rows = luminosities.shape[0]
        seds = self.alloc_seds(n_rows)

        bol = band_indices == BOL_BAND_INDEX
        zero_lum = (luminosities == 0.0) & ~bol

        def _planck_block(rest_wavs, radius_phot, temperature_phot):
            """One Planck / numexpr eval for a shared wavelength grid."""
            rest_wavs = np.asarray(rest_wavs, dtype=float)
            if rest_wavs.ndim == 1:
                rest_wavs = rest_wavs[None, :]
            radius_phot = np.asarray(radius_phot, dtype=float)[:, None]
            temperature_phot = np.asarray(temperature_phot, dtype=float)[:, None]
            block = ne.evaluate(
                'fc * radius_phot**2 / rest_wavs**5 / '
                'expm1(xc / rest_wavs / temperature_phot)',
                local_dict={
                    'fc': fc,
                    'xc': xc,
                    'radius_phot': radius_phot,
                    'rest_wavs': rest_wavs,
                    'temperature_phot': temperature_phot,
                })
            block[np.isnan(block)] = 0.0
            return block

        rest_wavs_dict = {}
        banded = (band_indices >= 0) & ~bol & ~zero_lum
        for bi in np.unique(band_indices[banded]):
            bi = int(bi)
            rest_wavs = rest_wavs_dict.setdefault(
                bi, self._sample_wavelengths[bi] * Azp1)
            idx = np.flatnonzero(banded & (band_indices == bi))
            block = _planck_block(
                rest_wavs, radius_all[idx], temperature_all[idx])
            seds[idx] = block

        freq_rows = (band_indices < 0) & ~bol & ~zero_lum
        if np.any(freq_rows):
            idx = np.flatnonzero(freq_rows)
            rest_wavs = (czp1 / frequencies[idx])[:, None]
            block = _planck_block(
                rest_wavs, radius_all[idx], temperature_all[idx])
            seds[idx, 0] = block[:, 0]

        seds = self.add_to_existing_seds(seds, **kwargs)

        # Units of `seds` is ergs / s / Angstrom.
        tor = {
            'sample_wavelengths': self._sample_wavelengths,
            self.key('seds'): seds,
            'luminosities_out': self._luminosities,
            'times_out': self._times
        }

        
        return tor
