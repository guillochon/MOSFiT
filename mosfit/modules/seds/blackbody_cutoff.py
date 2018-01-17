"""Definitions for the `BlackbodyCutoff` class."""
from math import pi

import numexpr as ne
import numpy as np
from astrocats.catalog.source import SOURCE
from astropy import constants as c
from astropy import units as u

from mosfit.constants import ANG_CGS, FOUR_PI
from mosfit.modules.seds.sed import SED


# Important: Only define one ``Module`` class per file.


class BlackbodyCutoff(SED):
    """Blackbody SED with cutoff.

    Blackbody spectral energy dist. for given temperature and radius,
    with a linear absorption function bluewards of a cutoff wavelength.
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2017arXiv170600825N'}
    ]

    C_CONST = c.c.cgs.value
    FLUX_CONST = FOUR_PI * (
        2.0 * c.h * c.c ** 2 * pi).cgs.value * u.Angstrom.cgs.scale
    X_CONST = (c.h * c.c / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    F_TERMS = 10

    def __init__(self, **kwargs):
        """Initialize module."""
        super(BlackbodyCutoff, self).__init__(**kwargs)
        self._nxcs = self.X_CONST * np.array(range(1, self.F_TERMS + 1))

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._luminosities = kwargs[self.key('luminosities')]
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._radius_phot = np.array(kwargs[self.key('radiusphot')])
        self._temperature_phot = np.array(kwargs[self.key('temperaturephot')])
        self._cutoff_wavelength = kwargs[self.key('cutoff_wavelength')]
        self._times = np.array(kwargs['rest_times'])
        xc = self.X_CONST  # noqa: F841
        fc = self.FLUX_CONST
        cc = self.C_CONST
        ac = ANG_CGS
        cwave_ac = self._cutoff_wavelength * ac
        cwave_ac2 = cwave_ac * cwave_ac
        cwave_ac3 = cwave_ac2 * cwave_ac  # noqa: F841
        zp1 = 1.0 + kwargs[self.key('redshift')]

        lt = len(self._times)

        seds = np.empty(lt, dtype=object)
        rp2 = self._radius_phot ** 2
        tp = self._temperature_phot

        evaled = False
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            # tpi = tp[li]
            # rp2i = rp2[li]
            if lum == 0.0:
                seds[li] = np.zeros(
                    len(self._sample_wavelengths[bi]) if bi >= 0 else 1)
                continue
            if bi >= 0:
                rest_wavs = self._sample_wavelengths[bi] * ac / zp1
            else:
                rest_wavs = np.array([cc / (self._frequencies[li] * zp1)])

            # Apply absorption to SED only bluewards of cutoff wavelength
            ab = rest_wavs < cwave_ac  # noqa: F841
            tpi = tp[li]  # noqa: F841
            rp2i = rp2[li]  # noqa: F841

            if not evaled:
                # Absorbed blackbody: 0% transmission at 0 Angstroms 100% at
                # >3000 Angstroms.
                sed = ne.evaluate(
                    "where(ab, fc * (rp2i / cwave_ac / "
                    "rest_wavs ** 4) / expm1(xc / rest_wavs / tpi), "
                    "fc * (rp2i / rest_wavs ** 5) / "
                    "expm1(xc / rest_wavs / tpi))"
                )
                evaled = True
            else:
                sed = ne.re_evaluate()

            sed[np.isnan(sed)] = 0.0
            seds[li] = sed

        uniq_times = np.unique(self._times)
        tsort = np.argsort(self._times)
        uniq_is = np.searchsorted(self._times, uniq_times, sorter=tsort)
        lu = len(uniq_times)

        norms = self._luminosities[
            uniq_is] / (fc / ac * rp2[uniq_is] * tp[uniq_is])

        rp2 = rp2[uniq_is].reshape(lu, 1)
        tp = tp[uniq_is].reshape(lu, 1)
        tp2 = tp * tp
        tp3 = tp2 * tp  # noqa: F841
        nxcs = self._nxcs  # noqa: F841

        f_blue_reds = ne.evaluate(
            "sum((exp(-nxcs / (cwave_ac * tp)) * ("
            "nxcs ** 2 + 2 * ("
            "nxcs * cwave_ac * tp + cwave_ac2 * tp2)) / ("
            "nxcs ** 3 * cwave_ac3)) + "
            "(6 * tp3 - exp(-nxcs / (cwave_ac * tp)) * ("
            "nxcs ** 3 + 3 * nxcs ** 2 * cwave_ac * tp + 6 * ("
            "nxcs * cwave_ac2 * tp2 + cwave_ac3 *"
            "tp3)) / cwave_ac3) / (nxcs ** 4), 1)"
        )

        norms /= f_blue_reds

        # Apply renormalisation
        seds *= norms[np.searchsorted(uniq_times, self._times)]

        seds = self.add_to_existing_seds(seds, **kwargs)

        return {'sample_wavelengths': self._sample_wavelengths,
                self.key('seds'): seds}
