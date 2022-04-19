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

def bbody(lam,T,R2,sup_lambda, power_lambda):
    '''
    Calculate the corresponding blackbody radiance for a set
    of wavelengths given a temperature and radiance.

    Parameters
    ---------------
    lam: Reference wavelengths in Angstroms
    T:   Temperature in Kelvin
    R2:   Radius in cm, squared

    Output
    ---------------
    Spectral radiance in units of erg/s/Angstrom

    (calculation and constants checked by Sebastian Gomez)
    '''

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10

    # Convert wavelength to cm
    lam_cm = lam * 1E-8

    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_lam, in units of (erg / s) / cm ^ 2 / cm
    exponential = (h * c) / (lam_cm * k_B * T)
    B_lam = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)

    # Multiply by the surface area
    A = 4*np.pi*R2

    # Output radiance in units of (erg / s) / Angstrom
    Radiance = B_lam * A / 1E8

    # Apply Supression below sup_lambda wavelength
    Radiance[lam < sup_lambda] *= (lam[lam < sup_lambda]/sup_lambda)**power_lambda

    return Radiance

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
        self._alpha = kwargs[self.key('alpha')]
        self._times = np.array(kwargs['rest_times'])
        xc = self.X_CONST  # noqa: F841
        fc = self.FLUX_CONST
        cc = self.C_CONST
        ac = ANG_CGS
        zp1 = 1.0 + kwargs[self.key('redshift')]

        lt = len(self._times)
        seds = np.empty(lt, dtype=object)
        rp2  = self._radius_phot ** 2
        tp   = self._temperature_phot

        evaled = False
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if lum == 0.0:
                seds[li] = np.zeros(
                    len(self._sample_wavelengths[bi]) if bi >= 0 else 1)
                continue
            if bi >= 0:
                rest_wavs = self._sample_wavelengths[bi] * ac / zp1
            else:
                rest_wavs = np.array([cc / (self._frequencies[li] * zp1)])

            #if float(tp[li]) <= float(9e9):
            cwave_ac = float(max(1, self._cutoff_wavelength)) * ac
            #else:
            #    cwave_ac = float(1) * ac

            # Apply absorption to SED only bluewards of cutoff wavelength
            ab   = rest_wavs < cwave_ac  # noqa: F841
            tpi  = tp[li]  # noqa: F841
            rp2i = rp2[li]  # noqa: F841

            # Exponent of suppresion and rest wavelength should sum to 5
            sup_power  = float(max(0,self._alpha))
            wavs_power = (5 - sup_power)

            if not evaled:
                # Absorbed blackbody: 0% transmission at 0 Angstroms 100% at
                # >3000 Angstroms.
                sed = ne.evaluate(
                    "where(ab, fc * (rp2i / cwave_ac ** sup_power/ "
                    "rest_wavs ** wavs_power) / expm1(xc / rest_wavs / tpi), "
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

        sample_wavelengths = np.linspace(100, 100000, 1000)
        STEF_CONST      = (4.0 * pi * c.sigma_sb).cgs.value
        wavelength_list = np.ones(len(self._times)) * np.array(self._cutoff_wavelength).astype(float)# * ac
        power_list      = np.ones(len(self._times)) * np.array(self._alpha).astype(float)
        wavelength_list[wavelength_list < 0] = 0
        power_list     [power_list      < 0] = 0

        norms = np.array([(R2 * STEF_CONST * T ** 4) / np.trapz(bbody(sample_wavelengths,T,R2,wave,power), sample_wavelengths) for T, R2, wave, power in zip(tp[uniq_is],rp2[uniq_is],wavelength_list[uniq_is],power_list[uniq_is])])

        # Apply renormalisation
        seds *= norms[np.searchsorted(uniq_times, self._times)]
        seds = self.add_to_existing_seds(seds, **kwargs)

        # Units of `seds` is ergs / s / Angstrom.
        return {'sample_wavelengths': self._sample_wavelengths,
                self.key('seds'): seds,
                'luminosities_out': self._luminosities,
                'power_list': power_list,
                'wavelength_list': wavelength_list,
                'times_out': self._times
                }