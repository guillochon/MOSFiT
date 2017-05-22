"""Definitions for the `LOSExtinction` class."""
from collections import OrderedDict

import astropy.constants as c
import astropy.units as u
import numpy as np
from mosfit.modules.seds.sed import SED

from extinction import apply as eapp
from extinction import odonnell94


# Important: Only define one ``Module`` class per file.


class LOSExtinction(SED):
    """Adds extinction to SED from both host galaxy and MW."""

    MW_RV = 3.1
    H_CGS = c.h.cgs.value
    C_CGS = c.c.cgs.value
    H_C_CGS = c.h.cgs.value * c.c.cgs.value
    ANG_CGS = u.Angstrom.cgs.scale
    KEV_CGS = u.keV.cgs.scale
    LYMAN = 912.0

    def __init__(self, **kwargs):
        """Initialize module."""
        super(LOSExtinction, self).__init__(**kwargs)

        self._ref_table = {
            '1': [
                {'bibcode': '1994ApJ...422..158O'}
            ],
            '2': [
                {'bibcode': '1983ApJ...270..119M'},
                {'bibcode': '1994AJ....107.2108R'}
            ]
        }

        self._mm83 = np.array(
            [[0.03, 17.3, 608.1, -2150.0],
             [0.1, 34.6, 267.9, -476.1],
             [0.284, 78.1, 18.8, 4.3],
             [0.4, 71.4, 66.8, -51.4],
             [0.532, 95.5, 145.8, -61.1],
             [0.707, 308.9, -380.6, 294.0],
             [0.867, 120.6, 169.3, -47.7],
             [1.303, 141.3, 146.8, -31.5],
             [1.84, 202.7, 104.7, -17.0],
             [2.471, 342.7, 18.7, 0.0],
             [3.21, 352.2, 18.7, 0.0],
             [4.038, 433.9, -2.4, 0.75],
             [7.111, 629.0, 30.9, 0.0],
             [8.331, 701.2, 25.2, 0.0]
             ])
        self._min_xray = 0.03
        self._max_xray = 10.0
        self._min_wavelength = 1.0 * self.C_CGS / (
            self._max_xray * self.KEV_CGS / self.H_CGS)
        self._almin = 1.0e-24 * (
            self._mm83[0, 1] + self._mm83[0, 2] * self._min_xray +
            self._mm83[0, 3] * self._min_xray ** 2) / self._min_xray ** 3
        self._almax = 1.0e-24 * (
            self._mm83[-1, 1] + self._mm83[-1, 2] * self._max_xray +
            self._mm83[-1, 3] * self._max_xray ** 2) / self._max_xray ** 3

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self.preprocess(**kwargs)
        zp1 = 1.0 + kwargs[self.key('redshift')]
        self._seds = kwargs[self.key('seds')]
        self._nh_host = kwargs[self.key('nhhost')]
        self._rv_host = kwargs[self.key('rvhost')]
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._band_rest_wavelengths = self._sample_wavelengths / zp1

        av_host = self._nh_host / 1.8e21

        extinct_cache = OrderedDict()
        for si, cur_band in enumerate(self._bands):
            bi = self._band_indices[si]
            # Extinct out host gal (using rest wavelengths)
            if bi >= 0:
                if bi not in extinct_cache:
                    extinct_cache[bi] = np.zeros_like(
                        self._band_rest_wavelengths[bi])
                    ind = self._ext_indices[bi]
                    if len(ind) > 0:
                        extinct_cache[bi][ind] = odonnell94(
                            self._band_rest_wavelengths[bi][ind],
                            av_host, self._rv_host)
                    ind = self._x_indices[bi]
                    if len(ind) > 0:
                        extinct_cache[bi][ind] = self.mm83(
                            self._nh_host,
                            self._band_rest_wavelengths[bi][ind])
                # Add host and MW contributions
                eapp(
                    self._mw_extinct[bi] + extinct_cache[bi],
                    self._seds[si], inplace=True)
            else:
                # wavelengths = np.array(
                #   [c.c.cgs.value / self._frequencies[si]])
                # Need extinction function for radio
                pass

        return {
            'sample_wavelengths': self._sample_wavelengths,
            self.key('seds'): self._seds,
            self.key('avhost'): av_host
        }

    def preprocess(self, **kwargs):
        """Preprocess module."""
        if self._preprocessed:
            return
        self._ebv = kwargs[self.key('ebv')]
        self._av_mw = self.MW_RV * self._ebv
        self._nh_mw = self._av_mw * 1.8e21
        # Pre-calculate LOS dust from MW for all bands
        self._mw_extinct = np.zeros_like(self._sample_wavelengths)
        self._ext_indices = []
        self._x_indices = []
        add_refs = set()
        for si, sw in enumerate(self._sample_wavelengths):
            self._ext_indices.append(
                self._sample_wavelengths[si] >= self.LYMAN)
            self._x_indices.append(
                (self._sample_wavelengths[si] >= self._min_wavelength) &
                (self._sample_wavelengths[si] < self.LYMAN))
            if len(self._ext_indices[si]) > 0:
                self._mw_extinct[si][self._ext_indices[si]] = odonnell94(
                    self._sample_wavelengths[si][self._ext_indices[si]],
                    self._av_mw, self.MW_RV)
                add_refs.add('1')
            if len(self._x_indices[si]) > 0:
                self._mw_extinct[si][self._x_indices[si]] = self.mm83(
                    self._nh_mw,
                    self._sample_wavelengths[si][self._x_indices[si]])
                add_refs.add('2')
        for ref in list(add_refs):
            self._REFERENCES.extend(self._ref_table[ref])
        self._preprocessed = True

    def mm83(self, nh, waves):
        """X-ray extinction in the ISM from Morisson & McCammon 1983."""
        y = np.array([self.H_C_CGS / (x * self.ANG_CGS * self.KEV_CGS)
                      for x in waves])
        i = np.array([np.searchsorted(self._mm83[:, 0], x) - 1 for x in y])
        al = [1.0e-24 * (self._mm83[x, 1] + self._mm83[x, 2] * y[j] +
                         self._mm83[x, 3] * y[j] ** 2) / y[j] ** 3
              for j, x in enumerate(i)]
        # For less than 0.03 keV assume cross-section scales as E^-3.
        # http://ned.ipac.caltech.edu/level5/Madau6/Madau1_2.html
        # See also Rumph, Boyer, & Vennes 1994.
        al = [al[j] if x < self._min_xray
              else self._almin * (self._min_xray / x) ** 3
              for j, x in enumerate(y)]
        al = [al[j] if x > self._max_xray
              else self._almax * (self._max_xray / x) ** 3
              for j, x in enumerate(y)]
        return nh * np.array(al)
