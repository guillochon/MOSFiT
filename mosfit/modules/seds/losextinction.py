"""Definitions for the `LOSExtinction` class."""
from collections import OrderedDict

import numpy as np
from mosfit.modules.seds.sed import SED

from extinction import apply as eapp
from extinction import odonnell94


# Important: Only define one ``Module`` class per file.


class LOSExtinction(SED):
    """Adds extinction to SED from both host galaxy and MW."""

    MW_RV = 3.1

    def __init__(self):
        """Initialize module."""
        self._mm83 = [[0.03, 0.0, 0.0, 0.0],
                      [0.1, 17.3, 608.1, -2150.0],
                      [0.284, 34.6, 267.9, -476.1],
                      [0.4, 78.1, 18.8, 4.3],
                      [0.532, 71.4, 66.8, -51.4],
                      [0.707, 95.5, 145.8, -61.1],
                      [0.867, 308.9, -380.6, 294.0],
                      [1.303, 120.6, 169.3, -47.7],
                      [1.84, 141.3, 146.8, -31.5],
                      [2.471, 202.7, 104.7, -17.0],
                      [3.21, 342.7, 18.7, 0.0],
                      [4.038, 352.2, 18.7, 0.0],
                      [7.111, 433.9, -2.4, 0.75],
                      [8.331, 629.0, 30.9, 0.0],
                      [10.0, 701.2, 25.2, 0.0]
                      ]

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
            if bi >= 0 and np.count_nonzero(self._ext_indices[si]) > 0:
                if bi not in extinct_cache:
                    extinct_cache[bi] = odonnell94(
                        self._band_rest_wavelengths[bi][self._ext_indices[si]],
                        av_host, self._rv_host)
                # Add host and MW contributions
                eapp(
                    self._mw_extinct[bi] + extinct_cache[bi],
                    self._seds[si][self._ext_indices[si]],
                    inplace=True)
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
        # Pre-calculate LOS dust from MW for all bands
        self._mw_extinct = np.zeros_like(self._sample_wavelengths)
        self._ext_indices = []
        for si, sw in enumerate(self._sample_wavelengths):
            self._ext_indices.append(
                self._sample_wavelengths[si] >= 1.0e4 / 11.0)
            if np.count_nonzero(self._ext_indices[si]) > 0:
                self._mw_extinct[si][self._ext_indices[si]] = odonnell94(
                    self._sample_wavelengths[si][self._ext_indices[si]],
                    self._av_mw, self.MW_RV)
        self._preprocessed = True
