import csv
import os
from math import log10, pi

import numpy as np
from astropy import units as u

from ..module import Module

CLASS_NAME = 'Band'


class Band(Module):
    """Band-pass filter
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._path = kwargs['path']
        self._band = kwargs['band']
        with open(os.path.join('friendlyfit', 'modules', self._path),
                  'r') as f:
            rows = []
            for row in csv.reader(f, delimiter='\t', skipinitialspace=True):
                rows.append([float(x) for x in row])
        self._wavelengths, self._transmissions = list(map(list, zip(*rows)))
        self._min_wave = min(self._wavelengths)
        self._max_wave = max(self._wavelengths)
        self._filter_integral = np.trapz(
            np.array(self._transmissions), np.array(self._wavelengths))

    def process(self, **kwargs):
        self._lumdist = (kwargs['lumdist'] * u.Mpc).cgs.value
        bi = kwargs['bands'].index(self._band)
        seds = kwargs['seds'][bi]
        wavs = kwargs['wavelengths'][bi]
        mags = []
        for sed in seds:
            itrans = np.interp(wavs, self._wavelengths, self._transmissions)
            eff_flux = np.trapz(
                np.array([x * y for x, y in zip(itrans, sed)]), np.array(wavs))
            eff_flux = eff_flux / self._filter_integral
            mags.append(self.abmag(eff_flux))
        return {'model_magnitudes': mags}

    def abmag(self, eff_flux):
        if eff_flux == 0.0:
            return np.nan
        return -48.6 - (100.0**0.2) * log10(eff_flux /
                                            (4.0 * pi * self._lumdist**2))

    def request(self, request):
        if request == 'band':
            return self._band
        elif request == 'wavelengths':
            return [self._min_wave, self._max_wave]
        return []
