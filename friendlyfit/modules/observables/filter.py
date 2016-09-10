import csv
import os
from math import log10, pi

import numpy as np
from astropy import units as u

from ..module import Module

CLASS_NAME = 'Filter'


class Filter(Module):
    """Band-pass filter
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._paths = kwargs['paths']
        self._bands = kwargs['bands']
        self._n_bands = len(self._bands)
        self._wavelengths = [[] for i in range(self._n_bands)]
        self._transmissions = [[] for i in range(self._n_bands)]
        self._min_waves = [0.0] * self._n_bands
        self._max_waves = [0.0] * self._n_bands
        self._filter_integrals = [0.0] * self._n_bands
        for i, path in enumerate(self._paths):
            with open(os.path.join('friendlyfit', 'modules', path), 'r') as f:
                rows = []
                for row in csv.reader(
                        f, delimiter='\t', skipinitialspace=True):
                    rows.append([float(x) for x in row])
            self._wavelengths[i], self._transmissions[i] = list(
                map(list, zip(*rows)))
            self._min_waves[i] = min(self._wavelengths[i])
            self._max_waves[i] = max(self._wavelengths[i])
            self._filter_integrals[i] = np.trapz(
                np.array(self._transmissions[i]),
                np.array(self._wavelengths[i]))

    def process(self, **kwargs):
        self._lumdist = (kwargs['lumdist'] * u.Mpc).cgs.value
        eff_fluxes = [[] for x in range(self._n_bands)]
        mags = []
        for bi, band in enumerate(self._bands):
            seds = kwargs['seds'][bi]
            wavs = kwargs['wavelengths'][bi]
            itrans = np.interp(wavs, self._wavelengths[bi],
                               self._transmissions[bi])
            eff_fluxes = [0.0] * len(seds)
            for si, sed in enumerate(seds):
                eff_flux = np.trapz([x * y for x, y in zip(itrans, sed)], wavs)
                eff_fluxes[si] = eff_flux / self._filter_integrals[bi]
            mags.extend(self.abmag(eff_fluxes))
        return {'model_magnitudes': mags}

    def abmag(self, eff_fluxes):
        return [(np.inf if x == 0.0 else
                 (-48.6 -
                  (100.0**0.2) * np.log10(x / (4.0 * pi * self._lumdist**2))))
                for x in eff_fluxes]

    def request(self, request):
        if request == 'bands':
            return self._bands
        elif request == 'wavelengths':
            return list(map(list, zip(*[self._min_waves, self._max_waves])))
        return []
