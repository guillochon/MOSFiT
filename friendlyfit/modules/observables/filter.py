import csv
import json
import os

import numexpr as ne
import numpy as np

from ...constants import AB_OFFSET, FOUR_PI, MAG_FAC, MPC_CGS
from ...utils import listify
from ..module import Module

CLASS_NAME = 'Filter'


class Filter(Module):
    """Band-pass filter.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bands = kwargs.get('bands', '')
        systems = kwargs.get('systems', '')
        instruments = kwargs.get('instruments', '')
        bands = listify(bands)
        systems = listify(systems)
        instruments = listify(instruments)

        band_list = []
        with open(
                os.path.join('friendlyfit', 'modules', 'observables',
                             'filterrules.json')) as f:
            filterrules = json.loads(f.read())
            for bi, band in enumerate(bands):
                for rule in filterrules:
                    if systems[bi] not in rule.get("systems", []):
                        continue
                    if instruments[bi] not in rule.get("instruments", []):
                        continue
                    for bnd in rule.get('filters', []):
                        if band == bnd or band == '':
                            band_list.append(rule['filters'][bnd])
                            band_list[-1]['systems'] = rule.get('systems', [])
                            band_list[-1]['instruments'] = rule.get(
                                'instruments', [])
                            band_list[-1]['name'] = bnd
                            if not band_list[-1].get('offset', ''):
                                band_list[-1]['offset'] = 0.0

        self._unique_bands = band_list
        self._band_names = [x['name'] for x in self._unique_bands]
        self._band_offsets = [x['offset'] for x in self._unique_bands]
        self._n_bands = len(self._unique_bands)
        self._band_wavelengths = [[] for i in range(self._n_bands)]
        self._transmissions = [[] for i in range(self._n_bands)]
        self._min_waves = [0.0] * self._n_bands
        self._max_waves = [0.0] * self._n_bands
        self._filter_integrals = [0.0] * self._n_bands

        for i, band in enumerate(self._unique_bands):
            with open(
                    os.path.join('friendlyfit', 'modules', 'observables',
                                 'filters', band['path']), 'r') as f:
                rows = []
                for row in csv.reader(f, delimiter=' ', skipinitialspace=True):
                    rows.append([float(x) for x in row[:2]])
            self._band_wavelengths[i], self._transmissions[i] = list(
                map(list, zip(*rows)))
            self._min_waves[i] = min(self._band_wavelengths[i])
            self._max_waves[i] = max(self._band_wavelengths[i])
            self._filter_integrals[i] = np.trapz(
                np.array(self._transmissions[i]),
                np.array(self._band_wavelengths[i]))

    def find_band_index(self, name, instrument='', system=''):
        for bi, band in enumerate(self._unique_bands):
            if (instrument in band.get('instruments', '') and
                    system in band.get('systems', '') and
                    name == band['name']):
                return bi
            if ('' in band.get('instruments', '') and
                    '' in band.get('systems', '') and
                    name == band['name']):
                return bi
        raise(ValueError('Cannot find band index!'))

    def process(self, **kwargs):
        self._dist_const = np.log10(FOUR_PI * (kwargs['lumdist'] * MPC_CGS)**2)
        self._luminosities = kwargs['luminosities']
        self._bands = kwargs['bands']
        self._systems = kwargs['systems']
        self._instruments = kwargs['instruments']
        eff_fluxes = []
        offsets = []
        for li, band in enumerate(self._luminosities):
            cur_band = self._bands[li]
            bi = self.find_band_index(cur_band, self._systems[li],
                                      self._instruments[li])
            sed = kwargs['seds'][li]
            wavs = kwargs['bandwavelengths'][bi]
            offsets.append(self._band_offsets[bi])
            dx = wavs[1] - wavs[0]
            itrans = np.interp(wavs, self._band_wavelengths[bi],
                               self._transmissions[bi])
            # if li == 0:
            #     ef = ne.evaluate('sum(itrans * sed)')
            # else:
            #     ef = ne.re_evaluate()
            # eff_fluxes.append(dx * ef)
            yvals = [x * y for x, y in zip(itrans, sed)]
            eff_fluxes.append(
                np.trapz(
                    yvals, dx=dx) / self._filter_integrals[bi])
        mags = self.abmag(eff_fluxes, offsets)
        return {'model_magnitudes': mags}

    def band_names(self):
        return self._band_names

    def abmag(self, eff_fluxes, offsets):
        return [(np.inf if x == 0.0 else
                 (y + AB_OFFSET - MAG_FAC * (np.log10(x) - self._dist_const)))
                for x, y in zip(eff_fluxes, offsets)]

    def request(self, request):
        if request == 'bandnames':
            return self._band_names
        elif request == 'bandwavelengths':
            return list(map(list, zip(*[self._min_waves, self._max_waves])))
        return []
