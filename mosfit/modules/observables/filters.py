import csv
import json
import os
import shutil
import urllib
from collections import OrderedDict

import numpy as np
from astropy.io.votable import parse as voparse
from mosfit.constants import AB_OFFSET, FOUR_PI, MAG_FAC, MPC_CGS
from mosfit.modules.module import Module
from mosfit.utils import listify, print_inline

CLASS_NAME = 'Filters'


class Filters(Module):
    """Band-pass filters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._preprocessed = False
        bands = kwargs.get('bands', '')
        systems = kwargs.get('systems', '')
        instruments = kwargs.get('instruments', '')
        bands = listify(bands)
        systems = listify(systems)
        instruments = listify(instruments)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        band_list = []
        with open(os.path.join(dir_path, 'filterrules.json')) as f:
            filterrules = json.loads(f.read(), object_pairs_hook=OrderedDict)
            for bi, band in enumerate(bands):
                for rule in filterrules:
                    sysinstperms = [[x, y]
                                    for x in rule.get("systems", [''])
                                    for y in rule.get("instruments", [''])]
                    for bnd in rule.get('filters', []):
                        if band == bnd or band == '':
                            for perm in sysinstperms:
                                band_list.append(rule['filters'][bnd])
                                band_list[-1]['systems'] = perm[0]
                                band_list[-1]['instruments'] = perm[1]
                                band_list[-1]['name'] = bnd
                                if not band_list[-1].get('offset', ''):
                                    band_list[-1]['offset'] = 0.0

        self._unique_bands = band_list
        self._band_insts = [x['instruments'] for x in self._unique_bands]
        self._band_systs = [x['systems'] for x in self._unique_bands]
        self._band_names = [x['name'] for x in self._unique_bands]
        self._band_offsets = [x['offset'] for x in self._unique_bands]
        self._n_bands = len(self._unique_bands)
        self._band_wavelengths = [[] for i in range(self._n_bands)]
        self._transmissions = [[] for i in range(self._n_bands)]
        self._min_waves = [0.0] * self._n_bands
        self._max_waves = [0.0] * self._n_bands
        self._filter_integrals = [0.0] * self._n_bands

        for i, band in enumerate(self._unique_bands):
            if self._pool.is_master():
                if 'SVO' in band:
                    path = os.path.join(dir_path, '..', '..', 'cache',
                                        band['SVO'].replace('/', '_') + '.dat')

                    xml_path = os.path.join(
                        dir_path, '..', '..', 'cache',
                        band['SVO'].replace('/', '_') + '.xml')
                    if not os.path.exists(xml_path):
                        print('Downloading bandpass {} from SVO.'.format(band[
                            'SVO']))
                        try:
                            response = urllib.request.urlopen(
                                'http://svo2.cab.inta-csic.es/svo/theory/fps3/'
                                'fps.php?PhotCalID=' + band['SVO'],
                                timeout=10)
                        except:
                            print_inline(
                                'Warning: Could not download SVO filter (are '
                                'you online?), using cached filter.')
                        else:
                            with open(xml_path, 'wb') as f:
                                shutil.copyfileobj(response, f)

                    if os.path.exists(xml_path):
                        vo_tab = voparse(xml_path)
                        vo_dat = vo_tab.get_first_table().array
                        vo_string = '\n'.join(
                            [' '.join([str(y) for y in x]) for x in vo_dat])
                        with open(path, 'w') as f:
                            f.write(vo_string)
                    else:
                        print('Error: Could not read SVO filter!')
                        raise RuntimeError
                else:
                    path = band['path']

                with open(os.path.join(dir_path, 'filters', path), 'r') as f:
                    rows = []
                    for row in csv.reader(
                            f, delimiter=' ', skipinitialspace=True):
                        rows.append([float(x) for x in row[:2]])
                for rank in range(1, self._pool.size + 1):
                    self._pool.comm.send(rows, dest=rank, tag=3)
            else:
                rows = self._pool.comm.recv(source=0, tag=3)

            self._band_wavelengths[i], self._transmissions[i] = list(
                map(list, zip(*rows)))
            self._min_waves[i] = min(self._band_wavelengths[i])
            self._max_waves[i] = max(self._band_wavelengths[i])
            self._filter_integrals[i] = np.trapz(self._transmissions[i],
                                                 self._band_wavelengths[i])

    def find_band_index(self, name, instrument='', system=''):
        for bi, band in enumerate(self._unique_bands):
            if (name == band['name'] and instrument in self._band_insts[bi] and
                    system in self._band_systs[bi]):
                return bi
            if (name == band['name'] and '' in self._band_insts[bi] and
                    '' in self._band_systs[bi]):
                return bi
        raise ValueError('Cannot find band index!')

    def process(self, **kwargs):
        self.preprocess(**kwargs)
        self._dist_const = np.log10(FOUR_PI * (kwargs['lumdist'] * MPC_CGS)**2)
        self._luminosities = kwargs['luminosities']
        self._systems = kwargs['systems']
        self._instruments = kwargs['instruments']
        eff_fluxes = []
        offsets = []
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            offsets.append(self._band_offsets[bi])
            wavs = kwargs['samplewavelengths'][bi]
            itrans = np.interp(wavs, self._band_wavelengths[bi],
                               self._transmissions[bi])
            yvals = [x * y for x, y in zip(itrans, kwargs['seds'][li])]
            eff_fluxes.append(
                np.trapz(
                    yvals, dx=self._dxs[bi]) / self._filter_integrals[bi])
        mags = self.abmag(eff_fluxes, offsets)
        return {'model_magnitudes': mags}

    def band_names(self):
        return self._band_names

    def abmag(self, eff_fluxes, offsets):
        return [(np.inf if x == 0.0 else
                 (AB_OFFSET - y - MAG_FAC * (np.log10(x) - self._dist_const)))
                for x, y in zip(eff_fluxes, offsets)]

    def request(self, request):
        if request == 'filters':
            return self
        elif request == 'band_wave_ranges':
            return list(map(list, zip(*[self._min_waves, self._max_waves])))
        return []

    def preprocess(self, **kwargs):
        if self._preprocessed:
            return
        self._bands = kwargs['bands']
        self._band_indices = list(map(self.find_band_index, self._bands))
        self._dxs = []
        for bi in self._band_indices:
            wavs = kwargs['samplewavelengths'][bi]
            self._dxs.append(wavs[1] - wavs[0])
        self._preprocessed = True
