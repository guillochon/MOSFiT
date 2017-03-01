import csv
import json
import os
import shutil
from collections import OrderedDict

import numpy as np
from astropy.io.votable import parse as voparse

from mosfit.constants import AB_OFFSET, FOUR_PI, MAG_FAC, MPC_CGS, C_CGS
from mosfit.modules.module import Module
from mosfit.utils import get_url_file_handle, listify, print_inline, syst_syns

CLASS_NAME = 'Photometry'


class Photometry(Module):
    """Band-pass filters.
    """

    def __init__(self, **kwargs):
        super(Photometry, self).__init__(**kwargs)
        self._preprocessed = False
        self._bands = []

        bands = kwargs.get('bands', '')
        bands = listify(bands)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        band_list = []

        if self._pool.is_master():
            with open(os.path.join(dir_path, 'filterrules.json')) as f:
                filterrules = json.load(f, object_pairs_hook=OrderedDict)
            for rank in range(1, self._pool.size + 1):
                self._pool.comm.send(filterrules, dest=rank, tag=5)
        else:
            filterrules = self._pool.comm.recv(source=0, tag=5)

        for bi, band in enumerate(bands):
            for rule in filterrules:
                sysinstperms = [
                    {
                        'systems': xx,
                        'instruments': yy,
                        'bandsets': zz
                    }
                    for xx in rule.get('systems', [''])
                    for yy in rule.get('instruments', [''])
                    for zz in rule.get('bandsets', [''])
                ]
                for bnd in rule.get('filters', []):
                    if band == bnd or band == '':
                        for perm in sysinstperms:
                            new_band = rule['filters'][bnd].copy()
                            new_band.update(perm.copy())
                            new_band['name'] = bnd
                            band_list.append(new_band)

        self._unique_bands = band_list
        self._band_insts = [x['instruments'] for x in self._unique_bands]
        self._band_bsets = [x['bandsets'] for x in self._unique_bands]
        self._band_systs = [x['systems'] for x in self._unique_bands]
        self._band_names = [x['name'] for x in self._unique_bands]
        self._n_bands = len(self._unique_bands)
        self._band_wavelengths = [[] for i in range(self._n_bands)]
        self._transmissions = [[] for i in range(self._n_bands)]
        self._min_waves = [0.0] * self._n_bands
        self._max_waves = [0.0] * self._n_bands
        self._filter_integrals = [0.0] * self._n_bands
        self._average_wavelengths = [0.0] * self._n_bands
        self._band_offsets = [0.0] * self._n_bands
        FLUX_STD = 3.361e-12 * C_CGS

        if self._pool.is_master():
            vo_tabs = {}

        for i, band in enumerate(self._unique_bands):
            if self._pool.is_master():
                systems = ['AB']
                zps = [0.0]
                if 'SVO' in band:
                    photsystem = self._band_systs[i]
                    if photsystem in syst_syns:
                        photsystem = syst_syns[photsystem]
                    if photsystem not in systems:
                        systems.append(photsystem)
                    zpfluxes = []
                    for sys in systems:
                        svopath = band['SVO'] + '/' + sys
                        path = os.path.join(dir_path, 'filters',
                                            svopath.replace('/', '_') + '.dat')

                        xml_path = os.path.join(
                            dir_path, 'filters',
                            svopath.replace('/', '_') + '.xml')
                        if not os.path.exists(xml_path):
                            print('Downloading bandpass {} from SVO.'.format(
                                svopath))
                            try:
                                response = get_url_file_handle(
                                    'http://svo2.cab.inta-csic.es'
                                    '/svo/theory/fps3/'
                                    'fps.php?PhotCalID=' + svopath,
                                    timeout=10)
                            except:
                                print_inline(
                                    'Warning: Could not download SVO filter '
                                    '(are you online?), using cached filter.')
                            else:
                                with open(xml_path, 'wb') as f:
                                    shutil.copyfileobj(response, f)

                        if os.path.exists(xml_path):
                            already_written = svopath in vo_tabs
                            if not already_written:
                                vo_tabs[svopath] = voparse(xml_path)
                            vo_tab = vo_tabs[svopath]
                            # need to account for zeropoint type

                            for resource in vo_tab.resources:
                                if len(resource.params) == 0:
                                    params = vo_tab.get_first_table().params
                                else:
                                    params = resource.params

                            oldzplen = len(zps)
                            for param in params:
                                if param.name == 'ZeroPoint':
                                    zpfluxes.append(param.value)
                                    if sys != 'AB':
                                        # 0th element is AB flux
                                        zps.append(2.5 * np.log10(
                                            zpfluxes[0] / zpfluxes[-1]))
                                else:
                                    continue
                            if sys != 'AB' and len(zps) == oldzplen:
                                raise RuntimeError(
                                    'ZeroPoint not found in XML.')

                            if not already_written:
                                vo_dat = vo_tab.get_first_table().array
                                bi = max(
                                    next((i for i, x in enumerate(vo_dat)
                                          if x[1]), 0) - 1, 0)
                                ei = -max(
                                    next((i
                                          for i, x in enumerate(
                                              reversed(vo_dat))
                                          if x[1]), 0) - 1, 0)
                                vo_dat = vo_dat[bi:ei if ei else len(vo_dat)]
                                vo_string = '\n'.join([
                                    ' '.join([str(y) for y in x])
                                    for x in vo_dat
                                ])
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
                    self._pool.comm.send(zps, dest=rank, tag=4)
            else:
                rows = self._pool.comm.recv(source=0, tag=3)
                zps = self._pool.comm.recv(source=0, tag=4)

            self._band_wavelengths[i], self._transmissions[i] = list(
                map(list, zip(*rows)))
            self._min_waves[i] = min(self._band_wavelengths[i])
            self._max_waves[i] = max(self._band_wavelengths[i])
            self._filter_integrals[i] = FLUX_STD * np.trapz(
                                    np.array(self._transmissions[i]) /
                                    np.array(self._band_wavelengths[i])**2,
                                                 self._band_wavelengths[i])
            self._average_wavelengths[i] = np.trapz([
                x * y
                for x, y in zip(self._transmissions[i], self._band_wavelengths[
                    i])
            ], self._band_wavelengths[i]) / self._filter_integrals[i]

            if 'offset' in band:
                self._band_offsets[i] = band['offset']
            elif 'SVO' in band:
                self._band_offsets[i] = zps[-1]

    def find_band_index(self, band, instrument='', bandset='', system=''):
        for i in range(4):
            for bi, bnd in enumerate(self._unique_bands):
                if (i == 0 and band == bnd['name'] and
                        instrument == self._band_insts[bi] and
                        bandset == self._band_bsets[bi] and
                        system == self._band_systs[bi]):
                    return bi
                elif (i == 1 and band == bnd['name'] and
                      instrument == self._band_insts[bi] and
                      system == self._band_systs[bi]):
                    return bi
                elif (i == 2 and band == bnd['name'] and
                      system == self._band_systs[bi]):
                    return bi
                elif (i == 3 and band == bnd['name'] and
                      '' == self._band_insts[bi] and
                      '' == self._band_bsets[bi] and
                      '' == self._band_systs[bi]):
                    return bi
        raise ValueError(
            'Cannot find band index for `{}` band of bandset `{}` with '
            'instrument `{}` in the `{}` system!'.format(band, bandset,
                                                         instrument, system))

    def process(self, **kwargs):
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._dist_const = FOUR_PI * (kwargs['lumdist'] * MPC_CGS)**2
        self._ldist_const = np.log10(self._dist_const)
        self._luminosities = kwargs['luminosities']
        self._systems = kwargs['systems']
        self._instruments = kwargs['instruments']
        self._bandsets = kwargs['bandsets']
        zp1 = 1.0 + kwargs['redshift']
        eff_fluxes = np.zeros_like(self._luminosities)
        offsets = np.zeros_like(self._luminosities)
        observations = np.zeros_like(self._luminosities)
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if bi >= 0:
                offsets[li] = self._band_offsets[bi]
                wavs = kwargs['sample_wavelengths'][bi]
                dx = wavs[1] - wavs[0]
                yvals = np.interp(
                    wavs, self._band_wavelengths[bi],
                    self._transmissions[bi]) * kwargs['seds'][li] * (
                    1e8 * C_CGS / wavs**2 ) / zp1
                eff_fluxes[li] = np.trapz(
                    yvals, dx=dx) / self._filter_integrals[bi]
            else:
                eff_fluxes[li] = kwargs['seds'][li][0]
        nbs = np.array(self._band_indices) < 0
        ybs = np.array(self._band_indices) >= 0
        observations[nbs] = eff_fluxes[nbs] / self._dist_const
        observations[ybs] = self.abmag(eff_fluxes[ybs], offsets[ybs])
        return {'model_observations': observations}

    def band_names(self):
        return self._band_names

    def abmag(self, eff_fluxes, offsets):
        mags = np.full(len(eff_fluxes), np.inf)
        mags[eff_fluxes !=
             0.0] = offsets[eff_fluxes != 0.0] - MAG_FAC * (
                 np.log10(eff_fluxes[eff_fluxes != 0.0]) - self._ldist_const)
        return mags

    def send_request(self, request):
        if request == 'photometry':
            return self
        elif request == 'band_wave_ranges':
            return list(map(list, zip(*[self._min_waves, self._max_waves])))
        return []
