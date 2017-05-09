"""Definitions for the `Photometry` class."""
import csv
import json
import os
import shutil
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.io.votable import parse as voparse

from mosfit.constants import C_CGS, FOUR_PI, MAG_FAC, MPC_CGS
from mosfit.modules.module import Module
from mosfit.utils import get_url_file_handle, listify, open_atomic, syst_syns


# Important: Only define one ``Module`` class per file.


class Photometry(Module):
    """Band-pass filters."""

    FLUX_STD = 3631 * u.Jy.cgs.scale / u.Angstrom.cgs.scale * C_CGS
    ANG_CGS = u.Angstrom.cgs.scale
    H_C_ANG_CGS = c.h.cgs.value * c.c.cgs.value / u.Angstrom.cgs.scale
    C_CGS = c.c.cgs.value
    H_CGS = c.h.cgs.value

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Photometry, self).__init__(**kwargs)

        bands = kwargs.get('bands', '')
        bands = listify(bands)

        self._dir_path = os.path.dirname(os.path.realpath(__file__))
        band_list = []

        if self._pool.is_master():
            with open(os.path.join(self._dir_path, 'filterrules.json')) as f:
                filterrules = json.load(f, object_pairs_hook=OrderedDict)
            for rank in range(1, self._pool.size + 1):
                self._pool.comm.send(filterrules, dest=rank, tag=5)
        else:
            filterrules = self._pool.comm.recv(source=0, tag=5)

        for bi, band in enumerate(bands):
            for rule in filterrules:
                if '@note' in rule:
                    continue
                sysinstperms = [
                    {
                        'systems': xx,
                        'instruments': yy,
                        'bandsets': zz,
                        'telescopes': tt,
                        'modes': mm
                    }
                    for xx in rule.get('systems', [''])
                    for yy in rule.get('instruments', [''])
                    for zz in rule.get('bandsets', [''])
                    for tt in rule.get('telescopes', [''])
                    for mm in rule.get('modes', [''])
                ]
                for bnd in rule.get('filters', []):
                    if band == bnd or band == '':
                        for perm in sysinstperms:
                            new_band = deepcopy(rule['filters'][bnd])
                            new_band.update(deepcopy(perm))
                            new_band['name'] = bnd
                            band_list.append(new_band)

        self._unique_bands = band_list
        self._band_insts = np.array(
            [x['instruments'] for x in self._unique_bands], dtype=object)
        self._band_bsets = np.array(
            [x['bandsets'] for x in self._unique_bands], dtype=object)
        self._band_systs = np.array(
            [x['systems'] for x in self._unique_bands], dtype=object)
        self._band_teles = np.array(
            [x['telescopes'] for x in self._unique_bands], dtype=object)
        self._band_modes = np.array(
            [x['modes'] for x in self._unique_bands], dtype=object)
        self._band_names = np.array(
            [x['name'] for x in self._unique_bands], dtype=object)
        self._n_bands = len(self._unique_bands)
        self._band_wavelengths = [[] for i in range(self._n_bands)]
        self._band_energies = [[] for i in range(self._n_bands)]
        self._transmissions = [[] for i in range(self._n_bands)]
        self._band_areas = [[] for i in range(self._n_bands)]
        self._min_waves = np.full(self._n_bands, 0.0)
        self._max_waves = np.full(self._n_bands, 0.0)
        self._imp_waves = [[0.0, 1.0] for i in range(self._n_bands)]
        self._filter_integrals = np.full(self._n_bands, 0.0)
        self._average_wavelengths = np.full(self._n_bands, 0.0)
        self._band_offsets = np.full(self._n_bands, 0.0)
        self._band_xunits = np.full(self._n_bands, 'Angstrom', dtype=object)
        self._band_yunits = np.full(self._n_bands, '', dtype=object)
        self._band_xu = np.full(self._n_bands, u.Angstrom.cgs.scale)
        self._band_yu = np.full(self._n_bands, 1.0)
        self._band_kinds = np.full(self._n_bands, 'magnitude', dtype=object)
        self._band_index_cache = {}

        for i, band in enumerate(self._unique_bands):
            self._band_xunits[i] = band.get('xunit', 'Angstrom')
            self._band_yunits[i] = band.get('yunit', '')
            self._band_xu[i] = u.Unit(self._band_xunits[i]).cgs.scale
            self._band_yu[i] = u.Unit(self._band_yunits[i]).cgs.scale

            if '{0}'.format(self._band_yunits[i]) == 'cm2':
                self._band_kinds[i] = 'countrate'

    def load_bands(self, band_indices):
        """Load band files."""
        prt = self._printer

        if self._pool.is_master():
            vo_tabs = OrderedDict()

        per = 0.0
        bc = 0
        band_set = set(band_indices)
        for i, band in enumerate(self._unique_bands):
            if len(band_indices) and i not in band_set:
                continue
            if self._pool.is_master():
                new_per = np.round(100.0 * float(bc) / len(band_set))
                if new_per > per:
                    per = new_per
                    prt.message('loading_bands', [per], inline=True)
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
                        path = os.path.join(self._dir_path, 'filters',
                                            svopath.replace('/', '_') + '.dat')

                        xml_path = os.path.join(
                            self._dir_path, 'filters',
                            svopath.replace('/', '_') + '.xml')
                        if not os.path.exists(xml_path):
                            prt.message('dl_svo', [svopath], inline=True)
                            try:
                                response = get_url_file_handle(
                                    'http://svo2.cab.inta-csic.es'
                                    '/svo/theory/fps3/'
                                    'fps.php?PhotCalID=' + svopath,
                                    timeout=10)
                            except Exception:
                                prt.message('cant_dl_svo', warning=True)
                            else:
                                with open_atomic(xml_path, 'wb') as f:
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
                                with open_atomic(path, 'w') as f:
                                    f.write(vo_string)
                        else:
                            raise RuntimeError(
                                prt.string('cant_read_svo'))
                    self._unique_bands[i]['origin'] = band['SVO']
                else:
                    self._unique_bands[i]['origin'] = band['path']
                    path = band['path']

                with open(os.path.join(
                        self._dir_path, 'filters', path), 'r') as f:
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

            xvals, yvals = list(
                map(list, zip(*rows)))
            xvals = np.array(xvals)
            yvals = np.array(yvals)

            if '{0}'.format(self._band_yunits[i]) == 'cm2':
                xscale = (c.h * c.c /
                          u.Angstrom).cgs.value / self._band_xu[i]
                self._band_energies[
                    i], self._band_areas[i] = xvals, yvals / xvals
                self._band_wavelengths[i] = xscale / self._band_energies[i]
                self._average_wavelengths[i] = np.trapz([
                    x * y
                    for x, y in zip(
                        self._band_areas[i], self._band_wavelengths[i])
                ], self._band_wavelengths[i]) / np.trapz(
                    self._band_areas[i], self._band_wavelengths[i])
            else:
                self._band_wavelengths[
                    i], self._transmissions[i] = xvals, yvals
                self._filter_integrals[i] = self.FLUX_STD * np.trapz(
                    np.array(self._transmissions[i]) /
                    np.array(self._band_wavelengths[i]) ** 2,
                    self._band_wavelengths[i])
                self._average_wavelengths[i] = np.trapz([
                    x * y
                    for x, y in zip(
                        self._transmissions[i], self._band_wavelengths[i])
                ], self._band_wavelengths[i]) / np.trapz(
                    self._transmissions[i], self._band_wavelengths[i])

                if 'offset' in band:
                    self._band_offsets[i] = band['offset']
                elif 'SVO' in band:
                    self._band_offsets[i] = zps[-1]

                # Do some sanity checks.
                if (self._band_offsets[i] != 0.0 and
                        self._band_systs[i] == 'AB'):
                    raise RuntimeError(
                        'Filters in AB system should always have offset = '
                        '0.0, not the case for `{}`'.format(
                            self._band_names[i]))

            self._min_waves[i] = min(self._band_wavelengths[i])
            self._max_waves[i] = max(self._band_wavelengths[i])
            self._imp_waves[i] = [self._min_waves[i], self._max_waves[i]]
            if len(self._transmissions[i]):
                new_wave = self._band_wavelengths[i][
                    np.argmax(self._transmissions[i])]
                self._imp_waves[i].append(new_wave)
            elif len(self._band_areas[i]):
                new_wave = self._band_wavelengths[i][
                    np.argmax(self._band_areas[i])]
                self._imp_waves[i].append(new_wave)
            bc = bc + 1

        if self._pool.is_master():
            prt.message('band_load_complete', inline=True)

    def find_band_index(
            self, band, telescope='', instrument='', mode='', bandset='',
            system=''):
        """Find the index corresponding to the provided band information."""
        bmatch = 0
        bbi = None
        cache_key = ':'.join([
            band, telescope, instrument, mode, bandset, system])
        if cache_key in self._band_index_cache:
            return self._band_index_cache[cache_key]
        for bi, bnd in enumerate(self._unique_bands):
            matches = [band == bnd['name'],
                       system == self._band_systs[bi],
                       mode == self._band_modes[bi],
                       instrument == self._band_insts[bi],
                       telescope == self._band_teles[bi],
                       bandset == self._band_bsets[bi]]
            lmatch = sum(matches)
            nbmatch = sum(
                [(band == bnd['name']) & (band != ''),
                 (system == self._band_systs[bi]) & (system != ''),
                 (mode == self._band_modes[bi]) & (mode != ''),
                 (instrument == self._band_insts[bi]) & (instrument != ''),
                 (telescope == self._band_teles[bi]) & (telescope != ''),
                 (bandset == self._band_bsets[bi]) & (bandset != '')])
            if lmatch > bmatch and nbmatch > 0:
                bmatch = lmatch
                bbi = bi
            if lmatch == 6 and nbmatch > 0:
                break
        if bbi is not None:
            self._band_index_cache[cache_key] = bbi
            return bbi
        raise ValueError(
            'Cannot find band index for `{}` band of bandset `{}` '
            'in mode `{}` with '
            'instrument `{}` on telescope `{}` in the `{}` system!'.format(
                band, bandset, mode, instrument, telescope, system))

    def preprocess(self, **kwargs):
        """Preprocess module."""
        if not self._preprocessed:
            self.load_bands(kwargs['all_band_indices'])
            self._preprocessed = True

    def process(self, **kwargs):
        """Process module."""
        self.preprocess(**kwargs)
        kwargs = self.prepare_input('luminosities', **kwargs)
        self._band_indices = kwargs['all_band_indices']
        self._observation_types = np.array(kwargs['observation_types'])
        self._dist_const = FOUR_PI * (kwargs['lumdist'] * MPC_CGS) ** 2
        self._ldist_const = np.log10(self._dist_const)
        self._luminosities = kwargs['luminosities']
        self._frequencies = kwargs['all_frequencies']
        zp1 = 1.0 + kwargs['redshift']
        eff_fluxes = np.zeros_like(self._luminosities)
        offsets = np.zeros_like(self._luminosities)
        observations = np.zeros_like(self._luminosities)
        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if bi >= 0:
                if self._observation_types[li] == 'magnitude':
                    offsets[li] = self._band_offsets[bi]
                    wavs = kwargs['sample_wavelengths'][bi]
                    yvals = np.interp(
                        wavs, self._band_wavelengths[bi],
                        self._transmissions[bi]) * kwargs['seds'][li] / zp1
                    eff_fluxes[li] = np.trapz(
                        yvals, wavs) / self._filter_integrals[bi]
                elif self._observation_types[li] == 'countrate':
                    wavs = np.array(kwargs['sample_wavelengths'][bi])
                    yvals = (np.interp(
                        wavs, self._band_wavelengths[bi], self._band_areas[
                            bi]) * kwargs['seds'][li] / zp1 / (
                                self.H_C_ANG_CGS / wavs) / self.ANG_CGS)
                    eff_fluxes[li] = np.trapz(yvals, wavs)
                else:
                    raise RuntimeError('Unknown observation kind.')
            else:
                eff_fluxes[li] = kwargs['seds'][li][0] / self.ANG_CGS * (
                    C_CGS / (self._frequencies[li] ** 2))
        nbs = self._observation_types != 'magnitude'
        ybs = self._observation_types == 'magnitude'
        observations[nbs] = eff_fluxes[nbs] / self._dist_const
        observations[ybs] = self.abmag(eff_fluxes[ybs], offsets[ybs])
        return {'model_observations': observations}

    def average_wavelengths(self, indices=None):
        """Return average wavelengths for specified band indices."""
        if indices:
            return [x for i, x in
                    enumerate(self._average_wavelengths) if i in indices]
        return self._average_wavelengths

    def bands(self, indices=None):
        """Return the list of unique band names."""
        if indices:
            return [x for i, x in
                    enumerate(self._band_names) if i in indices]
        return self._band_names

    def instruments(self, indices=None):
        """Return the list of instruments."""
        if indices:
            return [x for i, x in
                    enumerate(self._band_insts) if i in indices]
        return self._band_insts

    def abmag(self, eff_fluxes, offsets):
        """Convert fluxes into AB magnitude."""
        mags = np.full(len(eff_fluxes), np.inf)
        ef_mask = eff_fluxes != 0.0
        mags[ef_mask] = - offsets[ef_mask] - MAG_FAC * (
            np.log10(eff_fluxes[ef_mask]) - self._ldist_const)
        return mags

    def set_variance_bands(self, band_pairs):
        """Set band (or pair of bands) that variance will be anchored to."""
        self._variance_bands = []
        for i, wave in enumerate(self._average_wavelengths):
            match_found = False
            for pwave, band in band_pairs:
                if wave == pwave:
                    self._variance_bands.append(band)
                    match_found = True
                    break
            if not match_found:
                for bpi, (pwave, band) in enumerate(band_pairs):
                    if wave < pwave:
                        if bpi > 0:
                            frac = ((wave - band_pairs[bpi - 1][0]) /
                                    (pwave - band_pairs[bpi - 1][0]))
                            self._variance_bands.append(
                                [frac, [x[1] for x in
                                        band_pairs[bpi - 1:bpi + 1]]])
                        else:
                            self._variance_bands.append(band)
                        break
                    if bpi == len(band_pairs) - 1:
                        self._variance_bands.append(band)

    def send_request(self, request):
        """Send requests to other modules."""
        if request == 'photometry':
            return self
        elif request == 'band_wave_ranges':
            return self._imp_waves
        elif request == 'average_wavelengths':
            return self._average_wavelengths
        elif request == 'variance_bands':
            return getattr(self, '_variance_bands', [])
        return []
