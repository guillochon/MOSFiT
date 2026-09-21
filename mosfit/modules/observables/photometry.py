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
from mosfit.constants import (
    ANG_CGS,
    BOL_BAND_INDEX,
    C_CGS,
    FOUR_PI,
    H_C_ANG_CGS,
    LBOL_SUN_CGS,
    MAG_FAC,
    MBOL_SUN_MAG,
    MPC_CGS,
)
from mosfit.modules.module import Module
from mosfit.utils import get_url_file_handle, listify, open_atomic, syst_syns


# Important: Only define one ``Module`` class per file.


def _sed_block(seds_in, idx):
    """Rows ``idx`` of a rectangular or object-dtype SED array."""
    arr = np.asarray(seds_in)
    idx = np.asarray(idx)
    if arr.dtype != object and arr.ndim == 2:
        return arr[idx]
    return np.stack([np.asarray(seds_in[int(i)], dtype=float) for i in idx])


def _sed_col0(seds_in, idx):
    """First wavelength sample of rows ``idx`` (radio / single-λ SEDs)."""
    arr = np.asarray(seds_in)
    idx = np.asarray(idx)
    if arr.dtype != object and arr.ndim == 2:
        return np.asarray(arr[idx, 0], dtype=float)
    return np.array([float(np.asarray(seds_in[int(i)]).ravel()[0])
                     for i in idx], dtype=float)


def luminosity_cgs_to_mbol(lum_cgs):
    """Bolometric luminosity [erg/s] → absolute bolometric magnitude.

    Calibration uses ``MBOL_SUN_MAG`` and ``LBOL_SUN_CGS`` consistent with the
    model engine luminosity units.
    """
    lum_cgs = np.asarray(lum_cgs, dtype=np.float64)
    out = np.full(np.shape(lum_cgs), np.nan, dtype=np.float64)
    pos = np.isfinite(lum_cgs) & (lum_cgs > 0.0)
    out[pos] = (
        MBOL_SUN_MAG -
        MAG_FAC * np.log10(lum_cgs[pos] / LBOL_SUN_CGS))
    return out


def _photometry_debug_enabled() -> bool:
    v = os.environ.get("MOSFIT_PHOTOMETRY_DEBUG", "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


_PHOTOMETRY_DEBUG_PATCH = (
    "sesn_valid_mask_scatter_magnitudes_onto_dense_observed_rows"
)


def _dense_measurements_aligned(kwargs, shape_model, measurement_key='magnitudes'):
    """Column same length as model rows: NaN off ``observed``; data on real rows.

    ``AllTimes`` does not emit ``all_magnitudes``; ``kwargs[measurement_key]``
    is usually the short transient list (N_real) while ``observed`` (N_dense)
    marks which dense rows have data. Then ``count_nonzero(observed)==N_real``.
    """
    col = np.full(int(shape_model[0]), np.nan, dtype=np.float64)
    mag = kwargs.get(measurement_key)
    obs = kwargs.get('observed')
    if mag is None or obs is None:
        return col
    mag = np.asarray(mag, dtype=np.float64).ravel()
    obs = np.asarray(obs, dtype=bool).ravel()
    if obs.shape[0] != col.shape[0]:
        return col
    ind = obs
    no = int(np.count_nonzero(ind))
    if no == mag.size:
        col[ind] = mag
    return col


class Photometry(Module):
    """Band-pass filters."""

    FLUX_STD = 3631 * u.Jy.cgs.scale / u.Angstrom.cgs.scale * C_CGS

    _debug_banner_printed = False
    _debug_invalid_mask_logged = False

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Photometry, self).__init__(**kwargs)

        bands = kwargs.get('bands', '')
        bands = listify(bands)

        self._dir_path = os.path.dirname(os.path.realpath(__file__))
        self._filter_run_path = os.path.join('modules', 'observables')
        # SVO downloads are cached under the process CWD. Do not mkdir in the
        # installed package tree; site-packages is often read-only.
        os.makedirs(
            os.path.join(self._filter_run_path, 'filters'), exist_ok=True)

        band_list = []

        has_comm = getattr(self._pool, 'comm', None) is not None
        if self._pool.is_master():
            rules_path = os.path.join(
                'modules', 'observables', 'filterrules.json')
            if not os.path.isfile(rules_path):
                rules_path = os.path.join(self._dir_path, 'filterrules.json')
            with open(rules_path) as f:
                filterrules = json.load(f, object_pairs_hook=OrderedDict)
            if has_comm:
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
        self._count_integrals = np.full(self._n_bands, 0.0)
        self._average_wavelengths = np.full(self._n_bands, 0.0)
        self._band_offsets = np.full(self._n_bands, 0.0)
        self._band_xunits = np.full(self._n_bands, 'Angstrom', dtype=object)
        self._band_yunits = np.full(self._n_bands, '', dtype=object)
        self._band_xu = np.full(self._n_bands, u.Angstrom.cgs.scale)
        self._band_yu = np.full(self._n_bands, 1.0)
        self._band_kinds = np.full(self._n_bands, 'magnitude', dtype=object)
        self._band_index_cache = {}
        self._warned_mismatch = False
        self._zps = np.full(self._n_bands, 0.0)
        self._trans_on_sample = None
        self._area_on_sample = None
        self._filter_sample_id = None

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
                path = None
                if 'SVO' in band:
                    photsystem = self._band_systs[i]
                    if photsystem in syst_syns:
                        photsystem = syst_syns[photsystem]
                    if photsystem not in systems:
                        systems.append(photsystem)
                    zpfluxes = []
                    for sys in systems:
                        svopath = band['SVO'] + '/' + sys
                        path = os.path.join(self._filter_run_path, 'filters',
                                            svopath.replace('/', '_') + '.dat')

                        xml_path = os.path.join(
                            self._filter_run_path, 'filters',
                            svopath.replace('/', '_') + '.xml')

                        xml_install_path = os.path.join(
                            self._dir_path, 'filters',
                            svopath.replace('/', '_') + '.xml')

                        if not os.path.exists(xml_path):
                            if not os.path.exists(xml_install_path):
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

                        if os.path.exists(xml_install_path):
                            already_written = svopath in vo_tabs
                            if not already_written:
                                vo_tabs[svopath] = voparse(xml_install_path)
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
                            if not os.path.exists(path):
                                with open_atomic(path, 'w') as f:
                                    f.write(vo_string)

                        elif os.path.exists(xml_path):
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
                                if not os.path.exists(path):
                                    with open_atomic(path, 'w') as f:
                                        f.write(vo_string)

                        else:
                            raise RuntimeError(
                                prt.string('cant_read_svo'))
                    self._unique_bands[i]['origin'] = band['SVO']
                elif all(x in band for x in [
                        'min_wavelength', 'max_wavelength',
                        'delta_wavelength']):
                    nbins = int(np.round((
                        band['max_wavelength'] -
                        band['min_wavelength']) / band[
                            'delta_wavelength'])) + 1
                    rows = np.array(
                        [np.linspace(
                            band['min_wavelength'], band['max_wavelength'],
                            nbins), np.full(nbins, 1.0)]).T.tolist()
                    self._unique_bands[i]['origin'] = 'generated'
                elif 'path' in band:
                    self._unique_bands[i]['origin'] = band['path']
                    path = band['path']
                else:
                    raise RuntimeError(prt.text('bad_filter_rule'))

                if path:
                    with open(path, 'r') as f:
                        rows = []
                        for row in csv.reader(
                                f, delimiter=' ', skipinitialspace=True):
                            rows.append([float(x) for x in row[:2]])
                has_comm = getattr(self._pool, 'comm', None) is not None
                if has_comm:
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
                self._average_wavelengths[i] = np.trapezoid([
                    x * y
                    for x, y in zip(
                        self._band_areas[i], self._band_wavelengths[i])
                ], self._band_wavelengths[i]) / np.trapezoid(
                    self._band_areas[i], self._band_wavelengths[i])
            else:
                self._band_wavelengths[
                    i], self._transmissions[i] = xvals, yvals
                # scale by zero-point flux (Flbda = Fnu*c/lbda^2)
                self._filter_integrals[i] = self.FLUX_STD * np.trapezoid(
                    np.array(self._transmissions[i]) /
                    np.array(self._band_wavelengths[i]) ** 2,
                    self._band_wavelengths[i])
                self._count_integrals[i] = self.FLUX_STD * np.trapezoid(
                    np.array(self._transmissions[i]) /
                    np.array(self._band_wavelengths[i]) ** 2 / (
                        H_C_ANG_CGS / self._band_wavelengths[i]),
                    self._band_wavelengths[i])
                self._average_wavelengths[i] = np.trapezoid([
                    x * y
                    for x, y in zip(
                        self._transmissions[i], self._band_wavelengths[i])
                ], self._band_wavelengths[i]) / np.trapezoid(
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
            self._imp_waves[i] = set([self._min_waves[i], self._max_waves[i]])
            if len(self._transmissions[i]):
                new_wave = self._band_wavelengths[i][
                    np.argmax(self._transmissions[i])]
                self._imp_waves[i].add(new_wave)
            elif len(self._band_areas[i]):
                new_wave = self._band_wavelengths[i][
                    np.argmax(self._band_areas[i])]
                self._imp_waves[i].add(new_wave)
            self._imp_waves[i] = list(sorted(self._imp_waves[i]))
            bc = bc + 1

        if self._pool.is_master():
            prt.message('band_load_complete', inline=True)
        self._filter_sample_id = None
        self._trans_on_sample = None
        self._area_on_sample = None

    def _ensure_filter_sample_cache(self, sample_wavelengths):
        """Interpolate filter curves onto ``sample_wavelengths`` once.

        Sample grids and native filter curves are fixed after setup; only the
        SED values change between likelihood / generative draws.
        """
        sid = id(sample_wavelengths)
        n = len(sample_wavelengths)
        if (getattr(self, '_filter_sample_id', None) == sid and
                getattr(self, '_trans_on_sample', None) is not None and
                len(self._trans_on_sample) == n):
            return
        trans_on = [None] * n
        area_on = [None] * n
        for bi in range(n):
            wavs = np.asarray(sample_wavelengths[bi], dtype=float)
            bw = np.asarray(self._band_wavelengths[bi], dtype=float)
            if bw.size == 0:
                continue
            trans = self._transmissions[bi]
            if len(trans):
                trans_on[bi] = np.interp(
                    wavs, bw, np.asarray(trans, dtype=float))
            areas = self._band_areas[bi]
            if len(areas):
                area_on[bi] = np.interp(
                    wavs, bw, np.asarray(areas, dtype=float))
        self._trans_on_sample = trans_on
        self._area_on_sample = area_on
        self._filter_sample_id = sid

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
        ltele, linst, lmode, lbset, lsyst = tuple([x.lower() for x in [
            telescope, instrument, mode, bandset, system]])

        for bi, bnd in enumerate(self._unique_bands):
            # Band name *must* match (case-sensitive), all other matches
            # optional and case-insensitive.
            if (band != bnd['name']) and (band != ''):
                continue
            nmismatches = sum(
                [(linst != self._band_insts[bi].lower()) & (
                    linst != '') & (self._band_insts[bi] != ''),
                 (ltele != self._band_teles[bi].lower()) & (
                    ltele != '') & (self._band_teles[bi] != '')])
            matches = [band == bnd['name'],
                       lsyst == self._band_systs[bi].lower(),
                       lmode == self._band_modes[bi].lower(),
                       linst == self._band_insts[bi].lower(),
                       ltele == self._band_teles[bi].lower(),
                       lbset == self._band_bsets[bi].lower()]
            lmatch = sum(matches)
            nbmatch = sum(
                [(band == bnd['name']) & (band != ''),
                 (lsyst == self._band_systs[bi].lower()) & (lsyst != ''),
                 (lmode == self._band_modes[bi].lower()) & (lmode != ''),
                 (linst == self._band_insts[bi].lower()) & (linst != ''),
                 (ltele == self._band_teles[bi].lower()) & (ltele != ''),
                 (lbset == self._band_bsets[bi].lower()) & (lbset != '')])
            if lmatch > bmatch and nbmatch > 0:
                bmatch = lmatch
                bbi = bi
                bmm = nmismatches
            if lmatch == 6 and nbmatch > 0:
                break
        if bbi is not None:
            if bmm > 0 and not self._warned_mismatch:
                self._printer.message('potential_mismatch', reps=[
                    band, instrument, telescope, self._band_insts[bbi],
                    self._band_teles[bbi]], warning=True)
                self._warned_mismatch = True
            self._band_index_cache[cache_key] = bbi
            return bbi
        raise ValueError(
            self._printer.text('band_not_found', reps=[
                band, bandset, mode, instrument, telescope, system]))

    def preprocess(self, **kwargs):
        """Preprocess module."""
        if not self._preprocessed:
            self.load_bands(kwargs['all_band_indices'])
            self._preprocessed = True

    def process(self, **kwargs):
        """Process module."""
        if _photometry_debug_enabled() and (not Photometry._debug_banner_printed):
            Photometry._debug_banner_printed = True
            print(
                "[MOSFIT_PHOTOMETRY_DEBUG] OK — patch {!r}".format(
                    _PHOTOMETRY_DEBUG_PATCH),
                flush=True)
            print(
                "    photometry.py: {}".format(
                    os.path.abspath(__file__)),
                flush=True)
        self.preprocess(**kwargs)
        kwargs = self.prepare_input(self.key('luminosities'), **kwargs)
        self._band_indices = kwargs['all_band_indices']
        self._observation_types = np.array(kwargs['observation_types'])
        self._observed = kwargs['observed']
        self._dist_const = FOUR_PI * (kwargs['lumdist'] * MPC_CGS) ** 2
        self._ldist_const = np.log10(self._dist_const)
        self._luminosities = kwargs[self.key('luminosities')]
        self._frequencies = kwargs['all_frequencies']
        self._zps = kwargs.get('all_zeropoints', np.zeros_like(
            self._luminosities))
        zp1 = 1.0 + kwargs['redshift']
        eff_fluxes = np.zeros_like(self._luminosities)
        offsets = np.zeros_like(self._luminosities)
        model_observations = np.zeros_like(self._luminosities)
        band_indices = np.asarray(self._band_indices)
        obs_types = self._observation_types
        seds_in = kwargs['seds']
        sample_wavelengths = kwargs['sample_wavelengths']
        frequencies = np.asarray(self._frequencies)
        self._ensure_filter_sample_cache(sample_wavelengths)

        freq_rows = (band_indices < 0) & (band_indices != BOL_BAND_INDEX)
        if np.any(freq_rows):
            idx = np.flatnonzero(freq_rows)
            sed0 = _sed_col0(seds_in, idx)
            eff_fluxes[idx] = sed0 / ANG_CGS * (
                C_CGS / (frequencies[idx] ** 2))

        unique_bis = np.unique(band_indices[band_indices >= 0])
        for bi in unique_bis:
            bi = int(bi)
            wavs = np.asarray(sample_wavelengths[bi])
            rows = band_indices == bi
            mag_rows = rows & (
                (obs_types == 'magnitude') | (obs_types == 'magcount'))
            cr_rows = rows & (obs_types == 'countrate')
            other = rows & ~mag_rows & ~cr_rows & (
                band_indices != BOL_BAND_INDEX)
            if np.any(other):
                raise RuntimeError('Unknown observation kind.')

            if np.any(mag_rows):
                idx = np.flatnonzero(mag_rows)
                offsets[idx] = self._band_offsets[bi]
                trans = self._trans_on_sample[bi]
                sed_block = _sed_block(seds_in, idx)
                yvals = trans * sed_block / zp1
                eff_fluxes[idx] = np.trapezoid(
                    yvals, wavs, axis=-1) / self._filter_integrals[bi]

            if np.any(cr_rows):
                idx = np.flatnonzero(cr_rows)
                areas = self._area_on_sample[bi]
                sed_block = _sed_block(seds_in, idx)
                yvals = areas * sed_block / zp1 / (
                    H_C_ANG_CGS / wavs) / ANG_CGS
                eff_fluxes[idx] = np.trapezoid(yvals, wavs, axis=-1)
        bi_arr = np.asarray(self._band_indices)
        phot_band = bi_arr >= 0
        nbs = np.logical_and(
            phot_band,
            self._observation_types != 'luminosity',
            np.logical_or(
                self._observation_types == 'countrate',
                self._observation_types == 'fluxdensity'))
        ybs = np.logical_and(
            phot_band,
            self._observation_types != 'luminosity',
            np.logical_or(
                self._observation_types == 'magnitude',
                self._observation_types == 'magcount'))
        cbs = self._observation_types == 'magcount'

        lum_mask = self._observation_types == 'luminosity'

        model_observations[nbs] = eff_fluxes[nbs] / self._dist_const
        model_observations[ybs] = self.abmag(eff_fluxes[ybs], offsets[ybs])
        model_observations[cbs] = 10.0 ** (-0.4 * (model_observations[
            cbs] - self._zps[cbs]))
        lumo_dense = np.asarray(self._luminosities, dtype=np.float64)
        model_observations[lum_mask] = lumo_dense[lum_mask]
        mbol_fit_mask = np.logical_and(
            bi_arr == BOL_BAND_INDEX,
            self._observation_types == 'magnitude')
        if np.any(mbol_fit_mask):
            model_observations[mbol_fit_mask] = luminosity_cgs_to_mbol(
                lumo_dense[mbol_fit_mask])

        # Get the per-point mask from SESNSedona, if present
        valid_mask = kwargs.get("sesn_valid_mask") 
        '''
        print(f"DEBUG photometry: sesn_valid_mask present = {valid_mask is not None}")
        if valid_mask is not None:
            print(f"DEBUG photometry: valid fraction = {np.mean(valid_mask):.2%}")'''
        if valid_mask is None:
            valid_mask = np.ones(len(model_observations), dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)
        
        if not np.all(valid_mask):
            filled = 0
            obs = kwargs.get('observed')
            n_obs = (int(np.sum(np.asarray(obs, dtype=bool)))
                     if obs is not None else -1)
            m_short = kwargs.get('magnitudes')
            short_n = (int(np.asarray(m_short).size) if m_short is not None
                       else -1)
            ali = kwargs.get('all_magnitudes')
            if ali is not None and (
                    np.asarray(ali).shape == model_observations.shape):
                m_arr = np.asarray(ali, dtype=np.float64)
            else:
                m_arr = _dense_measurements_aligned(
                    kwargs, model_observations.shape, 'magnitudes')
            shape_ok = m_arr.shape == model_observations.shape and np.any(
                np.isfinite(m_arr))
            if shape_ok:
                sel = (~valid_mask) & np.isfinite(m_arr)
                model_observations[sel] = m_arr[sel]
                filled = int(np.sum(sel))
            if _photometry_debug_enabled() and (
                    not Photometry._debug_invalid_mask_logged):
                Photometry._debug_invalid_mask_logged = True
                n_inv = int(np.sum(~valid_mask))
                n_inv_obs = -1
                if obs is not None:
                    obs_b = np.asarray(obs, dtype=bool).ravel()
                    if obs_b.shape == valid_mask.shape:
                        n_inv_obs = int(np.sum((~valid_mask) & obs_b))
                print(
                    "[MOSFIT_PHOTOMETRY_DEBUG] sesn_valid_mask substitution "
                    "(one-time): n_invalid_dense={} n_invalid_at_observed_rows={} "
                    "n_substituted_using_catalog_mag={} scatter_ok={}  "
                    "n_dense={} sum(observed)={} len(magnitudes_short)={}".format(
                        n_inv,
                        n_inv_obs,
                        filled,
                        bool(shape_ok and filled > 0),
                        int(model_observations.size), n_obs, short_n),
                    flush=True,
                )
                if not shape_ok or filled == 0:
                    print(
                        "    model_obs {}, all_magnitudes {}".format(
                            np.shape(model_observations),
                            None if ali is None else np.shape(ali)),
                        flush=True,
                    )

        out = {
            'model_observations': model_observations,
            # forward mask so outputs can see it
            'valid_mask': valid_mask,
        }
        preset_mag = kwargs.get('emulator_preset_systematic_mag')
        if preset_mag is not None:
            out['emulator_preset_systematic_mag'] = np.asarray(
                preset_mag, dtype=np.float64)
        return out

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

    def telescopes(self, indices=None):
        """Return the list of telescopes."""
        if indices:
            return [x for i, x in
                    enumerate(self._band_teles) if i in indices]
        return self._band_teles

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
