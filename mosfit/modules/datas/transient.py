"""Definitions for the `Transient` class."""
from collections import OrderedDict

import numpy as np
from astrocats.catalog.utils import is_number
from astrocats.catalog.source import SOURCE
from mosfit.modules.module import Module
from mosfit.utils import listify


# Important: Only define one ``Module`` class per file.


class Transient(Module):
    """Structure to store transient data."""

    _REFERENCES = [
        {SOURCE.BIBCODE: '2017arXiv171002145G'}
    ]
    _EX_REPS = {
        'rad': 'radio',
        'xray': 'x-ray'
    }

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Transient, self).__init__(**kwargs)
        self._keys = kwargs.get('keys', '')
        self._data_determined_parameters = []

    def process(self, **kwargs):
        """Process module."""
        return self._data

    def set_data(self,
                 all_data,
                 req_key_values={},
                 subtract_minimum_keys=[],
                 smooth_times=-1,
                 extrapolate_time=0.0,
                 limit_fitting_mjds=False,
                 exclude_bands=[],
                 exclude_instruments=[],
                 exclude_systems=[],
                 exclude_sources=[],
                 exclude_kinds=[],
                 band_list=[],
                 band_telescopes=[],
                 band_systems=[],
                 band_instruments=[],
                 band_modes=[],
                 band_bandsets=[]):
        """Set transient data."""
        prt = self._printer

        self._all_data = all_data
        self._data = OrderedDict()
        if not self._all_data:
            return
        name = list(self._all_data.keys())[0]
        self._data['name'] = name
        numeric_keys = set()

        ex_kinds = [self._EX_REPS.get(
            x.lower(), x.lower()) for x in exclude_kinds]

        # Construct some source dictionaries for exclusion rules
        src_dict = OrderedDict()
        sources = self._all_data[name].get('sources', [])
        for src in sources:
            if SOURCE.BIBCODE in src:
                src_dict[src[SOURCE.ALIAS]] = src[SOURCE.BIBCODE]
            if SOURCE.ARXIVID in src:
                src_dict[src[SOURCE.ALIAS]] = src[SOURCE.ARXIVID]
            if SOURCE.NAME in src:
                src_dict[src[SOURCE.ALIAS]] = src[SOURCE.NAME]

        for key in self._keys:
            subkeys = self._keys[key]
            req_subkeys = [
                x for x in subkeys
                if not isinstance(subkeys, dict) or 'required' in listify(
                    subkeys[x])
            ]
            num_subkeys = [
                x for x in subkeys if 'numeric' in listify(subkeys[x])
            ]
            boo_subkeys = [
                x for x in subkeys if 'boolean' in listify(subkeys[x])
            ]
            exc_subkeys = [
                x for x in subkeys if 'exclude' in listify(subkeys[x])
            ]

            if (key not in self._all_data[name] and not
                    self._model.is_parameter_fixed_by_user(key)):
                if subkeys.get('value', None) == 'recommended':
                    self._unset_recommended_keys.add(key)
                continue

            subdata = self._all_data[name].get(key)

            if subdata is None:
                continue

            # Only include data that contains all subkeys
            for entry in subdata:
                if any([x not in entry for x in req_subkeys]):
                    continue
                if any([x in entry for x in exc_subkeys]):
                    continue
                if any([
                        x in entry and ((isinstance(entry[x], list) and any([
                            not is_number(y) or np.isnan(float(y))
                            for y in entry[x]
                        ])) or not isinstance(entry[x], list) and (
                            not is_number(entry[x]) or np.isnan(
                                float(entry[x]))))
                        for x in num_subkeys
                ]):
                    continue

                skip_key = False
                if 'frequency' not in entry:
                    for qkey in req_key_values:
                        if qkey in entry and entry[qkey] != '':
                            if entry[qkey] not in req_key_values[qkey]:
                                skip_key = True
                            break

                if key == 'photometry':
                    if ('fluxdensity' in entry and
                        'magnitude' not in entry and
                            'countrate' not in entry):
                        self._kinds_needed.add('radio')
                        if ('radio' in ex_kinds or
                            (not len(ex_kinds) or 'none' not in ex_kinds) and
                                'radio' not in self._model._kinds_supported):
                            continue
                    if (('countrate' in entry or
                         'unabsorbedflux' in entry or
                         'flux' in entry) and
                        'magnitude' not in entry and
                            'fluxdensity' not in entry):
                        self._kinds_needed.add('x-ray')
                        if ('x-ray' in ex_kinds or
                            (not len(ex_kinds) or 'none' not in ex_kinds) and
                                'x-ray' not in self._model._kinds_supported):
                            continue
                    if 'magnitude' in entry:
                        # For now, magnitudes are not excludable.
                        self._kinds_needed |= set(
                            ['infrared', 'optical', 'ultraviolet'])

                    skip_entry = False

                    for x in subkeys:
                        if limit_fitting_mjds is not False and x == 'time':
                            val = np.mean([
                                float(x) for x in listify(
                                    entry.get(x, None))])
                            if (val < limit_fitting_mjds[0] or
                                    val > limit_fitting_mjds[1]):
                                skip_entry = True
                                break
                        if exclude_bands is not False and x == 'band':
                            if (entry.get(x, '') in exclude_bands and
                                (not exclude_instruments or entry.get(
                                    'instrument', '') in
                                 exclude_instruments) and (
                                     not exclude_systems or entry.get(
                                    'system', '') in exclude_systems)):
                                skip_entry = True
                                break
                        if (exclude_instruments is not False and
                                x == 'instrument'):
                            if (entry.get(x, '') in exclude_instruments and
                                (not exclude_bands or
                                 entry.get('band', '') in
                                 exclude_bands) and (
                                     not exclude_systems or entry.get(
                                    'system', '') in exclude_systems)):
                                skip_entry = True
                                break
                        if (exclude_systems is not False and
                                x == 'system'):
                            if (entry.get(x, '') in exclude_systems and
                                (not exclude_bands or
                                 entry.get('band', '') in
                                 exclude_bands) and (
                                     not exclude_instruments or entry.get(
                                    'instrument', '') in exclude_instruments)):
                                skip_entry = True
                                break
                        if (exclude_sources is not False and
                                x == 'source'):
                            val = entry.get(x, '')
                            if (any([x in exclude_sources
                                     for x in val.split(',')]) or
                                any([src_dict.get(x, '') in exclude_sources
                                     for x in val.split(',')])):
                                skip_entry = True
                                break
                    if skip_entry:
                        continue

                    if ((('magnitude' in entry) != ('band' in entry)) or
                        ((('fluxdensity' in entry) != (
                            'frequency' in entry)) and (
                                'magnitude' not in entry)) or
                        (('countrate' in entry) and
                         ('magnitude' not in entry) and
                         ('instrument' not in entry))):
                        continue

                for x in subkeys:
                    falseval = (
                        False if x in boo_subkeys else None if
                        x in num_subkeys else '')
                    if x == 'value':
                        if not skip_key:
                            self._data[key] = entry.get(x, falseval)
                    else:
                        plural = self._model.plural(x)
                        val = entry.get(x, falseval)
                        if x in num_subkeys:
                            val = None if val is None else np.mean([
                                float(x) for x in listify(val)])
                        if not skip_key:
                            self._data.setdefault(plural, []).append(val)
                            if x in num_subkeys:
                                numeric_keys.add(plural)
                        else:
                            self._data.setdefault(
                                'unmatched_' + plural, []).append(val)

        if 'times' not in self._data or not any([x in self._data for x in [
                'magnitudes', 'frequencies', 'countrates']]):
            prt.message('no_fittable_data', [name])
            return False

        for key in list(self._data.keys()):
            if isinstance(self._data[key], list):
                self._data[key] = np.array(self._data[key])
                if key not in numeric_keys:
                    continue
                num_values = [
                    x for x in self._data[key] if isinstance(x, float)
                ]
                if len(num_values):
                    self._data['min_' + key] = min(num_values)
                    self._data['max_' + key] = max(num_values)
            else:
                if is_number(self._data[key]):
                    self._data[key] = float(self._data[key])
                    self._data_determined_parameters.append(key)

        if 'times' in self._data and smooth_times >= 0:
            obs = list(
                zip(*(self._data['telescopes'], self._data['systems'],
                      self._data['modes'], self._data['instruments'],
                      self._data['bandsets'], self._data['bands'], self._data[
                          'frequencies'], self._data['u_frequencies'])))
            if len(band_list):
                b_teles = band_telescopes if len(band_telescopes) == len(
                    band_list) else ([band_telescopes[0] for x in band_list]
                                     if len(band_telescopes) else
                                     ['' for x in band_list])
                b_systs = band_systems if len(band_systems) == len(
                    band_list) else ([band_systems[0] for x in band_list]
                                     if len(band_systems) else
                                     ['' for x in band_list])
                b_modes = band_modes if len(band_modes) == len(
                    band_list) else ([band_modes[0] for x in band_list]
                                     if len(band_modes) else
                                     ['' for x in band_list])
                b_insts = band_instruments if len(band_instruments) == len(
                    band_list) else ([band_instruments[0] for x in band_list]
                                     if len(band_instruments) else
                                     ['' for x in band_list])
                b_bsets = band_bandsets if len(band_bandsets) == len(
                    band_list) else ([band_bandsets[0] for x in band_list]
                                     if len(band_bandsets) else
                                     ['' for x in band_list])
                b_freqs = [None for x in band_list]
                b_u_freqs = ['' for x in band_list]
                obs.extend(
                    list(
                        zip(*(b_teles, b_systs, b_modes, b_insts, b_bsets,
                              band_list, b_freqs, b_u_freqs))))

            uniqueobs = []
            for o in obs:
                to = tuple(o)
                if to not in uniqueobs:
                    uniqueobs.append(to)

            minet, maxet = (extrapolate_time, extrapolate_time) if isinstance(
                extrapolate_time, (float, int)) else (
                    (tuple(extrapolate_time) if len(extrapolate_time) == 2 else
                     (extrapolate_time[0], extrapolate_time[0])))
            mint, maxt = (min(self._data['times']) - minet,
                          max(self._data['times']) + maxet)
            alltimes = list(
                sorted(
                    set([x for x in self._data['times']] + list(
                        np.linspace(mint, maxt, max(smooth_times, 2))))))
            currobslist = list(
                zip(*(
                    self._data['times'], self._data['telescopes'],
                    self._data['systems'], self._data['modes'],
                    self._data['instruments'], self._data['bandsets'],
                    self._data['bands'], self._data['frequencies'],
                    self._data['u_frequencies'])))

            obslist = []
            for ti, t in enumerate(alltimes):
                new_per = np.round(100.0 * float(ti) / len(alltimes), 1)
                prt.message('construct_obs_array', [new_per], inline=True)
                for o in uniqueobs:
                    newobs = tuple([t] + list(o))
                    if newobs not in currobslist:
                        obslist.append(newobs)

            obslist.sort()

            if len(obslist):
                (self._data['extra_times'], self._data['extra_telescopes'],
                 self._data['extra_systems'], self._data['extra_modes'],
                 self._data['extra_instruments'], self._data['extra_bandsets'],
                 self._data['extra_bands'], self._data['extra_frequencies'],
                 self._data['extra_u_frequencies']) = zip(*obslist)

        for qkey in subtract_minimum_keys:
            if 'upperlimits' in self._data:
                new_vals = np.array(self._data[qkey])[
                    np.array(self._data['upperlimits']) != True]  # noqa E712
                if len(new_vals):
                    self._data['min_' + qkey] = min(new_vals)
                    self._data['max_' + qkey] = max(new_vals)
            minv = self._data['min_' + qkey]
            self._data[qkey] = [x - minv for x in self._data[qkey]]
            if 'extra_' + qkey in self._data:
                self._data['extra_' + qkey] = [
                    x - minv for x in self._data['extra_' + qkey]
                ]

        return True

    def get_data_determined_parameters(self):
        """Return list of parameters determined by data."""
        return self._data_determined_parameters

    def send_request(self, request):
        """Send requests to other modules."""
        if request == 'min_times':
            return self._data['min_times']
        return []
