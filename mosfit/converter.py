"""A class for converting ASCII inputs to JSON."""
import os
import re
from collections import Counter
from itertools import permutations

import numpy as np
from astrocats.catalog.entry import Entry
from astrocats.catalog.key import KEY_TYPES
from astrocats.catalog.photometry import PHOTOMETRY, set_pd_mag_from_counts
from astrocats.catalog.utils import is_number
from astropy.io.ascii import Cds, Latex, read
from six import string_types

from mosfit.utils import entabbed_json_dump


class Converter(object):
    """Convert ASCII formats to Open Catalog JSON schemas."""

    def __init__(self, printer, **kwargs):
        """Initialize."""
        self._printer = printer
        emagstrs = [
            'magnitude error', 'error', 'e mag', 'e magnitude', 'dmag',
            'mag err', 'magerr']
        self._header_keys = {
            PHOTOMETRY.TIME: ['time', 'mjd', 'jd'],
            PHOTOMETRY.BAND: ['passband', 'band', 'filter'],
            PHOTOMETRY.TELESCOPE: ['tel', 'telescope'],
            PHOTOMETRY.INSTRUMENT: ['inst', 'instrument'],
            PHOTOMETRY.SYSTEM: ['system'],
            PHOTOMETRY.MAGNITUDE: ['mag', 'magnitude'],
            PHOTOMETRY.E_MAGNITUDE: emagstrs,
            PHOTOMETRY.E_LOWER_MAGNITUDE: [
                ' '.join(y) for y in (
                    list(i for s in [
                        list(permutations(['minus'] + x.split()))
                        for x in emagstrs] for i in s) +
                    list(i for s in [
                        list(permutations(['lower'] + x.split()))
                        for x in emagstrs] for i in s))],
            PHOTOMETRY.E_UPPER_MAGNITUDE: [
                ' '.join(y) for y in (
                    list(i for s in [
                        list(permutations(['plus'] + x.split()))
                        for x in emagstrs] for i in s) +
                    list(i for s in [
                        list(permutations(['upper'] + x.split()))
                        for x in emagstrs] for i in s))],
            PHOTOMETRY.UPPER_LIMIT: ['upper limit', 'upperlimit', 'l_mag'],
            PHOTOMETRY.COUNT_RATE: ['counts', 'flux', 'count rate'],
            PHOTOMETRY.E_COUNT_RATE: [
                'e_counts', 'count error', 'count rate error'],
            PHOTOMETRY.ZERO_POINT: ['zero point', 'self._zp'],
            'reference': ['reference', 'bibcode', 'source', 'origin'],
            'event': ['event', 'transient', 'name', 'supernova']
        }
        self._critical_keys = [
            PHOTOMETRY.TIME, PHOTOMETRY.MAGNITUDE, PHOTOMETRY.BAND,
            PHOTOMETRY.E_MAGNITUDE,
            PHOTOMETRY.COUNT_RATE, PHOTOMETRY.E_COUNT_RATE,
            PHOTOMETRY.ZERO_POINT]
        self._optional_keys = [PHOTOMETRY.ZERO_POINT]
        self._mc_keys = [PHOTOMETRY.MAGNITUDE, PHOTOMETRY.COUNT_RATE]
        self._dep_keys = [
            PHOTOMETRY.E_MAGNITUDE, PHOTOMETRY.BAND, PHOTOMETRY.E_COUNT_RATE]
        self._bool_keys = [PHOTOMETRY.UPPER_LIMIT]
        self._specify_keys = [PHOTOMETRY.BAND]
        self._use_mc = False
        for key in self._header_keys.keys():
            for val in self._header_keys[key]:
                for i in range(val.count(' ')):
                    rep = val.replace(' ', '_', i + 1)
                    if rep not in self._header_keys[key]:
                        self._header_keys[key].append(rep)
                for i in range(val.count(' ')):
                    rep = val.replace(' ', '', i + 1)
                    if rep not in self._header_keys[key]:
                        self._header_keys[key].append(rep)

    def generate_event_list(self, event_list):
        """Generate a list of events and/or convert events to JSON format."""
        prt = self._printer
        cidict = {}
        metadict = {}

        new_event_list = []
        for event in event_list:
            new_events = []
            if ('.' in event and os.path.isfile(event) and
                    not event.endswith('.json')):
                prt.message('converting_to_json', [event])

                with open(event, 'r') as f:
                    ftxt = f.read()

                # Try a couple of table formats from astropy.
                table = None
                try:
                    table = read(ftxt, Reader=Cds, guess=False)
                except Exception:
                    pass
                else:
                    prt.message('convert_cds')
                    flines = [table.colnames] + [
                        list(x) for x in np.array(table).tolist()]

                try:
                    table = read(ftxt, Reader=Latex, guess=False)
                except Exception:
                    pass
                else:
                    prt.message('convert_latex')
                    flines = [table.colnames] + [
                        list(x) for x in np.array(table).tolist()]

                if table is None:
                    fsplit = ftxt.splitlines()
                    fsplit = [x.replace(',', '\t').replace('&', '\t')
                              for x in fsplit]
                    flines = [
                        [y.replace('"', '').replace("'", '') for y in
                         re.split(
                             '''\s(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', x)]
                        for x in fsplit]
                    flines = [[
                        x.strip(' #$()') for x in y] for y in flines]
                    flines = [list(filter(None, x)) for x in flines]

                    # Find the most frequent column count. These are probably
                    # the tables we wish to read.
                    flens = [len(x) for x in flines]
                    ncols = Counter(flens).most_common(1)[0][0]

                    newlines = []
                    for fi, fl in enumerate(flines):
                        if flens[fi] == ncols:
                            newlines.append(list(fl))
                    flines = newlines

                # If none of the rows contain numeric data, the file
                # is likely a list of transients.
                if (len(flines) and
                    (not any(any([is_number(x) for x in y])
                             for y in flines) or
                     len(flines) == 1)):
                    new_events = [
                        it for s in flines for it in s]

                # If last row is numeric, then likely this is a single
                # transient.
                elif (len(flines) > 1 and
                        any([is_number(x) for x in flines[-1]])):

                    # Check that each row has the same number of columns.
                    if len(set([len(x) for x in flines])) > 1:
                        raise ValueError(
                            'Number of columns in each row not '
                            'consistent!')
                    new_event_name = '.'.join(event.split(
                        '.')[:-1]).split('/')[-1]
                    text = prt.message(
                        'is_event_name', [new_event_name], prt=False)
                    is_name = prt.prompt(text, message=False,
                                         kind='bool', color='!m')
                    if not is_name:
                        new_event_name = ''
                        while new_event_name.strip() == '':
                            new_event_name = prt.prompt(
                                'enter_name', kind='string', color='!m')
                    new_events = [new_event_name + '.json']

                    if len(cidict) and len(new_event_list):
                        text = prt.message(
                            'is_event_same', [''.join(
                                new_event_list[-1].split('.')[:-1])],
                            prt=False)
                        is_same = prt.prompt(text, message=False,
                                             kind='bool', color='!m')
                        if not is_same:
                            cidict = {}
                            metadict = {}

                    # If the first row has no numbers it is likely a header.
                    if not len(cidict):
                        self.assign_columns(cidict, flines, metadict)

                    # Create a new event, populate the photometry, and dump
                    # to a JSON file in the run directory.
                    entry = Entry(name=new_event_name)
                    perms = 1
                    for key in cidict:
                        if type(cidict[key]) == 'list':
                            perms = len(cidict[key])

                    for row in flines[self._first_data:]:
                        photodict = {}
                        for pi in range(perms):
                            sources = set()
                            for key in cidict:
                                if key in self._bool_keys:
                                    rval = row[cidict[key]]
                                    if type(rval) != 'bool':
                                        try:
                                            rval = bool(float(rval))
                                        except Exception:
                                            rval = True
                                    if not rval:
                                        continue
                                    row[cidict[key]] = rval
                                elif key == 'reference':
                                    if (isinstance(cidict[key],
                                                   string_types) and
                                            len(cidict[key]) == 19):
                                        new_src = entry.add_source(
                                            bibcode=cidict[key])
                                        sources.update(new_src)
                                        row[
                                            cidict[key]] = new_src
                                val = cidict[key]
                                if isinstance(val, list) and not isinstance(
                                        val, string_types):
                                    val = val[pi]
                                    if isinstance(val, string_types):
                                        photodict[key] = val
                                    else:
                                        photodict[key] = row[val]
                                else:
                                    if isinstance(val, string_types):
                                        photodict[key] = val
                                    else:
                                        photodict[key] = row[val]
                            if self._data_type == 2:
                                if self._zp:
                                    photodict[PHOTOMETRY.ZERO_POINT] = self._zp
                                set_pd_mag_from_counts(
                                    photodict,
                                    c=row[cidict[PHOTOMETRY.COUNT_RATE]],
                                    ec=row[cidict[PHOTOMETRY.E_COUNT_RATE]],
                                    zp=self._zp)
                            if not len(sources):
                                photodict[
                                    PHOTOMETRY.SOURCE] = entry.add_source(
                                        name='MOSFiT paper')

                            # Skip placeholder values.
                            if float(photodict.get(
                                    PHOTOMETRY.MAGNITUDE, 0.0)) > 50.0:
                                continue

                            # Add the photometry.
                            entry.add_photometry(**photodict)

                    entry.sanitize()
                    oentry = entry._ordered(entry)

                    with open(new_events[0], 'w') as f:
                        entabbed_json_dump(
                            {new_event_name: oentry}, f,
                            separators=(',', ':'))

                new_event_list.extend(new_events)
            else:
                new_event_list.append(event)

        return new_event_list

    def assign_columns(self, cidict, flines, metadict):
        """Assign columns based on header."""
        used_cis = {}
        ckeys = list(self._critical_keys)
        dkeys = list(self._dep_keys)
        hkeys = np.array([])
        prt = self._printer

        for fi, fl in enumerate(flines):
            if not any([is_number(x) for x in fl]):
                hkeys = np.array(fl)
                # Try to associate column names with common header
                # keys.
                for ci, col in enumerate(fl):
                    for key in self._header_keys:
                        if any([x == col.lower()
                                for x in self._header_keys[key]]):
                            if ci in used_cis:
                                # There is a conflict, ask user.
                                del(cidict[used_cis[ci]])
                            else:
                                cidict[key] = ci
                                used_cis[ci] = key
                            break
            else:
                self._first_data = fi
                break

        # See which keys we collected. If we are missing any
        # critical keys, ask the user which column they are.

        # First ask the user if this data is in magnitudes or
        # in counts.
        self._data_type = 1
        if (PHOTOMETRY.MAGNITUDE in cidict and
                PHOTOMETRY.COUNT_RATE not in cidict):
            self._data_type = 1
        elif (PHOTOMETRY.MAGNITUDE not in cidict and
              PHOTOMETRY.COUNT_RATE in cidict):
            self._data_type = 2
        else:
            self._data_type = None
            while self._data_type is None:
                self._data_type = prt.prompt(
                    'counts_or_mags', kind='option',
                    options=['Magnitudes', 'Counts (fluxes)'],
                    none_string=None,
                    color='!m')
        if self._data_type == 1:
            ckeys.remove(PHOTOMETRY.COUNT_RATE)
            ckeys.remove(PHOTOMETRY.E_COUNT_RATE)
            ckeys.remove(PHOTOMETRY.ZERO_POINT)
            if (PHOTOMETRY.E_LOWER_MAGNITUDE in cidict and
                    PHOTOMETRY.E_UPPER_MAGNITUDE in cidict):
                ckeys.remove(PHOTOMETRY.E_MAGNITUDE)
            dkeys.remove(PHOTOMETRY.E_COUNT_RATE)
        else:
            ckeys.remove(PHOTOMETRY.MAGNITUDE)
            ckeys.remove(PHOTOMETRY.E_MAGNITUDE)
            dkeys.remove(PHOTOMETRY.E_MAGNITUDE)

        columns = np.array(flines[self._first_data:]).T.tolist()
        colstrs = np.array([
            ', '.join(x[:3]) + ', ...' for x in columns])
        colinds = np.setdiff1d(np.arange(len(colstrs)),
                               list(cidict.values()))
        ignore = prt.message('ignore_column', prt=False)
        specify = prt.message('specify_column', prt=False)
        for key in ckeys:
            if key in cidict:
                continue
            if key in dkeys and self._use_mc:
                continue
            if key.type == KEY_TYPES.NUMERIC:
                lcolinds = [x for xi, x in enumerate(colinds)
                            if any(is_number(y) for y in columns[x])]
            else:
                lcolinds = colinds
            select = False
            selects = []
            while select is False:
                mc = 1
                if key in self._mc_keys:
                    text = prt.message(
                        'one_per_line', [key, key, key], color='!m',
                        prt=False)
                    mc = None
                    while mc is None:
                        mc = prt.prompt(
                            text, kind='option', message=False,
                            none_string=None,
                            options=[
                                'One `{}` per row'.format(key),
                                'Multiple `{}s` per row'.format(
                                    key)], color='!m')
                if mc == 1:
                    text = prt.message(
                        'no_matching_column', [key], prt=False)
                    select = prt.prompt(
                        text, message=False,
                        kind='option', none_string=(
                            ignore if key in self._optional_keys else
                            specify if key in self._specify_keys
                            else None),
                        options=colstrs[lcolinds], color='!m')
                else:
                    self._use_mc = True
                    select = False
                    llcolinds = np.array(lcolinds)
                    selmap = np.array(range(1, len(lcolinds) + 1))
                    while select is not None:
                        text = prt.message(
                            'select_mc_column', [key], prt=False)
                        select = prt.prompt(
                            text, message=False,
                            kind='option',
                            none_string='No more `{}` columns.'.format(key),
                            options=colstrs[llcolinds], color='!m')
                        if select is not None:
                            selects.append(selmap[select])
                            selmap = np.delete(selmap, select - 1)
                            llcolinds = np.delete(llcolinds, select - 1)
                        else:
                            break
                        for dk in dkeys:
                            dksel = None
                            while dksel is None:
                                text = prt.message(
                                    'select_dep_column', [dk, key], prt=False)
                                sk = dk in self._specify_keys
                                dksel = prt.prompt(
                                    text, message=False,
                                    kind='option',
                                    none_string=specify if sk else None,
                                    options=colstrs[llcolinds], color='!m')
                                if dksel is None and sk:
                                    spectext = prt.message(
                                        'specify_value', [dk], prt=False)
                                    val = ''
                                    while val.strip() is '':
                                        val = prt.prompt(
                                            spectext, message=False,
                                            kind='string', color='!m')
                                    selects.append(val)
                                    break
                                elif dksel is not None:
                                    selects.append(selmap[dksel])
                                    selmap = np.delete(selmap, dksel - 1)
                                    llcolinds = np.delete(
                                        llcolinds, dksel - 1)

            if select is not None:
                cidict[key] = colinds[select - 1]
                colinds = np.delete(colinds, select - 1)
            elif len(selects):
                allk = [key] + dkeys
                for ki, k in enumerate(allk):
                    cidict[k] = [
                        colinds[s - 1] if isinstance(s, int) else s
                        for s in selects[ki::len(allk)]]
                ind = np.array(cidict[key])
                if len(hkeys):
                    metadict.setdefault(key, []).extend(hkeys[ind])
                for s in selects:
                    if not isinstance(s, int):
                        continue
                    colinds = np.delete(colinds, s - 1)
            elif key in self._specify_keys:
                text = prt.message('specify_value', [key], prt=False)
                cidict[key] = ''
                while cidict[key].strip() is '':
                    cidict[key] = prt.prompt(
                        text, message=False, kind='string', color='!m')

        self._zp = ''
        if self._data_type == 2 and PHOTOMETRY.ZERO_POINT not in cidict:
            while not is_number(self._zp):
                self._zp = prt.prompt(
                    'zeropoint', kind='string', color='!m')
