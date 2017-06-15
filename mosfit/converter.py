"""A class for converting ASCII inputs to JSON."""
import os
import re
from collections import Counter, OrderedDict
from decimal import Decimal
from itertools import permutations

import numpy as np
from astrocats.catalog.entry import Entry
from astrocats.catalog.key import KEY_TYPES, Key
from astrocats.catalog.photometry import PHOTOMETRY, set_pd_mag_from_counts
from astrocats.catalog.source import SOURCE
from astrocats.catalog.utils import is_number, jd_to_mjd
from astropy.io.ascii import Cds, Latex, read
from astropy.time import Time as astrotime
from six import string_types

from mosfit.utils import entabbed_json_dump, is_date


class Converter(object):
    """Convert ASCII formats to Open Catalog JSON schemas."""

    _MONTH_IDS = OrderedDict((
        ('January', '01'),
        ('February', '02'),
        ('March', '03'),
        ('April', '04'),
        ('June', '06'),
        ('July', '07'),
        ('August', '08'),
        ('September', '09'),
        ('October', '10'),
        ('November', '11'),
        ('December', '12'),
        ('Jan', '01'),
        ('Feb', '02'),
        ('Mar', '03'),
        ('Apr', '04'),
        ('May', '05'),
        ('Jun', '06'),
        ('Jul', '07'),
        ('Aug', '08'),
        ('Sep', '09'),
        ('Oct', '10'),
        ('Nov', '11'),
        ('Dec', '12')
    ))

    _DEFAULT_SOURCE = 'MOSFiT paper'

    def __init__(self, printer, require_source=False, **kwargs):
        """Initialize."""
        self._printer = printer
        self._require_source = require_source

        self._rsource = {SOURCE.NAME: self._DEFAULT_SOURCE}

        self._emagstrs = [
            'magnitude error', 'error', 'e mag', 'e magnitude', 'dmag',
            'mag err', 'magerr', 'mag error', 'err']
        self._header_keys = OrderedDict((
            (PHOTOMETRY.TIME, ['time', 'mjd', ('jd', 'jd')]),
            (PHOTOMETRY.SYSTEM, ['system']),
            (PHOTOMETRY.MAGNITUDE, ['mag', 'magnitude']),
            (PHOTOMETRY.E_MAGNITUDE, self._emagstrs),
            (PHOTOMETRY.TELESCOPE, ['tel', 'telescope']),
            (PHOTOMETRY.INSTRUMENT, ['inst', 'instrument']),
            (PHOTOMETRY.BAND, ['passband', 'band', 'filter']),
            (PHOTOMETRY.E_LOWER_MAGNITUDE, [
                ' '.join(y) for y in (
                    list(i for s in [
                        list(permutations(['minus'] + x.split()))
                        for x in self._emagstrs] for i in s) +
                    list(i for s in [
                        list(permutations(['lower'] + x.split()))
                        for x in self._emagstrs] for i in s))]),
            (PHOTOMETRY.E_UPPER_MAGNITUDE, [
                ' '.join(y) for y in (
                    list(i for s in [
                        list(permutations(['plus'] + x.split()))
                        for x in self._emagstrs] for i in s) +
                    list(i for s in [
                        list(permutations(['upper'] + x.split()))
                        for x in self._emagstrs] for i in s))]),
            (PHOTOMETRY.UPPER_LIMIT, ['upper limit', 'upperlimit', 'l_mag']),
            (PHOTOMETRY.COUNT_RATE, ['counts', 'flux', 'count rate']),
            (PHOTOMETRY.E_COUNT_RATE, [
                'e_counts', 'count error', 'count rate error']),
            (PHOTOMETRY.ZERO_POINT, ['zero point', 'self._zp']),
            ('reference', ['reference', 'bibcode', 'source', 'origin']),
            ('event', ['event', 'transient', 'name', 'supernova'])
        ))
        self._critical_keys = [
            PHOTOMETRY.TIME, PHOTOMETRY.MAGNITUDE, PHOTOMETRY.BAND,
            PHOTOMETRY.E_MAGNITUDE,
            PHOTOMETRY.COUNT_RATE, PHOTOMETRY.E_COUNT_RATE,
            PHOTOMETRY.ZERO_POINT]
        self._helpful_keys = [PHOTOMETRY.INSTRUMENT, PHOTOMETRY.TELESCOPE]
        self._optional_keys = [PHOTOMETRY.ZERO_POINT]
        self._mc_keys = [PHOTOMETRY.MAGNITUDE, PHOTOMETRY.COUNT_RATE]
        self._dep_keys = [
            PHOTOMETRY.E_MAGNITUDE, PHOTOMETRY.BAND, PHOTOMETRY.E_COUNT_RATE]
        self._bool_keys = [PHOTOMETRY.UPPER_LIMIT]
        self._specify_keys = [
            PHOTOMETRY.BAND, PHOTOMETRY.INSTRUMENT, PHOTOMETRY.TELESCOPE]
        self._band_names = [
            'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'u', 'g', 'r', 'i', 'z',
            'y', 'W1', 'W2', 'M2'
        ]
        self._use_mc = False
        self._month_rep = re.compile(
            r'\b(' + '|'.join(self._MONTH_IDS.keys()) + r')\b')
        self._converted = []

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
        intro_shown = False

        new_event_list = []
        for event in event_list:
            new_events = []
            if ('.' in event and os.path.isfile(event) and
                    not event.endswith('.json')):
                if not intro_shown:
                    prt.message('converter_info')
                    intro_shown = True

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
                    for i in range(len(flines)):
                        flines[i] = [str(x) for x in flines[i]]

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
                    fsplit = [x.replace('$', '').replace(',', '\t')
                              .replace('&', '\t').replace('\\pm', '\t')
                              .strip(' ()')
                              for x in fsplit]
                    flines = [
                        [y.replace('"', '').replace("'", '') for y in
                         re.split(
                             '''\s+(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''',
                             x)]
                        for x in fsplit]

                    flines = [[
                        x.strip(' #$()\\') for x in y] for y in flines]

                    # Find band columns if they exist and insert error columns
                    # if they don't exist.
                    for fi, fl in enumerate(list(flines)):
                        flcopy = list(fl)
                        offset = 0
                        if not any([is_number(x) for x in fl]):
                            for fci, fc in enumerate(fl):
                                if (fc in self._band_names and
                                    (fci == len(fl) - 1 or
                                     fl[fci + 1].lower()
                                     not in self._emagstrs)):
                                    flcopy.insert(fci + 1 + offset, 'e mag')
                                    offset += 1
                        flines[fi] = flcopy

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
                    (not any(any([is_number(x) or x == '' for x in y])
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
                                         kind='bool', default='y')
                    if not is_name:
                        new_event_name = ''
                        while new_event_name.strip() == '':
                            new_event_name = prt.prompt(
                                'enter_name', kind='string')
                    new_events = [new_event_name + '.json']

                    if len(cidict) and len(new_event_list):
                        text = prt.message(
                            'is_event_same', [''.join(
                                new_event_list[-1].split('.')[:-1])],
                            prt=False)
                        is_same = prt.prompt(text, message=False,
                                             kind='bool')
                        if not is_same:
                            cidict = {}

                    # If the first row has no numbers it is likely a header.
                    if not len(cidict):
                        self.assign_columns(cidict, flines)

                    # Create a new event, populate the photometry, and dump
                    # to a JSON file in the run directory.
                    entry = Entry(name=new_event_name)
                    perms = 1
                    for key in cidict:
                        if isinstance(cidict[key], list) and not isinstance(
                                cidict[key], string_types):
                            if cidict[key][0] != 'j':
                                perms = len(cidict[key])

                    # Clean up the data a bit now that we know the column
                    # identities.

                    # Strip common prefixes/suffixes from band names
                    if PHOTOMETRY.BAND in cidict:
                        bi = cidict[PHOTOMETRY.BAND]
                        for d in [True, False]:
                            if not isinstance(bi, int):
                                break
                            strip_cols = []
                            lens = [len(x[bi])
                                    for x in flines[self._first_data:]]
                            llen = min(lens)
                            ra = range(llen) if d else range(-1, -llen - 1, -1)
                            for li in ra:
                                letter = None
                                for row in list(flines[self._first_data:]):
                                    if letter is None:
                                        letter = row[bi][li]
                                    elif row[bi][li] != letter:
                                        letter = None
                                        break
                                if letter is not None:
                                    strip_cols.append(li)
                                else:
                                    break
                            print(strip_cols)
                            if len(strip_cols) == llen:
                                break
                            for ri in range(len(flines[self._first_data:])):
                                flines[self._first_data + ri][bi] = ''.join(
                                    [c for i, c in enumerate(flines[
                                        self._first_data + ri][bi])
                                     if (i if d else i - len(flines[
                                         self._first_data + ri][bi])) not in
                                     strip_cols])

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
                                elif (isinstance(key, Key) and
                                        key.type == KEY_TYPES.TIME and
                                        isinstance(cidict[key], list) and not
                                        isinstance(cidict[key],
                                                   string_types)):
                                    tval = np.array(row)[np.array(cidict[key][
                                        1:], dtype=int)]
                                    if cidict[key][0] == 'j':
                                        date = '-'.join([x.zfill(2) for x in
                                                         tval])
                                        date = self._month_rep.sub(
                                            lambda x: self._MONTH_IDS[
                                                x.group()], date)
                                        photodict[key] = str(
                                            astrotime(date, format='isot').mjd)
                                    elif cidict[key][0] == 'jd':
                                        photodict[key] = str(
                                            jd_to_mjd(Decimal(tval[-1])))
                                    continue

                                val = cidict[key]
                                if (isinstance(val, list) and not
                                        isinstance(val, string_types)):
                                    val = val[pi]
                                    if isinstance(val, string_types):
                                        if val != '':
                                            photodict[key] = val
                                    else:
                                        photodict[key] = row[val]
                                else:
                                    if isinstance(val, string_types):
                                        if val != '':
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
                                if self._require_source:
                                    if (self._rsource.get(SOURCE.NAME, '') ==
                                            self._DEFAULT_SOURCE):
                                        self._rsource = {}
                                        sopts = [
                                            ('Bibcode', 'b'), ('Name', 'n')]
                                        text = prt.message(
                                            'select_source', [new_event_name],
                                            prt=False)
                                        skind = prt.prompt(
                                            text, kind='option',
                                            options=sopts, default='b',
                                            none_string=None)
                                        if skind == 'b':
                                            bibcode = ''
                                            while len(bibcode) != 19:
                                                bibcode = prt.prompt(
                                                    'bibcode',
                                                    kind='string',
                                                    allow_blank=False
                                                )
                                            self._rsource[
                                                SOURCE.BIBCODE] = bibcode
                                        elif skind == 'n':
                                            last_name = prt.prompt(
                                                'last_name', kind='string'
                                            )
                                            self._rsource[
                                                SOURCE.NAME] = (
                                                    last_name.strip().title() +
                                                    ' et al., in preparation')

                                photodict[
                                    PHOTOMETRY.SOURCE] = entry.add_source(
                                        **self._rsource)

                            if any([x in photodict.get(
                                    PHOTOMETRY.MAGNITUDE, '')
                                    for x in ['<', '>']]):
                                photodict[PHOTOMETRY.UPPER_LIMIT] = True
                                photodict[
                                    PHOTOMETRY.MAGNITUDE] = photodict[
                                        PHOTOMETRY.MAGNITUDE].strip('<>')

                            # Skip entries for which key values are not
                            # expected type.
                            if not all([
                                is_number(photodict.get(x, ''))
                                for x in photodict.keys() if
                                (PHOTOMETRY.get_key_by_name(x).type ==
                                 KEY_TYPES.NUMERIC)]):
                                continue

                            # Skip placeholder values.
                            if float(photodict.get(
                                    PHOTOMETRY.MAGNITUDE, 0.0)) > 50.0:
                                continue

                            # Add system if specified by user.
                            if (self._system is not None and
                                    PHOTOMETRY.SYSTEM not in photodict):
                                photodict[PHOTOMETRY.SYSTEM] = self._system

                            # Remove keys not in the `PHOTOMETRY` class.
                            for key in list(photodict.keys()):
                                if key not in PHOTOMETRY.vals():
                                    del(photodict[key])

                            # Add the photometry.
                            entry.add_photometry(**photodict)

                    entry.sanitize()
                    oentry = entry._ordered(entry)

                    with open(new_events[0], 'w') as f:
                        entabbed_json_dump(
                            {new_event_name: oentry}, f,
                            separators=(',', ':'))

                    self._converted.append([new_event_name, new_events[0]])

                new_event_list.extend(new_events)
            else:
                new_event_list.append(event)

        return new_event_list

    def assign_columns(self, cidict, flines):
        """Assign columns based on header."""
        used_cis = {}
        akeys = list(self._critical_keys) + list(self._helpful_keys)
        dkeys = list(self._dep_keys)
        prt = self._printer

        for fi, fl in enumerate(flines):
            if not any([is_number(x) for x in fl]):
                # Try to associate column names with common header keys.
                conflict_keys = []
                conflict_cis = []
                for ci, col in enumerate(fl):
                    for key in self._header_keys:
                        if any([(x[0] if isinstance(x, tuple)
                                 else x) == col.lower()
                                for x in self._header_keys[key]]):
                            if key in cidict or ci in used_cis:
                                # There is a conflict, ask user.
                                conflict_keys.append(key)
                                conflict_cis.append(ci)
                            else:
                                ind = [
                                    (x[0] if isinstance(x, tuple) else x)
                                    for x in self._header_keys[key]].index(
                                        col.lower())
                                match = self._header_keys[key][ind]
                                cidict[key] = [match[-1], ci] if isinstance(
                                    match, tuple) else ci
                                used_cis[ci] = key
                            break

                for cki, ck in enumerate(conflict_keys):
                    if ck in cidict:
                        ci = cidict[ck]
                        del(cidict[ck])
                        del(used_cis[ci])
            else:
                self._first_data = fi
                break

        # Look for columns that are band names if no magnitude column was
        # found.
        if PHOTOMETRY.MAGNITUDE not in cidict:
            # Delete `E_MAGNITUDE` and `BAND` if they exist (we'll need to find
            # for each column).
            key = PHOTOMETRY.MAGNITUDE
            ekey = PHOTOMETRY.E_MAGNITUDE
            bkey = PHOTOMETRY.BAND
            if ekey in cidict:
                ci = cidict[ekey]
                del(cidict[used_cis[ci]])
                del(used_cis[ci])
            if bkey in cidict:
                ci = cidict[bkey]
                del(cidict[used_cis[ci]])
                del(used_cis[ci])
            for fi, fl in enumerate(flines):
                if not any([is_number(x) for x in fl]):
                    # Try to associate column names with common header keys.
                    for ci, col in enumerate(fl):
                        if ci in used_cis:
                            continue
                        if col in self._band_names:
                            cidict.setdefault(key, []).append(ci)
                            used_cis[ci] = key
                            cidict.setdefault(bkey, []).append(col)
                        elif col.lower() in self._emagstrs:
                            cidict.setdefault(ekey, []).append(ci)
                            used_cis[ci] = ekey

        # See which keys we collected. If we are missing any critical keys, ask
        # the user which column they are.

        # First ask the user if this data is in magnitudes or in counts.
        self._data_type = 1
        if (PHOTOMETRY.MAGNITUDE in cidict and
                PHOTOMETRY.COUNT_RATE not in cidict):
            self._data_type = 1
        elif (PHOTOMETRY.MAGNITUDE not in cidict and
              PHOTOMETRY.COUNT_RATE in cidict):
            self._data_type = 2
        else:
            self._data_type = prt.prompt(
                'counts_or_mags', kind='option',
                options=['Magnitudes', 'Counts (fluxes)'],
                none_string=None)
        if self._data_type == 1:
            akeys.remove(PHOTOMETRY.COUNT_RATE)
            akeys.remove(PHOTOMETRY.E_COUNT_RATE)
            akeys.remove(PHOTOMETRY.ZERO_POINT)
            if (PHOTOMETRY.E_LOWER_MAGNITUDE in cidict and
                    PHOTOMETRY.E_UPPER_MAGNITUDE in cidict):
                akeys.remove(PHOTOMETRY.E_MAGNITUDE)
            dkeys.remove(PHOTOMETRY.E_COUNT_RATE)
        else:
            akeys.remove(PHOTOMETRY.MAGNITUDE)
            akeys.remove(PHOTOMETRY.E_MAGNITUDE)
            dkeys.remove(PHOTOMETRY.E_MAGNITUDE)

        columns = np.array(flines[self._first_data:]).T.tolist()
        colstrs = np.array([
            ', '.join(x[:5]) + ', ...' for x in columns])
        print(colstrs)
        colinds = np.setdiff1d(np.arange(len(colstrs)),
                               list([x[-1] if isinstance(x, list)
                                     else x for x in cidict.values()]))
        ignore = prt.message('ignore_column', prt=False)
        specify = prt.message('specify_column', prt=False)
        for key in akeys:
            if key in cidict:
                continue
            if key in dkeys and self._use_mc:
                continue
            if key.type == KEY_TYPES.NUMERIC:
                lcolinds = [x for x in colinds
                            if any(is_number(y) for y in columns[x])]
            elif key.type == KEY_TYPES.TIME:
                lcolinds = [x for x in colinds
                            if any(is_date(y) or is_number(y)
                                   for y in columns[x])]
            elif key.type == KEY_TYPES.STRING:
                lcolinds = [x for x in colinds
                            if any(not is_number(y) for y in columns[x])]
            else:
                lcolinds = list(colinds)
            select = False
            selmap = np.array(range(len(lcolinds)))
            selects = []
            while select is False:
                mc = 1
                if key in self._mc_keys:
                    text = prt.message(
                        'one_per_line', [key, key, key], prt=False)
                    mc = prt.prompt(
                        text, kind='option', message=False,
                        none_string=None,
                        options=[
                            'One `{}` per row'.format(key),
                            'Multiple `{}s` per row'.format(
                                key)])
                if mc == 1:
                    text = prt.message(
                        'no_matching_column', [key], prt=False)
                    ns = (
                        ignore if key in (
                            self._optional_keys + self._helpful_keys) else
                        specify if key in self._specify_keys
                        else None)
                    if len(colstrs[lcolinds]):
                        select = prt.prompt(
                            text, message=False,
                            kind='option', none_string=ns,
                            default=('j' if ns is None and
                                     len(colstrs[lcolinds]) > 1
                                     else None if ns is None else 'n'),
                            options=colstrs[lcolinds].tolist() + (
                                [('Multiple columns need to be joined.', 'j')]
                                if len(colstrs[lcolinds]) > 1 else []))
                    else:
                        select = None
                    if select == 'j':
                        select = None
                        jsel = None
                        selects.append('j')
                        while jsel != 'd' and len(lcolinds):
                            jsel = prt.prompt(
                                'join_which_columns', default='d',
                                kind='option', none_string=None,
                                options=colstrs[lcolinds].tolist() + [
                                    ('All columns to be joined '
                                     'have been selected.', 'd')
                                ])
                            if jsel != 'd':
                                selects.append(lcolinds[jsel - 1])
                                selmap = np.delete(selmap, jsel - 1)
                                lcolinds = np.delete(lcolinds, jsel - 1)
                else:
                    self._use_mc = True
                    select = False
                    while select is not None:
                        text = prt.message(
                            'select_mc_column', [key], prt=False)
                        select = prt.prompt(
                            text, message=False,
                            kind='option', default='n',
                            none_string='No more `{}` columns.'.format(key),
                            options=colstrs[lcolinds].tolist())
                        if select is not None and select is not False:
                            selects.append(lcolinds[select - 1])
                            selmap = np.delete(selmap, select - 1)
                            lcolinds = np.delete(lcolinds, select - 1)
                        else:
                            break
                        for dk in dkeys:
                            dksel = None
                            while dksel is None:
                                text = prt.message(
                                    'select_dep_column', [dk, key], prt=False)
                                sk = dk in self._specify_keys
                                if not sk:
                                    dksel = prt.prompt(
                                        text, message=False,
                                        kind='option', none_string=None,
                                        options=colstrs[lcolinds].tolist())
                                    if dksel is not None:
                                        selects.append(lcolinds[dksel - 1])
                                        selmap = np.delete(selmap, dksel - 1)
                                        lcolinds = np.delete(
                                            lcolinds, dksel - 1)
                                else:
                                    spectext = prt.message(
                                        'specify_mc_value', [dk, key],
                                        prt=False)
                                    val = ''
                                    while val.strip() is '':
                                        val = prt.prompt(
                                            spectext, message=False,
                                            kind='string')
                                    selects.append(val)
                                    break

            if select is not None:
                iselect = int(select)
                cidict[key] = lcolinds[iselect - 1]
                colinds = np.delete(colinds, np.argwhere(
                    colinds == lcolinds[iselect - 1]))
            elif len(selects):
                if selects[0] == 'j':
                    cidict[key] = selects
                else:
                    allk = [key] + dkeys
                    for ki, k in enumerate(allk):
                        cidict[k] = [
                            lcolinds[s - 1] if isinstance(s, int) else s
                            for s in selects[ki::len(allk)]]
                    for s in selects:
                        if not isinstance(s, int):
                            continue
                        colinds = np.delete(colinds, np.argwhere(
                            colinds == lcolinds[s - 1]))
            elif key in self._specify_keys:
                msg = ('specify_value_blank' if key in self._helpful_keys else
                       'specify_value')
                text = prt.message(msg, [key], prt=False)
                cidict[key] = prt.prompt(
                    text, message=False, kind='string', allow_blank=(
                        key in self._helpful_keys))

        self._zp = ''
        if self._data_type == 2 and PHOTOMETRY.ZERO_POINT not in cidict:
            while not is_number(self._zp):
                self._zp = prt.prompt('zeropoint', kind='string')

        self._system = None
        if self._data_type == 1 and PHOTOMETRY.SYSTEM not in cidict:
            systems = ['AB', 'Vega']
            self._system = prt.prompt(
                'system', kind='option', options=systems,
                none_string='Use default for all bands.',
                default='n')
            if self._system is not None:
                self._system = systems[int(self._system) - 1]

        if (PHOTOMETRY.INSTRUMENT not in cidict and
                PHOTOMETRY.TELESCOPE not in cidict):
            prt.message('instrument_recommended', warning=True)

    def get_converted(self):
        """Get a list of events that were converted from ASCII to JSON."""
        return self._converted
