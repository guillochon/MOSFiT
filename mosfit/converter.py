# -*- coding: UTF-8 -*-
"""A class for converting ASCII inputs to JSON."""
import csv
import os
import re
from collections import Counter, OrderedDict
from decimal import Decimal
from itertools import chain, product

import inflect
import numpy as np
from astrocats.catalog.catalog import Catalog
from astrocats.catalog.entry import ENTRY, Entry
from astrocats.catalog.key import KEY_TYPES, Key
from astrocats.catalog.photometry import (PHOTOMETRY, set_pd_mag_from_counts,
                                          set_pd_mag_from_flux_density)
from astrocats.catalog.quantity import QUANTITY
from astrocats.catalog.source import SOURCE
from astrocats.catalog.utils import jd_to_mjd
from astropy.io.ascii import Cds, Latex, read
from astropy.time import Time as astrotime
from mosfit.constants import KS_DAYS
from mosfit.utils import (entabbed_json_dump, get_mosfit_hash, is_bibcode,
                          is_date, is_datum, is_number, listify, name_clean,
                          replace_multiple)
from six import string_types


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

    _TRUE_VALS = ['t', 'true', 'T', 'True', '1', 'y', 'Y']
    _FALSE_VALS = ['f', 'false', 'F', 'False', '0', 'n', 'N']
    _EMPTY_VALS = ['nodata']

    def __init__(self, printer, require_source=False, **kwargs):
        """Initialize."""
        import pickle

        self._path = os.path.dirname(os.path.realpath(__file__))
        self._inflect = inflect.engine()
        self._printer = printer
        self._require_source = require_source

        self._estrs = [
            'err', '_err', 'err_', 'ERR', 'e_', '_e', '(err)', 'error',
            'uncertainty', 'sigma']
        self._emagstrs = self._estrs + [
            'magnitude error', 'e mag', 'e magnitude', 'dmag',
            'mag err', 'magerr', 'mag error']
        self._ecntstrs = self._estrs + [
            'flux error', 'e flux', 'e counts', 'count err', 'flux err',
            'countrate error', 'countrate err', 'e_flux']
        self._band_names = [
            'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'K_s', "Ks", "K'", 'u',
            'g', 'r', 'i', 'z', 'y', 'W1', 'W2', 'M2', "u'", "g'", "r'", "i'",
            "z'", 'C', 'Y', 'Open'
        ]
        ebands = [a + b for a, b in chain(
            product(self._ecntstrs, self._band_names),
            product(self._band_names, self._estrs))]
        self._emagstrs += ebands
        self._ecntstrs += ebands
        key_cache_path = os.path.join(
            self._path, 'cache', 'key_cache_{}.pickle'.format(
                get_mosfit_hash()))
        hks_loaded = False
        if os.path.isfile(key_cache_path):
            try:
                self._header_keys = pickle.load(open(key_cache_path, 'rb'))
                hks_loaded = True
            except Exception:
                printer.message('bad_header_pickle', warning=True)
                if hasattr(self, '_header_keys'):
                    del(self._header_keys)
        if not hks_loaded:
            self._header_keys = OrderedDict((
                (PHOTOMETRY.TIME, [
                    'time', 'mjd', ('jd', 'jd'), ('julian date', 'jd'),
                    ('date', 'yyyy-mm-dd'), 'day', (
                        'kiloseconds', 'kiloseconds')]),
                (PHOTOMETRY.SYSTEM, ['system', 'magsys', 'magnitude system']),
                (PHOTOMETRY.MAGNITUDE, [
                 'vega mag', 'ab mag', 'mag', 'magnitude']),
                (PHOTOMETRY.E_MAGNITUDE, self._emagstrs),
                (PHOTOMETRY.TELESCOPE, ['tel', 'telescope']),
                (PHOTOMETRY.INSTRUMENT, ['inst', 'instrument']),
                (PHOTOMETRY.OBSERVER, ['observer']),
                (PHOTOMETRY.OBSERVATORY, ['observatory']),
                (PHOTOMETRY.BAND, ['passband', 'band', 'filter', 'filt',
                                   'flt']),
                (PHOTOMETRY.E_LOWER_MAGNITUDE, [a + ' ' + b for a, b in chain(
                    product(self._emagstrs, ['minus', 'lower']),
                    product(['minus', 'lower'], self._emagstrs))]),
                (PHOTOMETRY.E_UPPER_MAGNITUDE, [a + ' ' + b for a, b in chain(
                    product(self._emagstrs, ['plus', 'upper']),
                    product(['plus', 'upper'], self._emagstrs))]),
                (PHOTOMETRY.UPPER_LIMIT, [
                    'upper limit', 'upperlimit', 'l_mag', 'limit']),
                (PHOTOMETRY.COUNT_RATE, [
                    'count', 'counts', 'flux', 'count rate']),
                (PHOTOMETRY.E_COUNT_RATE, self._ecntstrs),
                (PHOTOMETRY.FLUX_DENSITY, ['flux density', 'fd', 'f_nu']),
                (PHOTOMETRY.E_FLUX_DENSITY, [
                    'e_flux_density', 'flux density error', 'e_fd',
                    'sigma_nu']),
                (PHOTOMETRY.U_FLUX_DENSITY, []),
                (PHOTOMETRY.ZERO_POINT, ['zero point', 'zp']),
                ('reference', ['reference', 'bibcode', 'source', 'origin']),
                (ENTRY.NAME, [
                    'event', 'transient', 'name', 'supernova', 'sne', 'id',
                    'identifier', 'object']),
                (ENTRY.REDSHIFT, ['redshift']),
                (ENTRY.LUM_DIST, [
                    'lumdist', 'luminosity distance', 'distance']),
                (ENTRY.COMOVING_DIST, ['comoving distance']),
                (ENTRY.RA, ['ra', 'right ascension', 'right_ascension']),
                (ENTRY.DEC, ['dec', 'declination']),
                (ENTRY.EBV, ['ebv', 'extinction']),
                # At the moment transient-specific keys are not in astrocats.
                ('claimedtype', [
                    'claimedtype', 'type', 'claimed_type', 'claimed type'])
            ))

        self._critical_keys = [
            PHOTOMETRY.TIME, PHOTOMETRY.MAGNITUDE, PHOTOMETRY.COUNT_RATE,
            PHOTOMETRY.FLUX_DENSITY,
            PHOTOMETRY.BAND, PHOTOMETRY.E_COUNT_RATE,
            PHOTOMETRY.E_FLUX_DENSITY, PHOTOMETRY.ZERO_POINT]
        self._helpful_keys = [
            PHOTOMETRY.E_MAGNITUDE, PHOTOMETRY.INSTRUMENT,
            PHOTOMETRY.TELESCOPE]
        self._optional_keys = [
            PHOTOMETRY.ZERO_POINT, PHOTOMETRY.E_MAGNITUDE,
            PHOTOMETRY.U_FLUX_DENSITY]
        self._mc_keys = [
            PHOTOMETRY.MAGNITUDE, PHOTOMETRY.COUNT_RATE,
            PHOTOMETRY.FLUX_DENSITY]
        self._dep_keys = [
            PHOTOMETRY.E_MAGNITUDE, PHOTOMETRY.E_COUNT_RATE,
            PHOTOMETRY.E_FLUX_DENSITY, PHOTOMETRY.U_FLUX_DENSITY,
            PHOTOMETRY.BAND]
        self._purge_non_numeric_keys = [
            PHOTOMETRY.E_MAGNITUDE, PHOTOMETRY.E_LOWER_MAGNITUDE,
            PHOTOMETRY.E_UPPER_MAGNITUDE, PHOTOMETRY.E_COUNT_RATE,
            PHOTOMETRY.E_LOWER_COUNT_RATE, PHOTOMETRY.E_UPPER_COUNT_RATE,
            PHOTOMETRY.E_FLUX, PHOTOMETRY.E_LOWER_FLUX,
            PHOTOMETRY.E_UPPER_FLUX, PHOTOMETRY.E_UNABSORBED_FLUX,
            PHOTOMETRY.E_LOWER_UNABSORBED_FLUX,
            PHOTOMETRY.E_UPPER_UNABSORBED_FLUX]
        self._positive_keys = [
            PHOTOMETRY.MAGNITUDE] + self._purge_non_numeric_keys
        self._bool_keys = [PHOTOMETRY.UPPER_LIMIT]
        self._specify_keys = [
            PHOTOMETRY.BAND, PHOTOMETRY.INSTRUMENT, PHOTOMETRY.TELESCOPE]
        self._entry_keys = [
            ENTRY.COMOVING_DIST, ENTRY.REDSHIFT, ENTRY.LUM_DIST,
            ENTRY.RA, ENTRY.DEC, ENTRY.EBV, 'claimedtype']
        self._use_mc = False
        self._month_rep = re.compile(
            r'\b(' + '|'.join(self._MONTH_IDS.keys()) + r')\b')
        self._converted = []

        if not hks_loaded:
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
            pickle.dump(self._header_keys, open(key_cache_path, 'wb'))

    def generate_event_list(self, event_list):
        """Generate a list of events and/or convert events to JSON format."""
        prt = self._printer
        cidict = OrderedDict()
        intro_shown = False
        check_all_files = None
        shared_sources = []

        new_event_list = []
        previous_file = None
        for event in event_list:
            rsource = {SOURCE.NAME: self._DEFAULT_SOURCE}
            use_self_source = None
            new_events = []
            toffset = Decimal('0')
            if ('.' in event and os.path.isfile(event) and
                    not event.endswith('.json')):
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
                    # Count to try and determine delimiter.
                    delims = [' ', '\t', ',', ';', '|', '&']
                    delimnames = [
                        'Space: ` `', 'Tab: `\t`', 'Comma: `,`',
                        'Semi-colon: `;`', 'Bar: `|`', 'Ampersand: `&`']
                    delim = None
                    delimcounts = [re.sub(
                        re.escape(y) + '+', y, re.sub(
                            ' ?[' + ''.join([re.escape(
                                x) for x in delims if x != y]) + ']' +
                            ' ?', '', ftxt)).count(
                                y) for y in delims]
                    maxdelimcount = max(delimcounts)
                    # Make sure at least one delimeter per line.
                    maxdelimavg = delimcounts[delimcounts.index(
                        maxdelimcount)] / len(ftxt.splitlines())
                    if maxdelimavg >= 1.0:
                        delim = delims[delimcounts.index(maxdelimcount)]
                    # If two delimiter options are close in count, ask user.
                    for i, x in enumerate(delimcounts):
                        if x > 0.5 * maxdelimcount and delims[i] != delim:
                            delim = None
                    if delim is None and maxdelimavg >= 1.0:
                        odelims = list(np.array(delimnames)[
                            np.array(delimcounts) > 0])
                        dchoice = prt.prompt(
                            'delim', kind='option', options=odelims,
                            none_string=prt.text('no_delimiter'))
                        if is_number(dchoice):
                            delim = delims[dchoice - 1]
                    if delim is not None:
                        ad = list(delims)
                        ad.remove(delim)
                        ad = ''.join(ad)

                    fsplit = ftxt.splitlines()

                # If none of the rows contain numeric data, the file
                # is likely a list of transient names.
                flines = list(fsplit)

                if (len(flines) and
                    (not any(any([is_datum(x.strip()) or x == ''
                                  for x in (
                                      y.split(delim) if delim is not None else
                                      listify(y))])
                             for y in flines) or
                     len(flines) == 1)):
                    new_events = [
                        it.strip() for s in flines for it in (
                            s.split(delim) if delim is not None else
                            listify(s))]
                    new_event_list.extend(new_events)
                    continue

                if delim is None:
                    raise ValueError(prt.text('delimiter_not_found'))

                if not intro_shown:
                    prt.message('converter_info')
                    intro_shown = True

                prt.message('converting_to_json', [event])

                if table is None:
                    # See if we need to append blank errors to upper limits.
                    tsplit = [
                        replace_multiple(x, ['$', '\\pm', '±', '-or+'], delim)
                        .strip(ad + '()# ').replace('′', "'")
                        for x in fsplit]

                    append_missing_errs = False
                    for fl in tsplit:
                        dfl = list(csv.reader([fl], delimiter=delim))[0]
                        if any([is_number(x.strip('(<>≤≥'))
                                for x in dfl]) and any([
                                any([y in x for y in [
                                    '(', '<', '>', '≥', '≤']])
                                for x in dfl]):
                            append_missing_errs = True
                            break

                    fsplit = [
                        replace_multiple(x, ['$', '\\pm', '±', '-or+'], delim)
                        .replace('(', delim + '(')
                        .strip(ad + '()# ').replace('′', "'")
                        for x in fsplit]
                    flines = []
                    for fs in fsplit:
                        # Replace repeated spaces if fixed-width
                        if delim in [' ']:
                            fsn = re.sub(r'(\s)\1+', r'\1', fs)
                        else:
                            fsn = fs
                        flines.append(list(
                            csv.reader([fsn], delimiter=delim))[0])

                    flines = [[
                        x.strip(ad + '#$()\\')
                        for x in y] for y in flines]

                    # Find band columns if they exist and insert error columns
                    # if they don't exist.
                    for fi, fl in enumerate(list(flines)):
                        flcopy = list(fl)
                        offset = 0
                        if not any([is_datum(x) for x in fl]):
                            for fci, fc in enumerate(fl):
                                if (fc in self._band_names and
                                    (fci == len(fl) - 1 or
                                     fl[fci + 1] not in self._emagstrs)):
                                    flcopy.insert(fci + 1 + offset, 'e_' + fc)
                                    offset += 1
                        flines[fi] = flcopy

                    # Append blank errors to upper limits.
                    if append_missing_errs:
                        # Find the most frequent column count. These are
                        # probably the tables we wish to read.
                        flens = [len(x) for x in flines]
                        ncols = Counter(flens).most_common(1)[0][0]

                        flines = [[x for y in [
                            ([z, '-'] if (any([w in z for w in [
                                '<', '>', '≤', '≥']]) or
                                z in self._EMPTY_VALS) else [z])
                            for z in fl] for x in y] if len(
                                fl) != ncols else fl for fl in flines]

                    # Find the most frequent column count. These are probably
                    # the tables we wish to read.
                    flens = [len(x) for x in flines]
                    ncols = Counter(flens).most_common(1)[0][0]

                    newlines = []
                    potential_name = None
                    for fi, fl in enumerate(flines):
                        if (len(fl) and flens[fi] == 1 and
                            fi < len(flines) - 1 and
                                flens[fi + 1] == ncols and not len(newlines)):
                            potential_name = fl[0]
                        if flens[fi] == ncols:
                            if potential_name is not None and any(
                                    [is_datum(x) for x in fl]):
                                newlines.append([potential_name] + list(fl))
                            else:
                                newlines.append(list(fl))
                    flines = newlines
                    for fi, fl in enumerate(flines):
                        if len(fl) == ncols and potential_name is not None:
                            if not any([is_datum(x) for x in fl]):
                                flines[fi] = ['name'] + list(fl)

                # If last row is numeric, then likely this is a file with
                # transient data.
                if (len(flines) > 1 and
                        any([is_datum(x) for x in flines[-1]])):

                    # Check that each row has the same number of columns.
                    if len(set([len(x) for x in flines])) > 1:
                        print(set([len(x) for x in flines]))
                        raise ValueError(
                            'Number of columns in each row not '
                            'consistent!')

                    if len(cidict) and len(
                            new_event_list) and check_all_files is None:
                        reps = [previous_file] if previous_file else [''.join(
                            new_event_list[-1].split('.')[:-1])]
                        check_all_files = not prt.prompt(
                            'check_all_files', reps=reps, kind='bool')

                    if check_all_files or check_all_files is None:
                        shared_sources = []

                    if len(cidict) and len(new_event_list) and check_all_files:
                        msg = ('is_file_same' if
                               previous_file else 'is_event_same')
                        reps = [previous_file] if previous_file else [''.join(
                            new_event_list[-1].split('.')[:-1])]
                        is_same = prt.prompt(msg, reps=reps, kind='bool')
                        if not is_same:
                            cidict = OrderedDict()

                    # If the first row has no numbers it is likely a header.
                    if not len(cidict):
                        self.assign_columns(cidict, flines)

                    perms = 1
                    for key in cidict:
                        if isinstance(cidict[key], list) and not isinstance(
                                cidict[key], string_types):
                            if cidict[key][0] != 'j':
                                perms = len(cidict[key])

                    # Get event name (if single event) or list of names from
                    # table.
                    event_names = []
                    if ENTRY.NAME in cidict:
                        for fi, fl in enumerate(flines):
                            flines[fi][cidict[ENTRY.NAME]] = name_clean(
                                fl[cidict[ENTRY.NAME]])
                        event_names = list(sorted(set([
                            x[cidict[ENTRY.NAME]] for x in flines[
                                self._first_data:]])))
                        new_events = [x + '.json' for x in event_names]
                    else:
                        new_event_name = '.'.join(event.split(
                            '.')[:-1]).split('/')[-1]
                        if check_all_files or check_all_files is None:
                            is_name = prt.prompt(
                                'is_event_name', reps=[new_event_name],
                                kind='bool', default='y')
                            if not is_name:
                                new_event_name = ''
                                while new_event_name.strip() == '':
                                    new_event_name = prt.prompt(
                                        'enter_name', kind='string')
                        event_names.append(new_event_name)
                        new_events = [new_event_name + '.json']

                    # Create a new event, populate the photometry, and dump
                    # to a JSON file in the run directory.
                    entries = OrderedDict([(x, Entry(name=x))
                                           for x in event_names])

                    # Clean up the data a bit now that we know the column
                    # identities.

                    # Strip common prefixes/suffixes from band names
                    if PHOTOMETRY.BAND in cidict:
                        bi = cidict[PHOTOMETRY.BAND]
                        for d in [True, False]:
                            if not isinstance(bi, (int, np.integer)):
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
                            if len(strip_cols) == llen:
                                break
                            for ri in range(len(flines[self._first_data:])):
                                flines[self._first_data + ri][bi] = ''.join(
                                    [c for i, c in enumerate(flines[
                                        self._first_data + ri][bi])
                                     if (i if d else i - len(flines[
                                         self._first_data + ri][bi])) not in
                                     strip_cols])

                    if (PHOTOMETRY.TIME in cidict and
                            (not isinstance(cidict[PHOTOMETRY.TIME], list) or
                             len(cidict[PHOTOMETRY.TIME]) <= 2)):
                        bi = cidict[PHOTOMETRY.TIME]

                        bistr = bi[0].upper() if isinstance(
                            bi, list) and not isinstance(
                                bi, string_types) else 'MJD'

                        if isinstance(bi, list) and not isinstance(
                            bi, string_types) and isinstance(
                                bi[0], string_types):
                            bi = bi[-1]

                        mmtimes = None
                        try:
                            mmtimes = [float(x[bi])
                                       for x in flines[self._first_data:]]
                            mintime, maxtime = min(mmtimes), max(mmtimes)
                        except Exception:
                            pass

                        if not mmtimes:
                            pass
                        elif (bistr == 'MJD' and mintime < 10000 or
                              bistr == 'JD' and mintime < 2410000 or
                                bistr in ['KILOSECONDS', 'SECONDS']):
                            while True:
                                pstr = ('s_offset' if bistr in [
                                    'KILOSECONDS', 'SECONDS'] else
                                    'small_time_offset')
                                try:
                                    response = prt.prompt(pstr, [
                                        bistr for x in range(3)],
                                        kind='string')
                                    if response is not None:
                                        toffset = Decimal(response)
                                    break
                                except Exception:
                                    pass
                        elif maxtime > 90000 and bistr != 'JD':
                            isjd = prt.prompt(
                                'large_time_offset',
                                kind='bool', default='y')
                            if isjd:
                                toffset = Decimal('-2400000.5')

                    for row in flines[self._first_data:]:
                        photodict = OrderedDict()
                        rname = (row[cidict[ENTRY.NAME]]
                                 if ENTRY.NAME in cidict else event_names[0])
                        for pi in range(perms):
                            sources = set()
                            for key in cidict:
                                if key in self._bool_keys:
                                    rval = row[cidict[key]]

                                    if rval in self._FALSE_VALS:
                                        rval = False
                                    elif rval in self._TRUE_VALS:
                                        rval = True

                                    if type(rval) != 'bool':
                                        try:
                                            rval = bool(rval)
                                        except Exception:
                                            pass

                                    if type(rval) != 'bool':
                                        try:
                                            rval = bool(float(rval))
                                        except Exception:
                                            rval = True

                                    if not rval:
                                        continue
                                    if rval:
                                        row[cidict[key]] = rval
                                elif key == 'reference':
                                    if (isinstance(row[cidict[key]],
                                                   string_types)):
                                        srcdict = OrderedDict()
                                        if is_bibcode(row[cidict[key]]):
                                            srcdict[SOURCE.BIBCODE] = row[
                                                cidict[key]]
                                        else:
                                            srcdict[SOURCE.NAME] = row[
                                                cidict[key]]
                                        new_src = entries[rname].add_source(
                                            **srcdict)
                                        sources.update(new_src)
                                        # row[cidict[key]] = new_src
                                elif key == ENTRY.NAME:
                                    continue
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
                                    elif cidict[key][0].upper() == 'JD':
                                        photodict[key] = str(
                                            jd_to_mjd(Decimal(tval[-1])))
                                    elif cidict[key][0].upper(
                                    ) == 'KILOSECONDS':
                                        photodict[key] = str(Decimal(
                                            KS_DAYS) * Decimal(tval[-1]))
                                    elif cidict[key][0].upper(
                                    ) == 'YYYY-MM-DD':
                                        photodict[key] = str(
                                            astrotime(
                                                tval[-1].replace('/', '-'),
                                                format='isot').mjd)
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
                                else:
                                    photodict[PHOTOMETRY.ZERO_POINT] = (
                                        row[cidict[PHOTOMETRY.ZERO_POINT][pi]]
                                        if isinstance(cidict[
                                            PHOTOMETRY.ZERO_POINT], list) else
                                        row[cidict[PHOTOMETRY.ZERO_POINT]])
                                zpp = photodict[PHOTOMETRY.ZERO_POINT]
                                cc = (
                                    row[cidict[PHOTOMETRY.COUNT_RATE][pi]] if
                                    isinstance(cidict[
                                        PHOTOMETRY.COUNT_RATE], list) else
                                    row[cidict[PHOTOMETRY.COUNT_RATE]])
                                ecc = (
                                    row[cidict[PHOTOMETRY.E_COUNT_RATE][pi]] if
                                    isinstance(cidict[
                                        PHOTOMETRY.E_COUNT_RATE], list) else
                                    row[cidict[PHOTOMETRY.E_COUNT_RATE]])
                                if '<' in cc:
                                    set_pd_mag_from_counts(
                                        photodict, ec=cc.strip('<'), zp=zpp)
                                else:
                                    set_pd_mag_from_counts(
                                        photodict, c=cc, ec=ecc, zp=zpp)
                            elif self._data_type == 3:
                                photodict[
                                    PHOTOMETRY.U_FLUX_DENSITY] = self._ufd
                                if PHOTOMETRY.U_FLUX_DENSITY in cidict:
                                    photodict[PHOTOMETRY.U_FLUX_DENSITY] = (
                                        row[cidict[
                                            PHOTOMETRY.U_FLUX_DENSITY][pi]]
                                        if isinstance(cidict[
                                            PHOTOMETRY.
                                            U_FLUX_DENSITY], list) else
                                        row[cidict[PHOTOMETRY.U_FLUX_DENSITY]])
                                if photodict[
                                        PHOTOMETRY.U_FLUX_DENSITY] == '':
                                    photodict[
                                        PHOTOMETRY.U_FLUX_DENSITY] = 'µJy'
                                fd = (
                                    row[cidict[PHOTOMETRY.FLUX_DENSITY][pi]] if
                                    isinstance(cidict[
                                        PHOTOMETRY.FLUX_DENSITY], list) else
                                    row[cidict[PHOTOMETRY.FLUX_DENSITY]])
                                efd = (
                                    row[cidict[
                                        PHOTOMETRY.E_FLUX_DENSITY][pi]] if
                                    isinstance(cidict[
                                        PHOTOMETRY.E_FLUX_DENSITY], list) else
                                    row[cidict[PHOTOMETRY.E_FLUX_DENSITY]])

                                mult = Decimal('1')
                                ufd = photodict[PHOTOMETRY.U_FLUX_DENSITY]
                                if ufd.lower() in [
                                        'mjy', 'millijy', 'millijansky']:
                                    mult = Decimal('1e3')
                                elif ufd.lower() in ['jy', 'jansky']:
                                    mult = Decimal('1e6')

                                if '<' in fd:
                                    set_pd_mag_from_flux_density(
                                        photodict, efd=str(
                                            Decimal(fd.strip('<')) * mult))
                                else:
                                    set_pd_mag_from_flux_density(
                                        photodict, fd=Decimal(fd) * mult,
                                        efd=Decimal(efd) * mult)
                            if not len(sources) and not len(shared_sources):
                                if use_self_source is None:
                                    sopts = [
                                        ('Bibcode', 'b'), ('DOI', 'd'),
                                        ('ArXiv ID', 'a'), ('Last name', 'l')]
                                    if self._require_source:
                                        sel_str = 'must_select_source'
                                    else:
                                        sel_str = 'select_source'
                                    text = prt.text(sel_str)
                                    skind = prt.prompt(
                                        text, kind='option',
                                        options=sopts, default='b',
                                        none_string=(
                                            None if self._require_source else
                                            ('None of the above, '
                                             'tag MOSFiT as source')))
                                    if skind == 'b':
                                        rsource = OrderedDict()
                                        bibcode = ''

                                        while len(bibcode) != 19:
                                            bibcode = prt.prompt(
                                                'bibcode',
                                                kind='string',
                                                allow_blank=False
                                            )
                                            bibcode = bibcode.strip()
                                            if not is_bibcode(bibcode):
                                                bibcode = ''
                                        rsource[
                                            SOURCE.BIBCODE] = bibcode
                                        use_self_source = False
                                    elif skind == 'd':
                                        rsource = OrderedDict()
                                        doi = prt.prompt(
                                            'doi', kind='string',
                                            allow_blank=False)
                                        rsource[SOURCE.DOI] = doi.strip()
                                        use_self_source = False
                                    elif skind == 'a':
                                        rsource = OrderedDict()
                                        arxiv = prt.prompt(
                                            'arxiv', kind='string',
                                            allow_blank=False)
                                        rsource[SOURCE.ARXIVID] = arxiv.strip()
                                        use_self_source = False
                                    elif skind == 'l':
                                        rsource = OrderedDict()
                                        last_name = prt.prompt(
                                            'last_name', kind='string',
                                            allow_blank=False
                                        )
                                        rsource[
                                            SOURCE.NAME] = (
                                                last_name.strip().title() +
                                                ' et al., in preparation')
                                        use_self_source = False
                                    elif skind == 'n':
                                        use_self_source = True

                                    shared_sources.append(rsource)

                            if len(sources) or len(shared_sources):
                                src_list = list(sources)
                                for src in shared_sources:
                                    src_list.append(entries[
                                        rname].add_source(**src))
                                photodict[PHOTOMETRY.SOURCE] = ','.join(
                                    src_list)

                            if any([x in photodict.get(
                                    PHOTOMETRY.MAGNITUDE, '')
                                    for x in ['<', '>', '≤', '≥']]):
                                photodict[PHOTOMETRY.UPPER_LIMIT] = True
                                photodict[
                                    PHOTOMETRY.MAGNITUDE] = photodict[
                                        PHOTOMETRY.MAGNITUDE].strip('<>≤≥')

                            if '<' in photodict.get(PHOTOMETRY.COUNT_RATE, ''):
                                photodict[PHOTOMETRY.UPPER_LIMIT] = True
                                photodict[
                                    PHOTOMETRY.COUNT_RATE] = photodict[
                                        PHOTOMETRY.COUNT_RATE].strip('<')
                                if PHOTOMETRY.E_COUNT_RATE in photodict:
                                    del(photodict[PHOTOMETRY.E_COUNT_RATE])

                            if '<' in photodict.get(
                                    PHOTOMETRY.FLUX_DENSITY, ''):
                                photodict[PHOTOMETRY.UPPER_LIMIT] = True
                                photodict[
                                    PHOTOMETRY.FLUX_DENSITY] = photodict[
                                        PHOTOMETRY.FLUX_DENSITY].strip('<')
                                if PHOTOMETRY.E_FLUX_DENSITY in photodict:
                                    del(photodict[PHOTOMETRY.E_FLUX_DENSITY])

                            # Apply offset time if set.
                            if (PHOTOMETRY.TIME in photodict and
                                    toffset != Decimal('0')):
                                photodict[PHOTOMETRY.TIME] = str(
                                    Decimal(photodict[PHOTOMETRY.TIME]) +
                                    toffset)

                            # Remove some attributes if not numeric.
                            for attr in self._purge_non_numeric_keys:
                                if not is_number(photodict.get(attr, 0)):
                                    del(photodict[attr])

                            # Skip entries for which key values are not
                            # expected type.
                            if not all([
                                is_number(photodict.get(x, 0))
                                for x in photodict.keys() if
                                (PHOTOMETRY.get_key_by_name(x).type ==
                                 KEY_TYPES.NUMERIC)]):
                                continue

                            # Remove some attributes if not positive.
                            for attr in self._positive_keys:
                                if float(photodict.get(attr, 1)) <= 0:
                                    del(photodict[attr])

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
                            entries[rname].add_photometry(
                                **photodict)

                            # Add other entry keys.
                            for key in self._entry_keys:
                                if key not in cidict:
                                    continue
                                qdict = OrderedDict()
                                qdict[QUANTITY.VALUE] = row[
                                    cidict[key]]
                                if len(sources) or len(shared_sources):
                                    qdict[QUANTITY.SOURCE] = ','.join(
                                        src_list)
                                entries[rname].add_quantity(key, **qdict)

                    merge_with_existing = None
                    for ei, entry in enumerate(entries):
                        entries[entry].sanitize()
                        if os.path.isfile(new_events[ei]):
                            if merge_with_existing is None:
                                merge_with_existing = prt.prompt(
                                    'merge_with_existing', default='y')
                            if merge_with_existing:
                                existing = Entry.init_from_file(
                                    catalog=None,
                                    name=event_names[ei],
                                    path=new_events[ei],
                                    merge=False,
                                    pop_schema=False,
                                    ignore_keys=[ENTRY.MODELS],
                                    compare_to_existing=False)
                                Catalog().copy_entry_to_entry(
                                    existing, entries[entry])

                        oentry = entries[entry]._ordered(entries[entry])
                        entabbed_json_dump(
                            {entry: oentry}, open(new_events[ei], 'w'),
                            separators=(',', ':'))

                    self._converted.extend([
                        [event_names[x], new_events[x]]
                        for x in range(len(event_names))])

                new_event_list.extend(new_events)
                previous_file = event
            else:
                new_event_list.append(event)

        return new_event_list

    def assign_columns(self, cidict, flines):
        """Assign columns based on header."""
        used_cis = OrderedDict()
        akeys = list(self._critical_keys) + list(self._helpful_keys)
        dkeys = list(self._dep_keys)
        prt = self._printer
        mpatt = re.compile('mag', re.IGNORECASE)

        for fi, fl in enumerate(flines):
            if not any([is_datum(x) for x in fl]):
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

        # Look for columns that are band names if no mag/counts/flux dens
        # column was found.
        if (not any([x in cidict for x in [
            PHOTOMETRY.MAGNITUDE, PHOTOMETRY.COUNT_RATE,
                PHOTOMETRY.FLUX_DENSITY]])):
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
                if not any([is_datum(x) for x in fl]):
                    # Try to associate column names with common header keys.
                    for ci, col in enumerate(fl):
                        if ci in used_cis:
                            continue
                        ccol = mpatt.sub('', col)
                        if ccol in self._band_names:
                            cidict.setdefault(key, []).append(ci)
                            used_cis[ci] = key
                            cidict.setdefault(bkey, []).append(ccol)
                        elif ccol in self._emagstrs:
                            cidict.setdefault(ekey, []).append(ci)
                            used_cis[ci] = ekey

        # See which keys we collected. If we are missing any critical keys, ask
        # the user which column they are.

        # First ask the user if this data is in magnitudes or in counts.
        self._data_type = 1
        if (PHOTOMETRY.MAGNITUDE in cidict and
                PHOTOMETRY.COUNT_RATE not in cidict and
                PHOTOMETRY.FLUX_DENSITY not in cidict):
            self._data_type = 1
        elif (PHOTOMETRY.MAGNITUDE not in cidict and
              PHOTOMETRY.COUNT_RATE in cidict and
              PHOTOMETRY.FLUX_DENSITY not in cidict):
            self._data_type = 2
        elif (PHOTOMETRY.MAGNITUDE not in cidict and
              PHOTOMETRY.COUNT_RATE not in cidict and
              PHOTOMETRY.FLUX_DENSITY in cidict):
            self._data_type = 3
        else:
            self._data_type = prt.prompt(
                'counts_mags_fds', kind='option',
                options=['Magnitudes', 'Counts (per second)',
                         'Flux Densities (Jansky)'],
                none_string='No Photometry', default='1')
        if self._data_type in [1, 3, 'n']:
            akeys.remove(PHOTOMETRY.COUNT_RATE)
            akeys.remove(PHOTOMETRY.E_COUNT_RATE)
            akeys.remove(PHOTOMETRY.ZERO_POINT)
            dkeys.remove(PHOTOMETRY.E_COUNT_RATE)
        if self._data_type in [2, 3, 'n']:
            akeys.remove(PHOTOMETRY.MAGNITUDE)
            akeys.remove(PHOTOMETRY.E_MAGNITUDE)
            dkeys.remove(PHOTOMETRY.E_MAGNITUDE)
        if self._data_type in [1, 2, 'n']:
            akeys.remove(PHOTOMETRY.FLUX_DENSITY)
            akeys.remove(PHOTOMETRY.E_FLUX_DENSITY)
            if (PHOTOMETRY.E_LOWER_FLUX_DENSITY in cidict and
                    PHOTOMETRY.E_UPPER_FLUX_DENSITY in cidict):
                akeys.remove(PHOTOMETRY.E_FLUX_DENSITY)
            dkeys.remove(PHOTOMETRY.E_FLUX_DENSITY)
            dkeys.remove(PHOTOMETRY.U_FLUX_DENSITY)
        if self._data_type == 'n':
            akeys.remove(PHOTOMETRY.TIME)
            akeys.remove(PHOTOMETRY.BAND)
            akeys.remove(PHOTOMETRY.INSTRUMENT)
            akeys.remove(PHOTOMETRY.TELESCOPE)

        # Make sure `E_` keys always appear after the actual measurements.
        if (PHOTOMETRY.MAGNITUDE in akeys and
                PHOTOMETRY.E_MAGNITUDE in akeys):
            akeys.remove(PHOTOMETRY.E_MAGNITUDE)
            akeys.insert(
                akeys.index(PHOTOMETRY.MAGNITUDE) + 1,
                PHOTOMETRY.E_MAGNITUDE)
        # Remove regular `E_` keys if both `E_LOWER_`/`E_UPPER_` exist.
        if (PHOTOMETRY.E_LOWER_MAGNITUDE in cidict and
                PHOTOMETRY.E_UPPER_MAGNITUDE in cidict):
            akeys.remove(PHOTOMETRY.E_MAGNITUDE)

        columns = np.array(flines[self._first_data:]).T.tolist()
        colstrs = np.array([
            ', '.join(x[:5]) + ', ...' for x in columns])
        colinds = np.setdiff1d(np.arange(
            len(colstrs)), list([x[-1] if (
                isinstance(x, list) and not isinstance(
                    x, string_types)) else x for x in cidict.values()]))
        ignore = prt.message('ignore_column', prt=False)
        specify = prt.message('specify_column', prt=False)
        for key in akeys:
            selected_cols = [
                y for y in [a for b in [
                    listify(x) for x in list(cidict.values())] for a in b]
                if isinstance(y, (int, np.integer))]
            if key in cidict:
                continue
            if key in dkeys and self._use_mc:
                continue
            if key.type == KEY_TYPES.NUMERIC:
                lcolinds = [x for x in colinds
                            if any(is_datum(y) for y in columns[x]) and
                            x not in selected_cols]
            elif key.type == KEY_TYPES.TIME:
                lcolinds = [x for x in colinds
                            if any(is_date(y) or is_datum(y)
                                   for y in columns[x]) and
                            x not in selected_cols]
            elif key.type == KEY_TYPES.STRING:
                lcolinds = [x for x in colinds
                            if any(not is_datum(y) for y in columns[x]) and
                            x not in selected_cols]
            else:
                lcolinds = [x for x in colinds if x not in selected_cols]
            select = False
            selects = []
            while select is False:
                mc = 1
                if key in self._mc_keys:
                    pkey = self._inflect.plural(key)
                    text = prt.message(
                        'one_per_line', [key, pkey, pkey],
                        prt=False)
                    mc = prt.prompt(
                        text, kind='option', message=False,
                        none_string=None,
                        options=[
                            'One `{}` per row'.format(key),
                            'Multiple `{}` per row'.format(pkey)])
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
                                lcolinds = np.delete(lcolinds, jsel - 1)
                else:
                    self._use_mc = True
                    select = False
                    while select is not None and len(lcolinds):
                        text = prt.message(
                            'select_mc_column', [key], prt=False)
                        select = prt.prompt(
                            text, message=False,
                            kind='option', default='n',
                            none_string='No more `{}` columns.'.format(key),
                            options=colstrs[lcolinds].tolist())
                        if select is not None and select is not False:
                            selects.append(lcolinds[select - 1])
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
                    kdkeys = [key] + dkeys
                    allk = list(OrderedDict.fromkeys(kdkeys).keys())
                    for ki, k in enumerate(allk):
                        cidict[k] = [
                            colinds[s - 1] if isinstance(s, (
                                int, np.integer)) else s
                            for s in selects[ki::len(allk)]]
                    for s in selects:
                        if not isinstance(s, (int, np.integer)):
                            continue
                        colinds = np.delete(colinds, np.argwhere(
                            colinds == s - 1))
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

        self._ufd = None
        if self._data_type == 3 and PHOTOMETRY.U_FLUX_DENSITY not in cidict:
            while ((self._ufd.lower() if self._ufd is not None else None)
                   not in ['µjy', 'mjy', 'jy', 'microjy', 'millijy', 'jy',
                           'microjansky', 'millijansky', 'jansky', '']):
                self._ufd = prt.prompt('u_flux_density', kind='string')

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
            PHOTOMETRY.TELESCOPE not in cidict and
                PHOTOMETRY.MAGNITUDE in cidict):
            prt.message('instrument_recommended', warning=True)

    def get_converted(self):
        """Get a list of events that were converted from ASCII to JSON."""
        return self._converted
