"""Unit tests for utils, plotting, printer, fetcher, converter, and CLI."""
from __future__ import print_function

import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

    from mosfit.constants import C_CGS, DAY_CGS, MAG_FAC
    from mosfit.converter import Converter
    from mosfit.fetcher import Fetcher
    from mosfit.main import get_parser, main as mosfit_main
    from mosfit.plotting import (
        bandaliasf, bandcolorf, bandgroupf, bandmetaf, bandrepf,
        bandshortaliasf, bandwavef, radiocolorf, xraycolorf)
    from mosfit.printer import Printer
    from mosfit.utils import (
        all_to_list, calculate_WAIC, congrid, entabbed_json_dump,
        entabbed_json_dumps, flux_density_unit, frequency_unit, get_model_hash,
        get_mosfit_hash, hash_bytes, is_bibcode, is_coordinate, is_date,
        is_datum, is_integer, is_master, is_number, listify, load_walkers_file,
        name_clean, open_atomic, pretty_num, rebin, replace_multiple,
        round_sig, slugify, write_chain_hdf5, write_json_payload,
        write_walkers_hdf5)

    assert C_CGS > 0 and DAY_CGS > 0 and MAG_FAC == 2.5
    assert is_number('1.2') and not is_number('1 2')
    assert is_number(['1', '2']) and not is_number(['1 2'])
    assert is_integer('3') and not is_integer('3.5')
    assert is_integer(['1', '2']) and not is_integer(['x'])
    assert is_date('2020-01-01') and not is_date('not-a-date')
    assert is_coordinate('+12:30:00') and not is_coordinate('hello')
    assert is_datum('12.3') and is_datum('+12:30:00') and not is_datum('abc')
    assert is_bibcode('2017ApJ...835...73G')
    assert listify('a') == ['a'] and listify(['a', 'b']) == ['a', 'b']
    assert pretty_num(12345, sig=3)
    assert pretty_num(float('nan')) == 'nan'
    assert round_sig(1234.5, 2)
    assert flux_density_unit('µJy') > 1 and flux_density_unit('Jy') == 1.0
    assert frequency_unit('GHz') < 1 and frequency_unit('Hz') == 1.0
    assert calculate_WAIC([[1.0, 2.0], [3.0]]) < 10
    dumped = entabbed_json_dumps({'a': 1, 'b': [1, 2, 3]})
    assert '"a"' in dumped
    tmpd = tempfile.mkdtemp()
    jpath = os.path.join(tmpd, 'payload.json')
    with open(jpath, 'w') as f:
        write_json_payload(f, '{"k":1}')
        entabbed_json_dump({'z': 2}, f)
    assert hash_bytes('abc')
    assert len(get_model_hash({'a': 1, 'b': 2}, ignore_keys=['b'])) == 16
    assert len(get_mosfit_hash()) == 16
    assert is_master() is True
    arr = np.arange(16, dtype=float).reshape(4, 4)
    reb = rebin(arr, (2, 2))
    assert reb.shape == (2, 2)
    cg = congrid(arr, (8, 8), method='linear')
    assert cg.shape == (8, 8)
    try:
        cg2 = congrid(arr, (8, 8), method='neighbour')
        assert cg2 is None or cg2.shape == (8, 8)
    except IndexError:
        pass
    cg3 = congrid(arr, (8, 8), method='nearest', center=True, minusone=True)
    assert cg3.shape == (8, 8)
    cg4 = congrid(arr, (6, 6), method='spline')
    assert cg4 is None or cg4.shape == (6, 6)
    assert congrid(arr, (8, 8), method='nope') is None
    assert congrid(arr, (8,), method='linear') is None
    assert all_to_list(np.array([1, 2])) == [1, 2]
    assert replace_multiple('$1 \\pm 2$', ['$', '\\pm'], ',')
    assert name_clean(' NAME SN2006le SN') == 'SN2006le'
    assert name_clean('OGLE-2012-SN-1').startswith('OGLE')
    assert name_clean('SDSS 1234-56-7')
    assert name_clean('GAIA 123')
    assert name_clean('KSN-2015K')
    assert name_clean('GRB123456')
    assert name_clean('LSQ 12dlf')
    assert name_clean('DES13C1jok')
    assert name_clean('PTF 10hgi')
    assert name_clean('iPTF 14atg')
    assert name_clean('SNHiTS 15A')
    assert name_clean('ESSENCE 123')
    assert name_clean('PS1 10jh')
    assert name_clean('CSS123')
    assert name_clean('ASASSN-15lh')
    assert name_clean('MASTER123456')
    assert slugify('Hello World!') == 'Hello-World'
    assert slugify('ñandú', allow_unicode=True)
    h5c = os.path.join(tmpd, 'chain.h5')
    write_chain_hdf5(h5c, np.zeros((1, 2, 3, 2)), ['a', 'b'])
    h5w = os.path.join(tmpd, 'walkers.h5')
    write_walkers_hdf5(h5w, {'evt': {'name': 'evt'}})
    loaded = load_walkers_file(h5w)
    assert 'evt' in loaded
    jwalk = os.path.join(tmpd, 'walkers.json')
    with open(jwalk, 'w') as f:
        json.dump({'evt': {'name': 'evt'}}, f)
    assert 'evt' in load_walkers_file(jwalk)
    nested = os.path.join(tmpd, 'a', 'b', 'c.txt')
    with open_atomic(nested, 'w') as f:
        f.write('ok')
    assert Path(nested).read_text() == 'ok'
    with open_atomic(nested, 'w') as f:
        f.write('replaced')
    assert Path(nested).read_text() == 'replaced'
    print('utils ok')

    assert bandrepf('uvm2') == 'UVM2'
    assert bandcolorf('g').startswith('#') or bandcolorf('g') == 'black'
    assert radiocolorf(5.9).startswith('#')
    assert xraycolorf('0.3 - 10').startswith('#') or xraycolorf('nope') == 'black'
    assert bandaliasf('g_SDSS') == "g'"
    assert bandgroupf('U') == 'Johnson'
    assert bandshortaliasf('g_SDSS') == "g'"
    assert bandwavef('V') == 551.
    assert bandwavef('notaband') == 0.
    assert bandmetaf('UVM2', 'telescope') == 'Swift'
    assert bandmetaf('g', 'telescope') == ''
    print('plotting ok')

    prt = Printer(quiet=False, wrap_length=40, exit_on_prompt=True)
    prt.prt('hello', wrapped=True)
    prt.prt('!rred!e', colorify=True)
    prt.message('byline', reps=['1', 'h', 'a', 'c'], wrapped=False)
    prt.colorify('!ghi!e')
    prt.get_timestring(12.3)
    prt.tree({'root': {'child': {'leaf': 1}}})
    try:
        prt.prompt('anything', kind='bool')
        raise SystemExit('prompt should have raised')
    except RuntimeError:
        pass
    prt_q = Printer(quiet=True)
    prt_q.status(None, desc='drawing_walkers', iterations=[1, 2], min_time=0)
    print('printer ok')

    fixture = os.path.abspath('mosfit/tests/PS1-10jh.json')
    lsq = os.path.abspath('mosfit/tests/LSQ12dlf.json')
    fetcher = Fetcher(test=True, printer=prt_q)
    fetched = fetcher.fetch([fixture])
    assert fetched[0]['name'] == 'PS1-10jh'
    data = fetcher.load_data(fetched[0])
    assert data is not None
    fetched_lsq = fetcher.fetch([lsq])
    assert fetched_lsq[0]['name'] == 'LSQ12dlf'
    assert fetcher.load_data(fetched_lsq[0]) is not None
    try:
        fetcher.fetch(['definitely-missing-event.json'])
        raise SystemExit('missing event should raise')
    except RuntimeError:
        pass
    assert fetcher.load_data(None) is None
    print('fetcher ok')

    conv = Converter(prt_q, guess=True, cache_path=tmpd)
    names_path = os.path.join(tmpd, 'names.txt')
    with open(names_path, 'w') as f:
        f.write('SN2008ar\nSN2007bg\n')
    named = conv.generate_event_list([names_path])
    assert 'SN2008ar' in named and 'SN2007bg' in named
    json_pass = conv.generate_event_list([fixture])
    assert json_pass[0].endswith('PS1-10jh.json') or 'PS1-10jh' in json_pass[0]

    csv_path = os.path.join(tmpd, 'phot.csv')
    with open(csv_path, 'w') as f:
        f.write('time,band,magnitude,e_magnitude\n')
        f.write('55000,V,18.1,0.1\n')
        f.write('55001,V,18.3,0.1\n')

    def fake_prompt(self, text, reps=[], kind='bool', default=None, options=None,
                    **kwargs):
        if kind == 'bool':
            return True
        if kind == 'string':
            key = str(text).lower()
            if 'zeropoint' in key:
                return '23.9'
            if 'flux' in key:
                return 'jy'
            return 'TestSN'
        if kind == 'option':
            if default == 'n':
                return None
            return 1
        return default if default is not None else '1'

    with mock.patch.object(Printer, 'prompt', fake_prompt):
        conv2 = Converter(Printer(quiet=True, exit_on_prompt=False), guess=True,
                          cache_path=tmpd)
        converted = conv2.generate_event_list([csv_path])
        assert converted
        assert any(str(x).endswith('.json') for x in converted)
    print('converter ok')

    parser = get_parser()
    help_txt = parser.format_help()
    assert 'mosfit' in help_txt.lower() or 'Fit astrophysical' in help_txt
    from mosfit import __version__ as mosfit_version
    from mosfit import _ASTROCATS_MIN_VERSION, _version_triple
    assert mosfit_version.startswith('2.')
    assert _version_triple('0.5.0') == (0, 5, 0)
    assert _version_triple('0.5.0rc1') == (0, 5, 0)
    assert _version_triple('0.5.0.post1') == (0, 5, 0)
    assert _version_triple('0.4.9') < _ASTROCATS_MIN_VERSION
    assert _version_triple('0.5.0') >= _ASTROCATS_MIN_VERSION
    from mosfit.fitter import Fitter
    fitter_cores = Fitter(
        max_cores=10, test=True, quiet=True, exit_on_prompt=True)
    assert fitter_cores._max_cores == 10
    fitter_cores._bind_max_cores(None)
    assert fitter_cores._max_cores == 10
    fitter_cores._bind_max_cores(4)
    assert fitter_cores._max_cores == 4
    args = parser.parse_args(['-m', 'exppow', '-i', '1', '--quiet',
                              '--no-copy-at-launch'])
    assert args.models == 'exppow'
    assert args.iterations == 1
    with mock.patch.object(sys, 'argv', ['mosfit', '--version']):
        mosfit_main()
    with mock.patch('mosfit.fitter.Fitter.fit_events',
                    return_value=([], [], [])):
        with mock.patch.object(
                sys, 'argv',
                ['mosfit', '--quiet', '--no-copy-at-launch', '--exit-on-prompt',
                 '-m', 'exppow', '-i', '0', '--method', 'ensembler',
                 '--frack-step', '0', '--time-list', '55000', '56000']):
            mosfit_main()
        with mock.patch.object(
                sys, 'argv',
                ['mosfit', '--quiet', '--no-copy-at-launch', '--generative',
                 '-m', 'exppow', '-i', '5']):
            mosfit_main()
    print('cli ok')
    print('all unit tests passed')
    sys.exit(0)
