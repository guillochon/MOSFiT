"""Miscellaneous utility functions.
"""
from __future__ import print_function

import json
import re
import signal
import sys
from builtins import input
from math import floor, log10
from textwrap import fill

if sys.version_info[:2] < (3, 3):
    old_print = print

    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        file = kwargs.get('file', sys.stdout)
        if flush and file is not None:
            file.flush()


syst_syns = {'': 'Vega', 'SDSS': 'AB', 'Standard': 'Vega'}


def get_url_file_handle(url, timeout=10):
    if sys.version_info[0] >= 3:
        from urllib.request import urlopen
    else:
        from urllib2 import urlopen
    return urlopen(url, timeout=timeout)


def is_number(s):
    if isinstance(s, bool):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_integer(s):
    if isinstance(s, list) and not isinstance(s, str):
        try:
            [int(x) for x in s]
            return True
        except ValueError:
            return False
    else:
        try:
            int(s)
            return True
        except ValueError:
            return False


def pretty_num(x, sig=4):
    return str('%g' % (round_sig(x, sig)))


def round_sig(x, sig=4):
    if x == 0.0:
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def listify(x):
    if not isinstance(x, list):
        return [x]
    return x


def print_inline(x, new_line=False):
    lines = x.split('\n')
    if not new_line:
        for line in lines:
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
    print(x, flush=True)


def print_wrapped(text, wrap_length=100):
    print(fill(text, wrap_length))


def prompt(text, wrap_length=100, kind='bool', options=None):
    if kind == 'bool':
        choices = ' (y/[n])'
    elif kind == 'select':
        choices = '\n' + '\n'.join([
            ' ' + str(i + 1) + '.  ' + options[i] for i in range(
                len(options))
        ] + [
            '[n]. None of the above, skip this event.\n'
            'Enter selection (' + ('1-' if len(options) > 1 else '') + str(
                len(options)) + '/[n]):'
        ])
    else:
        raise ValueError('Unknown prompt kind.')

    prompt_txt = (text + choices).split('\n')
    for txt in prompt_txt[:-1]:
        ptxt = fill(txt, wrap_length, replace_whitespace=False)
        print(ptxt)
    user_choice = input(
        fill(
            prompt_txt[-1], wrap_length, replace_whitespace=False) + " ")
    if kind == 'bool':
        return user_choice in ["Y", "y", "Yes", "yes"]
    elif kind == 'select':
        if (is_integer(user_choice) and
                int(user_choice) in list(range(1, len(options) + 1))):
            return options[int(user_choice) - 1]
        return False


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def entabbed_json_dump(string, f, **kwargs):
    if sys.version_info[:2] >= (3, 3):
        json.dump(
            string,
            f,
            indent='\t',
            separators=kwargs['separators'],
            ensure_ascii=False)
        return
    newstr = json.dumps(
        string, indent=4, separators=kwargs['separators'], ensure_ascii=False)
    newstr = re.sub(
        '\n +',
        lambda match: '\n' + '\t' * (len(match.group().strip('\n')) / 4),
        newstr)
    f.write(newstr)


def flux_density_unit(unit):
    if unit == 'µJy':
        return 1.0/(1.0e-6*1.0e-23)
    return 1.0


def frequency_unit(unit):
    if unit == 'GHz':
        return 1.0/1.0e9
    return 1.0


def is_master():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True
