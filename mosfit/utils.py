# -*- coding: UTF-8 -*-
"""Miscellaneous utility functions.
"""
from __future__ import print_function

import hashlib
import json
import re
import signal
import sys
from builtins import input
from collections import OrderedDict
from math import floor, log10
from textwrap import fill

import numpy as np

if sys.version_info[:2] < (3, 3):
    old_print = print

    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        file = kwargs.get('file', sys.stdout)
        if flush and file is not None:
            file.flush()


syst_syns = {'': 'Vega', 'SDSS': 'AB', 'Standard': 'Vega', 'Landolt': 'Vega'}


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
    elif kind == 'string':
        choices = ''
    else:
        raise ValueError('Unknown prompt kind.')

    prompt_txt = (text + choices).split('\n')
    for txt in prompt_txt[:-1]:
        ptxt = fill(txt, wrap_length, replace_whitespace=False)
        print(ptxt)
    user_input = input(
        fill(
            prompt_txt[-1], wrap_length, replace_whitespace=False) + " ")
    if kind == 'bool':
        return user_input in ["Y", "y", "Yes", "yes"]
    elif kind == 'select':
        if (is_integer(user_input) and
                int(user_input) in list(range(1, len(options) + 1))):
            return options[int(user_input) - 1]
        return False
    elif kind == 'string':
        return user_input


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def entabbed_json_dumps(string, **kwargs):
    if sys.version_info[:2] >= (3, 3):
        return json.dumps(
            string,
            indent='\t',
            separators=kwargs['separators'],
            ensure_ascii=False)
        return
    newstr = json.dumps(
        string,
        indent=4,
        separators=kwargs['separators'],
        ensure_ascii=False,
        encoding='utf8')
    newstr = re.sub(
        '\n +',
        lambda match: '\n' + '\t' * (len(match.group().strip('\n')) / 4),
        newstr)
    return newstr


def entabbed_json_dump(string, f, **kwargs):
    f.write(entabbed_json_dumps(string, **kwargs))


def calculate_WAIC(scores):
    """WAIC from Gelman
    http://www.stat.columbia.edu/~gelman/research/published/waic_understand3
    """
    fscores = [x for y in scores for x in y]
    # Technically needs to be multiplied by -2, but this makes score easily
    # relatable to maximum likelihood score.
    return np.mean(fscores) - np.var(fscores)


def flux_density_unit(unit):
    if unit == 'µJy':
        return 1.0 / (1.0e-6 * 1.0e-23)
    return 1.0


def frequency_unit(unit):
    if unit == 'GHz':
        return 1.0 / 1.0e9
    return 1.0


def get_model_hash(modeldict, ignore_keys=[]):
    """Return a unique hash for the given model
    """
    newdict = OrderedDict()
    for key in modeldict:
        if key not in ignore_keys:
            newdict[key] = modeldict[key]
    string_rep = json.dumps(newdict, sort_keys=True)
    return hashlib.sha512(string_rep.encode()).hexdigest()[:16]


def get_mosfit_hash(salt=''):
    import fnmatch
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))

    matches = []
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, '*.py'):
            matches.append(os.path.join(root, filename))

    matches = list(sorted(list(matches)))
    code_str = salt
    for match in matches:
        with open(match, 'r') as f:
            code_str += f.read()

    return hashlib.sha512(code_str.encode()).hexdigest()[:16]


def is_master():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True
