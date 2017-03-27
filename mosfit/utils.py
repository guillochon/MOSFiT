# -*- coding: UTF-8 -*-
"""Miscellaneous utility functions."""

import hashlib
import json
import re
import sys
from collections import OrderedDict
from math import floor, isnan, log10

import numpy as np
from six import string_types

syst_syns = {'': 'Vega', 'SDSS': 'AB', 'Standard': 'Vega', 'Landolt': 'Vega'}


def get_url_file_handle(url, timeout=10):
    """Get file handle from urllib request of URL."""
    if sys.version_info[0] >= 3:
        from urllib.request import urlopen
    else:
        from urllib2 import urlopen
    return urlopen(url, timeout=timeout)


def is_number(s):
    """Check if input is numeric."""
    if isinstance(s, bool):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_integer(s):
    """Check if input is an integer."""
    if isinstance(s, list) and not isinstance(s, string_types):
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
    """Convert number into string with specified significant digits."""
    if isnan(x) or not np.isfinite(x):
        return str(x)
    return str('%g' % (round_sig(x, sig)))


def round_sig(x, sig=4):
    """Round number with specified significant digits."""
    if x == 0.0:
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def listify(x):
    """Convert item to a `list` of items if it isn't already a `list`."""
    if not isinstance(x, list):
        return [x]
    return x


def entabbed_json_dumps(string, **kwargs):
    """Produce entabbed string for JSON output.

    This is necessary because Python 2 does not allow tabs to be used in its
    JSON dump(s) functions.
    """
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
    """Write `entabbed_json_dumps` output to file handle."""
    f.write(entabbed_json_dumps(string, **kwargs))


def calculate_WAIC(scores):
    """WAIC from Gelman.

    http://www.stat.columbia.edu/~gelman/research/published/waic_understand3
    """
    fscores = [x for y in scores for x in y]
    # Technically needs to be multiplied by -2, but this makes score easily
    # relatable to maximum likelihood score.
    return np.mean(fscores) - np.var(fscores)


def flux_density_unit(unit):
    """Return coeffiecent to convert µJy to Jy."""
    if unit == 'µJy':
        return 1.0 / (1.0e-6 * 1.0e-23)
    return 1.0


def frequency_unit(unit):
    """Return coeffiecent to convert GHz to Hz."""
    if unit == 'GHz':
        return 1.0 / 1.0e9
    return 1.0


def hash_bytes(input_string):
    """Return a hash bytestring.

    Necessary to have consistent behavior between Python 2 & 3.
    """
    if sys.version_info[0] < 3:
        return bytes(input_string)
    return input_string.encode()


def get_model_hash(modeldict, ignore_keys=[]):
    """Return a unique hash for the given model."""
    newdict = OrderedDict()
    for key in modeldict:
        if key not in ignore_keys:
            newdict[key] = modeldict[key]
    string_rep = json.dumps(newdict, sort_keys=True)

    return hashlib.sha512(hash_bytes(string_rep)).hexdigest()[:16]


def get_mosfit_hash(salt=''):
    """Return a unique hash for the MOSFiT code."""
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

    return hashlib.sha512(hash_bytes(code_str)).hexdigest()[:16]


def is_master():
    """Determine if we are the master process."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True
