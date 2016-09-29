"""Miscellaneous utility functions.
"""
from __future__ import print_function

import sys
from math import floor, log10

if sys.version_info[:2] < (3, 3):
    old_print = print

    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        file = kwargs.get('file', sys.stdout)
        if flush and file is not None:
            file.flush()


def is_number(s):
    try:
        float(s)
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
    if not new_line:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    print(x, flush=True)
