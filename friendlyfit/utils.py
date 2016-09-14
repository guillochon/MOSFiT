"""Miscellaneous utility functions.
"""
from math import floor, log10


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
