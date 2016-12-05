"""Miscellaneous utility functions.
"""
from __future__ import print_function

import signal
import sys
from math import floor, log10
from textwrap import wrap

if sys.version_info[:2] < (3, 3):
    old_print = print

    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        file = kwargs.get('file', sys.stdout)
        if flush and file is not None:
            file.flush()


def is_number(s):
    if isinstance(s, bool):
        return False
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
    lines = x.split('\n')
    if not new_line:
        for line in lines:
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
    print(x, flush=True)


def print_wrapped(text, wrap_length=100):
    print('\n'.join(wrap(text, wrap_length)))


def prompt(text, wrap_length=100, kind='bool'):
    if kind == 'bool':
        choices = ' (y/[n])'
    else:
        raise ValueError('Unknown prompt kind.')
    prompt_txt = wrap(text + choices, wrap_length)
    for txt in prompt_txt[:-1]:
        print(txt)
    user_choice = input(prompt_txt[-1] + " ")
    if kind == 'bool':
        return user_choice in ["Y", "y", "Yes", "yes"]


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
