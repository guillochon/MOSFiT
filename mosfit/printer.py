"""Defines the `Printer` class."""
from __future__ import print_function

import datetime
import sys
from builtins import input
from textwrap import fill

import numpy as np

from .utils import calculate_WAIC, is_integer, pretty_num

if sys.version_info[:2] < (3, 3):
    old_print = print  # noqa

    def print(*args, **kwargs):
        """Replacement for print function in Python 2.x."""
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        file = kwargs.get('file', sys.stdout)
        if flush and file is not None:
            file.flush()


class Printer(object):
    """Print class for MOSFiT."""

    class bcolors(object):
        """Special formatting characters."""

        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def __init__(self, pool=None, wrap_length=100):
        """Initialize printer, setting wrap length."""
        self._wrap_length = wrap_length
        self._pool = pool

    def inline(self, x, new_line=False, error=False):
        """Print inline, erasing underlying pre-existing text."""
        lines = x.split('\n')
        if error:
            sys.stdout.write(self.bcolors.FAIL)
        if not new_line:
            for line in lines:
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
        print(x, flush=True)
        if error:
            sys.stdout.write(self.bcolors.ENDC)

    def wrapped(self, text, wrap_length=None, master_only=True):
        """Print text wrapped to either the specified length or the default."""
        if wrap_length and is_integer(wrap_length):
            wl = wrap_length
        else:
            wl = self._wrap_length
        if master_only and self._pool and not self._pool.is_master():
            return
        print(fill(text, wl))

    def prompt(self, text, wrap_length=None, kind='bool', options=None):
        """Prompt the user for input and return a value based on response."""
        if wrap_length and is_integer(wrap_length):
            wl = wrap_length
        else:
            wl = self._wrap_length

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
            ptxt = fill(txt, wl, replace_whitespace=False)
            print(ptxt)
        user_input = input(
            fill(
                prompt_txt[-1], wl, replace_whitespace=False) + " ")
        if kind == 'bool':
            return user_input in ["Y", "y", "Yes", "yes"]
        elif kind == 'select':
            if (is_integer(user_input) and
                    int(user_input) in list(range(1, len(options) + 1))):
                return options[int(user_input) - 1]
            return False
        elif kind == 'string':
            return user_input

    def status(self,
               fitter,
               desc='',
               scores='',
               progress='',
               acor=None,
               fracking=False,
               messages=[]):
        """Print status message showing state of fitting process."""
        outarr = [fitter._event_name]
        if desc:
            outarr.append(desc)
        if isinstance(scores, list):
            scorestring = 'Fracking' if fracking else 'Best'
            scorestring += ' scores: [ ' + ', '.join([
                pretty_num(max(x))
                if not np.isnan(max(x)) and np.isfinite(max(x)) else 'NaN'
                for x in scores
            ]) + ' ]'
            outarr.append(scorestring)
            if not fracking:
                scorestring = 'WAIC: ' + pretty_num(calculate_WAIC(scores))
                outarr.append(scorestring)
        if isinstance(progress, list):
            progressstring = 'Progress: [ {}/{} ]'.format(*progress)
            outarr.append(progressstring)
        if fitter._emcee_est_t + fitter._bh_est_t > 0.0:
            if fitter._bh_est_t > 0.0 or not fitter._fracking:
                tott = fitter._emcee_est_t + fitter._bh_est_t
            else:
                tott = 2.0 * fitter._emcee_est_t
            timestring = self.get_timestring(tott)
            outarr.append(timestring)
        if acor is not None:
            acorcstr = pretty_num(acor[1], sig=3)
            if acor[0] <= 0.0:
                acorstring = (self.bcolors.FAIL +
                              'Chain too short for acor ({})'.format(
                                  acorcstr) + self.bcolors.ENDC)
            else:
                acortstr = pretty_num(acor[0], sig=3)
                if fitter._travis:
                    col = ''
                elif acor[1] < 5.0:
                    col = self.bcolors.FAIL
                elif acor[1] < 10.0:
                    col = self.bcolors.WARNING
                else:
                    col = self.bcolors.OKGREEN
                acorstring = col
                acorstring = acorstring + 'Acor Tau: {} ({}x)'.format(acortstr,
                                                                      acorcstr)
                acorstring = acorstring + (self.bcolors.ENDC if col else '')
            outarr.append(acorstring)

        if not isinstance(messages, list):
            raise ValueError('`messages` must be list!')
        outarr.extend(messages)

        line = ''
        lines = ''
        li = 0
        for i, item in enumerate(outarr):
            oldline = line
            line = line + (' | ' if li > 0 else '') + item
            li = li + 1
            if len(line) > self._wrap_length:
                li = 1
                lines = lines + '\n' + oldline
                line = item

        lines = lines + '\n' + line

        self.inline(lines, new_line=fitter._travis)

    def get_timestring(self, t):
        """Return estimated time remaining.

        Return a string showing the estimated remaining time based upon
        elapsed times for emcee and fracking.
        """
        td = str(datetime.timedelta(seconds=int(round(t))))
        return ('Estimated time left: [ ' + td + ' ]')
