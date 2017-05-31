# -*- encoding: utf-8 -*-
"""Defines the `Printer` class."""
from __future__ import print_function, unicode_literals

import datetime
import json
import os
import re
import sys
from builtins import input, str
from collections import OrderedDict
from textwrap import fill

import numpy as np
from scipy import ndimage

from .utils import (calculate_WAIC, congrid, is_integer, open_atomic,
                    pretty_num, rebin)

if sys.version_info[:2] < (3, 3):
    old_print = print  # noqa

    def print(*args, **kwargs):
        """Replace print function in Python 2.x."""
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        file = kwargs.get('file', sys.stdout)
        if flush and file is not None:
            file.flush()


class Printer(object):
    """Print class for MOSFiT."""

    class ansi(object):
        """Special formatting characters."""

        BLUE = '\033[0;94m'
        BOLD = '\033[0;1m'
        CYAN = '\033[0;96m'
        END = '\033[0m'
        GREEN = '\033[0;92m'
        HEADER = '\033[0;95m'
        MAGENTA = '\033[1;35m'
        ORANGE = '\033[38;5;214m'
        RED = '\033[0;91m'
        UNDERLINE = '\033[4m'
        YELLOW = '\033[0;93m'

        codes = {
            '!b': BLUE,
            '!c': CYAN,
            '!e': END,
            '!g': GREEN,
            '!m': MAGENTA,
            '!o': ORANGE,
            '!r': RED,
            '!u': UNDERLINE,
            '!y': YELLOW
        }

    def __init__(self, pool=None, wrap_length=100, quiet=False, fitter=None,
                 language='en'):
        """Initialize printer, setting wrap length."""
        self._wrap_length = wrap_length
        self._quiet = quiet
        self._pool = pool
        self._fitter = fitter
        self._language = language
        self._was_inline = False

        self.set_strings()

    def set_strings(self):
        """Set pre-defined list of strings."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'strings.json')) as f:
            strings = json.load(f, object_pairs_hook=OrderedDict)
        if self._language == 'en':
            self._strings = strings
            return
        lsf = os.path.join(dir_path, 'strings-' + self._language + '.json')
        if os.path.isfile(lsf):
            with open(lsf) as f:
                self._strings = json.load(f)
            if set(self._strings.keys()) == set(strings):
                return

        try:
            from googletrans import Translator  # noqa
        except Exception:
            self.prt(
                'The `--language` option requires the `Googletrans` package. '
                'Please install with `pip install googletrans`.', wrapped=True)
            self._strings = strings
            pass
        else:
            self.prt(self.translate(
                'Building strings for `{}`, please wait...'
                .format(self._language)), wrapped=True)
            self._strings = OrderedDict()
            for key in strings:
                self._strings[key] = self.translate(strings[key])
            with open_atomic(lsf, 'w') as f:
                json.dump(self._strings, f)

    def set_language(self, language):
        """Set language."""
        self._language = language

    def colorify(self, text):
        """Add colors to text."""
        output = text
        for code in self.ansi.codes:
            # Windows doesn't support ANSI codes in Python, simple delete color
            # commands if on Windows.
            if os.name == 'nt':
                output = output.replace(code, '')
            else:
                output = output.replace(code, self.ansi.codes[code])
        return output

    def _lines(
        self, text, colorify=False, center=False, width=None,
        warning=False, error=False, prefix=True, color='', inline=False,
            wrap_length=None, wrapped=False, master_only=True):
        """Generate lines for output."""
        if self._quiet:
            return []
        if master_only and self._pool and not self._pool.is_master():
            return []
        if warning:
            if prefix:
                text = '!y' + self._strings['warning'] + ': ' + text + '!e'
        if error:
            if prefix:
                text = '!r' + self._strings['error'] + ': ' + text + '!e'
        if color:
            text = color + text + '!e'
        tspl = text.split('\n')
        if wrapped:
            if not wrap_length or not is_integer(wrap_length):
                wrap_length = self._wrap_length
            ntspl = []
            for line in tspl:
                ntspl.extend(fill(line, wrap_length).split('\n'))
            tspl = ntspl
        return tspl

    def prt(self, text='', **kwargs):
        """Print text without modification."""
        warning = kwargs.get('warning', False)
        error = kwargs.get('error', False)
        color = kwargs.get('color', '')
        inline = kwargs.get('inline', False)
        center = kwargs.get('center', False)
        width = kwargs.get('width', None)
        colorify = kwargs.get('colorify', True)
        tspl = self._lines(text, **kwargs)
        if warning or error or color:
            colorify = True
        if inline and self._fitter is not None:
            inline = not self._fitter._test
        rlines = []
        for ri, line in enumerate(tspl):
            rline = line
            if colorify:
                rline = self.colorify(rline)
            if center:
                tlen = len(repr(rline)) - len(line) - line.count('!')
                rline = rline.center(width + tlen)
            rlines.append(rline)

        if inline:
            if self._was_inline:
                for line in tspl:
                    sys.stdout.write("\033[F")
                    sys.stdout.write("\033[K")
            self._was_inline = True
        else:
            self._was_inline = False

        for rline in rlines:
            try:
                print(rline, flush=True)
            except UnicodeEncodeError:
                print(rline.encode('ascii', 'replace').decode(), flush=True)

    def string(self, text, **kwargs):
        """Return message string."""
        center = kwargs.get('center', False)
        width = kwargs.get('width', None)
        tspl = self._lines(text, **kwargs)
        lines = []
        for ri, line in enumerate(tspl):
            rline = line
            if center:
                tlen = len(repr(rline)) - len(line) - line.count('!')
                rline = rline.center(width + tlen)
            lines.append(rline)
        return '\n'.join(lines)

    def message(self, name, reps=[], wrapped=True, inline=False,
                warning=False, error=False, prefix=True, center=False,
                colorify=True, width=None, prt=True, color=''):
        """Print a message from a dictionary of strings."""
        if name in self._strings:
            text = self._strings[name]
        else:
            text = '< Message not found [' + ''.join(
                ['{} ' for x in range(len(reps))]).strip() + '] >'
        text = text.format(*reps)
        if prt:
            self.prt(
                text, center=center, colorify=colorify, width=width,
                prefix=prefix, inline=inline, wrapped=wrapped,
                warning=warning, error=error, color=color)
        return text

    def prompt(self, text, wrap_length=None, kind='bool', default=None,
               none_string='None of the above.', colorify=True, single=False,
               options=None, translate=True, message=True, color='',
               allow_blank=True):
        """Prompt the user for input and return a value based on response."""
        if wrap_length and is_integer(wrap_length):
            wl = wrap_length
        else:
            wl = self._wrap_length

        if kind in ['select', 'option', 'bool']:
            if kind == 'bool':
                options = [('', 'y'), ('', 'n')]
                if default is None:
                    default = 'n'
                single = True
                none_string = None
            if none_string is not None:
                options.append((none_string, 'n'))
            if default is None and none_string is None:
                default = (
                    str(options[0][-1]) if isinstance(options[0], tuple)
                    else '1')

        while True:
            if kind in ['select', 'option', 'bool']:
                opt_sels = [
                    (str(x[-1]) if isinstance(x, tuple) else str(xi + 1))
                    for xi, x in enumerate(options)]
                opt_strs = [
                    (('[' + str(x) + '.]') if x == default else x + '.')
                    for x in opt_sels]
                opt_nops = [(('[' + str(x) + ']') if x == default else x)
                            for x in opt_sels]
                opt_labs = [(opt[0] if isinstance(opt, tuple) else opt)
                            for opt in options]
                opt_labs = [x.decode('utf-8') if not isinstance(x, str)
                            else x for x in opt_labs]
                new_opts = dict(zip(*(opt_sels, opt_labs)))

                if single:
                    choices = ' (' + '/'.join(opt_nops) + ')'
                else:
                    msp = max([len(x) for x in opt_strs])
                    carr = [
                        opt_strs[i].rjust(
                            (msp + 1) if opt_sels[i] == default
                            else msp) + (u' ' if opt_sels[i] == default
                                         else u'  ') + opt_labs[i]
                        for i in range(len(options))
                    ]
                    sel_str = ''
                    beg_range = None
                    for oi, opt in enumerate(opt_sels):
                        if is_integer(opt):
                            if beg_range is None:
                                sel_str += opt_nops[oi]
                                beg_range = opt
                            else:
                                if (oi + 1 >= len(opt_sels) or
                                        not is_integer(opt_sels[oi + 1])):
                                    if (oi > 0 and
                                            opt_sels[oi - 1] == beg_range):
                                        sel_str += '/' + opt_nops[oi]
                                    else:
                                        sel_str += '-' + opt_nops[oi]
                        else:
                            if oi != 0:
                                sel_str += '/'
                            sel_str += opt_nops[oi]
                    carr += ['Enter selection (' + sel_str + '):']
                    choices = '\n' + '\n'.join(carr)
            elif kind == 'string':
                choices = ''
            else:
                raise ValueError('Unknown prompt kind.')

            if message and text in self._strings:
                text = self.message(text, prt=False)
            textchoices = text + choices
            if translate:
                textchoices = self.translate(textchoices)
            prompt_txt = (textchoices).split('\n')
            for txt in prompt_txt[:-1]:
                ptxt = fill(txt, wl, replace_whitespace=False)
                self.prt(ptxt, color=color)

            inp_text = fill(
                prompt_txt[-1], wl, replace_whitespace=False) + " "
            if colorify:
                inp_text = self.colorify(color + inp_text + "!e")
            user_input = input(inp_text)

            yes = ['y', 'yes', 'yep']
            nos = ['n', 'no', 'nope']

            uil = user_input.lower()
            if kind == 'bool':
                if uil == '':
                    return default in yes
                if uil in yes:
                    return True
                if uil in nos:
                    return False
                continue
            elif kind == 'select':
                if (none_string is not None and default == 'n' and
                        uil in nos + ['']):
                    return None
                if uil == '' and default in new_opts:
                    return new_opts[default]
                if uil in opt_sels:
                    return new_opts[uil]
                continue
            elif kind == 'option':
                if (none_string is not None and default == 'n' and
                        uil in nos + ['']):
                    return None
                if uil == '' and default in new_opts:
                    return int(default) if is_integer(default) else default
                if is_integer(user_input) and user_input in opt_sels:
                    return int(user_input)
                if uil in opt_sels:
                    return uil
                continue
            elif kind == 'string':
                if not allow_blank and user_input == '':
                    continue
                return user_input

    def status(self,
               desc='',
               scores='',
               accepts='',
               progress='',
               acor=None,
               psrf=None,
               fracking=False,
               messages=[],
               kmat=None,
               make_space=False,
               convergence_type='',
               convergence_criteria=''):
        """Print status message showing state of fitting process."""
        if self._quiet:
            return

        fitter = self._fitter
        outarr = [fitter._event_name]
        if desc:
            if desc == 'burning':
                descstr = '!o' + self._strings.get(
                    desc, '?') + '!e'
            else:
                descstr = self._strings.get(desc, '?')
            outarr.append(descstr)
        if isinstance(scores, list):
            scorestring = self._strings[
                'fracking_scores'] if fracking else self._strings[
                    'score_ranges']
            scorestring += ': [ ' + ', '.join([
                '...'.join([
                    pretty_num(min(x))
                    if not np.isnan(min(x)) and np.isfinite(min(x))
                    else 'NaN',
                    pretty_num(max(x))
                    if not np.isnan(max(x)) and np.isfinite(max(x))
                    else 'NaN']) if len(x) > 1 else pretty_num(x[0])
                for x in scores
            ]) + ' ]'
            outarr.append(scorestring)
            if not fracking:
                scorestring = 'WAIC: ' + pretty_num(calculate_WAIC(scores))
                outarr.append(scorestring)
        if isinstance(accepts, list):
            scorestring = self._strings['moves_accepted'] + ': [ '
            scorestring += ', '.join([
                ('!r' if x < 0.01 else '!y' if x < 0.1 else '!g') +
                '{:.0%}'.format(x) + '!e'
                for x in accepts
            ]) + ' ]'
            outarr.append(scorestring)
        if isinstance(progress, list):
            if progress[1]:
                progressstring = (
                    self._strings['progress'] +
                    ': [ {}/{} ]'.format(*progress))
            else:
                progressstring = (
                    self._strings['progress'] + ': [ {} ]'.format(progress[0]))
            outarr.append(progressstring)
        if fitter._emcee_est_t < 0.0:
            txt = self.message('run_until_converged', [
                convergence_type, convergence_criteria], prt=False)
            outarr.append(txt)
        elif fitter._emcee_est_t + fitter._bh_est_t > 0.0:
            if fitter._bh_est_t > 0.0 or not fitter._fracking:
                tott = fitter._emcee_est_t + fitter._bh_est_t
            else:
                tott = 2.0 * fitter._emcee_est_t
            timestring = self.get_timestring(tott)
            outarr.append(timestring)
        if acor is not None:
            acorcstr = pretty_num(acor[1], sig=3)
            if acor[0] <= 0.0:
                acorstring = ('!rChain too short for `acor` ({})!e'.format(
                              acorcstr))
            else:
                acortstr = pretty_num(acor[0], sig=3)
                acorbstr = str(int(acor[2]))
                if acor[1] < 2.0:
                    col = '!r'
                elif acor[1] < 5.0:
                    col = '!y'
                else:
                    col = '!g'
                acorstring = col
                acorstring = acorstring + 'Acor Tau (i > {}): {} ({}x)'.format(
                    acorbstr, acortstr, acorcstr)
                acorstring = acorstring + ('!e' if col else '')
            outarr.append(acorstring)
        if psrf is not None and psrf[0] != np.inf:
            psrfstr = pretty_num(psrf[0], sig=4)
            psrfbstr = str(int(psrf[1]))
            if psrf[0] > 2.0:
                col = '!r'
            elif psrf[0] > 1.2:
                col = '!y'
            else:
                col = '!g'
            psrfstring = col
            psrfstring = psrfstring + 'PSRF (i > {}): {}'.format(
                psrfbstr, psrfstr)
            psrfstring = psrfstring + ('!e' if col else '')
            outarr.append(psrfstring)

        if not isinstance(messages, list):
            raise ValueError('`messages` must be list!')
        outarr.extend(messages)

        kmat_extra = 0
        if kmat is not None and kmat.shape[0] > 1:
            smat = ndimage.filters.gaussian_filter(
                kmat, 0.1 * len(kmat) / 7.0, mode='nearest', truncate=2.0)
            try:
                kmat_scaled = congrid(smat, (14, 7), minusone=True,
                                      bounds_error=True)
            except Exception:
                kmat_scaled = rebin(smat, (14, 7))
            kmat_scaled = np.log(kmat_scaled)
            kmat_scaled /= np.max(kmat_scaled) - np.min(kmat_scaled)
            kmat_pers = [np.percentile(kmat_scaled, x) for x in (20, 50, 80)]
            kmat_dimi = range(len(kmat_scaled))
            kmat_dimj = range(len(kmat_scaled[0]))
            doodle = '\n╔' + ('═' * len(kmat_scaled)) + '╗   \n'
            doodle += '║' + '║   \n║'.join(
                [''.join([self.ascii_fill(kmat_scaled[i, j], kmat_pers)
                          for i in kmat_dimi]) for j in kmat_dimj]) + '║'
            doodle += '\n╚' + ('═' * len(kmat_scaled)) + '╝   '
            doodle = doodle.splitlines()

            kmat_extra = len(doodle[-1])

        line = ''
        lines = ''
        li = 0
        for i, item in enumerate(outarr):
            oldline = line
            line = line + (' | ' if li > 0 else '') + item
            li = li + 1
            if len(line) > self._wrap_length - kmat_extra:
                li = 1
                lines = lines + '\n' + oldline
                line = item

        lines = lines + '\n' + line

        if kmat is not None and kmat.shape[0] > 1:
            lines = self._lines(lines)
            loff = int(np.floor((len(kmat_scaled[0]) - len(lines)) / 2.0)) + 2
            for li, line in enumerate(doodle):
                if li < loff:
                    continue
                elif li > loff + len(lines) - 1:
                    break
                doodle[li] += lines[li - loff]
            lines = '\n'.join(doodle)

        self.prt(lines, colorify=True, inline=not make_space)
        if make_space:
            self._was_inline = True

    def get_timestring(self, t):
        """Return estimated time remaining.

        Return a string showing the estimated remaining time based upon
        elapsed times for emcee and fracking.
        """
        td = str(datetime.timedelta(seconds=int(round(t))))
        return (self._strings['estimated_time'] + ': [ ' + td + ' ]')

    def translate(self, text):
        """Translate text to another language."""
        if self._language != 'en':
            try:
                from googletrans import Translator
                translator = Translator()
                ttext, reps = self.rep_ansi(text)
                ttext = translator.translate(ttext, dest=self._language).text
                text = ttext.format(*reps)
            except Exception:
                pass
        return text

    def rep_ansi(self, text):
        """Replace ANSI codes and return the list of codes."""
        patt = re.compile(r'({})'.format(
            '|'.join(['\{.*?\}'] + list(self.ansi.codes.keys()))))
        stext = patt.sub("{}", text)
        matches = patt.findall(text)
        return stext, matches

    def tree(self, my_tree):
        """Pretty print the module dependency trees for each root."""
        for root in my_tree:
            tree_str = json.dumps({root: my_tree[root]},
                                  indent='─ ',
                                  separators=('', ''))
            tree_str = ''.join(
                c for c in tree_str if c not in ['{', '}', '"'])
            tree_str = '\n'.join([
                x.rstrip() for x in tree_str.split('\n') if
                x.strip('─ ') != ''])
            tree_str = '\n'.join(
                [x[::-1].replace('─ ─', '├', 1)[::-1].replace('─', '│') if
                 x.startswith('─ ─') else x.replace('─ ', '') + ':'
                 for x in tree_str.split('\n')])
            lines = ['  ' + x for x in tree_str.split('\n')]
            ll = len(lines)
            for li, line in enumerate(lines):
                if (li < ll - 1 and
                        lines[li + 1].count('│') < line.count('│')):
                    lines[li] = line.replace('├', '└')
            for li, line in enumerate(reversed(lines)):
                if li == 0:
                    lines[ll - li - 1] = line.replace(
                        '│', ' ').replace('├', '└')
                    continue
                lines[ll - li - 1] = ''.join([
                    x if ci > len(lines[ll - li]) - 1 or x not in ['│', '├'] or
                    lines[ll - li][ci] != ' ' else x.replace(
                        '│', ' ').replace(
                            '├', '└') for ci, x in enumerate(line)])
            tree_str = '\n'.join(lines)
            self.prt(tree_str)

    def ascii_fill(self, value, pers):
        """Print a character based on range from 0 - 1."""
        if np.isnan(value) or value < pers[0]:
            return ' '
        if np.isnan(value) or value < pers[1]:
            return '.'
        elif value < pers[2]:
            return '*'
        else:
            return '#'
