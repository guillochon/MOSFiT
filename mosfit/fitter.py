import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import OrderedDict
from difflib import get_close_matches

import emcee
import numpy as np
from astrocats.catalog.entry import ENTRY, Entry
from astrocats.catalog.model import MODEL
from astrocats.catalog.photometry import PHOTOMETRY
from astrocats.catalog.realization import REALIZATION
from mosfit.__init__ import __version__
from mosfit.constants import LIKELIHOOD_FLOOR
from mosfit.utils import (get_url_file_handle, is_number, pretty_num,
                          print_inline, print_wrapped, prompt)
from schwimmbad import MPIPool, SerialPool

from .model import Model

warnings.filterwarnings("ignore")


def draw_walker(test=True):
    global model
    return model.draw_walker(test)


def likelihood(x):
    global model
    return model.likelihood(x)


def prior(x):
    global model
    return model.prior(x)


def frack(x):
    global model
    return model.frack(x)


class Fitter():
    """Fit transient events with the provided model.
    """

    def __init__(self):
        pass

    def fit_events(self,
                   events=[''],
                   models=[],
                   plot_points='',
                   max_time='',
                   band_list=[],
                   band_systems=[],
                   band_instruments=[],
                   band_bandsets=[],
                   iterations=1000,
                   num_walkers=50,
                   num_temps=2,
                   parameter_paths=[],
                   fracking=True,
                   frack_step=20,
                   wrap_length=100,
                   travis=False,
                   post_burn=500,
                   smooth_times=-1,
                   extrapolate_time=0.0,
                   limit_fitting_mjds=False,
                   exclude_bands=[],
                   exclude_instruments=[],
                   suffix=''):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._travis = travis
        self._wrap_length = wrap_length

        self._event_name = 'Batch'
        for event in events:
            self._event_name = ''
            self._event_path = ''
            if event:
                try:
                    pool = MPIPool()
                except:
                    pool = SerialPool()
                if pool.is_master():
                    path = ''
                    # If the event name ends in .json, assume a path
                    if event.endswith('.json'):
                        path = event
                        self._event_name = event.replace('.json',
                                                         '').split('/')[-1]
                    # If not (or the file doesn't exist), download from OSC
                    if not path or not os.path.exists(path):
                        names_path = os.path.join(dir_path, 'cache',
                                                  'names.min.json')
                        input_name = event.replace('.json', '')
                        print_wrapped(
                            'Event `{}` interpreted as supernova '
                            'name, downloading list of supernova '
                            'aliases...'.format(input_name),
                            wrap_length=self._wrap_length)
                        try:
                            response = get_url_file_handle(
                                'https://sne.space/astrocats/astrocats/'
                                'supernovae/output/names.min.json',
                                timeout=10)
                        except:
                            print_wrapped(
                                'Warning: Could not download SN names (are '
                                'you online?), using cached list.',
                                wrap_length=self._wrap_length)
                            raise
                        else:
                            with open(names_path, 'wb') as f:
                                shutil.copyfileobj(response, f)
                        if os.path.exists(names_path):
                            with open(names_path, 'r') as f:
                                names = json.load(
                                    f, object_pairs_hook=OrderedDict)
                        else:
                            print('Error: Could not read list of SN names!')
                            raise RuntimeError

                        if event in names:
                            self._event_name = event
                        else:
                            for name in names:
                                if (event in names[name] or
                                        'SN' + event in names[name]):
                                    self._event_name = name
                                    break
                        if not self._event_name:
                            namekeys = []
                            for name in names:
                                namekeys.extend(names[name])
                            matches = set(
                                get_close_matches(
                                    event, namekeys, n=5, cutoff=0.8))
                            # matches = []
                            if len(matches) < 5 and is_number(event[0]):
                                print_wrapped(
                                    'Could not find event, performing '
                                    'extended name search...',
                                    wrap_length=self._wrap_length)
                                snprefixes = set()
                                for name in names:
                                    ind = re.search("\d", name)
                                    if ind and ind.start() > 0:
                                        snprefixes.add(name[:ind.start()])
                                snprefixes = list(snprefixes)
                                for prefix in snprefixes:
                                    testname = prefix + event
                                    new_matches = get_close_matches(
                                        testname, namekeys, cutoff=0.95, n=1)
                                    if len(new_matches):
                                        matches.add(new_matches[0])
                                    if len(matches) == 5:
                                        break
                            if len(matches):
                                response = prompt(
                                    'No exact match to given event '
                                    'found. Did you mean one of the '
                                    'following events?',
                                    kind='select',
                                    options=list(matches))
                                if response:
                                    for name in names:
                                        if response in names[name]:
                                            self._event_name = name
                                            break
                        if not self._event_name:
                            print_wrapped('Could not find event by that name, '
                                          'skipping!', self._wrap_length)
                            continue
                        urlname = self._event_name + '.json'

                        print_wrapped(
                            'Found event by primary name `{}` in the OSC, '
                            'downloading data...'.format(self._event_name),
                            wrap_length=self._wrap_length)
                        name_path = os.path.join(dir_path, 'cache', urlname)
                        try:
                            response = get_url_file_handle(
                                'https://sne.space/astrocats/astrocats/'
                                'supernovae/output/json/' + urlname,
                                timeout=10)
                        except:
                            print_wrapped(
                                'Warning: Could not download data for `{}`, '
                                'will attempt to use cached data.'.format(
                                    self._event_name),
                                wrap_length=self._wrap_length)
                        else:
                            with open(name_path, 'wb') as f:
                                shutil.copyfileobj(response, f)
                        path = name_path

                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            data = json.load(f, object_pairs_hook=OrderedDict)
                        print_wrapped('Event file:', self._wrap_length)
                        print_wrapped('  ' + path, self._wrap_length)
                    else:
                        print_wrapped(
                            'Error: Could not find data for `{}` locally or '
                            'on the OSC.'.format(self._event_name),
                            self._wrap_length)
                        raise RuntimeError

                    for rank in range(1, pool.size + 1):
                        pool.comm.send(self._event_name, dest=rank, tag=0)
                        pool.comm.send(path, dest=rank, tag=1)
                        pool.comm.send(data, dest=rank, tag=2)
                else:
                    self._event_name = pool.comm.recv(source=0, tag=0)
                    path = pool.comm.recv(source=0, tag=1)
                    data = pool.comm.recv(source=0, tag=2)
                    pool.wait()

                self._event_path = path

                if pool.is_master():
                    pool.close()

            for mod_name in models:
                for parameter_path in parameter_paths:
                    try:
                        pool = MPIPool()
                    except:
                        pool = SerialPool()
                    self._model = Model(
                        model=mod_name,
                        parameter_path=parameter_path,
                        wrap_length=wrap_length,
                        pool=pool)

                    if not event:
                        print('No event specified, generating dummy data.')
                        self._event_name = mod_name
                        gen_args = {
                            'name': mod_name,
                            'max_time': max_time,
                            'plot_points': plot_points,
                            'band_list': band_list,
                            'band_systems': band_systems,
                            'band_instruments': band_instruments,
                            'band_bandsets': band_bandsets
                        }
                        data = self.generate_dummy_data(**gen_args)

                    success = self.load_data(
                        data,
                        event_name=self._event_name,
                        iterations=iterations,
                        fracking=fracking,
                        post_burn=post_burn,
                        smooth_times=smooth_times,
                        extrapolate_time=extrapolate_time,
                        limit_fitting_mjds=limit_fitting_mjds,
                        exclude_bands=exclude_bands,
                        exclude_instruments=exclude_instruments,
                        band_list=band_list,
                        band_systems=band_systems,
                        band_instruments=band_instruments,
                        band_bandsets=band_bandsets,
                        pool=pool)

                    if success:
                        self.fit_data(
                            event_name=self._event_name,
                            iterations=iterations,
                            num_walkers=num_walkers,
                            num_temps=num_temps,
                            fracking=fracking,
                            frack_step=frack_step,
                            post_burn=post_burn,
                            pool=pool,
                            suffix=suffix)

                    if pool.is_master():
                        pool.close()

    def load_data(self,
                  data,
                  event_name='',
                  iterations=2000,
                  fracking=True,
                  post_burn=500,
                  smooth_times=-1,
                  extrapolate_time=0.0,
                  limit_fitting_mjds=False,
                  exclude_bands=[],
                  exclude_instruments=[],
                  band_list=[],
                  band_systems=[],
                  band_instruments=[],
                  band_bandsets=[],
                  pool=''):
        """Fit the data for a given event with this model using a combination
        of emcee and fracking.
        """
        fixed_parameters = []
        for task in self._model._call_stack:
            cur_task = self._model._call_stack[task]
            self._model._modules[task].set_event_name(event_name)
            if cur_task['kind'] == 'data':
                success = self._model._modules[task].set_data(
                    data,
                    req_key_values={'band': self._model._bands},
                    subtract_minimum_keys=['times'],
                    smooth_times=smooth_times,
                    extrapolate_time=extrapolate_time,
                    limit_fitting_mjds=limit_fitting_mjds,
                    exclude_bands=exclude_bands,
                    exclude_instruments=exclude_instruments,
                    band_list=band_list,
                    band_systems=band_systems,
                    band_instruments=band_instruments,
                    band_bandsets=band_bandsets)
                if not success:
                    return False
                fixed_parameters.extend(self._model._modules[task]
                                        .get_data_determined_parameters())

        self._model.determine_free_parameters(fixed_parameters)

        self._model.exchange_requests()

        # Run through once to set all inits
        outputs = self._model.run_stack(
            [0.0 for x in range(self._model._num_free_parameters)],
            root='output')

        # Collect observed band info
        if pool.is_master() and 'filters' in self._model._modules:
            print_wrapped('Bands being used for current transient:',
                          self._wrap_length)
            bis = list(set(outputs['all_band_indices']))
            ois = []
            for bi in bis:
                ois.append(
                    any([
                        y
                        for x, y in zip(outputs['all_band_indices'], outputs[
                            'observed']) if x == bi
                    ]))
            band_len = max([
                len(self._model._modules['filters']._unique_bands[bi]['SVO'])
                for bi in bis
            ])
            filts = self._model._modules['filters']
            ubs = filts._unique_bands
            filterrows = [
                s[3]
                for s in list(
                    sorted([(ubs[bis[i]]['systems'], ubs[bis[i]][
                        'bandsets'], filts._average_wavelengths[bis[i]], (
                            ' ' + (' ' if ois[i] else '*') + ubs[bis[i]]['SVO']
                            .ljust(band_len) + ' [' + ','.join(
                                set(
                                    filter(None, (ubs[bis[i]]['bandsets'], ubs[
                                        bis[i]]['systems'])))) + ']'
                        ).replace(' []', '')) for i in range(len(bis))]))
            ]
            if not all(ois):
                filterrows.append('  (* = Not observed in this band)')
            print('\n'.join(filterrows))

        self._event_name = event_name
        self._emcee_est_t = 0.0
        self._bh_est_t = 0.0
        self._fracking = fracking
        self._burn_in = max(iterations - post_burn, 0)

        return True

    def fit_data(self,
                 event_name='',
                 iterations=2000,
                 frack_step=20,
                 num_walkers=50,
                 num_temps=2,
                 fracking=True,
                 post_burn=500,
                 pool='',
                 suffix=''):
        """Fit the data for a given event with this model using a combination
        of emcee and fracking.
        """

        global model
        model = self._model

        if not pool.is_master():
            try:
                pool.wait()
            except KeyboardInterrupt:
                pass
            return

        ntemps, ndim, nwalkers = (num_temps, model._num_free_parameters,
                                  num_walkers)

        test_walker = iterations > 0
        lnprob = None
        pool_size = max(pool.size, 1)

        print('{} dimensions in problem.\n\n'.format(ndim))
        p0 = [[] for x in range(ntemps)]

        for i, pt in enumerate(p0):
            while len(p0[i]) < nwalkers:
                self.print_status(
                    desc='Drawing initial walkers',
                    progress=[i * nwalkers + len(p0[i]), nwalkers * ntemps])

                nmap = nwalkers - len(p0[i])
                p0[i].extend(pool.map(draw_walker, [test_walker] * nmap))

        sampler = emcee.PTSampler(
            ntemps, nwalkers, ndim, likelihood, prior, pool=pool)

        print_inline('Initial draws completed!')
        print('\n\n')
        p = list(p0)

        if fracking:
            frack_iters = max(round(iterations / frack_step), 1)
            bmax = int(round(self._burn_in / float(frack_step)))
            loop_step = frack_step
        else:
            frack_iters = 1
            loop_step = iterations

        acort = 1.0
        acorc = 1.0
        try:
            for b in range(frack_iters):
                if fracking and b >= bmax:
                    loop_step = iterations - self._burn_in
                emi = 0
                st = time.time()
                for p, lnprob, lnlike in sampler.sample(
                        p, iterations=min(loop_step, iterations)):
                    # Redraw bad walkers
                    for ti, tprob in enumerate(lnprob):
                        for wi, wprob in enumerate(tprob):
                            if wprob <= LIKELIHOOD_FLOOR or np.isnan(wprob):
                                print('Warning: Bad walker position detected, '
                                      'indicates variable mismatch.')
                                p[ti][wi] = draw_walker()
                    emi = emi + 1
                    prog = b * frack_step + emi
                    try:
                        acorc = min(max(0.1 * float(prog) / acort, 1.0), 10.0)
                        acort = max([
                            max(x) for x in sampler.get_autocorr_time(c=acorc)
                        ])
                    except:
                        pass
                    acor = [acort, acorc]
                    self._emcee_est_t = float(time.time() - st) / emi * (
                        iterations - (b * frack_step + emi))
                    self.print_status(
                        desc='Running PTSampler',
                        scores=[max(x) for x in lnprob],
                        progress=[prog, iterations],
                        acor=acor)
                if fracking and b >= bmax:
                    break
                if fracking and b < bmax:
                    self.print_status(
                        desc='Running Fracking',
                        scores=[max(x) for x in lnprob],
                        progress=[(b + 1) * frack_step, iterations],
                        acor=acor)
                    ijperms = [[x, y]
                               for x in range(ntemps) for y in range(nwalkers)]
                    ijprobs = np.array([
                        1.0
                        # lnprob[x][y]
                        for x in range(ntemps) for y in range(nwalkers)
                    ])
                    ijprobs -= max(ijprobs)
                    ijprobs = [np.exp(0.1 * x) for x in ijprobs]
                    ijprobs /= sum([x for x in ijprobs if not np.isnan(x)])
                    nonzeros = len([x for x in ijprobs if x > 0.0])
                    selijs = [
                        ijperms[x]
                        for x in np.random.choice(
                            range(len(ijperms)),
                            pool_size,
                            p=ijprobs,
                            replace=(pool_size > nonzeros))
                    ]

                    bhwalkers = [p[i][j] for i, j in selijs]

                    st = time.time()
                    seeds = [
                        round(time.time() * 1000.0) % 4294900000 + x
                        for x in range(len(bhwalkers))
                    ]
                    frack_args = list(zip(bhwalkers, seeds))
                    bhs = pool.map(frack, frack_args)
                    for bhi, bh in enumerate(bhs):
                        if -bh.fun > lnprob[selijs[bhi][0]][selijs[bhi][1]]:
                            p[selijs[bhi][0]][selijs[bhi][1]] = bh.x
                    self._bh_est_t = float(time.time() - st) * (bmax - b - 1)
                    scores = [-x.fun for x in bhs]
                    self.print_status(
                        desc='Running Fracking',
                        scores=scores,
                        progress=[(b + 1) * frack_step, iterations])
        except KeyboardInterrupt:
            pool.close()
            print(self._wrap_length)
            if (not prompt(
                    'You have interrupted the Monte Carlo. Do you wish to '
                    'save the incomplete run to disk? Previous results will '
                    'be overwritten.', self._wrap_length)):
                sys.exit()
        except:
            raise

        print_wrapped('Saving output to disk...', self._wrap_length)
        if self._event_path:
            entry = Entry.init_from_file(
                catalog=None,
                name=self._event_name,
                path=self._event_path,
                merge=False,
                pop_schema=False)
        else:
            entry = Entry(name=self._event_name)

        source = entry.add_source(name='MOSFiT paper')
        model_setup = {}
        for ti, task in enumerate(model._call_stack):
            task_copy = model._call_stack[task].copy()
            if (task_copy['kind'] == 'parameter' and
                    task in model._parameter_json):
                task_copy.update(model._parameter_json[task])
            model_setup[task] = task_copy
        modeldict = {
            MODEL.NAME: self._model._model_name,
            MODEL.SETUP: model_setup,
            MODEL.CODE: 'MOSFiT',
            MODEL.DATE: time.strftime("%Y/%m/%d"),
            MODEL.VERSION: __version__ + ' [' +
            subprocess.getoutput('git rev-parse --short HEAD').strip() + ']',
            MODEL.SOURCE: source
        }
        modelnum = entry.add_model(**modeldict)

        for xi, x in enumerate(p[0]):
            output = model.run_stack(x, root='output')
            for i in range(len(output['times'])):
                if not np.isfinite(output['model_magnitudes'][i]):
                    continue
                photodict = {
                    PHOTOMETRY.BAND: output['bands'][i],
                    PHOTOMETRY.TIME: output['times'][i] + output['min_times'],
                    PHOTOMETRY.MAGNITUDE: output['model_magnitudes'][i],
                    PHOTOMETRY.MODEL: modelnum,
                    PHOTOMETRY.SOURCE: source,
                    PHOTOMETRY.REALIZATION: str(xi + 1)
                }
                if output['systems'][i]:
                    photodict[PHOTOMETRY.SYSTEM] = output['systems'][i]
                if output['bandsets'][i]:
                    photodict[PHOTOMETRY.BAND_SET] = output['bandsets'][i]
                if output['instruments'][i]:
                    photodict[PHOTOMETRY.INSTRUMENT] = output['instruments'][i]
                entry.add_photometry(
                    compare_against_existing=False, **photodict)

            parameters = OrderedDict()
            pi = 0
            for ti, task in enumerate(model._call_stack):
                if task not in model._free_parameters:
                    continue
                output = model._modules[task].process(**{'fraction': x[pi]})
                value = list(output.values())[0]
                paramdict = {
                    'value': value,
                    'fraction': x[pi],
                    'latex': model._modules[task].latex(),
                    'log': model._modules[task].is_log()
                }
                parameters.update({model._modules[task].name(): paramdict})
                pi = pi + 1
            realdict = {REALIZATION.PARAMETERS: parameters}
            if lnprob is not None:
                realdict[REALIZATION.SCORE] = str(lnprob[0][xi])
            entry[ENTRY.MODELS][0].add_realization(**realdict)

        entry.sanitize()
        oentry = entry._ordered(entry)

        if not os.path.exists(model.MODEL_OUTPUT_DIR):
            os.makedirs(model.MODEL_OUTPUT_DIR)

        with open(os.path.join(model.MODEL_OUTPUT_DIR, 'walkers.json'),
                  'w') as flast, open(
                      os.path.join(model.MODEL_OUTPUT_DIR, self._event_name + (
                          ('_' + suffix)
                          if suffix else '') + '.json'), 'w') as f:
            json.dump(oentry, flast, indent='\t', separators=(',', ':'))
            json.dump(oentry, f, indent='\t', separators=(',', ':'))

        return (p, lnprob)

    def generate_dummy_data(self,
                            name,
                            max_time=1000.,
                            plot_points=100,
                            band_list=[],
                            band_systems=[],
                            band_instruments=[],
                            band_bandsets=[]):
        time_list = np.linspace(0.0, max_time, plot_points)
        band_list_all = ['V'] if len(band_list) == 0 else band_list
        times = np.repeat(time_list, len(band_list_all))

        # Create lists of systems/instruments if not provided.
        if isinstance(band_systems, str):
            band_systems = [band_systems for x in range(len(band_list_all))]
        if isinstance(band_instruments, str):
            band_instruments = [
                band_instruments for x in range(len(band_list_all))
            ]
        if isinstance(band_bandsets, str):
            band_bandsets = [band_bandsets for x in range(len(band_list_all))]
        if len(band_systems) < len(band_list_all):
            rep_val = '' if len(band_systems) == 0 else band_systems[-1]
            band_systems = band_systems + [
                rep_val
                for x in range(len(band_list_all) - len(band_systems))
            ]
        if len(band_instruments) < len(band_list_all):
            rep_val = '' if len(band_instruments) == 0 else band_instruments[
                -1]
            band_instruments = band_instruments + [
                rep_val
                for x in range(len(band_list_all) - len(band_instruments))
            ]
        if len(band_bandsets) < len(band_list_all):
            rep_val = '' if len(band_bandsets) == 0 else band_bandsets[-1]
            band_bandsets = band_bandsets + [
                rep_val
                for x in range(len(band_list_all) - len(band_bandsets))
            ]

        bands = [i for s in [band_list_all for x in time_list] for i in s]
        systs = [i for s in [band_systems for x in time_list] for i in s]
        insts = [i for s in [band_instruments for x in time_list] for i in s]
        bsets = [i for s in [band_bandsets for x in time_list] for i in s]

        data = {name: {'photometry': []}}
        for ti, tim in enumerate(times):
            band = bands[ti]
            if isinstance(band, dict):
                band = band['name']

            photodict = {
                'time': tim,
                'band': band,
                'magnitude': 0.0,
                'e_magnitude': 0.0
            }
            if systs[ti]:
                photodict['system'] = systs[ti]
            if insts[ti]:
                photodict['instrument'] = insts[ti]
            if bsets[ti]:
                photodict['bandset'] = bsets[ti]
            data[name]['photometry'].append(photodict)

        return data

    def print_status(self, desc='', scores='', progress='', acor=''):
        """Prints a status message showing the current state of the fitting process.
        """

        class bcolors:
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'

        outarr = [self._event_name]
        if desc:
            outarr.append(desc)
        if isinstance(scores, list):
            scorestring = 'Best scores: [ ' + ', '.join(
                [pretty_num(x) if not np.isnan(x) else 'NaN'
                 for x in scores]) + ' ]'
            outarr.append(scorestring)
        if isinstance(progress, list):
            progressstring = 'Progress: [ {}/{} ]'.format(*progress)
            outarr.append(progressstring)
        if self._emcee_est_t + self._bh_est_t > 0.0:
            if self._bh_est_t > 0.0 or not self._fracking:
                tott = self._emcee_est_t + self._bh_est_t
            else:
                tott = 2.0 * self._emcee_est_t
            timestring = self.get_timestring(tott)
            outarr.append(timestring)
        if isinstance(acor, list):
            acortstr = pretty_num(acor[0], sig=3)
            acorcstr = pretty_num(acor[1], sig=2)
            if self._travis:
                col = ''
            elif acor[1] < 5.0:
                col = bcolors.FAIL
            elif acor[1] < 10.0:
                col = bcolors.WARNING
            else:
                col = bcolors.OKGREEN
            acorstring = col
            acorstring = acorstring + 'Acor Tau: {} ({}x)'.format(acortstr,
                                                                  acorcstr)
            acorstring = acorstring + bcolors.ENDC if col else ''
            outarr.append(acorstring)

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

        print_inline(lines, new_line=self._travis)

    def get_timestring(self, t):
        """Return a string showing the estimated remaining time based upon
        elapsed times for emcee and fracking.
        """
        td = str(datetime.timedelta(seconds=round(t)))
        return ('Estimated time left: [ ' + td + ' ]')
