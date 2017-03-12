# -*- coding: UTF-8 -*-
"""Definitions for `Fitter` class."""
import io
import json
import os
import re
import shutil
import sys
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from difflib import get_close_matches

import dropbox
import emcee
import numpy as np
import scipy
from emcee.autocorr import AutocorrError
from mosfit.__init__ import __version__
from mosfit.printer import Printer
from mosfit.utils import (calculate_WAIC, entabbed_json_dump,
                          entabbed_json_dumps, flux_density_unit,
                          frequency_unit, get_model_hash, get_url_file_handle,
                          is_number, pretty_num)
from schwimmbad import MPIPool, SerialPool
from six import string_types

from astrocats.catalog.entry import ENTRY, Entry
from astrocats.catalog.model import MODEL
from astrocats.catalog.photometry import PHOTOMETRY
from astrocats.catalog.quantity import QUANTITY
from astrocats.catalog.realization import REALIZATION

from .model import Model

warnings.filterwarnings("ignore")


class MOSSampler(emcee.PTSampler):
    """Override PTSampler methods."""

    def get_autocorr_time(self, min_step=0, **kwargs):
        """Return a matrix of autocorrelation lengths.

        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.
        Any arguments will be passed to :func:`autocorr.integrate_time`.
        """
        acors = np.zeros((self.ntemps, self.dim))

        for i in range(self.ntemps):
            x = np.mean(self._chain[i, :, min_step:, :], axis=0)
            acors[i, :] = emcee.autocorr.integrated_time(x, **kwargs)
        return acors


def draw_walker(test=True):
    """Draw a walker from the global model variable."""
    global model
    return model.draw_walker(test)  # noqa: F821


def likelihood(x):
    """Return a likelihood score using the global model variable."""
    global model
    return model.likelihood(x)  # noqa: F821


def prior(x):
    """Return a prior score using the global model variable."""
    global model
    return model.prior(x)  # noqa: F821


def frack(x):
    """Frack at the specified parameter combination."""
    global model
    return model.frack(x)  # noqa: F821


class Fitter(object):
    """Fit transient events with the provided model."""

    def __init__(self):
        """Initialize `Fitter`."""
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
                   num_temps=1,
                   parameter_paths=[],
                   fracking=True,
                   frack_step=50,
                   wrap_length=100,
                   travis=False,
                   post_burn=500,
                   smooth_times=-1,
                   extrapolate_time=0.0,
                   limit_fitting_mjds=False,
                   exclude_bands=[],
                   exclude_instruments=[],
                   suffix='',
                   offline=False,
                   upload=False,
                   upload_token='',
                   check_upload_quality=False,
                   printer=None,
                   variance_for_each=[],
                   **kwargs):
        """Fit a list of events with a list of models."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._travis = travis
        self._wrap_length = wrap_length

        if not printer:
            self._printer = Printer(wrap_length=wrap_length)
        else:
            self._printer = printer

        prt = self._printer

        self._event_name = 'Batch'
        for event in events:
            self._event_name = ''
            self._event_path = ''
            if event:
                try:
                    pool = MPIPool()
                except Exception:
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
                        prt.wrapped(
                            'Event `{}` interpreted as supernova '
                            'name, downloading list of supernova '
                            'aliases...'.format(input_name))
                        if not offline:
                            try:
                                response = get_url_file_handle(
                                    'https://sne.space/astrocats/astrocats/'
                                    'supernovae/output/names.min.json',
                                    timeout=10)
                            except Exception:
                                prt.wrapped(
                                    'Warning: Could not download SN names ('
                                    'are you online?), using cached list.')
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
                            if offline:
                                print('Try omitting the `--offline` flag.')
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
                                prt.wrapped(
                                    'Could not find event, performing '
                                    'extended name search...')
                                snprefixes = set(('SN19', 'SN20'))
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
                                if travis:
                                    response = list(matches)[0]
                                else:
                                    response = prt.prompt(
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
                            prt.wrapped(
                                'Could not find event by that name, skipping!')
                            continue
                        urlname = self._event_name + '.json'
                        name_path = os.path.join(dir_path, 'cache', urlname)

                        if not offline:
                            prt.wrapped(
                                'Found event by primary name `{}` in the OSC, '
                                'downloading data...'.format(self._event_name))
                            try:
                                response = get_url_file_handle(
                                    'https://sne.space/astrocats/astrocats/'
                                    'supernovae/output/json/' + urlname,
                                    timeout=10)
                            except Exception:
                                prt.wrapped(
                                    'Warning: Could not download data for '
                                    ' `{}`, '
                                    'will attempt to use cached data.'.format(
                                        self._event_name))
                            else:
                                with open(name_path, 'wb') as f:
                                    shutil.copyfileobj(response, f)
                        path = name_path

                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            data = json.load(f, object_pairs_hook=OrderedDict)
                        prt.wrapped('Event file:')
                        prt.wrapped('  ' + path)
                    else:
                        prt.wrapped(
                            'Error: Could not find data for `{}` locally or '
                            'on the OSC.'.format(self._event_name))
                        if offline:
                            prt.wrapped(
                                'Try omitting the `--offline` flag.')
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
                    except Exception:
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
                        variance_for_each=variance_for_each,
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
                            suffix=suffix,
                            upload=upload,
                            upload_token=upload_token,
                            check_upload_quality=check_upload_quality)

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
                  variance_for_each=[],
                  pool=''):
        """Load the data for the specified event."""
        prt = self._printer
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

        # Run through once to set all inits.
        outputs = self._model.run_stack(
            [0.0 for x in range(self._model._num_free_parameters)],
            root='output')

        # Create any data-dependent free parameters.
        self._model.create_data_dependent_free_parameters(
            variance_for_each, outputs)

        # Determine free parameters again as above may have changed them.
        self._model.determine_free_parameters(fixed_parameters)

        self._model.exchange_requests()

        # Collect observed band info
        if pool.is_master() and 'photometry' in self._model._modules:
            prt.wrapped('Bands being used for current transient:')
            bis = list(
                filter(lambda a: a != -1, set(outputs['all_band_indices'])))
            ois = []
            for bi in bis:
                ois.append(
                    any([
                        y
                        for x, y in zip(outputs['all_band_indices'], outputs[
                            'observed']) if x == bi
                    ]))
            band_len = max([
                len(self._model._modules['photometry']._unique_bands[bi][
                    'SVO']) for bi in bis
            ])
            filts = self._model._modules['photometry']
            ubs = filts._unique_bands
            filterarr = [(ubs[bis[i]]['systems'], ubs[bis[i]]['bandsets'],
                          filts._average_wavelengths[bis[i]],
                          filts._band_offsets[bis[i]], ois[i], bis[i])
                         for i in range(len(bis))]
            filterrows = [(
                ' ' + (' ' if s[-2] else '*') + ubs[s[-1]]['SVO']
                .ljust(band_len) + ' [' + ', '.join(
                    list(
                        filter(None, (
                            'Bandset: ' + s[1] if s[1] else '',
                            'System: ' + s[0] if s[0] else '',
                            'AB offset: ' + pretty_num(s[3]))))) +
                ']').replace(' []', '') for s in list(sorted(filterarr))]
            if not all(ois):
                filterrows.append('  (* = Not observed in this band)')
            print('\n'.join(filterrows))

        self._event_name = event_name
        self._emcee_est_t = 0.0
        self._bh_est_t = 0.0
        self._fracking = fracking
        self._post_burn = post_burn
        self._burn_in = max(iterations - post_burn, 0)

        return True

    def fit_data(self,
                 event_name='',
                 iterations=2000,
                 frack_step=20,
                 num_walkers=50,
                 num_temps=1,
                 fracking=True,
                 post_burn=500,
                 pool='',
                 suffix='',
                 upload=False,
                 upload_token='',
                 check_upload_quality=True):
        """Fit the data for a given event.

        Fitting performed using a combination of emcee and fracking.
        """
        global model
        model = self._model
        prt = self._printer

        upload_this = upload and iterations > 0

        if not pool.is_master():
            try:
                pool.wait()
            except (KeyboardInterrupt, SystemExit):
                pass
            return

        ntemps, ndim, nwalkers = (num_temps, model._num_free_parameters,
                                  num_walkers)

        test_walker = iterations > 0
        lnprob = None
        pool_size = max(pool.size, 1)
        # Derived so only half a walker redrawn with Gaussian distribution.
        redraw_mult = 1.5 * np.sqrt(
            2) * scipy.special.erfinv(float(nwalkers - 1) / nwalkers)

        print('{} dimensions in problem.\n\n'.format(ndim))
        p0 = [[] for x in range(ntemps)]

        for i, pt in enumerate(p0):
            while len(p0[i]) < nwalkers:
                prt.status(
                    self,
                    desc='Drawing initial walkers',
                    progress=[i * nwalkers + len(p0[i]), nwalkers * ntemps])

                nmap = nwalkers - len(p0[i])
                p0[i].extend(pool.map(draw_walker, [test_walker] * nmap))

        sampler = MOSSampler(
            ntemps, nwalkers, ndim, likelihood, prior, pool=pool)

        prt.inline('Initial draws completed!')
        print('\n\n')
        p = list(p0)

        try:
            st = time.time()
            tft = 0.0  # Total fracking time
            acor = None

            # The argument of the for loop runs emcee, after each iteration of
            # emcee the contents of the for loop are executed.
            for emi, (p, lnprob, lnlike
                      ) in enumerate(sampler.sample(
                          p, iterations=iterations)):
                emi1 = emi + 1
                messages = []

                # First, redraw any walkers with scores significantly worse
                # than their peers (only during burn-in).
                if emi <= self._burn_in:
                    pmedian = [np.median(x) for x in lnprob]
                    pmead = [np.mean([abs(y - pmedian) for y in x])
                             for x in lnprob]
                    redraw_count = 0
                    bad_redraws = 0
                    for ti, tprob in enumerate(lnprob):
                        for wi, wprob in enumerate(tprob):
                            if (wprob <= pmedian[ti] -
                                redraw_mult * pmead[ti] or
                                    np.isnan(wprob)):
                                redraw_count = redraw_count + 1
                                dxx = np.random.normal(scale=0.01, size=ndim)
                                tar_x = np.array(p[np.random.randint(ntemps)][
                                    np.random.randint(nwalkers)])
                                new_x = np.clip(tar_x + dxx, 0.0, 1.0)
                                new_like = likelihood(new_x)
                                new_prob = new_like + prior(new_x)
                                if new_prob > wprob or np.isnan(wprob):
                                    p[ti][wi] = new_x
                                    lnlike[ti][wi] = new_like
                                    lnprob[ti][wi] = new_prob
                                else:
                                    bad_redraws = bad_redraws + 1
                    if redraw_count > 0:
                        messages.append('{:.1%} redraw, {}/{} success'.format(
                            redraw_count / (nwalkers * ntemps), redraw_count -
                            bad_redraws, redraw_count))

                low = 10
                asize = emi - self._burn_in
                if asize >= 1:
                    acorc = max(1, min(10, int(np.floor(asize / low))))
                    acort = -1.0
                    aa = 1
                    for a in range(acorc, 0, -1):
                        try:
                            acort = max([
                                max(x)
                                for x in sampler.get_autocorr_time(
                                    low=low, c=a, min_step=self._burn_in)
                            ])
                        except AutocorrError:
                            continue
                        else:
                            aa = a
                            break
                    acor = [acort, aa]

                self._emcee_est_t = float(time.time() - st - tft) / emi1 * (
                    iterations - emi1) + tft / emi1 * max(0, self._burn_in -
                                                          emi1)

                # Perform fracking if we are still in the burn in phase and
                # iteration count is a multiple of the frack step.
                frack_now = (fracking and emi1 <= self._burn_in and
                             emi1 % frack_step == 0)

                scores = [
                    np.array(x) + np.array(y) for x, y in zip(lnprob, lnlike)
                ]
                prt.status(
                    self,
                    desc='Fracking' if frack_now else 'Walking',
                    scores=scores,
                    progress=[emi1, iterations],
                    acor=acor,
                    messages=messages)

                if not frack_now:
                    continue

                # Fracking starts here
                sft = time.time()
                ijperms = [[x, y] for x in range(ntemps)
                           for y in range(nwalkers)]
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

                seeds = [
                    int(round(time.time() * 1000.0)) % 4294900000 + x
                    for x in range(len(bhwalkers))
                ]
                frack_args = list(zip(bhwalkers, seeds))
                bhs = list(pool.map(frack, frack_args))
                for bhi, bh in enumerate(bhs):
                    (wi, ti) = tuple(selijs[bhi])
                    if -bh.fun > lnprob[wi][ti] + lnlike[wi][ti]:
                        p[wi][ti] = bh.x
                        lnprob[wi][ti] = likelihood(bh.x)
                        lnlike[wi][ti] = prior(bh.x)
                scores = [[-x.fun for x in bhs]]
                prt.status(
                    self,
                    desc='Fracking Results',
                    scores=scores,
                    fracking=True,
                    progress=[emi1, iterations])
                tft = tft + time.time() - sft
        except (KeyboardInterrupt, SystemExit):
            pool.close()
            if (not prt.prompt(
                    'You have interrupted the Monte Carlo. Do you wish to '
                    'save the incomplete run to disk? Previous results will '
                    'be overwritten.', self._wrap_length)):
                sys.exit()
        except Exception:
            raise

        prt.wrapped('Saving output to disk...')
        if self._event_path:
            entry = Entry.init_from_file(
                catalog=None,
                name=self._event_name,
                path=self._event_path,
                merge=False,
                pop_schema=False,
                ignore_keys=[ENTRY.MODELS],
                compare_to_existing=False)
            new_photometry = []
            for photo in entry.get(ENTRY.PHOTOMETRY, []):
                if PHOTOMETRY.REALIZATION not in photo:
                    new_photometry.append(photo)
            if len(new_photometry):
                entry[ENTRY.PHOTOMETRY] = new_photometry
        else:
            entry = Entry(name=self._event_name)

        if upload:
            uentry = Entry(name=self._event_name)
            usource = uentry.add_source(name='MOSFiT paper')
            data_keys = set()
            for task in model._call_stack:
                if model._call_stack[task]['kind'] == 'data':
                    data_keys.update(
                        list(model._call_stack[task].get('keys', {}).keys()))
            entryhash = entry.get_hash(keys=list(sorted(list(data_keys))))

        source = entry.add_source(name='MOSFiT paper')
        model_setup = OrderedDict()
        for ti, task in enumerate(model._call_stack):
            task_copy = model._call_stack[task].copy()
            if (task_copy['kind'] == 'parameter' and
                    task in model._parameter_json):
                task_copy.update(model._parameter_json[task])
            model_setup[task] = task_copy
        modeldict = OrderedDict(
            [(MODEL.NAME, self._model._model_name), (MODEL.SETUP, model_setup),
             (MODEL.CODE, 'MOSFiT'), (MODEL.DATE, time.strftime("%Y/%m/%d")),
             (MODEL.VERSION, __version__), (MODEL.SOURCE, source)])

        if iterations > 0:
            WAIC = calculate_WAIC(scores)
            modeldict[MODEL.SCORE] = {
                QUANTITY.VALUE: str(WAIC),
                QUANTITY.KIND: 'WAIC'
            }
            if acor:
                modeldict[MODEL.CONVERGENCE] = {
                    QUANTITY.VALUE: str(aa),
                    QUANTITY.KIND: 'autocorrelationtimes'
                }
            modeldict[MODEL.STEPS] = str(emi1)

        if upload:
            umodeldict = deepcopy(modeldict)
            umodeldict[MODEL.SOURCE] = usource
            modelhash = get_model_hash(
                umodeldict, ignore_keys=[MODEL.DATE, MODEL.SOURCE])
            umodelnum = uentry.add_model(**umodeldict)
            if check_upload_quality:
                if WAIC < 0.0:
                    prt.wrapped(
                        'WAIC score `{}` below 0.0, not uploading this fit.'.
                        format(pretty_num(WAIC)))
                    upload_this = False

        modelnum = entry.add_model(**modeldict)

        ri = 1
        pout = sampler.chain[:, :, -1, :]
        lnprobout = sampler.lnprobability[:, :, -1]
        lnlikeout = sampler.lnlikelihood[:, :, -1]
        if acor:
            actc = int(np.ceil(acort))
            for i in range(1, np.int(float(emi - self._burn_in) / actc)):
                pout = np.concatenate(
                    (sampler.chain[:, :, -i * actc, :], pout), axis=1)
                lnprobout = np.concatenate(
                    (sampler.lnprobability[:, :, -i * actc], lnprobout),
                    axis=1)
                lnlikeout = np.concatenate(
                    (sampler.lnlikelihood[:, :, -i * actc], lnlikeout), axis=1)
        for xi, x in enumerate(pout):
            for yi, y in enumerate(pout[xi]):
                # Only produce LCs for end walker state.
                if yi <= nwalkers:
                    output = model.run_stack(y, root='output')
                    for i in range(len(output['times'])):
                        if not np.isfinite(output['model_observations'][i]):
                            continue
                        photodict = {
                            PHOTOMETRY.TIME:
                            output['times'][i] + output['min_times'],
                            PHOTOMETRY.MODEL: modelnum,
                            PHOTOMETRY.SOURCE: source,
                            PHOTOMETRY.REALIZATION: str(ri)
                        }
                        if output['observation_types'][i] == 'magnitude':
                            photodict[PHOTOMETRY.BAND] = output['bands'][i]
                            photodict[PHOTOMETRY.MAGNITUDE] = output[
                                'model_observations'][i]
                        if output['observation_types'][i] == 'fluxdensity':
                            photodict[PHOTOMETRY.FREQUENCY] = output[
                                'frequencies'][i] * frequency_unit('GHz')
                            photodict[PHOTOMETRY.FLUX_DENSITY] = output[
                                'model_observations'][
                                    i] * flux_density_unit('µJy')
                            photodict[PHOTOMETRY.U_FREQUENCY] = 'GHz'
                            photodict[PHOTOMETRY.U_FLUX_DENSITY] = 'µJy'
                        if output['systems'][i]:
                            photodict[PHOTOMETRY.SYSTEM] = output['systems'][i]
                        if output['bandsets'][i]:
                            photodict[PHOTOMETRY.BAND_SET] = output[
                                'bandsets'][i]
                        if output['instruments'][i]:
                            photodict[PHOTOMETRY.INSTRUMENT] = output[
                                'instruments'][i]
                        entry.add_photometry(
                            compare_to_existing=False, **photodict)

                        if upload_this:
                            uphotodict = deepcopy(photodict)
                            uphotodict[PHOTOMETRY.SOURCE] = umodelnum
                            uentry.add_photometry(
                                compare_to_existing=False, **uphotodict)
                else:
                    output = model.run_stack(y, root='objective')

                parameters = OrderedDict()
                derived_keys = set()
                pi = 0
                for ti, task in enumerate(model._call_stack):
                    if task not in model._free_parameters:
                        continue
                    poutput = model._modules[task].process(
                        **{'fraction': y[pi]})
                    value = list(poutput.values())[0]
                    paramdict = {
                        'value': value,
                        'fraction': y[pi],
                        'latex': model._modules[task].latex(),
                        'log': model._modules[task].is_log()
                    }
                    parameters.update({model._modules[task].name(): paramdict})
                    # Dump out any derived parameter keys
                    derived_keys.update(model._modules[task].get_derived_keys(
                    ))
                    pi = pi + 1

                for key in list(sorted(list(derived_keys))):
                    parameters.update({key: {'value': output[key]}})

                realdict = {REALIZATION.PARAMETERS: parameters}
                if lnprobout is not None and lnlike is not None:
                    realdict[REALIZATION.SCORE] = str(
                        lnprobout[xi][yi] + lnlikeout[xi][yi])
                realdict[REALIZATION.ALIAS] = str(ri)
                entry[ENTRY.MODELS][0].add_realization(**realdict)
                urealdict = deepcopy(realdict)
                if upload_this:
                    uentry[ENTRY.MODELS][0].add_realization(**urealdict)
                ri = ri + 1

        entry.sanitize()
        oentry = entry._ordered(entry)

        if not os.path.exists(model.MODEL_OUTPUT_DIR):
            os.makedirs(model.MODEL_OUTPUT_DIR)

        with io.open(
                os.path.join(model.MODEL_OUTPUT_DIR, 'walkers.json'),
                'w') as flast, io.open(
                    os.path.join(model.MODEL_OUTPUT_DIR, self._event_name + (
                        ('_' + suffix)
                        if suffix else '') + '.json'), 'w') as feven:
            entabbed_json_dump(oentry, flast, separators=(',', ':'))
            entabbed_json_dump(oentry, feven, separators=(',', ':'))

        if upload_this:
            uentry.sanitize()
            prt.wrapped('Uploading fit...')
            prt.wrapped(
                'Data hash: ' + entryhash + ', model hash: ' + modelhash)
            upath = '/' + '_'.join(
                [self._event_name, entryhash, modelhash]) + '.json'
            ouentry = {self._event_name: uentry._ordered(uentry)}
            upayload = entabbed_json_dumps(ouentry, separators=(',', ':'))
            try:
                dbx = dropbox.Dropbox(upload_token)
                dbx.files_upload(
                    upayload.encode(),
                    upath,
                    mode=dropbox.files.WriteMode.overwrite)
                prt.wrapped(
                    'Uploading complete!')
            except Exception:
                if self._travis:
                    pass
                else:
                    raise

        return (pout, lnprobout, lnlikeout)

    def generate_dummy_data(self,
                            name,
                            max_time=1000.,
                            plot_points=100,
                            band_list=[],
                            band_systems=[],
                            band_instruments=[],
                            band_bandsets=[]):
        """Generate simulated data based on priors."""
        time_list = np.linspace(0.0, max_time, plot_points)
        band_list_all = ['V'] if len(band_list) == 0 else band_list
        times = np.repeat(time_list, len(band_list_all))

        # Create lists of systems/instruments if not provided.
        if isinstance(band_systems, string_types):
            band_systems = [band_systems for x in range(len(band_list_all))]
        if isinstance(band_instruments, string_types):
            band_instruments = [
                band_instruments for x in range(len(band_list_all))
            ]
        if isinstance(band_bandsets, string_types):
            band_bandsets = [band_bandsets for x in range(len(band_list_all))]
        if len(band_systems) < len(band_list_all):
            rep_val = '' if len(band_systems) == 0 else band_systems[-1]
            band_systems = band_systems + [
                rep_val for x in range(len(band_list_all) - len(band_systems))
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
