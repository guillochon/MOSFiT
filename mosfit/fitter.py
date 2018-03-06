# -*- coding: UTF-8 -*-
"""Definitions for `Fitter` class."""
import gc
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import scipy
from astrocats.catalog.entry import ENTRY, Entry
from astrocats.catalog.model import MODEL
from astrocats.catalog.photometry import PHOTOMETRY
from astrocats.catalog.quantity import QUANTITY
from astrocats.catalog.realization import REALIZATION
from astrocats.catalog.source import SOURCE
from emcee.autocorr import AutocorrError
from mosfit.converter import Converter
from mosfit.fetcher import Fetcher
from mosfit.mossampler import MOSSampler
from mosfit.printer import Printer
from mosfit.utils import (all_to_list, calculate_WAIC, entabbed_json_dump,
                          entabbed_json_dumps, flux_density_unit,
                          frequency_unit, get_model_hash, listify, open_atomic,
                          pretty_num, slugify, speak)
from schwimmbad import MPIPool, SerialPool
from six import string_types

from .model import Model

warnings.filterwarnings("ignore")


def draw_walker(test=True, walkers_pool=[], replace=False):
    """Draw a walker from the global model variable."""
    global model
    return model.draw_walker(test, walkers_pool, replace)  # noqa: F821


def ln_likelihood(x):
    """Return ln(likelihood) using the global model variable."""
    global model
    return model.ln_likelihood(x)  # noqa: F821


def ln_prior(x):
    """Return ln(prior) using the global model variable."""
    global model
    return model.ln_prior(x)  # noqa: F821


def frack(x):
    """Frack at the specified parameter combination."""
    global model
    return model.frack(x)  # noqa: F821


class Fitter(object):
    """Fit transient events with the provided model."""

    _MAX_ACORC = 5
    _REPLACE_AGE = 20
    _DEFAULT_SOURCE = {SOURCE.BIBCODE: '2017arXiv171002145G'}

    def __init__(self,
                 cuda=False,
                 exit_on_prompt=False,
                 language='en',
                 limiting_magnitude=None,
                 offline=False,
                 prefer_cache=False,
                 open_in_browser=False,
                 pool=None,
                 quiet=False,
                 test=False,
                 wrap_length=100,
                 **kwargs):
        """Initialize `Fitter` class."""
        self._pool = SerialPool() if pool is None else pool
        self._printer = Printer(
            pool=self._pool, wrap_length=wrap_length, quiet=quiet, fitter=self,
            language=language, exit_on_prompt=exit_on_prompt)
        self._fetcher = Fetcher(test=test, open_in_browser=open_in_browser,
                                printer=self._printer)

        self._cuda = cuda
        self._limiting_magnitude = limiting_magnitude
        self._offline = offline
        self._prefer_cache = prefer_cache
        self._open_in_browser = open_in_browser
        self._quiet = quiet
        self._test = test
        self._wrap_length = wrap_length

        if self._cuda:
            try:
                import pycuda.autoinit  # noqa: F401
                import skcuda.linalg as linalg
                linalg.init()
            except ImportError:
                pass

    def fit_events(self,
                   events=[],
                   models=[],
                   max_time='',
                   band_list=[],
                   band_systems=[],
                   band_instruments=[],
                   band_bandsets=[],
                   band_sampling_points=17,
                   iterations=5000,
                   num_walkers=None,
                   num_temps=1,
                   parameter_paths=['parameters.json'],
                   fracking=True,
                   frack_step=50,
                   burn=None,
                   post_burn=None,
                   gibbs=False,
                   smooth_times=-1,
                   extrapolate_time=0.0,
                   limit_fitting_mjds=False,
                   exclude_bands=[],
                   exclude_instruments=[],
                   exclude_systems=[],
                   exclude_sources=[],
                   exclude_kinds=[],
                   suffix='',
                   upload=False,
                   write=False,
                   upload_token='',
                   check_upload_quality=False,
                   variance_for_each=[],
                   user_fixed_parameters=[],
                   convergence_type='psrf',
                   convergence_criteria=None,
                   save_full_chain=False,
                   draw_above_likelihood=False,
                   maximum_walltime=False,
                   start_time=False,
                   print_trees=False,
                   maximum_memory=np.inf,
                   speak=False,
                   return_fits=True,
                   extra_outputs=[],
                   walker_paths=[],
                   catalogs=[],
                   exit_on_prompt=False,
                   download_recommended_data=False,
                   local_data_only=False,
                   **kwargs):
        """Fit a list of events with a list of models."""
        global model
        if start_time is False:
            start_time = time.time()
        self._start_time = start_time
        self._maximum_walltime = maximum_walltime
        self._maximum_memory = maximum_memory
        self._debug = False
        self._speak = speak
        self._download_recommended_data = download_recommended_data
        self._local_data_only = local_data_only

        self._draw_above_likelihood = draw_above_likelihood

        prt = self._printer

        event_list = listify(events)
        model_list = listify(models)

        if len(model_list) and not len(event_list):
            event_list = ['']

        # Exclude catalogs not included in catalog list.
        self._fetcher.add_excluded_catalogs(catalogs)

        if not len(event_list) and not len(model_list):
            prt.message('no_events_models', warning=True)

        # If the input is not a JSON file, assume it is either a list of
        # transients or that it is the data from a single transient in tabular
        # form. Try to guess the format first, and if that fails ask the user.
        self._converter = Converter(prt, require_source=upload)
        event_list = self._converter.generate_event_list(event_list)

        event_list = [x.replace('‑', '-') for x in event_list]

        entries = [[] for x in range(len(event_list))]
        ps = [[] for x in range(len(event_list))]
        lnprobs = [[] for x in range(len(event_list))]

        # Load walker data if provided a list of walker paths.
        walker_data = []

        if len(walker_paths):
            try:
                pool = MPIPool()
            except (ImportError, ValueError):
                pool = SerialPool()
            if pool.is_master():
                prt.message('walker_file')
                wfi = 0
                for walker_path in walker_paths:
                    if os.path.exists(walker_path):
                        prt.prt('  {}'.format(walker_path))
                        with open(walker_path, 'r') as f:
                            all_walker_data = json.load(
                                f, object_pairs_hook=OrderedDict)

                        # Support both the format where all data stored in a
                        # single-item dictionary (the OAC format) and the older
                        # MOSFiT format where the data was stored in the
                        # top-level dictionary.
                        if ENTRY.NAME not in all_walker_data:
                            all_walker_data = all_walker_data[
                                list(all_walker_data.keys())[0]]

                        models = all_walker_data.get(ENTRY.MODELS, [])
                        choice = None
                        if len(models) > 1:
                            model_opts = [
                                '{}-{}-{}'.format(
                                    x['code'], x['name'], x['date'])
                                for x in models]
                            choice = prt.prompt(
                                'select_model_walkers', kind='select',
                                message=True, options=model_opts)
                            choice = model_opts.index(choice)
                        elif len(models) == 1:
                            choice = 0

                        if choice is not None:
                            walker_data.extend([
                                [wfi, x[REALIZATION.PARAMETERS]]
                                for x in models[choice][
                                    MODEL.REALIZATIONS]])

                        if not len(walker_data):
                            prt.message('no_walker_data')
                    else:
                        prt.message('no_walker_data')
                        if self._offline:
                            prt.message('omit_offline')
                        raise RuntimeError
                    wfi = wfi + 1

                for rank in range(1, pool.size + 1):
                    pool.comm.send(walker_data, dest=rank, tag=3)
            else:
                walker_data = pool.comm.recv(source=0, tag=3)
                pool.wait()

            if pool.is_master():
                pool.close()

        self._event_name = 'Batch'
        self._event_path = ''
        self._event_data = {}

        try:
            pool = MPIPool()
        except (ImportError, ValueError):
            pool = SerialPool()
        if pool.is_master():
            fetched_events = self._fetcher.fetch(
                event_list, offline=self._offline,
                prefer_cache=self._prefer_cache)

            for rank in range(1, pool.size + 1):
                pool.comm.send(fetched_events, dest=rank, tag=0)
            pool.close()
        else:
            fetched_events = pool.comm.recv(source=0, tag=0)
            pool.wait()

        for ei, event in enumerate(fetched_events):
            if event is not None:
                self._event_name = event.get('name', 'Batch')
                self._event_path = event.get('path', '')
                if not self._event_path:
                    continue
                self._event_data = self._fetcher.load_data(event)
                if not self._event_data:
                    continue

            if model_list:
                lmodel_list = model_list
            else:
                lmodel_list = ['']

            entries[ei] = [None for y in range(len(lmodel_list))]
            ps[ei] = [None for y in range(len(lmodel_list))]
            lnprobs[ei] = [None for y in range(len(lmodel_list))]

            if (event is not None and (not self._event_data or
                ENTRY.PHOTOMETRY not in self._event_data[
                    list(self._event_data.keys())[0]])):
                prt.message('no_photometry', [self._event_name])
                continue

            for mi, mod_name in enumerate(lmodel_list):
                for parameter_path in parameter_paths:
                    try:
                        pool = MPIPool()
                    except (ImportError, ValueError):
                        pool = SerialPool()
                    self._model = Model(
                        model=mod_name,
                        data=self._event_data,
                        parameter_path=parameter_path,
                        wrap_length=self._wrap_length,
                        test=self._test,
                        printer=prt,
                        fitter=self,
                        pool=pool,
                        print_trees=print_trees)

                    if not self._model._model_name:
                        prt.message('no_models_avail', [
                            self._event_name], warning=True)
                        continue

                    if not event:
                        prt.message('gen_dummy')
                        self._event_name = mod_name
                        gen_args = {
                            'name': mod_name,
                            'max_time': max_time,
                            'band_list': band_list,
                            'band_systems': band_systems,
                            'band_instruments': band_instruments,
                            'band_bandsets': band_bandsets
                        }
                        self._event_data = self.generate_dummy_data(**gen_args)

                    success = False
                    alt_name = None
                    while not success:
                        self._model.reset_unset_recommended_keys()
                        success = self._model.load_data(
                            self._event_data,
                            event_name=self._event_name,
                            smooth_times=smooth_times,
                            extrapolate_time=extrapolate_time,
                            limit_fitting_mjds=limit_fitting_mjds,
                            exclude_bands=exclude_bands,
                            exclude_instruments=exclude_instruments,
                            exclude_systems=exclude_systems,
                            exclude_sources=exclude_sources,
                            exclude_kinds=exclude_kinds,
                            band_list=band_list,
                            band_systems=band_systems,
                            band_instruments=band_instruments,
                            band_bandsets=band_bandsets,
                            band_sampling_points=band_sampling_points,
                            variance_for_each=variance_for_each,
                            user_fixed_parameters=user_fixed_parameters,
                            pool=pool)

                        if not success:
                            break

                        if self._local_data_only:
                            break

                        # If our data is missing recommended keys, offer the
                        # user option to pull the missing data from online and
                        # merge with existing data.
                        urk = self._model.get_unset_recommended_keys()
                        ptxt = prt.text('acquire_recommended', [
                            ', '.join(list(urk))])
                        while event and len(urk) and (
                            alt_name or self._download_recommended_data or
                            prt.prompt(
                                ptxt, [', '.join(urk)], kind='bool')):
                            try:
                                pool = MPIPool()
                            except (ImportError, ValueError):
                                pool = SerialPool()
                            if pool.is_master():
                                en = (alt_name if alt_name
                                      else self._event_name)
                                extra_event = self._fetcher.fetch(
                                    en, offline=self._offline,
                                    prefer_cache=self._prefer_cache)[0]
                                extra_data = self._fetcher.load_data(
                                    extra_event)

                                for rank in range(1, pool.size + 1):
                                    pool.comm.send(extra_data, dest=rank,
                                                   tag=4)
                                pool.close()
                            else:
                                extra_data = pool.comm.recv(
                                    source=0, tag=4)
                                pool.wait()

                            if extra_data is not None:
                                extra_data = extra_data[
                                    list(extra_data.keys())[0]]

                                for key in urk:
                                    new_val = extra_data.get(key)
                                    self._event_data[list(
                                        self._event_data.keys())[0]][
                                            key] = new_val
                                    if new_val is not None and len(
                                            new_val):
                                        prt.message('extra_value', [
                                            key, str(new_val[0].get(
                                                QUANTITY.VALUE))])
                                success = False
                                prt.message('reloading_merged')
                                break
                            else:
                                text = prt.text(
                                    'extra_not_found', [self._event_name])
                                alt_name = prt.prompt(text, kind='string')
                                if not alt_name:
                                    break

                    if success:
                        self._walker_data = walker_data

                        entry, p, lnprob = self.fit_data(
                            event_name=self._event_name,
                            iterations=iterations,
                            num_walkers=num_walkers,
                            num_temps=num_temps,
                            burn=burn,
                            post_burn=post_burn,
                            fracking=fracking,
                            frack_step=frack_step,
                            gibbs=gibbs,
                            pool=pool,
                            suffix=suffix,
                            write=write,
                            upload=upload,
                            upload_token=upload_token,
                            check_upload_quality=check_upload_quality,
                            convergence_type=convergence_type,
                            convergence_criteria=convergence_criteria,
                            save_full_chain=save_full_chain,
                            extra_outputs=extra_outputs)
                        if return_fits:
                            entries[ei][mi] = deepcopy(entry)
                            ps[ei][mi] = deepcopy(p)
                            lnprobs[ei][mi] = deepcopy(lnprob)

                    if pool.is_master():
                        pool.close()

                    # Remove global model variable and garbage collect.
                    try:
                        model
                    except NameError:
                        pass
                    else:
                        del(model)
                    del(self._model)
                    gc.collect()

        return (entries, ps, lnprobs)

    def fit_data(self,
                 event_name='',
                 iterations=2000,
                 frack_step=20,
                 num_walkers=None,
                 num_temps=1,
                 burn=None,
                 post_burn=None,
                 fracking=True,
                 gibbs=False,
                 pool=None,
                 suffix='',
                 write=False,
                 upload=False,
                 upload_token='',
                 check_upload_quality=True,
                 convergence_type='psrf',
                 convergence_criteria=None,
                 save_full_chain=False,
                 extra_outputs=[]):
        """Fit the data for a given event.

        Fitting performed using a combination of emcee and fracking.
        """
        if self._speak:
            speak('Fitting ' + event_name, self._speak)
        from mosfit.__init__ import __version__
        global model
        model = self._model
        prt = self._printer

        self._emcee_est_t = 0.0
        self._bh_est_t = 0.0
        if burn is not None:
            self._burn_in = min(burn, iterations)
        elif post_burn is not None:
            self._burn_in = max(iterations - post_burn, 0)
        else:
            self._burn_in = int(np.round(iterations / 2))

        if pool is not None:
            self._pool = pool

        upload_model = upload and iterations > 0

        if upload:
            try:
                import dropbox
            except ImportError:
                if self._test:
                    pass
                else:
                    prt.message('install_db', error=True)
                    raise

        if not self._pool.is_master():
            try:
                self._pool.wait()
            except (KeyboardInterrupt, SystemExit):
                pass
            return (None, None, None)

        ntemps, ndim = (num_temps, model._num_free_parameters)

        if num_walkers:
            nwalkers = num_walkers
        else:
            nwalkers = 2 * ndim

        test_walker = iterations > 0
        lnprob = None
        lnlike = None
        pool_size = max(self._pool.size, 1)
        # Derived so only half a walker redrawn with Gaussian distribution.
        redraw_mult = 0.5 * np.sqrt(
            2) * scipy.special.erfinv(float(nwalkers - 1) / nwalkers)

        prt.message('nmeas_nfree', [model._num_measurements, ndim])
        if test_walker:
            if model._num_measurements <= ndim:
                prt.message('too_few_walkers', warning=True)
            if nwalkers < 10 * ndim:
                prt.message('want_more_walkers', [10 * ndim, nwalkers],
                            warning=True)
        p0 = [[] for x in range(ntemps)]

        # Generate walker positions based upon loaded walker data, if
        # available.
        walkers_pool = []
        nmodels = len(set([x[0] for x in self._walker_data]))
        wp_extra = 0
        while len(walkers_pool) < len(self._walker_data):
            appended_walker = False
            for walk in self._walker_data:
                if (len(walkers_pool) + wp_extra) % nmodels != walk[0]:
                    continue
                new_walk = np.full(model._num_free_parameters, None)
                for k, key in enumerate(model._free_parameters):
                    param = model._modules[key]
                    walk_param = walk[1].get(key, None)
                    if walk_param is None:
                        continue
                    if param:
                        new_walk[k] = param.fraction(walk_param['value'])
                walkers_pool.append(new_walk)
                appended_walker = True
            if not appended_walker:
                wp_extra += 1

        # Draw walker positions. This is either done from the priors or from
        # loaded walker data. If some parameters are not available from the
        # loaded walker data they will be drawn from their priors instead.
        pool_len = len(walkers_pool)
        for i, pt in enumerate(p0):
            dwscores = []
            while len(p0[i]) < nwalkers:
                prt.status(
                    desc='drawing_walkers',
                    progress=[
                        i * nwalkers + len(p0[i]) + 1, nwalkers * ntemps])

                if self._pool.size == 0 or pool_len:
                    p, score = draw_walker(
                        test_walker, walkers_pool,
                        replace=pool_len < ntemps * nwalkers)
                    p0[i].append(p)
                    dwscores.append(score)
                else:
                    nmap = min(nwalkers - len(p0[i]), max(self._pool.size, 10))
                    dws = self._pool.map(draw_walker, [test_walker] * nmap)
                    p0[i].extend([x[0] for x in dws])
                    dwscores.extend([x[1] for x in dws])

                if self._draw_above_likelihood is not False:
                    self._draw_above_likelihood = np.mean(dwscores)

        prt.message('initial_draws', inline=True)
        p = list(p0)

        sli = 1.0  # Keep track of how many times chain halved
        emi = 0
        tft = 0.0  # Total fracking time
        acor = None
        aacort = -1
        aa = 0
        psrf = np.inf
        s_exception = None
        kmat = None
        all_chain = np.array([])
        scores = np.ones((ntemps, nwalkers)) * -np.inf
        ages = np.zeros((ntemps, nwalkers), dtype=int)
        oldp = p

        max_chunk = 1000
        kmat_chunk = 5
        iter_chunks = int(np.ceil(float(iterations) / max_chunk))
        iter_arr = [max_chunk if xi < iter_chunks - 1 else
                    iterations - max_chunk * (iter_chunks - 1)
                    for xi, x in enumerate(range(iter_chunks))]
        # Make sure a chunk separation is located at self._burn_in
        chunk_is = sorted(set(
            np.concatenate(([0, self._burn_in], np.cumsum(iter_arr)))))
        iter_arr = np.diff(chunk_is)

        # The argument of the for loop runs emcee, after each iteration of
        # emcee the contents of the for loop are executed.
        converged = False
        exceeded_walltime = False
        ici = 0

        try:
            if iterations > 0:
                sampler = MOSSampler(
                    ntemps, nwalkers, ndim, ln_likelihood, ln_prior,
                    pool=self._pool)
                st = time.time()
            while (iterations > 0 and (
                    convergence_criteria is not None or ici < len(iter_arr))):
                slr = int(np.round(sli))
                ic = (max_chunk if convergence_criteria is not None else
                      iter_arr[ici])
                if exceeded_walltime:
                    break
                if (convergence_criteria is not None and converged and
                        emi > iterations):
                    break
                for li, (
                        p, lnprob, lnlike) in enumerate(
                            sampler.sample(
                                p, iterations=ic, gibbs=gibbs if
                                emi >= self._burn_in else True)):
                    if (self._maximum_walltime is not False and
                            time.time() - self._start_time >
                            self._maximum_walltime):
                        prt.message('exceeded_walltime', warning=True)
                        exceeded_walltime = True
                        break
                    emi = emi + 1
                    emim1 = emi - 1
                    messages = []

                    # Increment the age of each walker if their positions are
                    # unchanged.
                    for ti in range(ntemps):
                        for wi in range(nwalkers):
                            if np.array_equal(p[ti][wi], oldp[ti][wi]):
                                ages[ti][wi] += 1
                            else:
                                ages[ti][wi] = 0

                    # Record then reset sampler proposal/acceptance counts.
                    accepts = list(
                        np.mean(sampler.nprop_accepted / sampler.nprop,
                                axis=1))
                    sampler.nprop = np.zeros(
                        (sampler.ntemps, sampler.nwalkers), dtype=np.float)
                    sampler.nprop_accepted = np.zeros(
                        (sampler.ntemps, sampler.nwalkers),
                        dtype=np.float)

                    # During burn-in only, redraw any walkers with scores
                    # significantly worse than their peers, or those that are
                    # stale (i.e. remained in the same position for a long
                    # time).
                    if emim1 <= self._burn_in:
                        pmedian = [np.median(x) for x in lnprob]
                        pmead = [np.mean([abs(y - pmedian) for y in x])
                                 for x in lnprob]
                        redraw_count = 0
                        bad_redraws = 0
                        for ti, tprob in enumerate(lnprob):
                            for wi, wprob in enumerate(tprob):
                                if (wprob <= pmedian[ti] -
                                    max(redraw_mult * pmead[ti],
                                        float(nwalkers)) or
                                        np.isnan(wprob) or
                                        ages[ti][wi] >= self._REPLACE_AGE):
                                    redraw_count = redraw_count + 1
                                    dxx = np.random.normal(
                                        scale=0.01, size=ndim)
                                    tar_x = np.array(
                                        p[np.random.randint(ntemps)][
                                            np.random.randint(nwalkers)])
                                    # Reflect if out of bounds.
                                    new_x = np.clip(np.where(
                                        np.where(tar_x + dxx < 1.0,
                                                 tar_x + dxx,
                                                 tar_x - dxx) > 0.0,
                                        tar_x + dxx, tar_x - dxx), 0.0, 1.0)
                                    new_like = ln_likelihood(new_x)
                                    new_prob = new_like + ln_prior(new_x)
                                    if new_prob > wprob or np.isnan(wprob):
                                        p[ti][wi] = new_x
                                        lnlike[ti][wi] = new_like
                                        lnprob[ti][wi] = new_prob
                                    else:
                                        bad_redraws = bad_redraws + 1
                        if redraw_count > 0:
                            messages.append(
                                '{:.0%} redraw, {}/{} success'.format(
                                    redraw_count / (nwalkers * ntemps),
                                    redraw_count - bad_redraws, redraw_count))

                    oldp = p.copy()

                    # Calculate the autocorrelation time.
                    low = 10
                    asize = 0.5 * (emim1 - self._burn_in) / low
                    if asize >= 0 and convergence_type == 'acor':
                        acorc = max(
                            1, min(self._MAX_ACORC,
                                   int(np.floor(0.5 * emi / low))))
                        aacort = -1.0
                        aa = 0
                        ams = self._burn_in
                        cur_chain = (np.concatenate(
                            (all_chain, sampler.chain[:, :, :li + 1:slr, :]),
                            axis=2) if len(all_chain) else
                            sampler.chain[:, :, :li + 1:slr, :])
                        for a in range(acorc, 1, -1):
                            ms = self._burn_in
                            if ms >= emi - low:
                                break
                            try:
                                acorts = sampler.get_autocorr_time(
                                    chain=cur_chain, low=low, c=a,
                                    min_step=int(np.round(float(ms) / sli)),
                                    max_walkers=5, fast=True)
                                acort = max([
                                    max(x)
                                    for x in acorts
                                ])
                            except AutocorrError:
                                continue
                            else:
                                aa = a
                                aacort = acort * sli
                                ams = ms
                                break
                        acor = [aacort, aa, ams]

                        actc = int(np.ceil(aacort / sli))
                        actn = np.int(float(emi - ams) / actc)

                        if (convergence_criteria is not None and
                            actn >= convergence_criteria and
                                emi > iterations):
                            prt.message('converged')
                            converged = True
                            break

                    # Calculate the PSRF (Gelman-Rubin statistic).
                    if li > 1 and emi > self._burn_in + 2:
                        cur_chain = (np.concatenate(
                            (all_chain, sampler.chain[:, :, :li + 1:slr, :]),
                            axis=2) if len(all_chain) else
                            sampler.chain[:, :, :li + 1:slr, :])
                        vws = np.zeros((ntemps, ndim))
                        for ti in range(ntemps):
                            for xi in range(ndim):
                                vchain = cur_chain[
                                    ti, :, int(np.floor(
                                        self._burn_in / sli)):, xi]
                                vws[ti][xi] = self.psrf(vchain)
                        psrf = np.max(vws)
                        if np.isnan(psrf):
                            psrf = np.inf

                        if (convergence_type == 'psrf' and
                            convergence_criteria is not None and
                            psrf < convergence_criteria and
                                emi > iterations):
                            prt.message('converged')
                            converged = True
                            break

                    if convergence_criteria is not None:
                        self._emcee_est_t = -1.0
                    else:
                        self._emcee_est_t = float(
                            time.time() - st - tft) / emi * (
                            iterations - emi) + tft / emi * max(
                                0, self._burn_in - emi)

                    # Perform fracking if we are still in the burn in phase
                    # and iteration count is a multiple of the frack step.
                    frack_now = (fracking and frack_step != 0 and
                                 emi <= self._burn_in and
                                 emi % frack_step == 0)

                    scores = [np.array(x) for x in lnprob]
                    if emim1 % kmat_chunk == 0:
                        sout = model.run_stack(
                            p[np.unravel_index(
                                np.argmax(lnprob), lnprob.shape)],
                            root='objective')
                        kmat = sout.get('kmat')
                        kdiag = sout.get('kdiagonal')
                        variance = sout.get('obandvs', sout.get('variance'))
                        if kdiag is not None and kmat is not None:
                            kmat[np.diag_indices_from(kmat)] += kdiag
                        elif kdiag is not None and kmat is None:
                            kmat = np.diag(kdiag + variance)
                    prt.status(
                        desc='fracking' if frack_now else
                        ('burning' if emi < self._burn_in else 'walking'),
                        scores=scores,
                        kmat=kmat,
                        accepts=accepts,
                        progress=[emi, None if
                                  convergence_criteria is not None else
                                  iterations],
                        acor=acor,
                        psrf=[psrf, self._burn_in],
                        messages=messages,
                        make_space=emim1 == 0,
                        convergence_type=convergence_type,
                        convergence_criteria=convergence_criteria)

                    if s_exception:
                        break

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
                    bhs = list(self._pool.map(frack, frack_args))
                    for bhi, bh in enumerate(bhs):
                        (wi, ti) = tuple(selijs[bhi])
                        if -bh.fun > lnprob[wi][ti]:
                            p[wi][ti] = bh.x
                            like = ln_likelihood(bh.x)
                            lnprob[wi][ti] = like + ln_prior(bh.x)
                            lnlike[wi][ti] = like
                    scores = [[-x.fun for x in bhs]]
                    prt.status(
                        desc='fracking_results',
                        scores=scores,
                        kmat=kmat,
                        fracking=True,
                        progress=[emi, None if
                                  convergence_criteria is not None else
                                  iterations],
                        convergence_type=convergence_type,
                        convergence_criteria=convergence_criteria)
                    tft = tft + time.time() - sft
                    if s_exception:
                        break

                if ici == 0:
                    all_chain = sampler.chain[:, :, :li + 1:slr, :]
                    all_lnprob = sampler.lnprobability[:, :, :li + 1:slr]
                    all_lnlike = sampler.lnlikelihood[:, :, :li + 1:slr]
                else:
                    all_chain = np.concatenate(
                        (all_chain, sampler.chain[:, :, :li + 1:slr, :]),
                        axis=2)
                    all_lnprob = np.concatenate(
                        (all_lnprob, sampler.lnprobability[:, :, :li + 1:slr]),
                        axis=2)
                    all_lnlike = np.concatenate(
                        (all_lnlike, sampler.lnlikelihood[:, :, :li + 1:slr]),
                        axis=2)

                mem_mb = (all_chain.nbytes + all_lnprob.nbytes +
                          all_lnlike.nbytes) / (1024. * 1024.)

                if self._debug:
                    prt.prt('Memory `{}`'.format(mem_mb), wrapped=True)

                if mem_mb > self._maximum_memory:
                    sfrac = float(
                        all_lnprob.shape[-1]) / all_lnprob[:, :, ::2].shape[-1]
                    all_chain = all_chain[:, :, ::2, :]
                    all_lnprob = all_lnprob[:, :, ::2]
                    all_lnlike = all_lnlike[:, :, ::2]
                    sli *= sfrac
                    if self._debug:
                        prt.prt(
                            'Memory halved, sli: {}'.format(sli),
                            wrapped=True)

                sampler.reset()
                gc.collect()
                ici = ici + 1

        except (KeyboardInterrupt, SystemExit):
            prt.message('ctrl_c', error=True, prefix=False, color='!r')
            s_exception = sys.exc_info()
        except Exception:
            raise

        if s_exception:
            self._pool.close()
            if (not prt.prompt('mc_interrupted')):
                sys.exit()

        msg_criteria = (
            1.1 if convergence_criteria is None else convergence_criteria)
        if (test_walker and convergence_type == 'psrf' and
                msg_criteria is not None and psrf > msg_criteria):
            prt.message('not_converged', [
                'default' if convergence_criteria is None else 'specified',
                msg_criteria], warning=True)

        prt.message('constructing')

        if write:
            if self._speak:
                speak(prt._strings['saving_output'], self._speak)

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

        uentry = Entry(name=self._event_name)
        data_keys = set()
        for task in model._call_stack:
            if model._call_stack[task]['kind'] == 'data':
                data_keys.update(
                    list(model._call_stack[task].get('keys', {}).keys()))
        entryhash = entry.get_hash(keys=list(sorted(list(data_keys))))

        # Accumulate all the sources and add them to each entry.
        sources = []
        for root in self._model._references:
            for ref in self._model._references[root]:
                sources.append(entry.add_source(**ref))
        sources.append(entry.add_source(**self._DEFAULT_SOURCE))
        source = ','.join(sources)

        usources = []
        for root in self._model._references:
            for ref in self._model._references[root]:
                usources.append(uentry.add_source(**ref))
        usources.append(uentry.add_source(**self._DEFAULT_SOURCE))
        usource = ','.join(usources)

        model_setup = OrderedDict()
        for ti, task in enumerate(model._call_stack):
            task_copy = deepcopy(model._call_stack[task])
            if (task_copy['kind'] == 'parameter' and
                    task in model._parameter_json):
                task_copy.update(model._parameter_json[task])
            model_setup[task] = task_copy
        modeldict = OrderedDict(
            [(MODEL.NAME, self._model._model_name), (MODEL.SETUP, model_setup),
             (MODEL.CODE, 'MOSFiT'), (MODEL.DATE, time.strftime("%Y/%m/%d")),
             (MODEL.VERSION, __version__), (MODEL.SOURCE, source)])

        WAIC = None
        if iterations > 0:
            WAIC = calculate_WAIC(scores)
            modeldict[MODEL.SCORE] = {
                QUANTITY.VALUE: str(WAIC),
                QUANTITY.KIND: 'WAIC'
            }
            modeldict[MODEL.CONVERGENCE] = []
            if psrf < np.inf:
                modeldict[MODEL.CONVERGENCE].append(
                    {
                        QUANTITY.VALUE: str(psrf),
                        QUANTITY.KIND: 'psrf'
                    }
                )
            if acor and aacort > 0:
                acortimes = '<' if aa < self._MAX_ACORC else ''
                acortimes += str(np.int(float(emi - ams) / actc))
                modeldict[MODEL.CONVERGENCE].append(
                    {
                        QUANTITY.VALUE: str(acortimes),
                        QUANTITY.KIND: 'autocorrelationtimes'
                    }
                )
            modeldict[MODEL.STEPS] = str(emi)

        umodeldict = deepcopy(modeldict)
        umodeldict[MODEL.SOURCE] = usource
        modelhash = get_model_hash(
            umodeldict, ignore_keys=[MODEL.DATE, MODEL.SOURCE])
        umodelnum = uentry.add_model(**umodeldict)
        if check_upload_quality:
            if WAIC is None:
                upload_model = False
            elif WAIC is not None and WAIC < 0.0:
                if upload:
                    prt.message('no_ul_waic', ['' if WAIC is None
                                               else pretty_num(WAIC)])
                upload_model = False

        modelnum = entry.add_model(**modeldict)

        ri = 1
        if len(all_chain):
            pout = all_chain[:, :, -1, :]
            lnprobout = all_lnprob[:, :, -1]
            lnlikeout = all_lnlike[:, :, -1]
        else:
            pout = p
            lnprobout = lnprob
            lnlikeout = lnlike

        # Here, we append to the vector of walkers from the full chain based
        # upon the value of acort (the autocorrelation timescale).
        if acor and aacort > 0 and aa == self._MAX_ACORC:
            actc0 = int(np.ceil(aacort))
            for i in range(1, np.int(float(emi - ams) / actc0)):
                pout = np.concatenate(
                    (all_chain[:, :, -i * actc, :], pout), axis=1)
                lnprobout = np.concatenate(
                    (all_lnprob[:, :, -i * actc], lnprobout), axis=1)
                lnlikeout = np.concatenate(
                    (all_lnlike[:, :, -i * actc], lnlikeout), axis=1)

        extras = OrderedDict()
        for xi, x in enumerate(pout):
            for yi, y in enumerate(pout[xi]):
                # Only produce LCs for end walker state.
                wcnt = xi * nwalkers + yi
                if wcnt > 0:
                    prt.message('outputting_walker', [
                        wcnt, nwalkers * ntemps], inline=True)
                if yi <= nwalkers:
                    output = model.run_stack(y, root='output')
                    if extra_outputs:
                        for key in extra_outputs:
                            new_val = output.get(key, [])
                            new_val = all_to_list(new_val)
                            extras.setdefault(key, []).append(new_val)
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
                            photodict[PHOTOMETRY.E_MAGNITUDE] = output[
                                'model_variances'][i]
                        if output['observation_types'][i] == 'fluxdensity':
                            photodict[PHOTOMETRY.FREQUENCY] = output[
                                'frequencies'][i] * frequency_unit('GHz')
                            photodict[PHOTOMETRY.FLUX_DENSITY] = output[
                                'model_observations'][
                                    i] * flux_density_unit('µJy')
                            photodict[
                                PHOTOMETRY.
                                E_LOWER_FLUX_DENSITY] = (
                                    photodict[PHOTOMETRY.FLUX_DENSITY] - (
                                        10.0 ** (
                                            np.log10(photodict[
                                                PHOTOMETRY.FLUX_DENSITY]) -
                                            output['model_variances'][
                                                i] / 2.5)) *
                                    flux_density_unit('µJy'))
                            photodict[
                                PHOTOMETRY.
                                E_UPPER_FLUX_DENSITY] = (10.0 ** (
                                    np.log10(photodict[
                                        PHOTOMETRY.FLUX_DENSITY]) +
                                    output['model_variances'][i] / 2.5) *
                                    flux_density_unit('µJy') -
                                    photodict[PHOTOMETRY.FLUX_DENSITY])
                            photodict[PHOTOMETRY.U_FREQUENCY] = 'GHz'
                            photodict[PHOTOMETRY.U_FLUX_DENSITY] = 'µJy'
                        if output['observation_types'][i] == 'countrate':
                            photodict[PHOTOMETRY.COUNT_RATE] = output[
                                'model_observations'][i]
                            photodict[
                                PHOTOMETRY.
                                E_LOWER_COUNT_RATE] = (
                                    photodict[PHOTOMETRY.COUNT_RATE] - (
                                        10.0 ** (
                                            np.log10(photodict[
                                                PHOTOMETRY.COUNT_RATE]) -
                                            output['model_variances'][
                                                i] / 2.5)))
                            photodict[
                                PHOTOMETRY.
                                E_UPPER_COUNT_RATE] = (10.0 ** (
                                    np.log10(photodict[
                                        PHOTOMETRY.COUNT_RATE]) +
                                    output['model_variances'][i] / 2.5) -
                                    photodict[PHOTOMETRY.COUNT_RATE])
                            photodict[PHOTOMETRY.U_COUNT_RATE] = 's^-1'
                        if ('model_upper_limits' in output and
                                output['model_upper_limits'][i]):
                            photodict[PHOTOMETRY.UPPER_LIMIT] = bool(output[
                                'model_upper_limits'][i])
                        if self._limiting_magnitude is not None:
                            photodict[PHOTOMETRY.SIMULATED] = True
                        if 'telescopes' in output and output['telescopes'][i]:
                            photodict[PHOTOMETRY.TELESCOPE] = output[
                                'telescopes'][i]
                        if 'systems' in output and output['systems'][i]:
                            photodict[PHOTOMETRY.SYSTEM] = output['systems'][i]
                        if 'bandsets' in output and output['bandsets'][i]:
                            photodict[PHOTOMETRY.BAND_SET] = output[
                                'bandsets'][i]
                        if 'instruments' in output and output[
                                'instruments'][i]:
                            photodict[PHOTOMETRY.INSTRUMENT] = output[
                                'instruments'][i]
                        if 'modes' in output and output['modes'][i]:
                            photodict[PHOTOMETRY.MODE] = output[
                                'modes'][i]
                        entry.add_photometry(
                            compare_to_existing=False, check_for_dupes=False,
                            **photodict)

                        uphotodict = deepcopy(photodict)
                        uphotodict[PHOTOMETRY.SOURCE] = umodelnum
                        uentry.add_photometry(
                            compare_to_existing=False,
                            check_for_dupes=False,
                            **uphotodict)
                else:
                    output = model.run_stack(y, root='objective')

                parameters = OrderedDict()
                derived_keys = set()
                pi = 0
                for ti, task in enumerate(model._call_stack):
                    # if task not in model._free_parameters:
                    #     continue
                    if model._call_stack[task]['kind'] != 'parameter':
                        continue
                    paramdict = OrderedDict((
                        ('latex', model._modules[task].latex()),
                        ('log', model._modules[task].is_log())
                    ))
                    if task in model._free_parameters:
                        poutput = model._modules[task].process(
                            **{'fraction': y[pi]})
                        value = list(poutput.values())[0]
                        paramdict['value'] = value
                        paramdict['fraction'] = y[pi]
                        pi = pi + 1
                    else:
                        if output.get(task, None) is not None:
                            paramdict['value'] = output[task]
                    parameters.update({model._modules[task].name(): paramdict})
                    # Dump out any derived parameter keys
                    derived_keys.update(model._modules[task].get_derived_keys(
                    ))

                for key in list(sorted(list(derived_keys))):
                    if (output.get(key, None) is not None and
                            key not in parameters):
                        parameters.update({key: {'value': output[key]}})

                realdict = {REALIZATION.PARAMETERS: parameters}
                if lnprobout is not None:
                    realdict[REALIZATION.SCORE] = str(lnprobout[xi][yi])
                realdict[REALIZATION.ALIAS] = str(ri)
                entry[ENTRY.MODELS][0].add_realization(**realdict)
                urealdict = deepcopy(realdict)
                uentry[ENTRY.MODELS][0].add_realization(**urealdict)
                ri = ri + 1
        prt.message('all_walkers_written', inline=True)

        entry.sanitize()
        oentry = {self._event_name: entry._ordered(entry)}
        uentry.sanitize()
        ouentry = {self._event_name: uentry._ordered(uentry)}

        uname = '_'.join(
            [self._event_name, entryhash, modelhash])

        if not os.path.exists(model.MODEL_OUTPUT_DIR):
            os.makedirs(model.MODEL_OUTPUT_DIR)

        if write:
            prt.message('writing_complete')
            with open_atomic(
                    os.path.join(model.MODEL_OUTPUT_DIR, 'walkers.json'),
                    'w') as flast, open_atomic(os.path.join(
                        model.MODEL_OUTPUT_DIR,
                        self._event_name + (
                            ('_' + suffix) if suffix else '') +
                        '.json'), 'w') as feven:
                entabbed_json_dump(oentry, flast, separators=(',', ':'))
                entabbed_json_dump(oentry, feven, separators=(',', ':'))

            if save_full_chain:
                prt.message('writing_full_chain')
                with open_atomic(
                    os.path.join(model.MODEL_OUTPUT_DIR,
                                 'chain.json'), 'w') as flast, open_atomic(
                        os.path.join(model.MODEL_OUTPUT_DIR,
                                     self._event_name + '_chain' + (
                                         ('_' + suffix) if suffix else '') +
                                     '.json'), 'w') as feven:
                    entabbed_json_dump(all_chain.tolist(),
                                       flast, separators=(',', ':'))
                    entabbed_json_dump(all_chain.tolist(),
                                       feven, separators=(',', ':'))

            if extra_outputs:
                prt.message('writing_extras')
                with open_atomic(os.path.join(
                    model.MODEL_OUTPUT_DIR, 'extras.json'),
                        'w') as flast, open_atomic(os.path.join(
                            model.MODEL_OUTPUT_DIR, self._event_name +
                            '_extras' + (('_' + suffix) if suffix else '') +
                            '.json'), 'w') as feven:
                    entabbed_json_dump(extras, flast, separators=(',', ':'))
                    entabbed_json_dump(extras, feven, separators=(',', ':'))

            prt.message('writing_model')
            with open_atomic(os.path.join(
                model.MODEL_OUTPUT_DIR, 'upload.json'),
                    'w') as flast, open_atomic(os.path.join(
                        model.MODEL_OUTPUT_DIR,
                        uname + (('_' + suffix) if suffix else '') +
                        '.json'), 'w') as feven:
                entabbed_json_dump(ouentry, flast, separators=(',', ':'))
                entabbed_json_dump(ouentry, feven, separators=(',', ':'))

        if upload_model:
            prt.message('ul_fit', [entryhash, modelhash])
            upayload = entabbed_json_dumps(ouentry, separators=(',', ':'))
            try:
                dbx = dropbox.Dropbox(upload_token)
                dbx.files_upload(
                    upayload.encode(),
                    '/' + uname + '.json',
                    mode=dropbox.files.WriteMode.overwrite)
                prt.message('ul_complete')
            except Exception:
                if self._test:
                    pass
                else:
                    raise

        if upload:
            for ce in self._converter.get_converted():
                dentry = Entry.init_from_file(
                    catalog=None,
                    name=ce[0],
                    path=ce[1],
                    merge=False,
                    pop_schema=False,
                    ignore_keys=[ENTRY.MODELS],
                    compare_to_existing=False)

                dentry.sanitize()
                odentry = {ce[0]: uentry._ordered(dentry)}
                dpayload = entabbed_json_dumps(odentry, separators=(',', ':'))
                text = prt.message('ul_devent', [ce[0]], prt=False)
                ul_devent = prt.prompt(text, kind='bool', message=False)
                if ul_devent:
                    dpath = '/' + slugify(
                        ce[0] + '_' + dentry[ENTRY.SOURCES][0].get(
                            SOURCE.BIBCODE, dentry[ENTRY.SOURCES][0].get(
                                SOURCE.NAME, 'NOSOURCE'))) + '.json'
                    try:
                        dbx = dropbox.Dropbox(upload_token)
                        dbx.files_upload(
                            dpayload.encode(),
                            dpath,
                            mode=dropbox.files.WriteMode.overwrite)
                        prt.message('ul_complete')
                    except Exception:
                        if self._test:
                            pass
                        else:
                            raise

        return (entry, pout, lnprobout)

    def generate_dummy_data(self,
                            name,
                            max_time=1000.,
                            band_list=[],
                            band_systems=[],
                            band_instruments=[],
                            band_bandsets=[]):
        """Generate simulated data based on priors."""
        # Just need 2 plot points for beginning and end.
        plot_points = 2

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

    def psrf(self, chain):
        """Calculate PSRF for a chain."""
        m = len(chain)
        n = len(chain[0])
        mom = np.mean(np.mean(chain, axis=1))
        b = n / float(m - 1) * np.sum(
            (np.mean(chain, axis=1) - mom) ** 2)
        w = np.mean(np.var(chain, axis=1, ddof=1))
        v = float(n - 1) / float(n) * w + (b / float(n))
        return np.sqrt(v / w)
