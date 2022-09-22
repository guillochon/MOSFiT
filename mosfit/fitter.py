# -*- coding: UTF-8 -*-
"""Definitions for `Fitter` class."""
import codecs
import gc
import json
import os
import time
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from astrocats.catalog.entry import ENTRY, Entry
from astrocats.catalog.model import MODEL
from astrocats.catalog.photometry import PHOTOMETRY
from astrocats.catalog.quantity import QUANTITY
from astrocats.catalog.realization import REALIZATION
from astrocats.catalog.source import SOURCE
from schwimmbad import MPIPool, SerialPool
from six import string_types

from mosfit.converter import Converter
from mosfit.fetcher import Fetcher
from mosfit.printer import Printer
from mosfit.samplers.ensembler import Ensembler
from mosfit.samplers.nester import Nester
from mosfit.samplers.ultranester import UltraNester
from mosfit.utils import (all_to_list, entabbed_json_dump, entabbed_json_dumps,
                          flux_density_unit, frequency_unit, get_model_hash,
                          listify, open_atomic, slugify, speak)

from .model import Model

warnings.filterwarnings("ignore")

def get_pool(method=None):
    try:
        if method == 'ultranest':
            raise ValueError('ultranest parallises with MPI already')
        return MPIPool()
    except (ImportError, ValueError):
        return SerialPool()

def draw_walker(test=True, walkers_pool=[], replace=False, weights=None):
    """Draw a walker from the global model variable."""
    global model
    return model.draw_walker(test, walkers_pool, replace,
                             weights)  # noqa: F821


def draw_from_icdf(x):
    """Draw a walker from the global model variable."""
    global model
    return model.draw_from_icdf(x)  # noqa: F821


def ln_likelihood(x):
    """Return ln(likelihood) using the global model variable."""
    global model
    return model.ln_likelihood(x)  # noqa: F821


def ln_likelihood_floored(x):
    """Return ln(likelihood) using the global model variable."""
    global model
    return model.ln_likelihood_floored(x)  # noqa: F821


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

    _DEFAULT_SOURCE = {SOURCE.BIBCODE: '2017arXiv171002145G'}

    def __init__(self,
                 cuda=False,
                 exit_on_prompt=False,
                 language='en',
                 limiting_magnitude=None,
                 prefer_fluxes=False,
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
            pool=self._pool,
            wrap_length=wrap_length,
            quiet=quiet,
            fitter=self,
            language=language,
            exit_on_prompt=exit_on_prompt)
        self._fetcher = Fetcher(
            test=test, open_in_browser=open_in_browser, printer=self._printer)

        self._cuda = cuda
        self._limiting_magnitude = limiting_magnitude
        self._prefer_fluxes = prefer_fluxes
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
                   time_list=[],
                   time_unit=None,
                   band_list=[],
                   band_systems=[],
                   band_instruments=[],
                   band_bandsets=[],
                   band_sampling_points=17,
                   iterations=10000,
                   num_walkers=None,
                   num_temps=1,
                   parameter_paths=['parameters.json'],
                   fracking=True,
                   frack_step=50,
                   burn=None,
                   post_burn=None,
                   gibbs=False,
                   slice_sampler_steps=-1,
                   smooth_times=-1,
                   extrapolate_time=0.0,
                   limit_fitting_mjds=False,
                   exclude_bands=[],
                   exclude_instruments=[],
                   exclude_systems=[],
                   exclude_sources=[],
                   exclude_kinds=[],
                   output_path='',
                   suffix='',
                   upload=False,
                   write=False,
                   upload_token='',
                   check_upload_quality=False,
                   variance_for_each=[],
                   user_fixed_parameters=[],
                   user_released_parameters=[],
                   convergence_type=None,
                   convergence_criteria=None,
                   save_full_chain=False,
                   draw_above_likelihood=False,
                   maximum_walltime=False,
                   start_time=False,
                   print_trees=False,
                   maximum_memory=np.inf,
                   speak=False,
                   return_fits=True,
                   extra_outputs=None,
                   walker_paths=[],
                   catalogs=[],
                   exit_on_prompt=False,
                   download_recommended_data=False,
                   local_data_only=False,
                   guess=True,
                   method=None,
                   seed=None,
                   cache_path='',
                   **kwargs):
        """Fit a list of events with a list of models."""
        global model
        if start_time is False:
            start_time = time.time()

        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        self._start_time = start_time
        self._maximum_walltime = maximum_walltime
        self._maximum_memory = maximum_memory
        self._debug = False
        self._speak = speak
        self._download_recommended_data = download_recommended_data
        self._local_data_only = local_data_only
        self._cache_path = cache_path

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
        self._converter = Converter(prt, require_source=upload, guess=guess,
                                    cache_path=cache_path)
        event_list = self._converter.generate_event_list(event_list)

        event_list = [x.replace('‑', '-') for x in event_list]

        entries = [[] for x in range(len(event_list))]
        ps = [[] for x in range(len(event_list))]
        lnprobs = [[] for x in range(len(event_list))]

        # Load walker data if provided a list of walker paths.
        walker_data = []

        if len(walker_paths):
            pool = get_pool(method=method)
            if pool.is_master():
                prt.message('walker_file')
                wfi = 0
                for walker_path in walker_paths:
                    if os.path.exists(walker_path):
                        prt.prt('  {}'.format(walker_path))
                        with codecs.open(
                                walker_path, 'r', encoding='utf-8') as f:
                            all_walker_data = json.load(
                                f, object_pairs_hook=OrderedDict)

                        # Support both the format where all data stored in a
                        # single-item dictionary (the OAC format) and the older
                        # MOSFiT format where the data was stored in the
                        # top-level dictionary.
                        if ENTRY.NAME not in all_walker_data:
                            all_walker_data = all_walker_data[list(
                                all_walker_data.keys())[0]]

                        models = all_walker_data.get(ENTRY.MODELS, [])
                        choice = None
                        if len(models) > 1:
                            model_opts = [
                                '{}-{}-{}'.format(x['code'], x['name'],
                                                  x['date']) for x in models
                            ]
                            choice = prt.prompt(
                                'select_model_walkers',
                                kind='select',
                                message=True,
                                options=model_opts)
                            choice = model_opts.index(choice)
                        elif len(models) == 1:
                            choice = 0

                        if choice is not None:
                            walker_data.extend([[
                                wfi, x[REALIZATION.PARAMETERS],
                                x.get(REALIZATION.WEIGHT)
                            ] for x in models[choice][MODEL.REALIZATIONS]])

                        for i in range(len(walker_data)):
                            if walker_data[i][2] is not None:
                                walker_data[i][2] = float(walker_data[i][2])

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

        pool = get_pool(method=method)
        if pool.is_master():
            fetched_events = self._fetcher.fetch(
                event_list,
                offline=self._offline,
                prefer_cache=self._prefer_cache,
                cache_path=self._cache_path)

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

            if (event is not None
                    and (not self._event_data
                         or ENTRY.PHOTOMETRY not in self._event_data[list(
                             self._event_data.keys())[0]])):
                prt.message('no_photometry', [self._event_name])
                continue

            for mi, mod_name in enumerate(lmodel_list):
                for parameter_path in parameter_paths:
                    pool = get_pool(method=method)
                    self._model = Model(
                        model=mod_name,
                        data=self._event_data,
                        parameter_path=parameter_path,
                        output_path=output_path,
                        wrap_length=self._wrap_length,
                        test=self._test,
                        printer=prt,
                        fitter=self,
                        pool=pool,
                        print_trees=print_trees)

                    if not self._model._model_name:
                        prt.message(
                            'no_models_avail', [self._event_name],
                            warning=True)
                        continue

                    if not event:
                        prt.message('gen_dummy')
                        self._event_name = mod_name
                        gen_args = {
                            'name': mod_name,
                            'max_time': max_time,
                            'time_list': time_list,
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
                            time_list=time_list,
                            time_unit=time_unit,
                            band_list=band_list,
                            band_systems=band_systems,
                            band_instruments=band_instruments,
                            band_bandsets=band_bandsets,
                            band_sampling_points=band_sampling_points,
                            variance_for_each=variance_for_each,
                            user_fixed_parameters=user_fixed_parameters,
                            user_released_parameters=user_released_parameters,
                            pool=pool)

                        if not success:
                            break

                        if self._local_data_only:
                            break

                        # If our data is missing recommended keys, offer the
                        # user option to pull the missing data from online and
                        # merge with existing data.
                        urk = self._model.get_unset_recommended_keys()
                        ptxt = prt.text('acquire_recommended',
                                        [', '.join(list(urk))])
                        while event and len(urk) and (
                                alt_name or self._download_recommended_data
                                or prt.prompt(
                                    ptxt, [', '.join(urk)], kind='bool')):
                            pool = get_pool(method=method)
                            if pool.is_master():
                                en = (alt_name
                                      if alt_name else self._event_name)
                                extra_event = self._fetcher.fetch(
                                    en,
                                    offline=self._offline,
                                    prefer_cache=self._prefer_cache,
                                    cache_path=self._cache_path)[0]
                                extra_data = self._fetcher.load_data(
                                    extra_event)

                                for rank in range(1, pool.size + 1):
                                    pool.comm.send(
                                        extra_data, dest=rank, tag=4)
                                pool.close()
                            else:
                                extra_data = pool.comm.recv(source=0, tag=4)
                                pool.wait()

                            if extra_data is not None:
                                extra_data = extra_data[list(
                                    extra_data.keys())[0]]

                                for key in urk:
                                    new_val = extra_data.get(key)
                                    self._event_data[list(
                                        self._event_data.keys())
                                                     [0]][key] = new_val
                                    if new_val is not None and len(new_val):
                                        prt.message('extra_value', [
                                            key,
                                            str(new_val[0].get(QUANTITY.VALUE))
                                        ])
                                success = False
                                prt.message('reloading_merged')
                                break
                            else:
                                text = prt.text('extra_not_found',
                                                [self._event_name])
                                alt_name = prt.prompt(text, kind='string')
                                if not alt_name:
                                    break

                    if success:
                        self._walker_data = walker_data

                        entry, p, lnprob = self.fit_data(
                            event_name=self._event_name,
                            method=method,
                            iterations=iterations,
                            num_walkers=num_walkers,
                            num_temps=num_temps,
                            burn=burn,
                            post_burn=post_burn,
                            fracking=fracking,
                            frack_step=frack_step,
                            gibbs=gibbs,
                            slice_sampler_steps=slice_sampler_steps,
                            pool=pool,
                            output_path=output_path,
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
                        del (model)
                    del (self._model)
                    gc.collect()

        return (entries, ps, lnprobs)

    def fit_data(self,
                 event_name='',
                 method=None,
                 iterations=None,
                 frack_step=20,
                 num_walkers=None,
                 num_temps=1,
                 burn=None,
                 post_burn=None,
                 fracking=True,
                 gibbs=False,
                 slice_sampler_steps=-1,
                 pool=None,
                 output_path='',
                 suffix='',
                 write=False,
                 upload=False,
                 upload_token='',
                 check_upload_quality=True,
                 convergence_type=None,
                 convergence_criteria=None,
                 save_full_chain=False,
                 extra_outputs=None):
        """Fit the data for a given event.

        Fitting performed using a combination of emcee and fracking.
        """
        if self._speak:
            speak('Fitting ' + event_name, self._speak)
        from mosfit.__init__ import __version__
        global model
        model = self._model
        prt = self._printer

        upload_model = upload and iterations > 0

        if pool is not None:
            self._pool = pool

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

        self._method = method

        if self._method == 'dynesty':
            self._sampler = Nester(self, model, iterations, burn, post_burn,
                                   num_walkers, convergence_criteria,
                                   convergence_type, gibbs, fracking,
                                   frack_step)
        elif self._method == 'ultranest' or self._method == 'ultranest-progressive':
            self._sampler = UltraNester(self, model, num_walkers=num_walkers,
                slice_sampler_steps=slice_sampler_steps, progressive=self._method == 'ultranest-progressive')
            if output_path != '':
                self._sampler._sampler_kwargs['log_dir'] = output_path
                self._sampler._sampler_kwargs['resume'] = True
        else:
            self._sampler = Ensembler(self, model, iterations, burn, post_burn,
                                      num_temps, num_walkers,
                                      convergence_criteria, convergence_type,
                                      gibbs, fracking, frack_step)

        self._sampler.run(self._walker_data)
        if self._method == 'ultranest' and self._sampler._sampler.mpi_rank != 0:
            # only run the code below with one process
            return self._sampler._sampler.comm.bcast(None, root=0)

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
        for root in model._references:
            for ref in model._references[root]:
                sources.append(entry.add_source(**ref))
        sources.append(entry.add_source(**self._DEFAULT_SOURCE))
        source = ','.join(sources)

        usources = []
        for root in model._references:
            for ref in model._references[root]:
                usources.append(uentry.add_source(**ref))
        usources.append(uentry.add_source(**self._DEFAULT_SOURCE))
        usource = ','.join(usources)

        model_setup = OrderedDict()
        for ti, task in enumerate(model._call_stack):
            task_copy = deepcopy(model._call_stack[task])
            if (task_copy['kind'] == 'parameter'
                    and task in model._parameter_json):
                task_copy.update(model._parameter_json[task])
            model_setup[task] = task_copy
        modeldict = OrderedDict([(MODEL.NAME, model._model_name),
                                 (MODEL.SETUP, model_setup),
                                 (MODEL.CODE, 'MOSFiT'),
                                 (MODEL.DATE, time.strftime("%Y/%m/%d")),
                                 (MODEL.VERSION, __version__),
                                 (MODEL.SOURCE, source)])

        self._sampler.prepare_output(check_upload_quality, upload)

        self._sampler.append_output(modeldict)

        umodeldict = deepcopy(modeldict)
        umodeldict[MODEL.SOURCE] = usource
        modelhash = get_model_hash(
            umodeldict, ignore_keys=[MODEL.DATE, MODEL.SOURCE])
        umodelnum = uentry.add_model(**umodeldict)

        if self._sampler._upload_model is not None:
            upload_model = self._sampler._upload_model

        modelnum = entry.add_model(**modeldict)

        samples, probs, weights = self._sampler.get_samples()

        extras = OrderedDict()
        samples_to_plot = self._sampler._nwalkers

        if isinstance(self._sampler, Nester):
            icdf = np.cumsum(np.concatenate(([0.0], weights)))
            draws = np.random.rand(samples_to_plot)
            indices = np.searchsorted(icdf, draws) - 1
        else:
            indices = list(range(samples_to_plot))

        ri = 0
        selected_extra = False
        for xi, x in enumerate(samples):
            ri = ri + 1
            prt.message(
                'outputting_walker', [ri, len(samples)],
                inline=True,
                min_time=0.2)
            if xi in indices:
                output = model.run_stack(x, root='output')
                if extra_outputs is not None:
                    if not extra_outputs and not selected_extra:
                        extra_options = list(output.keys())
                        prt.message('available_keys')
                        for opt in extra_options:
                            prt.prt('- {}'.format(opt))
                        selected_extra = True
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
                        photodict[PHOTOMETRY.
                                  MAGNITUDE] = output['model_observations'][i]
                        photodict[PHOTOMETRY.
                                  E_MAGNITUDE] = output['model_variances'][i]
                    elif output['observation_types'][i] == 'magcount':
                        if output['model_observations'][i] == 0.0:
                            continue
                        photodict[PHOTOMETRY.BAND] = output['bands'][i]
                        photodict[PHOTOMETRY.
                                  COUNT_RATE] = output['model_observations'][i]
                        photodict[PHOTOMETRY.
                                  E_COUNT_RATE] = output['model_variances'][i]
                        photodict[PHOTOMETRY.MAGNITUDE] = -2.5 * np.log10(
                            output['model_observations']
                            [i]) + output['all_zeropoints'][i]
                        photodict[PHOTOMETRY.E_UPPER_MAGNITUDE] = 2.5 * (
                            np.log10(output['model_observations'][i] +
                                     output['model_variances'][i]) - np.log10(
                                         output['model_observations'][i]))
                        if (output['model_variances'][i] >
                                output['model_observations'][i]):
                            photodict[PHOTOMETRY.UPPER_LIMIT] = True
                        else:
                            photodict[PHOTOMETRY.E_LOWER_MAGNITUDE] = 2.5 * (
                                np.log10(output['model_observations'][i]) -
                                np.log10(output['model_observations'][i] -
                                         output['model_variances'][i]))
                    elif output['observation_types'][i] == 'fluxdensity':
                        photodict[PHOTOMETRY.FREQUENCY] = output[
                            'frequencies'][i] * frequency_unit('GHz')
                        photodict[PHOTOMETRY.FLUX_DENSITY] = output[
                            'model_observations'][i] * flux_density_unit('µJy')
                        photodict[PHOTOMETRY.E_LOWER_FLUX_DENSITY] = (
                            photodict[PHOTOMETRY.FLUX_DENSITY] -
                            (10.0**
                             (np.log10(photodict[PHOTOMETRY.FLUX_DENSITY]) -
                              output['model_variances'][i] / 2.5)) *
                            flux_density_unit('µJy'))
                        photodict[PHOTOMETRY.E_UPPER_FLUX_DENSITY] = (
                            10.0**(np.log10(photodict[PHOTOMETRY.FLUX_DENSITY])
                                   + output['model_variances'][i] / 2.5) *
                            flux_density_unit('µJy') -
                            photodict[PHOTOMETRY.FLUX_DENSITY])
                        photodict[PHOTOMETRY.U_FREQUENCY] = 'GHz'
                        photodict[PHOTOMETRY.U_FLUX_DENSITY] = 'µJy'
                    elif output['observation_types'][i] == 'countrate':
                        photodict[PHOTOMETRY.
                                  COUNT_RATE] = output['model_observations'][i]
                        photodict[PHOTOMETRY.E_LOWER_COUNT_RATE] = (
                            photodict[PHOTOMETRY.COUNT_RATE] -
                            (10.0**(np.log10(photodict[PHOTOMETRY.COUNT_RATE])
                                    - output['model_variances'][i] / 2.5)))
                        photodict[PHOTOMETRY.E_UPPER_COUNT_RATE] = (
                            10.0**(np.log10(photodict[PHOTOMETRY.COUNT_RATE]) +
                                   output['model_variances'][i] / 2.5) -
                            photodict[PHOTOMETRY.COUNT_RATE])
                        photodict[PHOTOMETRY.U_COUNT_RATE] = 's^-1'
                    if ('model_upper_limits' in output
                            and output['model_upper_limits'][i]):
                        photodict[PHOTOMETRY.UPPER_LIMIT] = bool(
                            output['model_upper_limits'][i])
                    if self._limiting_magnitude is not None:
                        photodict[PHOTOMETRY.SIMULATED] = True
                    if 'telescopes' in output and output['telescopes'][i]:
                        photodict[
                            PHOTOMETRY.TELESCOPE] = output['telescopes'][i]
                    if 'systems' in output and output['systems'][i]:
                        photodict[PHOTOMETRY.SYSTEM] = output['systems'][i]
                    if 'bandsets' in output and output['bandsets'][i]:
                        photodict[PHOTOMETRY.BAND_SET] = output['bandsets'][i]
                    if 'instruments' in output and output['instruments'][i]:
                        photodict[
                            PHOTOMETRY.INSTRUMENT] = output['instruments'][i]
                    if 'modes' in output and output['modes'][i]:
                        photodict[PHOTOMETRY.MODE] = output['modes'][i]
                    entry.add_photometry(
                        compare_to_existing=False,
                        check_for_dupes=False,
                        **photodict)

                    uphotodict = deepcopy(photodict)
                    uphotodict[PHOTOMETRY.SOURCE] = umodelnum
                    uentry.add_photometry(
                        compare_to_existing=False,
                        check_for_dupes=False,
                        **uphotodict)
            else:
                output = model.run_stack(x, root='objective')

            parameters = OrderedDict()
            derived_keys = set()
            pi = 0
            for ti, task in enumerate(model._call_stack):
                # if task not in model._free_parameters:
                #     continue
                if model._call_stack[task]['kind'] != 'parameter':
                    continue
                paramdict = OrderedDict(
                    (('latex', model._modules[task].latex()),
                     ('log', model._modules[task].is_log())))
                if task in model._free_parameters:
                    poutput = model._modules[task].process(
                        **{'fraction': x[pi]})
                    value = list(poutput.values())[0]
                    paramdict['value'] = value
                    paramdict['fraction'] = x[pi]
                    pi = pi + 1
                else:
                    if output.get(task, None) is not None:
                        paramdict['value'] = output[task]
                parameters.update({model._modules[task].name(): paramdict})
                # Dump out any derived parameter keys
                derived_keys.update(model._modules[task].get_derived_keys())

            for key in list(sorted(list(derived_keys))):
                if (output.get(key, None) is not None
                        and key not in parameters):
                    parameters.update({key: {'value': output[key]}})

            realdict = {REALIZATION.PARAMETERS: parameters}
            if probs is not None:
                realdict[REALIZATION.SCORE] = str(probs[xi])
            else:
                realdict[REALIZATION.SCORE] = str(
                    ln_likelihood(x) + ln_prior(x))
            realdict[REALIZATION.ALIAS] = str(ri)
            realdict[REALIZATION.WEIGHT] = str(weights[xi])
            entry[ENTRY.MODELS][0].add_realization(
                check_for_dupes=False, **realdict)
            urealdict = deepcopy(realdict)
            uentry[ENTRY.MODELS][0].add_realization(
                check_for_dupes=False, **urealdict)
        prt.message('all_walkers_written', inline=True)

        entry.sanitize()
        oentry = {self._event_name: entry._ordered(entry)}
        uentry.sanitize()
        ouentry = {self._event_name: uentry._ordered(uentry)}

        uname = '_'.join([self._event_name, entryhash, modelhash])

        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(model.get_products_path()):
            os.makedirs(model.get_products_path())

        if write:
            prt.message('writing_complete')
            with open_atomic(
                    os.path.join(model.get_products_path(), 'walkers.json'),
                    'w') as flast, open_atomic(
                        os.path.join(
                            model.get_products_path(), self._event_name + (
                                ('_' + suffix) if suffix else '') + '.json'),
                        'w') as feven:
                entabbed_json_dump(oentry, flast, separators=(',', ':'))
                entabbed_json_dump(oentry, feven, separators=(',', ':'))

            if save_full_chain:
                prt.message('writing_full_chain')
                my_chain = np.asarray(self._sampler._all_chain.tolist())
                pi = 0
                param_names = []
                for ti, task in enumerate(model._call_stack):
                    if model._call_stack[task]['kind'] != 'parameter':
                        continue
                    if task in model._free_parameters:
                        poutput = model._modules[task].process(
                            **{'fraction': my_chain[:, :, :, pi]})
                        value = list(poutput.values())[0]
                        my_chain[:, :, :, pi] = value
                        param_names.append(task)
                        pi = pi + 1
                my_chain = my_chain.tolist()
                my_chain.append(param_names)
                with open_atomic(
                        os.path.join(model.get_products_path(), 'chain.json'),
                        'w') as flast, open_atomic(
                            os.path.join(
                                model.get_products_path(),
                                self._event_name + '_chain' +
                                (('_' + suffix) if suffix else '') + '.json'),
                            'w') as feven:
                    entabbed_json_dump(
                        my_chain,
                        flast,
                        separators=(',', ':'))
                    entabbed_json_dump(
                        my_chain,
                        feven,
                        separators=(',', ':'))

            if extra_outputs is not None:
                prt.message('writing_extras')
                with open_atomic(
                        os.path.join(model.get_products_path(), 'extras.json'),
                        'w') as flast, open_atomic(
                            os.path.join(
                                model.get_products_path(),
                                self._event_name + '_extras' +
                                (('_' + suffix) if suffix else '') + '.json'),
                            'w') as feven:
                    entabbed_json_dump(extras, flast, separators=(',', ':'))
                    entabbed_json_dump(extras, feven, separators=(',', ':'))

            prt.message('writing_model')
            with open_atomic(
                    os.path.join(model.get_products_path(), 'upload.json'),
                    'w') as flast, open_atomic(
                        os.path.join(
                            model.get_products_path(), uname + (
                                ('_' + suffix) if suffix else '') + '.json'),
                        'w') as feven:
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

        if self._method == 'ultranest': #and self._sampler._sampler.mpi_size > 1:
            # send results to other MPI processes (above)
            self._sampler._sampler.comm.bcast((entry, samples, probs), root=0)
        return (entry, samples, probs)

    def nester(self):
        """Use nested sampling to determine posteriors."""
        pass

    def generate_dummy_data(self,
                            name,
                            max_time=1000.,
                            time_list=[],
                            band_list=[],
                            band_systems=[],
                            band_instruments=[],
                            band_bandsets=[]):
        """Generate simulated data based on priors."""
        # Just need 2 plot points for beginning and end.
        plot_points = 2

        times = list(
            sorted(
                set(list(np.linspace(0.0, max_time, plot_points)) +
                    time_list)))
        band_list_all = ['V'] if len(band_list) == 0 else band_list
        times = np.repeat(times, len(band_list_all))

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
            rep_val = '' if len(
                band_instruments) == 0 else band_instruments[-1]
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

        bands = [i for s in [band_list_all for x in times] for i in s]
        systs = [i for s in [band_systems for x in times] for i in s]
        insts = [i for s in [band_instruments for x in times] for i in s]
        bsets = [i for s in [band_bandsets for x in times] for i in s]

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
