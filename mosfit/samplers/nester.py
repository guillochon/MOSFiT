# -*- coding: UTF-8 -*-
"""Definitions for `Nester` class."""

import gc
import pickle
import sys

import numpy as np
from astrocats.catalog.model import MODEL
from astrocats.catalog.quantity import QUANTITY
from mosfit.samplers.sampler import Sampler
from mosfit.utils import pretty_num


class Nester(Sampler):
    """Fit transient events with the provided model."""

    _MAX_ACORC = 5
    _REPLACE_AGE = 20

    def __init__(
        self, fitter, model=None, iterations=2000, burn=None, post_burn=None,
            num_walkers=None, convergence_criteria=None,
            convergence_type='psrf', gibbs=False, fracking=True,
            frack_step=20, **kwargs):
        """Initialize `Nester` class."""
        super(Nester, self).__init__(fitter, num_walkers=num_walkers, **kwargs)

        self._model = model
        self._iterations = iterations
        self._burn = burn
        self._post_burn = post_burn
        self._cc = convergence_criteria
        self._ct = convergence_type
        self._gibbs = gibbs
        self._fracking = fracking
        self._frack_step = frack_step

        self._upload_model = None
        self._ntemps = 1

    def _get_best_kmat(self):
        """Get the kernel matrix associated with best current scoring model."""
        sout = self._model.run_stack(
            self._results.samples[np.unravel_index(
                np.argmax(self._results.logl),
                self._results.logl.shape)],
            root='objective')

        kmat = sout.get('kmat')
        kdiag = sout.get('kdiagonal')
        variance = sout.get('obandvs', sout.get('variance'))
        if kdiag is not None and kmat is not None:
            kmat[np.diag_indices_from(kmat)] += kdiag
        elif kdiag is not None and kmat is None:
            kmat = np.diag(kdiag + variance)

        return kmat

    def append_output(self, modeldict):
        """Append output from the nester to the model description."""
        modeldict[MODEL.SCORE] = {
            QUANTITY.VALUE: pretty_num(self._logz, sig=6),
            QUANTITY.E_VALUE: pretty_num(self._e_logz, sig=6),
            QUANTITY.KIND: 'Log(z)'
        }
        modeldict[MODEL.STEPS] = str(self._niter)

    def prepare_output(self, check_upload_quality, upload):
        """Prepare output for writing to disk and uploading."""
        self._pout = [self._results.samples]
        self._lnprobout = [self._results.logl]
        self._weights = [np.exp(self._results.logwt - max(
            self._results.logwt))]
        tweight = np.sum(self._weights)
        self._weights = [x / tweight for x in self._weights]

        if check_upload_quality:
            pass

    def run(self, walker_data):
        """Use nested sampling to determine posteriors."""
        from dynesty import DynamicNestedSampler
        from dynesty.dynamicsampler import stopping_function, weight_function
        from mosfit.fitter import ln_likelihood, draw_from_icdf

        prt = self._printer

        if len(walker_data):
            prt.message('nester_not_use_walkers', warning=True)

        ndim = self._model._num_free_parameters

        if self._num_walkers:
            self._nwalkers = self._num_walkers
        else:
            self._nwalkers = 2 * ndim

        self._nlive = 20 * ndim

        self._lnprob = None
        self._lnlike = None

        prt.message('nmeas_nfree', [self._model._num_measurements, ndim])

        nested_dlogz_init = self._cc
        post_thresh = self._cc

        max_iter = self._iterations if self._ct is None else np.inf
        if max_iter <= 0:
            return

        s_exception = None
        iter_denom = None if self._ct is not None else self._iterations

        # Save a few things from the dynesty run for diagnostic purposes.
        scales = []

        try:
            sampler = DynamicNestedSampler(
                ln_likelihood, draw_from_icdf, ndim,
                pool=self._pool, sample='rwalk',
                queue_size=max(self._pool.size, 1))
            # Perform initial sample.
            ncall = sampler.ncall
            self._niter = sampler.it - 1
            for li, res in enumerate(sampler.sample_initial(
                dlogz=nested_dlogz_init, nlive=self._nlive
            )):
                ncall0 = ncall
                (worst, ustar, vstar, loglstar, logvol,
                 logwt, self._logz, logzvar, h, nc, worst_it,
                 propidx, propiter, eff, delta_logz) = res

                ncall += nc
                self._niter += 1
                max_iter -= 1

                if max_iter < 0:
                    break

                if (self._fitter._maximum_walltime is not False and
                        self.time_running() >
                        self._fitter._maximum_walltime):
                    prt.message('exceeded_walltime', warning=True)
                    break

                self._results = sampler.results

                scales.append(sampler.results.scale)

                kmat = self._get_best_kmat()
                # The above added 1 call.
                ncall += 1

                self._e_logz = np.sqrt(logzvar)
                prt.status(
                    self, 'baseline', kmat=kmat,
                    iterations=[self._niter, iter_denom],
                    nc=ncall - ncall0, ncall=ncall, eff=eff,
                    logz=[self._logz, self._e_logz,
                          delta_logz, nested_dlogz_init],
                    loglstar=[loglstar],
                    time_running=self.time_running(),
                    maximum_walltime=self._fitter._maximum_walltime)

            if max_iter >= 0:
                prt.status(
                    self, 'starting_batches', kmat=kmat,
                    iterations=[self._niter, iter_denom],
                    nc=ncall - ncall0, ncall=ncall, eff=eff,
                    logz=[self._logz, self._e_logz,
                          delta_logz, nested_dlogz_init],
                    loglstar=[loglstar],
                    time_running=self.time_running(),
                    maximum_walltime=self._fitter._maximum_walltime)

            n = 0
            while max_iter >= 0:
                n += 1
                if (self._fitter._maximum_walltime is not False and
                        self.time_running() >
                        self._fitter._maximum_walltime):
                    prt.message('exceeded_walltime', warning=True)
                    break

                self._results = sampler.results

                scales.append(sampler.results.scale)

                stop, stop_vals = stopping_function(
                    self._results, return_vals=True, args={
                        'post_thresh': post_thresh})
                stop_post, stop_evid, stop_val = stop_vals
                if not stop:
                    logl_bounds = weight_function(self._results)
                    self._logz, self._e_logz = self._results.logz[
                        -1], self._results.logzerr[-1]
                    for res in sampler.sample_batch(
                            logl_bounds=logl_bounds,
                            nlive_new=int(np.ceil(self._nlive / 2))):
                        (worst, ustar, vstar, loglstar, nc,
                         worst_it, propidx, propiter, eff) = res
                        ncall0 = ncall

                        ncall += nc
                        self._niter += 1
                        max_iter -= 1

                        self._results = sampler.results

                        kmat = self._get_best_kmat()
                        # The above added 1 call.
                        ncall += 1

                        prt.status(
                            self, 'batching', kmat=kmat,
                            iterations=[self._niter, iter_denom],
                            batch=n, nc=ncall - ncall0, ncall=ncall, eff=eff,
                            logz=[self._logz, self._e_logz], loglstar=[
                                logl_bounds[0], loglstar,
                                logl_bounds[1]], stop=stop_val,
                            time_running=self.time_running(),
                            maximum_walltime=self._fitter._maximum_walltime)

                        if max_iter < 0:
                            break
                    sampler.combine_runs()
                else:
                    break

                # self._results.summary()
                # prt.nester_status(self, desc='sampling')

        except (KeyboardInterrupt, SystemExit):
            prt.message('ctrl_c', error=True, prefix=False, color='!r')
            s_exception = sys.exc_info()
        except Exception:
            print('Scale history:')
            print(scales)
            pickle.dump(sampler.results, open(
                self._fitter._event_name + '-dynesty.pickle', 'wb'))
            self._pool.close()
            raise

        if max_iter < 0:
            prt.message('max_iter')

        if s_exception is not None:
            self._pool.close()
            if (not prt.prompt('mc_interrupted')):
                sys.exit()

        sampler.reset()
        gc.collect()
