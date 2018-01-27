# -*- coding: UTF-8 -*-
"""Definitions for `Nester` class."""

import gc
import sys
import time

import numpy as np
from mosfit.samplers.sampler import Sampler


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
        self._WAIC = None
        self._ntemps = 1

    def append_output(self, modeldict):
        """Append output from the nester to the model description."""
        if self._iterations > 0:
            pass

    def prepare_output(self, check_upload_quality, upload):
        """Prepare output for writing to disk and uploading."""
        self._pout = [self._results.samples_u]
        self._lnprobout = [self._results.logl]
        self._weights = [np.exp(self._results.logwt)]
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

        ndim = self._model._num_free_parameters

        if self._num_walkers:
            self._nwalkers = self._num_walkers
        else:
            self._nwalkers = 20 * ndim

        self._lnprob = None
        self._lnlike = None

        prt.message('nmeas_nfree', [self._model._num_measurements, ndim])

        self._all_chain = np.array([])
        self._scores = np.ones((1, self._nwalkers)) * -np.inf

        nested_dlogz_init = 0.1

        max_iter = self._iterations
        if max_iter <= 0:
            return

        try:
            sampler = DynamicNestedSampler(
                ln_likelihood, draw_from_icdf, ndim,
                pool=self._pool, sample='rwalk', nlive=self._nwalkers)
            # Perform initial sample.
            ncall = sampler.ncall
            niter = sampler.it - 1
            for li, res in enumerate(sampler.sample_initial(
                dlogz=nested_dlogz_init
            )):
                ncall0 = ncall
                (worst, ustar, vstar, loglstar, logvol,
                 logwt, logz, logzvar, h, nc, worst_it,
                 propidx, propiter, eff, delta_logz) = res

                ncall += nc
                niter += 1
                max_iter -= 1

                if max_iter < 0:
                    prt.message('max_iter')
                    break

                self._results = sampler.results

                sout = self._model.run_stack(
                    self._results.samples_u[np.unravel_index(
                        np.argmax(self._results.logl),
                        self._results.logl.shape)],
                    root='objective')
                # The above added 1 call.
                ncall += 1

                kmat = sout.get('kmat')
                kdiag = sout.get('kdiagonal')
                variance = sout.get('obandvs', sout.get('variance'))
                if kdiag is not None and kmat is not None:
                    kmat[np.diag_indices_from(kmat)] += kdiag
                elif kdiag is not None and kmat is None:
                    kmat = np.diag(kdiag + variance)

                logzerr = np.sqrt(logzvar)
                prt.status(
                    self, 'baseline', kmat=kmat,
                    progress=[niter, self._iterations],
                    batch=0, nc=ncall - ncall0, ncall=ncall, eff=eff,
                    logz=[logz, logzerr,
                          delta_logz if delta_logz < 1.e6 else np.inf,
                          nested_dlogz_init],
                    loglstar=[loglstar])

            prt.status(
                self, 'starting_batches', kmat=kmat,
                progress=[niter, self._iterations],
                batch=0, nc=ncall - ncall0, ncall=ncall, eff=eff,
                logz=[logz, logzerr,
                      delta_logz if delta_logz < 1.e6 else np.inf,
                      nested_dlogz_init],
                loglstar=[loglstar])

            n = 1
            while max_iter >= 0:
                ncall0 = ncall
                if (self._fitter._maximum_walltime is not False and
                        time.time() - self._start_time >
                        self._fitter._maximum_walltime):
                    prt.message('exceeded_walltime', warning=True)
                    break

                self._results = sampler.results

                sout = self._model.run_stack(
                    self._results.samples_u[np.unravel_index(
                        np.argmax(self._results.logl),
                        self._results.logl.shape)],
                    root='objective')
                # The above added 1 call.
                ncall += 1

                stop, stop_vals = stopping_function(
                    self._results, return_vals=True)
                stop_post, stop_evid, stop_val = stop_vals
                if not stop:
                    logl_bounds = weight_function(self._results)
                    lnz, lnzerr = self._results.logz[
                        -1], self._results.logzerr[-1]
                    for res in sampler.sample_batch(
                            logl_bounds=logl_bounds,
                            nlive_new=np.ceil(self._nwalkers / 2)):
                        (worst, ustar, vstar, loglstar, nc,
                         worst_it, propidx, propiter, eff) = res

                        ncall += nc
                        niter += 1

                        prt.status(
                            self, 'batching', kmat=kmat,
                            progress=[niter, self._iterations],
                            batch=n, nc=ncall - ncall0, ncall=ncall, eff=eff,
                            logz=[lnz, lnzerr],
                            loglstar=[
                                logl_bounds[0], loglstar,
                                logl_bounds[1]],
                            stop=stop_val)
                    sampler.combine_runs()
                else:
                    break

            if max_iter < 0:
                prt.message('max_iter')

                # self._results.summary()
                # prt.nester_status(self, desc='sampling')

        except (KeyboardInterrupt, SystemExit):
            prt.message('ctrl_c', error=True, prefix=False, color='!r')
            s_exception = sys.exc_info()
        except Exception:
            raise

        try:
            s_exception
        except NameError:
            pass
        else:
            self._pool.close()
            if (not prt.prompt('mc_interrupted')):
                sys.exit()

        sampler.reset()
        gc.collect()
