# -*- coding: UTF-8 -*-
"""Definitions for `Nester` class."""

import gc
import sys
import time

import numpy as np
import scipy
from astrocats.catalog.model import MODEL
from astrocats.catalog.quantity import QUANTITY
from mosfit.samplers.sampler import Sampler
from mosfit.utils import pretty_num


class Ensembler(Sampler):
    """Fit transient events with the provided model."""

    _MAX_ACORC = 5
    _REPLACE_AGE = 20

    def __init__(
        self, fitter, model=None, iterations=2000, burn=None, post_burn=None,
            num_temps=1, num_walkers=None, convergence_criteria=None,
            convergence_type='psrf', gibbs=False, fracking=True,
            frack_step=20, **kwargs):
        """Initialize `Nester` class."""
        super(Ensembler, self).__init__(fitter, **kwargs)

        self._model = model
        self._iterations = iterations
        self._burn = burn
        self._post_burn = post_burn
        self._num_temps = num_temps
        self._num_walkers = num_walkers
        self._cc = convergence_criteria
        self._ct = convergence_type
        self._gibbs = gibbs
        self._fracking = fracking
        self._frack_step = frack_step

        self._upload_model = None
        self._WAIC = None

    def get_samples(self):
        """Return samples from nester."""
        samples = self._pout
        probs = self._lnprobout

        return samples, probs

    def append_output(self, modeldict):
        """Append output from the ensembler to the model description."""
        if self._iterations > 0:
            pass

    def prepare_output(self, check_upload_quality, upload):
        """Prepare output for writing to disk and uploading."""
        prt = self._printer

        if check_upload_quality:
            pass

    def run(self, walker_data):
        """Use nested sampling to determine posteriors."""
        from dynesty import NestedSampler
        from mosfit.fitter import draw_walker, frack, ln_likelihood, ln_prior

        prt = self._printer

        ndim = self._model._num_free_parameters

        if self._num_walkers:
            self._nwalkers = self._num_walkers
        else:
            self._nwalkers = 2 * ndim

        self._lnprob = None
        self._lnlike = None
        pool_size = max(self._pool.size, 1)

        prt.message('nmeas_nfree', [self._model._num_measurements, ndim])
        p0 = [[] for x in range(1)]

        self._p = list(p0)

        self._all_chain = np.array([])
        self._scores = np.ones((self._ntemps, self._nwalkers)) * -np.inf

        oldp = self._p

        # The argument of the for loop runs emcee, after each iteration of
        # emcee the contents of the for loop are executed.
        exceeded_walltime = False

        try:
            if self._iterations > 0:
                sampler = NestedSampler(
                    ln_likelihood, ptform, ndim, pool=self._pool)
            while self._iterations > 0 and self._cc is not None:
                if exceeded_walltime:
                    break
                for li, (
                        self._p, self._lnprob, self._lnlike) in enumerate(
                            sampler.sample(
                                self._p, iterations=self._iterations)):
                    if (self._fitter._maximum_walltime is not False and
                            time.time() - self._start_time >
                            self._fitter._maximum_walltime):
                        prt.message('exceeded_walltime', warning=True)
                        exceeded_walltime = True
                        break

                    prt.nester_status(self, desc='walking')

                sampler.reset()
                gc.collect()

        except (KeyboardInterrupt, SystemExit):
            prt.message('ctrl_c', error=True, prefix=False, color='!r')
            s_exception = sys.exc_info()
        except Exception:
            raise

        if s_exception:
            self._pool.close()
            if (not prt.prompt('mc_interrupted')):
                sys.exit()
