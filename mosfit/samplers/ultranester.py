# -*- coding: UTF-8 -*-
"""Definitions for `UltraNester` class."""

import numpy as np
from astrocats.catalog.model import MODEL
from astrocats.catalog.quantity import QUANTITY
from mosfit.samplers.sampler import Sampler
from mosfit.utils import pretty_num


class UltraNester(Sampler):
    """Fit transient events with the provided model."""

    _MAX_ACORC = 5
    _REPLACE_AGE = 20

    def __init__(
        self, fitter, model=None, sampler_kwargs={}, run_kwargs={'frac_remain':0.5},
            num_walkers=None, hotstart=False, slice_sampler_steps=-1, **kwargs):
        """Initialize `UltraNester` class."""
        super(UltraNester, self).__init__(fitter, num_walkers=num_walkers, **kwargs)

        self._model = model
        self._sampler_kwargs = sampler_kwargs
        self._run_kwargs = run_kwargs
        self._slice_sampler_steps = slice_sampler_steps
        if num_walkers is not None:
            self._run_kwargs['min_num_live_points'] = num_walkers
        self._hotstart = hotstart

        self._upload_model = None
        self._ntemps = 1
        self._nwalkers = self._num_walkers

    def _get_best_kmat(self):
        """Get the kernel matrix associated with best current scoring model."""
        max_like_index = np.argmax(self._results['weighted_samples']['logl'])
        max_like_point = self._results['weighted_samples']['points'][max_like_index, :]
        sout = self._model.run_stack(max_like_point, root='objective')

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
        if self._hotstart:
            self._pout = [self._results['weighted_samples']['points'][:,:-1]]
        else:
            self._pout = [self._results['weighted_samples']['points']]
        self._lnprobout = [self._results['weighted_samples']['logl']]
        self._weights = [self._results['weighted_samples']['weights']]

        if check_upload_quality:
            pass

    def run(self, walker_data):
        """Use nested sampling to determine posteriors."""
        from ultranest import ReactiveNestedSampler
        from mosfit.fitter import ln_likelihood, draw_from_icdf

        prt = self._printer

        if len(walker_data):
            prt.message('nester_not_use_walkers', warning=True)

        parameters = self._model._free_parameters
        prt.message('nmeas_nfree', [self._model._num_measurements, len(parameters)])

        if self._hotstart:
            import ultranest.hotstart
            import os

            if 'log_dir' in self._sampler_kwargs:
                upoints_path = os.path.join(self._sampler_kwargs['log_dir'], 'chains', 'laplaceapprox_samples.txt')
                try:
                    upoints = np.loadtxt(upoints_path)
                except IOError:
                    max_like_path = os.path.join(self._sampler_kwargs['log_dir'], 'chains', 'max_likelihood.txt')
                    cov_path = os.path.join(self._sampler_kwargs['log_dir'], 'chains', 'laplaceapprox.txt')
                    try:
                        optu = np.loadtxt(max_like_path)
                        cov = np.loadtxt(cov_path)
                    except IOError:
                        import snowline
                        os.makedirs(os.path.join(self._sampler_kwargs['log_dir'], 'chains'), exist_ok=True)
                        fitsampler = snowline.ReactiveImportanceSampler(parameters, ln_likelihood, transform=draw_from_icdf)
                        fitsampler.laplace_approximate()
                        optu = fitsampler.optu
                        cov = fitsampler.cov
                        np.savetxt(max_like_path, optu)
                        np.savetxt(cov_path, cov)

                    import scipy.stats
                    upoints = scipy.stats.multivariate_t.rvs(loc=optu, shape=cov, df=2, size=100000)
                    mask = np.logical_and(upoints > 0, upoints < 1).all(axis=1)
                    upoints = upoints[mask,:][:10000]
                    np.savetxt(upoints_path, upoints)

                uweights = np.ones(len(upoints)) / len(upoints)

            aux_parameters, aux_loglike, aux_transform, vectorized = ultranest.hotstart.get_auxiliary_contbox_parameterization(
                parameters, ln_likelihood, transform=draw_from_icdf, 
                upoints=upoints, uweights=uweights, vectorized=False)

            self._sampler = ultranest.ReactiveNestedSampler(
                aux_parameters, aux_loglike, aux_transform, vectorized=vectorized,
                **self._sampler_kwargs
            )
        else:
            self._sampler = ReactiveNestedSampler(parameters, ln_likelihood, transform=draw_from_icdf,
                **self._sampler_kwargs)
        if self._slice_sampler_steps > 0:
            prt.message('enabling_slice_sampling', [self._slice_sampler_steps], warning=True)
            import ultranest.stepsampler
            self._sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=self._slice_sampler_steps,
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            )
        self._results = self._sampler.run(**self._run_kwargs)
        self._logz = self._results['logz']
        self._e_logz = self._results['logzerr']
        self._niter = self._results['niter']
        if self._hotstart:
            self._all_chain = self._results['samples'][:,:-1][None][None]
        else:
            self._all_chain = self._results['samples'][None][None]
        
        if 'log_dir' in self._sampler_kwargs:
            self._sampler.plot()
