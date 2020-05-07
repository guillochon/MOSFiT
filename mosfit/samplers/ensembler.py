# -*- coding: UTF-8 -*-
"""Definitions for `Ensembler` class."""

import gc
import sys
import time

import numpy as np
import scipy
from astrocats.catalog.model import MODEL
from astrocats.catalog.quantity import QUANTITY
from emcee.autocorr import AutocorrError
from mosfit.mossampler import MOSSampler
from mosfit.samplers.sampler import Sampler
from mosfit.utils import calculate_WAIC, pretty_num


class Ensembler(Sampler):
    """Fit transient events with the provided model."""

    _MAX_ACORC = 5
    _REPLACE_AGE = 20

    def __init__(
        self, fitter, model=None, iterations=2000, burn=None, post_burn=None,
            num_temps=1, num_walkers=None, convergence_criteria=None,
            convergence_type='psrf', gibbs=False, fracking=True,
            frack_step=20, **kwargs):
        """Initialize `Ensembler` class."""
        super(Ensembler, self).__init__(
            fitter, num_walkers=num_walkers, **kwargs)

        self._model = model
        self._iterations = iterations
        self._burn = burn
        self._post_burn = post_burn
        self._num_temps = num_temps
        self._cc = convergence_criteria
        self._ct = convergence_type
        self._gibbs = gibbs
        self._fracking = fracking
        self._frack_step = frack_step

        self._upload_model = None
        self._WAIC = None

    def append_output(self, modeldict):
        """Append output from the ensembler to the model description."""
        self._WAIC = None
        if self._iterations > 0:
            self._WAIC = calculate_WAIC(self._scores)
            modeldict[MODEL.SCORE] = {
                QUANTITY.VALUE: str(self._WAIC),
                QUANTITY.KIND: 'WAIC'
            }
            modeldict[MODEL.CONVERGENCE] = []
            if self._psrf < np.inf:
                modeldict[MODEL.CONVERGENCE].append(
                    {
                        QUANTITY.VALUE: str(self._psrf),
                        QUANTITY.KIND: 'psrf'
                    }
                )
            if self._acor and self._aacort > 0:
                acortimes = '<' if self._aa < self._MAX_ACORC else ''
                acortimes += str(np.int(float(self._emi -
                                              self._ams) / self._actc))
                modeldict[MODEL.CONVERGENCE].append(
                    {
                        QUANTITY.VALUE: str(acortimes),
                        QUANTITY.KIND: 'autocorrelationtimes'
                    }
                )
            modeldict[MODEL.STEPS] = str(self._emi)

    def prepare_output(self, check_upload_quality, upload):
        """Prepare output for writing to disk and uploading."""
        prt = self._printer

        if check_upload_quality:
            if self._WAIC is None:
                self._upload_model = False
            elif self._WAIC is not None and self._WAIC < 0.0:
                if upload:
                    prt.message('no_ul_waic', ['' if self._WAIC is None
                                               else pretty_num(self._WAIC)])
                self._upload_model = False

        if len(self._all_chain):
            self._pout = self._all_chain[:, :, -1, :]
            self._lnprobout = self._all_lnprob[:, :, -1]
            self._lnlikeout = self._all_lnlike[:, :, -1]
        else:
            self._pout = self._p
            self._lnprobout = self._lnprob
            self._lnlikeout = self._lnlike

        weight = 1.0 / (self._nwalkers * self._ntemps)
        self._weights = np.full_like(self._lnlikeout, weight)

        # Here, we append to the vector of walkers from the full chain based
        # upon the value of acort (the autocorrelation timescale).
        if self._acor and self._aacort > 0 and self._aa == self._MAX_ACORC:
            actc0 = int(np.ceil(self._aacort))
            for i in range(1, np.int(float(self._emi - self._ams) / actc0)):
                self._pout = np.concatenate(
                    (self._all_chain[:, :, -i * self._actc, :], self._pout),
                    axis=1)
                self._lnprobout = np.concatenate(
                    (self._all_lnprob[:, :, -i * self._actc],
                     self._lnprobout), axis=1)
                self._lnlikeout = np.concatenate(
                    (self._all_lnlike[:, :, -i * self._actc],
                     self._lnlikeout), axis=1)
                self._weights = np.full_like(self._lnlikeout, weight)

    def run(self, walker_data):
        """Use ensemble sampling to determine posteriors."""
        from mosfit.fitter import draw_walker, frack, ln_likelihood, ln_prior

        prt = self._printer

        self._emcee_est_t = 0.0
        self._bh_est_t = 0.0
        if self._burn is not None:
            self._burn_in = min(self._burn, self._iterations)
        elif self._post_burn is not None:
            self._burn_in = max(self._iterations - self._post_burn, 0)
        else:
            self._burn_in = int(np.round(self._iterations / 2))

        self._ntemps, ndim = (
            self._num_temps, self._model._num_free_parameters)

        if self._num_walkers:
            self._nwalkers = self._num_walkers
        else:
            self._nwalkers = 2 * ndim

        test_walker = self._iterations > 0
        self._lnprob = None
        self._lnlike = None
        pool_size = max(self._pool.size, 1)
        # Derived so only half a walker redrawn with Gaussian distribution.
        redraw_mult = 0.5 * np.sqrt(
            2) * scipy.special.erfinv(float(
                self._nwalkers - 1) / self._nwalkers)

        prt.message('nmeas_nfree', [self._model._num_measurements, ndim])
        if test_walker:
            if self._model._num_measurements <= ndim:
                prt.message('too_few_walkers', warning=True)
            if self._nwalkers < 10 * ndim:
                prt.message('want_more_walkers', [10 * ndim, self._nwalkers],
                            warning=True)
        p0 = [[] for x in range(self._ntemps)]

        # Generate walker positions based upon loaded walker data, if
        # available.
        walkers_pool = []
        walker_weights = []
        nmodels = len(set([x[0] for x in walker_data]))
        wp_extra = 0
        while len(walkers_pool) < len(walker_data):
            appended_walker = False
            for walk in walker_data:
                if (len(walkers_pool) + wp_extra) % nmodels != walk[0]:
                    continue
                new_walk = np.full(self._model._num_free_parameters, None)
                for k, key in enumerate(self._model._free_parameters):
                    param = self._model._modules[key]
                    walk_param = walk[1].get(key)
                    if walk_param is None or 'value' not in walk_param:
                        continue
                    if param:
                        val = param.fraction(
                            walk_param['value'], self._iterations != 0)
                        if not np.isnan(val):
                            new_walk[k] = val
                walkers_pool.append(new_walk)
                walker_weights.append(walk[2])
                appended_walker = True
            if not appended_walker:
                wp_extra += 1

        # Make sure weights are normalized.
        if None not in walker_weights:
            totw = np.sum(walker_weights)
            walker_weights = [x / totw for x in walker_weights]

        # Draw walker positions. This is either done from the priors or from
        # loaded walker data. If some parameters are not available from the
        # loaded walker data they will be drawn from their priors instead.
        pool_len = len(walkers_pool)
        for i, pt in enumerate(p0):
            dwscores = []
            while len(p0[i]) < self._nwalkers:
                prt.status(
                    self,
                    desc='drawing_walkers',
                    iterations=[
                        i * self._nwalkers + len(p0[i]) + 1,
                        self._nwalkers * self._ntemps])

                if self._pool.size == 0 or pool_len:
                    self._p, score = draw_walker(
                        test_walker, walkers_pool,
                        replace=pool_len < self._ntemps * self._nwalkers,
                        weights=walker_weights)
                    p0[i].append(self._p)
                    dwscores.append(score)
                else:
                    nmap = min(self._nwalkers -
                               len(p0[i]), max(self._pool.size, 10))
                    dws = self._pool.map(draw_walker, [test_walker] * nmap)
                    p0[i].extend([x[0] for x in dws])
                    dwscores.extend([x[1] for x in dws])

                if self._fitter._draw_above_likelihood is not False:
                    self._fitter._draw_above_likelihood = np.mean(dwscores)

        prt.message('initial_draws', inline=True)
        self._p = list(p0)

        self._emi = 0
        self._acor = None
        self._aacort = -1
        self._aa = 0
        self._psrf = np.inf
        self._all_chain = np.array([])
        self._scores = np.ones((self._ntemps, self._nwalkers)) * -np.inf

        tft = 0.0  # Total self._fracking time
        sli = 1.0  # Keep track of how many times chain halved
        s_exception = None
        kmat = None
        ages = np.zeros((self._ntemps, self._nwalkers), dtype=int)
        oldp = self._p

        max_chunk = 1000
        kmat_chunk = 5
        iter_chunks = int(np.ceil(float(self._iterations) / max_chunk))
        iter_arr = [max_chunk if xi < iter_chunks - 1 else
                    self._iterations - max_chunk * (iter_chunks - 1)
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
            if self._iterations > 0:
                sampler = MOSSampler(
                    self._ntemps, self._nwalkers, ndim, ln_likelihood,
                    ln_prior, pool=self._pool)
            while (self._iterations > 0 and (
                    self._cc is not None or ici < len(iter_arr))):
                slr = int(np.round(sli))
                ic = (max_chunk if self._cc is not None else
                      iter_arr[ici])
                if exceeded_walltime:
                    break
                if (self._cc is not None and converged and
                        self._emi > self._iterations):
                    break
                for li, (
                        self._p, self._lnprob, self._lnlike) in enumerate(
                            sampler.sample(
                                self._p, iterations=ic, gibbs=self._gibbs if
                                self._emi >= self._burn_in else True)):
                    if (self._fitter._maximum_walltime is not False and
                            self.time_running() >
                            self._fitter._maximum_walltime):
                        prt.message('exceeded_walltime', warning=True)
                        exceeded_walltime = True
                        break
                    self._emi = self._emi + 1
                    emim1 = self._emi - 1
                    messages = []

                    # Increment the age of each walker if their positions are
                    # unchanged.
                    for ti in range(self._ntemps):
                        for wi in range(self._nwalkers):
                            if np.array_equal(self._p[ti][wi], oldp[ti][wi]):
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

                    # During self._burn-in only, redraw any walkers with scores
                    # significantly worse than their peers, or those that are
                    # stale (i.e. remained in the same position for a long
                    # time).
                    if emim1 <= self._burn_in:
                        pmedian = [np.median(x) for x in self._lnprob]
                        pmead = [np.mean([abs(y - pmedian) for y in x])
                                 for x in self._lnprob]
                        redraw_count = 0
                        bad_redraws = 0
                        for ti, tprob in enumerate(self._lnprob):
                            for wi, wprob in enumerate(tprob):
                                if (wprob <= pmedian[ti] -
                                    max(redraw_mult * pmead[ti],
                                        float(self._nwalkers)) or
                                        np.isnan(wprob) or
                                        ages[ti][wi] >= self._REPLACE_AGE):
                                    redraw_count = redraw_count + 1
                                    dxx = np.random.normal(
                                        scale=0.01, size=ndim)
                                    tar_x = np.array(
                                        self._p[np.random.randint(
                                            self._ntemps)][
                                            np.random.randint(self._nwalkers)])
                                    # Reflect if out of bounds.
                                    new_x = np.clip(np.where(
                                        np.where(tar_x + dxx < 1.0,
                                                 tar_x + dxx,
                                                 tar_x - dxx) > 0.0,
                                        tar_x + dxx, tar_x - dxx), 0.0, 1.0)
                                    new_like = ln_likelihood(new_x)
                                    new_prob = new_like + ln_prior(new_x)
                                    if new_prob > wprob or np.isnan(wprob):
                                        self._p[ti][wi] = new_x
                                        self._lnlike[ti][wi] = new_like
                                        self._lnprob[ti][wi] = new_prob
                                    else:
                                        bad_redraws = bad_redraws + 1
                        if redraw_count > 0:
                            messages.append(
                                '{:.0%} redraw, {}/{} success'.format(
                                    redraw_count /
                                    (self._nwalkers * self._ntemps),
                                    redraw_count - bad_redraws, redraw_count))

                    oldp = self._p.copy()

                    # Calculate the autocorrelation time.
                    low = 10
                    asize = 0.5 * (emim1 - self._burn_in) / low
                    if asize >= 0 and self._ct == 'acor':
                        acorc = max(
                            1, min(self._MAX_ACORC,
                                   int(np.floor(0.5 * self._emi / low))))
                        self._aacort = -1.0
                        self._aa = 0
                        self._ams = self._burn_in
                        cur_chain = (np.concatenate(
                            (self._all_chain,
                             sampler.chain[:, :, :li + 1:slr, :]),
                            axis=2) if len(self._all_chain) else
                            sampler.chain[:, :, :li + 1:slr, :])
                        for a in range(acorc, 1, -1):
                            ms = self._burn_in
                            if ms >= self._emi - low:
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
                                self._aa = a
                                self._aacort = acort * sli
                                self._ams = ms
                                break
                        self._acor = [self._aacort, self._aa, self._ams]

                        self._actc = int(np.ceil(self._aacort / sli))
                        actn = np.int(
                            float(self._emi - self._ams) / self._actc)

                        if (self._cc is not None and
                            actn >= self._cc and
                                self._emi > self._iterations):
                            prt.message('converged')
                            converged = True
                            break

                    # Calculate the PSRF (Gelman-Rubin statistic).
                    if li > 1 and self._emi > self._burn_in + 2:
                        cur_chain = (np.concatenate(
                            (self._all_chain,
                             sampler.chain[:, :, :li + 1:slr, :]),
                            axis=2) if len(self._all_chain) else
                            sampler.chain[:, :, :li + 1:slr, :])
                        vws = np.zeros((self._ntemps, ndim))
                        for ti in range(self._ntemps):
                            for xi in range(ndim):
                                vchain = cur_chain[
                                    ti, :, int(np.floor(
                                        self._burn_in / sli)):, xi]
                                vws[ti][xi] = self.psrf(vchain)
                        self._psrf = np.max(vws)
                        if np.isnan(self._psrf):
                            self._psrf = np.inf

                        if (self._ct == 'psrf' and
                            self._cc is not None and
                            self._psrf < self._cc and
                                self._emi > self._iterations):
                            prt.message('converged')
                            converged = True
                            break

                    if self._cc is not None:
                        self._emcee_est_t = -1.0
                    else:
                        self._emcee_est_t = float(
                            self.time_running() - tft) / self._emi * (
                            self._iterations - self._emi
                        ) + tft / self._emi * max(
                                0, self._burn_in - self._emi)

                    # Perform self._fracking if we are still in the self._burn
                    # in phase and iteration count is a multiple of the frack
                    # step.
                    frack_now = (self._fracking and self._frack_step != 0 and
                                 self._emi <= self._burn_in and
                                 self._emi % self._frack_step == 0)

                    self._scores = [np.array(x) for x in self._lnprob]
                    if emim1 % kmat_chunk == 0:
                        sout = self._model.run_stack(
                            self._p[np.unravel_index(
                                np.argmax(self._lnprob), self._lnprob.shape)],
                            root='objective')
                        kmat = sout.get('kmat')
                        kdiag = sout.get('kdiagonal')
                        variance = sout.get('obandvs', sout.get('variance'))
                        if kdiag is not None and kmat is not None:
                            kmat[np.diag_indices_from(kmat)] += kdiag
                        elif kdiag is not None and kmat is None:
                            kmat = np.diag(kdiag + variance)
                    prt.status(
                        self,
                        desc='fracking' if frack_now else
                        ('burning' if self._emi < self._burn_in
                         else 'walking'),
                        scores=self._scores,
                        kmat=kmat,
                        accepts=accepts,
                        iterations=[self._emi, None if
                                    self._cc is not None else
                                    self._iterations],
                        acor=self._acor,
                        psrf=[self._psrf, self._burn_in],
                        messages=messages,
                        make_space=emim1 == 0,
                        convergence_type=self._ct,
                        convergence_criteria=self._cc,
                        time_running=self.time_running(),
                        maximum_walltime=self._fitter._maximum_walltime)

                    if s_exception:
                        break

                    if not frack_now:
                        continue

                    # Fracking starts here
                    sft = time.time()
                    ijperms = [[x, y] for x in range(self._ntemps)
                               for y in range(self._nwalkers)]
                    ijprobs = np.array([
                        1.0
                        # self._lnprob[x][y]
                        for x in range(self._ntemps) for y in range(
                            self._nwalkers)
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

                    bhwalkers = [self._p[i][j] for i, j in selijs]

                    seeds = [
                        int(round(time.time() * 1000.0)) % 4294900000 + x
                        for x in range(len(bhwalkers))
                    ]
                    frack_args = list(zip(bhwalkers, seeds))
                    bhs = list(self._pool.map(frack, frack_args))
                    for bhi, bh in enumerate(bhs):
                        (wi, ti) = tuple(selijs[bhi])
                        if -bh.fun > self._lnprob[wi][ti]:
                            self._p[wi][ti] = bh.x
                            like = ln_likelihood(bh.x)
                            self._lnprob[wi][ti] = like + ln_prior(bh.x)
                            self._lnlike[wi][ti] = like
                    self._scores = [[-x.fun for x in bhs]]
                    prt.status(
                        self,
                        desc='fracking_results',
                        scores=self._scores,
                        kmat=kmat,
                        fracking=True,
                        iterations=[self._emi, None if
                                    self._cc is not None else
                                    self._iterations],
                        convergence_type=self._ct,
                        convergence_criteria=self._cc,
                        time_running=self.time_running(),
                        maximum_walltime=self._fitter._maximum_walltime)
                    tft = tft + time.time() - sft
                    if s_exception:
                        break

                if ici == 0:
                    self._all_chain = sampler.chain[:, :, :li + 1:slr, :]
                    self._all_lnprob = sampler.lnprobability[:, :, :li + 1:slr]
                    self._all_lnlike = sampler.lnlikelihood[:, :, :li + 1:slr]
                else:
                    self._all_chain = np.concatenate(
                        (self._all_chain, sampler.chain[:, :, :li + 1:slr, :]),
                        axis=2)
                    self._all_lnprob = np.concatenate(
                        (self._all_lnprob,
                         sampler.lnprobability[:, :, :li + 1:slr]),
                        axis=2)
                    self._all_lnlike = np.concatenate(
                        (self._all_lnlike,
                         sampler.lnlikelihood[:, :, :li + 1:slr]),
                        axis=2)

                mem_mb = (self._all_chain.nbytes + self._all_lnprob.nbytes +
                          self._all_lnlike.nbytes) / (1024. * 1024.)

                if self._fitter._debug:
                    prt.prt('Memory `{}`'.format(mem_mb), wrapped=True)

                if mem_mb > self._fitter._maximum_memory:
                    sfrac = float(
                        self._all_lnprob.shape[-1]) / self._all_lnprob[
                            :, :, ::2].shape[-1]
                    self._all_chain = self._all_chain[:, :, ::2, :]
                    self._all_lnprob = self._all_lnprob[:, :, ::2]
                    self._all_lnlike = self._all_lnlike[:, :, ::2]
                    sli *= sfrac
                    if self._fitter._debug:
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

        if s_exception is not None:
            self._pool.close()
            if (not prt.prompt('mc_interrupted')):
                sys.exit()

        msg_criteria = (
            1.1 if self._cc is None else self._cc)
        if (test_walker and self._ct == 'psrf' and
                msg_criteria is not None and self._psrf > msg_criteria):
            prt.message('not_converged', [
                'default' if self._cc is None else 'specified',
                msg_criteria], warning=True)