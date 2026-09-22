# -*- coding: UTF-8 -*-
"""Definitions for `Sampler` class."""

import numpy as np
import time


class Sampler(object):
    """Sample the posterior distribution of a model against an observation."""

    _MIN_WEIGHT = 1e-4

    def __init__(self, fitter, num_walkers=None, **kwargs):
        """Initialize `Sampler` class."""
        self._printer = kwargs.get('printer')
        self._fitter = fitter
        self._pool = self._fitter._pool
        self._printer = self._fitter._printer

        self._num_walkers = num_walkers

        # A generative run (`-i 0`, typically with no event) draws from the
        # priors and never samples a posterior. Samplers set this so that
        # output paths which assume a fitted chain can step aside.
        self._generative = False

    def draw_from_priors(self, nwalkers):
        """Populate the output arrays with draws from the priors.

        Used when there is nothing to fit: every draw is kept, unweighted and
        unscored, which is what a generative run wants.
        """
        from mosfit.fitter import draw_walker

        prt = self._printer
        draws = []
        while len(draws) < nwalkers:
            prt.status(
                self, desc='drawing_walkers',
                iterations=[len(draws) + 1, nwalkers])
            if self._pool.size == 0:
                draws.append(draw_walker(False)[0])
            else:
                nmap = min(nwalkers - len(draws), max(self._pool.size, 10))
                draws.extend(
                    [x[0] for x in self._pool.map(draw_walker,
                                                  [False] * nmap)])
        prt.message('initial_draws', inline=True)

        self._generative = True
        self._pout = [np.array(draws[:nwalkers])]
        self._lnprobout = None
        self._lnlikeout = None
        self._weights = None

    def get_samples(self):
        """Return samples from ensembler."""
        samples = np.array([a for b in self._pout for a in b])
        if self._lnprobout is None:
            return samples, None, np.array([
                1.0 / len(samples) for x in samples])
        probs = np.array([a for b in self._lnprobout for a in b])
        weights = np.array([a for b in self._weights for a in b])

        min_weight = self._MIN_WEIGHT / len(samples)

        sel = weights > min_weight
        samples = samples[sel]
        probs = probs[sel]
        weights = weights[sel]

        wsis = np.argsort(weights)

        samples = samples[wsis]
        probs = probs[wsis]
        weights = weights[wsis]

        return samples, probs, weights

    def run(self):
        """Run the sampler."""
        pass

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

    def time_running(self):
        return time.time() - self._fitter._start_time
