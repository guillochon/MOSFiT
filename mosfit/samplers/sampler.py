# -*- coding: UTF-8 -*-
"""Definitions for `Sampler` class."""

import numpy as np


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
