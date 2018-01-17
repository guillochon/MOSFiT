# -*- coding: UTF-8 -*-
"""Definitions for `Sampler` class."""

import numpy as np


class Sampler(object):
    """Fit transient events with the provided model."""

    def __init__(self, fitter, **kwargs):
        """Initialize `Sampler` class."""
        self._printer = kwargs.get('printer')
        self._fitter = fitter
        self._pool = self._fitter._pool
        self._printer = self._fitter._printer

    def get_samples(self):
        """Return samples from the sampler."""
        pass

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
