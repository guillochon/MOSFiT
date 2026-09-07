"""Exercise the vendored parallel-tempered sampler and MOSSampler."""
from __future__ import print_function

import os
import sys

import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    from mosfit.mossampler import MOSSampler
    from mosfit.samplers.ptsampler import PTLikePrior, PTSampler, default_beta_ladder
    from mosfit.samplers.sampler import Sampler
    from emcee.autocorr import AutocorrError

    betas = default_beta_ladder(3, ntemps=4)
    assert len(betas) == 4
    betas2 = default_beta_ladder(200, Tmax=20.0)
    assert len(betas2) >= 2
    try:
        default_beta_ladder(2)
        raise SystemExit('expected ValueError')
    except ValueError:
        pass

    def logl(x):
        return -0.5 * np.sum(x * x)

    def logp(x):
        if np.any(np.abs(x) > 8.0):
            return -np.inf
        return 0.0

    wrap = PTLikePrior(logl, logp)
    val = wrap(np.zeros(2))
    assert val[0] == 0.0
    wrap(np.array([100.0, 0.0]))

    ntemps, nwalkers, dim = 2, 8, 2
    rng = np.random.RandomState(0)
    p0 = 0.1 * rng.randn(ntemps, nwalkers, dim)
    sampler = PTSampler(ntemps, nwalkers, dim, logl, logp, a=2.0)
    last = None
    for p, lnprob, lnlike in sampler.sample(p0, iterations=8):
        last = (p, lnprob, lnlike)
    assert last is not None
    assert sampler.chain.shape[0] == ntemps

    mos = MOSSampler(ntemps, nwalkers, dim, logl, logp)
    chain = np.zeros((ntemps, nwalkers, 80, dim))
    chain[:, :, 0] = 0.05 * rng.randn(ntemps, nwalkers, dim)
    for t in range(1, 80):
        chain[:, :, t] = 0.8 * chain[:, :, t - 1] + 0.2 * rng.randn(
            ntemps, nwalkers, dim)
    mos._chain = chain
    try:
        mos.get_autocorr_time(chain=chain, max_walkers=4)
    except (AutocorrError, ValueError):
        pass
    series = np.cumsum(rng.randn(400))
    try:
        tau = mos.integrated_time(series, c=5)
        assert np.isfinite(tau)
    except AutocorrError:
        pass

    class _F(object):
        _pool = type('P', (), {'size': 0})()
        _printer = None
        _start_time = 0.0

    samp = Sampler(_F(), num_walkers=4)
    chain2 = rng.randn(5, 20)
    psrf = samp.psrf(chain2)
    assert psrf > 0
    print('ptsampler tests passed')
    sys.exit(0)
