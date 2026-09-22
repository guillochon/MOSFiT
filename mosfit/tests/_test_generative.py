"""Generative runs (`-i 0`, no event) must work under every sampler.

The nested sampler has nothing to nest against when there is no likelihood to
sample, and used to return from `run()` without leaving results behind, so
`prepare_output()` raised `AttributeError: 'Nester' object has no attribute
'_results'`. Prior draws are the right answer there, not a crash.
"""
from __future__ import print_function

import os
import sys

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    import mosfit

    n_walkers = 6

    for method in ('dynesty', 'ensembler'):
        fitter = mosfit.fitter.Fitter(
            quiet=True, test=True, exit_on_prompt=True, prefer_cache=True)
        entries, ps, lnprobs = fitter.fit_events(
            events=[],
            models=['exppow'],
            iterations=0,
            num_walkers=n_walkers,
            method=method,
            fracking=False,
            write=False)
        assert entries, method
        model = entries[0][0]['models'][0]
        reals = model['realizations']

        # Every draw the user asked for is rendered once. Nested sampling
        # resamples against its weights, which is wrong for equally weighted
        # prior draws: it would duplicate some and drop others.
        assert len(reals) == n_walkers, (method, len(reals))
        assert sorted(int(r['alias']) for r in reals) == list(
            range(1, n_walkers + 1)), method

        # Draws come from the priors, so they should differ from each other.
        values = {r['parameters']['mejecta']['value'] for r in reals}
        assert len(values) == n_walkers, (method, values)

        weights = {r['weight'] for r in reals}
        assert len(weights) == 1, (method, weights)

        # Nothing was sampled, so there is no evidence or WAIC to report.
        assert 'score' not in model, method

        print('generative {} ok'.format(method))

    sys.exit(0)
