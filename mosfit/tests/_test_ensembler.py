"""Short ensembler (emcee) smoke on dummy exppow data."""
from __future__ import print_function

import os
import sys

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    import mosfit

    fitter = mosfit.fitter.Fitter(
        quiet=True, test=True, exit_on_prompt=True, prefer_cache=True)
    entries, ps, lnprobs = fitter.fit_events(
        events=[],
        models=['exppow'],
        iterations=2,
        num_walkers=24,
        method='ensembler',
        fracking=False,
        write=False)
    assert entries
    print('ensembler smoke ok', [[y['models'][0]['score']['value']
                                  for y in x] for x in entries])
    sys.exit(0)
