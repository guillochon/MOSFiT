"""Load and fit one event for every built-in transient type.

`mosfit/tests/events/sim_<model>.json` holds a small catalog-format light
curve that the matching model produced itself (see `make_event_fixtures.py`),
so every built-in type is exercised over the whole public path: resolve the
event on disk, ingest its photometry, evaluate a likelihood, and run a short
ensemble fit.

`slsn` and `tde` are additionally fit against real observed events in
`_test_lsq12dlf.py` and `_test_pool_likelihood.py`.
"""
from __future__ import print_function

import json
import os
import sys
from pathlib import Path

import numpy as np

# Short ensemble run: enough to build walkers, score them, and return an
# entry, without turning the suite into a real fit.
ITERATIONS = 2

# Prior draws allowed while looking for a parameter vector that satisfies a
# model's constraints. Models such as `csmni` and `nsbh_generative` reject the
# middle of their prior box, so a fixed probe vector is not enough.
DRAWS = 40

# Fixed so a failure here is reproducible.
SEED = 20260915


def built_in_models(models_root, skip):
    """Return every packaged model that ships a model definition."""
    return sorted(
        p.name for p in models_root.iterdir()
        if p.is_dir() and (p / (p.name + '.json')).is_file()
        and p.name not in skip)


def fixture_photometry(fixture):
    """Return the photometry rows stored in a fixture."""
    with open(fixture, 'r') as f:
        data = json.load(f)
    return data[list(data)[0]]['photometry']


def load_event(model_name, fixture, pool):
    """Resolve a fixture off disk and load it into its model."""
    from mosfit.fitter import Fitter
    from mosfit.model import Model

    fitter = Fitter(
        test=True, quiet=True, exit_on_prompt=True, prefer_cache=True)
    event = fitter._fetcher.fetch([str(fixture)])[0]
    fitter._event_name = event['name']
    fitter._event_path = event['path']
    fitter._event_data = fitter._fetcher.load_data(event)
    if fitter._event_data is None:
        raise RuntimeError('fetcher returned no data for {}'.format(fixture))
    model = Model(
        model=model_name,
        data=fitter._event_data,
        test=True,
        printer=fitter._printer,
        fitter=fitter,
        pool=pool)
    if not model.load_data(
            fitter._event_data, event_name=fitter._event_name, pool=pool):
        raise RuntimeError('load_data returned False')
    return model


def finite_likelihood(model, rng):
    """Return the first finite ln(likelihood) found among prior draws."""
    ndim = model.get_num_free_parameters()
    for _ in range(DRAWS):
        draw = model.draw_from_icdf(rng.uniform(0.0, 1.0, ndim))
        value = float(model.ln_likelihood(draw))
        if np.isfinite(value):
            return value
    raise RuntimeError(
        'no finite ln_likelihood in {} prior draws'.format(DRAWS))


def check_type(model_name, fixture, pool):
    """Run one transient type end to end, returning a one-line summary."""
    from mosfit.fitter import Fitter

    photometry = fixture_photometry(fixture)
    model = load_event(model_name, fixture, pool)

    ndim = model.get_num_free_parameters()
    if ndim <= 0:
        raise RuntimeError('model has no free parameters')
    if model._num_measurements != len(photometry):
        raise RuntimeError('ingested {} of {} photometry points'.format(
            model._num_measurements, len(photometry)))

    ln_like = finite_likelihood(model, np.random.default_rng(SEED))

    # Now the same event through the public fitting entry point.
    fitter = Fitter(
        quiet=True, test=True, exit_on_prompt=True, prefer_cache=True)
    entries, _, _ = fitter.fit_events(
        events=[str(fixture)],
        models=[model_name],
        iterations=ITERATIONS,
        method='ensembler',
        fracking=False,
        write=False)
    if not entries or not entries[0]:
        raise RuntimeError('fit_events returned no entry')
    score = float(entries[0][0]['models'][0]['score']['value'])
    if not np.isfinite(score):
        raise RuntimeError('fit score is {}'.format(score))

    return 'ndim {:>2}  nobs {:>3}  lnL {:>14.3f}  score {:>14.3f}'.format(
        ndim, model._num_measurements, ln_like, score)


def main():
    """Check every built-in transient type."""
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    from schwimmbad import SerialPool

    # Share the generator's skip list and fixture naming, so a model added to
    # one is picked up by the other.
    from mosfit.tests.make_event_fixtures import SKIP, fixture_path

    # Seed the global RNG the ensemble sampler draws from, so a failing
    # transient type fails the same way on a re-run.
    np.random.seed(SEED)

    names = built_in_models(Path('mosfit') / 'models', SKIP)
    if not names:
        print('no built-in models found', file=sys.stderr)
        return 1

    failed = []
    pool = SerialPool()
    for name in names:
        fixture = Path(fixture_path(name))
        print('type', name, flush=True)
        if not fixture.is_file():
            failed.append(
                '{}: no fixture at {}; run `python '
                'mosfit/tests/make_event_fixtures.py {}`'.format(
                    name, fixture, name))
            print('  FAILED', failed[-1], flush=True)
            continue
        try:
            print(' ', check_type(name, fixture, pool), flush=True)
        except Exception as exc:
            failed.append('{}: {}: {}'.format(name, type(exc).__name__, exc))
            print('  FAILED', failed[-1], flush=True)

    if failed:
        print('failed transient types:', file=sys.stderr)
        for line in failed:
            print(' ', line, file=sys.stderr)
        return 1
    print('all {} transient types loaded and fit'.format(len(names)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
