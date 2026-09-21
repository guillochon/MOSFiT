"""Fit a transient handed to MOSFiT as a comma-separated table.

Catalog JSON is only one of the input formats MOSFiT accepts: `Converter`
turns ASCII tables (CSV and friends) into catalog entries before fitting.
`mosfit/tests/events/sim_default.csv` is the same simulated event as
`sim_default.json`, written as a table, so this test can convert it, check
the conversion reproduces the JSON fixture's photometry, and confirm both
routes into the model give the same likelihood.

`exit_on_prompt` is set throughout, so any column the converter cannot
identify on its own fails the test instead of waiting for a human.
"""
from __future__ import print_function

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

MODEL = 'default'
EVENT = 'sim_' + MODEL

# Photometry keys the conversion is expected to round-trip.
COMPARED = ['time', 'band', 'system', 'instrument', 'telescope', 'magnitude',
            'e_magnitude', 'upperlimit']

ITERATIONS = 2
DRAWS = 40
SEED = 20260915


def photometry_of(path):
    """Return the photometry rows of a catalog-format JSON file."""
    with open(str(path), 'r') as f:
        data = json.load(f)
    return data[list(data)[0]]['photometry']


def quantity_of(path, key):
    """Return the single value of an event-level quantity."""
    with open(str(path), 'r') as f:
        data = json.load(f)
    values = data[list(data)[0]][key]
    if len(values) != 1:
        raise RuntimeError('{} has {} {} values, expected 1'.format(
            path, len(values), key))
    return values[0]['value']


def _sorted_rows(photometry):
    """Return photometry in a canonical order.

    `Entry.sanitize()` reorders the rows it writes, so the two files are
    compared as sets of observations rather than in file order.
    """
    return sorted(photometry,
                  key=lambda row: tuple(str(row.get(k, '')) for k in COMPARED))


def compare_photometry(converted, expected):
    """Raise unless the converted photometry matches the JSON fixture."""
    if len(converted) != len(expected):
        raise RuntimeError('converted {} points, fixture has {}'.format(
            len(converted), len(expected)))
    for i, (got, want) in enumerate(
            zip(_sorted_rows(converted), _sorted_rows(expected))):
        for key in COMPARED:
            if got.get(key) != want.get(key):
                raise RuntimeError(
                    'row {} key {!r}: converted {!r}, fixture {!r}'.format(
                        i, key, got.get(key), want.get(key)))


def load_model(path, pool):
    """Load a catalog JSON file into the model under test."""
    from mosfit.fitter import Fitter
    from mosfit.model import Model

    fitter = Fitter(
        test=True, quiet=True, exit_on_prompt=True, prefer_cache=True)
    event = fitter._fetcher.fetch([str(path)])[0]
    data = fitter._fetcher.load_data(event)
    fitter._event_name = event['name']
    fitter._event_path = event['path']
    fitter._event_data = data
    model = Model(
        model=MODEL, data=data, test=True, printer=fitter._printer,
        fitter=fitter, pool=pool)
    if not model.load_data(data, event_name=event['name'], pool=pool):
        raise RuntimeError('load_data returned False for {}'.format(path))
    return model


def matching_likelihoods(converted_path, fixture_path, pool):
    """Return the likelihood both input routes give for one parameter draw."""
    fixture_model = load_model(fixture_path, pool)
    converted_model = load_model(converted_path, pool)

    ndim = fixture_model.get_num_free_parameters()
    if converted_model.get_num_free_parameters() != ndim:
        raise RuntimeError('converted event freed {} parameters, fixture {}'
                           .format(converted_model.get_num_free_parameters(),
                                   ndim))

    rng = np.random.default_rng(SEED)
    for _ in range(DRAWS):
        draw = fixture_model.draw_from_icdf(rng.uniform(0.0, 1.0, ndim))
        from_fixture = float(fixture_model.ln_likelihood(draw))
        if not np.isfinite(from_fixture):
            continue
        from_csv = float(converted_model.ln_likelihood(draw))
        # `sanitize()` reorders the rows, so the two sums agree only up to
        # floating-point summation order.
        if not np.isclose(from_csv, from_fixture, rtol=1.0e-10, atol=0.0):
            raise RuntimeError(
                'converted event gives lnL {}, fixture gives {}'.format(
                    from_csv, from_fixture))
        return from_fixture
    raise RuntimeError(
        'no finite ln_likelihood in {} prior draws'.format(DRAWS))


def main():
    """Convert the CSV fixture, fit it, and check it against the JSON one."""
    repo = Path(__file__).resolve().parents[2]
    os.chdir(str(repo))
    from schwimmbad import SerialPool

    import mosfit

    events = repo / 'mosfit' / 'tests' / 'events'
    csv_fixture = events / (EVENT + '.csv')
    json_fixture = events / (EVENT + '.json')
    for path in (csv_fixture, json_fixture):
        if not path.is_file():
            print('missing fixture {}'.format(path), file=sys.stderr)
            return 1

    # `Converter` writes the catalog JSON it produces into the working
    # directory, so run the conversion somewhere disposable.
    workdir = tempfile.mkdtemp(prefix='mosfit-csv-')
    try:
        local_csv = os.path.join(workdir, csv_fixture.name)
        shutil.copy(str(csv_fixture), local_csv)
        os.chdir(workdir)

        np.random.seed(SEED)
        fitter = mosfit.fitter.Fitter(
            quiet=True, test=True, exit_on_prompt=True, prefer_cache=True)
        entries, _, _ = fitter.fit_events(
            events=[local_csv],
            models=[MODEL],
            iterations=ITERATIONS,
            method='ensembler',
            fracking=False,
            write=False)

        converted = fitter._converter.get_converted()
        print('converted', converted)
        if len(converted) != 1 or converted[0][0] != EVENT:
            raise RuntimeError(
                'expected one converted event named {}, got {}'.format(
                    EVENT, converted))
        converted_path = os.path.join(workdir, converted[0][1])
        if not os.path.isfile(converted_path):
            raise RuntimeError(
                'converter wrote no JSON at {}'.format(converted_path))

        if not entries or not entries[0]:
            raise RuntimeError('fit_events returned no entry for the CSV')
        score = float(entries[0][0]['models'][0]['score']['value'])
        if not np.isfinite(score):
            raise RuntimeError('fit score is {}'.format(score))

        compare_photometry(
            photometry_of(converted_path), photometry_of(json_fixture))
        for key in ('redshift', 'ebv'):
            got = quantity_of(converted_path, key)
            want = quantity_of(json_fixture, key)
            if got != want:
                raise RuntimeError('{}: converted {!r}, fixture {!r}'.format(
                    key, got, want))

        ln_like = matching_likelihoods(
            converted_path, json_fixture, SerialPool())
    except Exception as exc:
        print('CSV input FAILED: {}: {}'.format(type(exc).__name__, exc),
              file=sys.stderr)
        return 1
    finally:
        os.chdir(str(repo))
        shutil.rmtree(workdir, ignore_errors=True)

    print('{} rows converted from CSV, lnL {:.3f}, fit score {:.3f}'.format(
        len(photometry_of(json_fixture)), ln_like, score))
    print('CSV input ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
