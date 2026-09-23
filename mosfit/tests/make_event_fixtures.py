"""Regenerate the simulated per-model event fixtures in ``tests/events``.

Every built-in model gets one small catalog-format JSON file holding a
synthetic light curve that the model itself produced, so the test suite can
exercise the full "read an event off disk and fit it" path for each transient
type without any network access or multi-megabyte data files.

The fixtures are committed, so this script only needs to be re-run when a
model's physics or parameter ranges change enough to make a fixture
unrepresentative::

    python mosfit/tests/make_event_fixtures.py          # all models
    python mosfit/tests/make_event_fixtures.py slsn tde # a subset

Draws are seeded per model, so re-running reproduces the same fixtures as long
as the model definitions are unchanged.
"""
from __future__ import print_function

import json
import os
import sys
from collections import OrderedDict

import numpy as np

# Photometry groups, each a (telescope, instrument, system, bands) set whose
# SVO filter responses are already cached under `modules/observables/filters`.
OPTICAL = OrderedDict([
    ('telescope', 'PAN-STARRS'), ('instrument', 'GPC'), ('system', 'AB'),
    ('bands', ['g', 'r', 'i', 'z']), ('limit', 23.0)])
ULTRAVIOLET = OrderedDict([
    ('telescope', 'Swift'), ('instrument', 'UVOT'), ('system', 'Vega'),
    ('bands', ['UVW2', 'UVM2', 'U']), ('limit', 21.5)])
NEAR_INFRARED = OrderedDict([
    ('telescope', ''), ('instrument', ''), ('system', 'AB'),
    ('bands', ['J', 'H']), ('limit', 21.0)])

# Per-model fixture recipe: how long the transient is watched, how many
# epochs, which photometry groups cover it, and the redshift and host
# extinction it sits behind. The values follow the kind of event each model is
# written for, so the synthetic light curves land in a plausible regime.
SUPERNOVA = dict(span=90.0, epochs=12, groups=[OPTICAL, ULTRAVIOLET],
                 redshift=0.03, ebv=0.04, texplosion=-6.0)
INTERACTING = dict(span=200.0, epochs=13, groups=[OPTICAL, ULTRAVIOLET],
                   redshift=0.06, ebv=0.03, texplosion=-12.0)
ENGINE = dict(span=150.0, epochs=13, groups=[OPTICAL, ULTRAVIOLET],
              redshift=0.12, ebv=0.02, texplosion=-10.0)
SUPERLUMINOUS = dict(span=220.0, epochs=14, groups=[OPTICAL, ULTRAVIOLET],
                     redshift=0.25, ebv=0.02, texplosion=-15.0)
TDE = dict(span=400.0, epochs=14, groups=[OPTICAL, ULTRAVIOLET],
           redshift=0.17, ebv=0.013, texplosion=-30.0)
KILONOVA = dict(span=14.0, epochs=11, groups=[OPTICAL, NEAR_INFRARED],
                redshift=0.0098, ebv=0.1, texplosion=-0.5)

FIXTURES = OrderedDict([
    ('bns', dict(KILONOVA, claimedtype='NS + NS')),
    ('bns_generative', dict(KILONOVA, claimedtype='NS + NS')),
    ('bns_magnetar', dict(KILONOVA, claimedtype='NS + NS')),
    ('csm', dict(INTERACTING, claimedtype='IIn')),
    # `csmni` overrides the base model's `luminositydistance` parameter with a
    # plain fixed one, so it never derives a distance from the event redshift.
    # Carrying an explicit `lumdist` keeps this fixture on the apparent
    # magnitude scale the other models use.
    ('csmni', dict(INTERACTING, claimedtype='IIn', lumdist=277.5)),
    ('default', dict(SUPERNOVA, claimedtype='Ib/c')),
    ('exppow', dict(SUPERNOVA, span=120.0, claimedtype='II')),
    ('fallback', dict(SUPERNOVA, span=150.0, claimedtype='Ic')),
    ('ia', dict(SUPERNOVA, span=80.0, redshift=0.02, claimedtype='Ia')),
    ('ic', dict(SUPERNOVA, claimedtype='Ic')),
    ('kilonova', dict(KILONOVA, claimedtype='Kilonova')),
    ('magnetar', dict(ENGINE, claimedtype='Ic-BL')),
    ('magni', dict(ENGINE, claimedtype='Ic-BL')),
    ('nsbh', dict(KILONOVA, claimedtype='BH + NS')),
    ('nsbh_generative', dict(KILONOVA, claimedtype='BH + NS')),
    ('rprocess', dict(KILONOVA, claimedtype='Kilonova')),
    ('shockni', dict(SUPERNOVA, claimedtype='IIb')),
    ('slsn', dict(SUPERLUMINOUS, claimedtype='SLSN-I')),
    ('slsnni', dict(SUPERLUMINOUS, claimedtype='SLSN-I')),
    ('tde', dict(TDE, claimedtype='TDE')),
    ('tde_shock', dict(TDE, claimedtype='TDE')),
])

# Built-in models that get no fixture. `sesn_sedona` needs the optional
# `sedona` (PyTorch) extra, so it is left out for the same reason
# `_test_models_likelihood.py` skips it. `_test_transient_types.py` reads this
# set too, so a model excluded here is not expected to have a fixture there.
SKIP = {'sesn_sedona'}

# One fixture is mirrored as a comma-separated table so the suite also covers
# MOSFiT's ASCII-to-catalog-JSON conversion, which is how most users hand it
# their own photometry. The header names are the ones `Converter` recognizes
# on its own, so the conversion runs without prompting.
CSV_MODEL = 'default'
CSV_COLUMNS = ['name', 'time', 'band', 'system', 'instrument', 'telescope',
               'magnitude', 'e_magnitude', 'upperlimit', 'reference',
               'redshift', 'ebv']
# No spaces or punctuation: `Converter` sniffs the delimiter by counting
# candidate characters across the whole file.
CSV_SOURCE = 'MOSFiT_simulated_photometry'

# First epoch of the synthetic campaign, an arbitrary but realistic MJD.
FIRST_MJD = 57000.0
# Target peak apparent magnitude used to pick among candidate prior draws.
TARGET_PEAK = 19.0
# Fraction of the campaign that should come out brighter than its group's
# limiting magnitude, so a fixture carries detections as well as the
# non-detections that bracket them.
TARGET_DETECTED = 0.75
# Number of prior draws to try per model before keeping the best one.
DRAWS = 150

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EVENTS = os.path.join(ROOT, 'mosfit', 'tests', 'events')


def event_name(model_name):
    """Return the fixture event name used for a given model."""
    return 'sim_' + model_name


def fixture_path(model_name):
    """Return the on-disk path of a model's fixture."""
    return os.path.join(EVENTS, event_name(model_name) + '.json')


def _epochs(recipe, rng):
    """Return (time, band, group) rows on a jittered observing cadence."""
    rows = []
    for gi, group in enumerate(recipe['groups']):
        # Offset each group slightly so the bands are not perfectly
        # simultaneous, as they would not be for real follow-up.
        offset = 0.04 * recipe['span'] * gi / max(len(recipe['groups']), 1)
        times = np.linspace(0.0, recipe['span'], recipe['epochs']) + offset
        times = times + rng.uniform(
            -0.01 * recipe['span'], 0.01 * recipe['span'], times.shape)
        for time in times:
            # Round here so the row times match what is written to JSON, and
            # so `_check_order` compares like with like.
            mjd = round(FIRST_MJD + float(time), 4)
            for band in group['bands']:
                rows.append((mjd, band, group))
    rows.sort(key=lambda row: row[0])
    return rows


def _skeleton(name, recipe, rows):
    """Return a catalog-format event with placeholder magnitudes."""
    photometry = []
    for time, band, group in rows:
        phot = OrderedDict([('time', '{:.4f}'.format(time)), ('band', band),
                            ('magnitude', '20.0'), ('e_magnitude', '0.05'),
                            ('system', group['system'])])
        if group['instrument']:
            phot['instrument'] = group['instrument']
        if group['telescope']:
            phot['telescope'] = group['telescope']
        phot['source'] = '1'
        photometry.append(phot)

    source = OrderedDict([
        ('name', 'MOSFiT simulated photometry'),
        ('reference', 'Generated by mosfit/tests/make_event_fixtures.py'),
        ('alias', '1')])
    event = OrderedDict([
        ('name', name),
        ('sources', [source]),
        ('alias', [OrderedDict([('value', name), ('source', '1')])]),
        ('claimedtype',
         [OrderedDict([('value', recipe['claimedtype']), ('source', '1')])]),
        ('ebv', [OrderedDict([('value', str(recipe['ebv'])),
                              ('source', '1')])]),
        ('redshift', [OrderedDict([('value', str(recipe['redshift'])),
                                   ('source', '1')])])])
    if recipe.get('lumdist') is not None:
        event['lumdist'] = [OrderedDict([
            ('value', str(recipe['lumdist'])), ('u_value', 'Mpc'),
            ('source', '1')])]
    event['photometry'] = photometry
    return event


def _load_model(model_name, data, name):
    """Return a `Model` loaded with the placeholder event."""
    from schwimmbad import SerialPool

    from mosfit.fitter import Fitter
    from mosfit.model import Model

    fitter = Fitter(
        test=True, quiet=True, exit_on_prompt=True, prefer_cache=True)
    fitter._event_name = name
    fitter._event_path = ''
    fitter._event_data = data
    pool = SerialPool()
    model = Model(model=model_name, data=data, test=True,
                  printer=fitter._printer, fitter=fitter, pool=pool)
    if not model.load_data(data, event_name=name, pool=pool):
        raise RuntimeError('load_data failed for {}'.format(model_name))
    return model


def _check_order(outputs, rows):
    """Confirm model outputs line up row-for-row with the fixture photometry.

    The fixtures are written by pasting `model_observations` back onto the
    placeholder rows, which is only valid while the model preserves the order
    (and completeness) of the photometry it was handed. Times come back
    relative to the first observation, so only the offsets are compared.
    """
    bands = list(outputs['all_bands'])
    times = np.array(outputs['all_times'], dtype=float)
    if len(bands) != len(rows):
        raise RuntimeError('model returned {} points for {} observations'
                           .format(len(bands), len(rows)))
    if not np.all(outputs['observed']):
        raise RuntimeError('model dropped some observations')
    for i, (time, band, _) in enumerate(rows):
        if bands[i] != band:
            raise RuntimeError('band {} at row {} became {}'.format(
                band, i, bands[i]))
        if abs((times[i] - times[0]) - (time - rows[0][0])) > 1.0e-6:
            raise RuntimeError('time at row {} is out of order'.format(i))


def _best_draw(model, recipe, rows, rng):
    """Return the prior draw whose light curve looks most like a transient.

    Draws are scored on how close the peak is to `TARGET_PEAK`, whether that
    peak falls early in the campaign (so both rise and decline are sampled),
    whether the curve actually varies, and whether most points come out
    brighter than their group's limiting magnitude.
    """
    free = model.free_parameter_names()
    tfrac = None
    if 'texplosion' in free:
        tfrac = model._modules['texplosion'].fraction(recipe['texplosion'])
    limits = np.array([row[2]['limit'] for row in rows], dtype=float)

    best = None
    for _ in range(DRAWS):
        draw = np.array(
            model.draw_from_icdf(
                rng.uniform(0.0, 1.0, model.get_num_free_parameters())))
        if tfrac is not None:
            draw[free.index('texplosion')] = tfrac
        try:
            outputs = model.run_stack(draw, root='output')
        except Exception:
            continue
        mags = np.array(outputs['model_observations'], dtype=float)
        if mags.size == 0 or not np.all(np.isfinite(mags)):
            continue
        _check_order(outputs, rows)
        times = np.array(outputs['all_times'], dtype=float)
        peak = float(np.min(mags))
        peak_frac = ((times[int(np.argmin(mags))] - times.min()) /
                     max(times.max() - times.min(), 1.0e-30))
        amplitude = float(np.max(mags) - np.min(mags))
        detected = float(np.mean(mags < limits))
        cost = (abs(peak - TARGET_PEAK) + 6.0 * abs(peak_frac - 0.25) +
                max(0.0, 2.0 - amplitude) +
                8.0 * max(0.0, TARGET_DETECTED - detected))
        if best is None or cost < best[0]:
            best = (cost, draw, mags)
    if best is None:
        raise RuntimeError('no finite light curve drawn for model')
    return best[1], best[2]


def _apply_photometry(event, rows, mags, rng):
    """Write noisy magnitudes (and non-detections) into the event."""
    n_detections = 0
    for phot, (_, _, group), mag in zip(event['photometry'], rows, mags):
        # Fainter points carry larger errors, as in real follow-up.
        # Anything past the group's limit is recorded as a non-detection.
        error = float(np.clip(0.02 + 0.06 * (mag - 17.0) / 5.0, 0.02, 0.25))
        observed = float(mag + rng.normal(0.0, error))
        if observed > group['limit']:
            phot['magnitude'] = '{:.3f}'.format(group['limit'])
            phot['upperlimit'] = True
            del phot['e_magnitude']
        else:
            phot['magnitude'] = '{:.3f}'.format(observed)
            phot['e_magnitude'] = '{:.3f}'.format(error)
            n_detections += 1
    return n_detections


def csv_path(model_name):
    """Return the on-disk path of a model's comma-separated fixture."""
    return os.path.join(EVENTS, event_name(model_name) + '.csv')


def write_csv(model_name, event):
    """Mirror a finished fixture as a comma-separated table."""
    name = event['name']
    redshift = event['redshift'][0]['value']
    ebv = event['ebv'][0]['value']

    lines = [','.join(CSV_COLUMNS)]
    for phot in event['photometry']:
        lines.append(','.join([
            name,
            phot['time'],
            phot['band'],
            phot['system'],
            phot.get('instrument', ''),
            phot.get('telescope', ''),
            phot['magnitude'],
            phot.get('e_magnitude', ''),
            'True' if phot.get('upperlimit') else 'False',
            CSV_SOURCE,
            redshift,
            ebv,
        ]))

    path = csv_path(model_name)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def build(model_name, recipe, seed):
    """Build and write one model's fixture, returning a short summary."""
    name = event_name(model_name)
    rng = np.random.default_rng(seed)
    rows = _epochs(recipe, rng)
    event = _skeleton(name, recipe, rows)
    model = _load_model(model_name, {name: event}, name)
    _, mags = _best_draw(model, recipe, rows, rng)
    n_detections = _apply_photometry(event, rows, mags, rng)

    path = fixture_path(model_name)
    with open(path, 'w') as f:
        json.dump({name: event}, f, indent='\t', separators=(',', ':'))
        f.write('\n')
    summary = '{} points, {} detections, {} bytes'.format(
        len(rows), n_detections, os.path.getsize(path))
    if model_name == CSV_MODEL:
        summary += ', + {}'.format(os.path.basename(write_csv(
            model_name, event)))
    return summary


def main(argv):
    """Regenerate every requested fixture."""
    os.chdir(ROOT)
    if not os.path.isdir(EVENTS):
        os.makedirs(EVENTS)

    wanted = argv[1:] or list(FIXTURES)
    unknown = [x for x in wanted if x not in FIXTURES]
    if unknown:
        print('unknown model(s): {}'.format(', '.join(unknown)),
              file=sys.stderr)
        return 1

    failed = []
    for model_name in wanted:
        if model_name in SKIP:
            continue
        # A per-model seed keeps one model's fixture stable when another's
        # recipe changes. `hash()` is salted per process, so derive it from
        # the name directly.
        seed = sum(ord(c) for c in model_name) * 7919
        try:
            print('{:<18}'.format(model_name),
                  build(model_name, FIXTURES[model_name], seed), flush=True)
        except Exception as exc:
            failed.append('{}: {}: {}'.format(
                model_name, type(exc).__name__, exc))
            print('{:<18} FAILED {}'.format(model_name, failed[-1]),
                  flush=True)

    if failed:
        print('failed fixtures:', file=sys.stderr)
        for line in failed:
            print(' ', line, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
