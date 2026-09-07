"""Load the LSQ12dlf catalog fixture and evaluate one SLSN likelihood."""
from __future__ import print_function

import os
import sys

import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    from schwimmbad import SerialPool

    from mosfit.fitter import Fitter, ln_likelihood
    from mosfit.model import Model

    fixture = os.path.join('mosfit', 'tests', 'LSQ12dlf.json')
    if not os.path.isfile(fixture):
        raise SystemExit('missing fixture {}'.format(fixture))

    fitter = Fitter(
        test=True, quiet=False, exit_on_prompt=True, prefer_cache=True)
    fetched = fitter._fetcher.fetch([fixture])
    event = fetched[0]
    if event.get('name') != 'LSQ12dlf':
        raise SystemExit('expected event name LSQ12dlf, got {!r}'.format(
            event.get('name')))
    fitter._event_name = event.get('name', 'LSQ12dlf')
    fitter._event_path = event.get('path', '')
    fitter._event_data = fitter._fetcher.load_data(event)
    pool = SerialPool()
    model = Model(
        model='slsn',
        data=fitter._event_data,
        test=True,
        printer=fitter._printer,
        fitter=fitter,
        pool=pool)
    ok = model.load_data(
        fitter._event_data,
        event_name=fitter._event_name,
        pool=pool)
    if not ok:
        raise SystemExit('load_data failed')
    import mosfit.fitter as ft
    ft.model = model
    x = np.full(model._num_free_parameters, 0.5)
    ll = float(ln_likelihood(x))
    print('LSQ12dlf slsn ln_likelihood', ll, 'ndim', model._num_free_parameters)
    if not np.isfinite(ll):
        raise SystemExit('ln_likelihood is not finite')
    print('LSQ12dlf slsn likelihood ok')
    sys.exit(0)
