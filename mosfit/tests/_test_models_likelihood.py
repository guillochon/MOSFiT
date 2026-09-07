"""Load every packaged model and evaluate one likelihood vector."""
from __future__ import print_function

import os
import sys
from pathlib import Path

import numpy as np
from schwimmbad import SerialPool

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    from mosfit.fitter import Fitter, ln_likelihood
    from mosfit.model import Model

    skip = {'sesn_sedona'}
    models_root = Path('mosfit') / 'models'
    names = sorted(
        p.name for p in models_root.iterdir()
        if p.is_dir() and (p / (p.name + '.json')).is_file()
        and p.name not in skip)

    failed = []
    pool = SerialPool()
    for name in names:
        print('model', name, flush=True)
        try:
            fitter = Fitter(
                test=True, quiet=True, exit_on_prompt=True, prefer_cache=True)
            dummy = fitter.generate_dummy_data(name=name)
            fitter._event_name = name
            fitter._event_path = ''
            fitter._event_data = dummy
            model = Model(
                model=name,
                data=dummy,
                test=True,
                printer=fitter._printer,
                fitter=fitter,
                pool=pool)
            ok = model.load_data(
                dummy, event_name=name, pool=pool)
            if not ok:
                raise RuntimeError('load_data returned False')
            import mosfit.fitter as ft
            ft.model = model
            x = np.full(model._num_free_parameters, 0.5)
            ll = float(ln_likelihood(x))
            print('  ndim', model._num_free_parameters, 'lnL', ll, flush=True)
        except Exception as exc:
            failed.append('{}: {}: {}'.format(name, type(exc).__name__, exc))
            print('  FAILED', failed[-1], flush=True)

    if failed:
        print('failed models:', file=sys.stderr)
        for line in failed:
            print(' ', line, file=sys.stderr)
        sys.exit(1)
    print('all models produced a likelihood')
    sys.exit(0)
