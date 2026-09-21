"""Serial vs 10-core ln_likelihood on a fixed TDE parameter vector."""
from __future__ import print_function

import os
import sys

import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    from schwimmbad import SerialPool

    from mosfit.fitter import (Fitter, attach_likelihood_pool, ln_likelihood,
                               pool_queue_size)
    from mosfit.model import Model

    fitter = Fitter(
        test=True, quiet=False, exit_on_prompt=True, prefer_cache=True)
    fetched = fitter._fetcher.fetch(['mosfit/tests/PS1-10jh.json'])
    event = fetched[0]
    fitter._event_name = event.get('name', 'PS1-10jh')
    fitter._event_path = event.get('path', '')
    fitter._event_data = fitter._fetcher.load_data(event)
    pool = SerialPool()
    model = Model(
        model='tde',
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
    ndim = model._num_free_parameters
    x = np.full(ndim, 0.5)
    serial = float(ln_likelihood(x))
    print('serial ln_likelihood', serial, 'ndim', ndim)
    workers = attach_likelihood_pool(SerialPool(), 10, model, method='dynesty')
    print('pool class', type(workers).__name__, 'size', workers.size,
          'queue', pool_queue_size(workers))
    mapped = list(workers.map(ln_likelihood, [x] * 10))
    workers.close()
    try:
        workers.join()
    except Exception:
        pass
    mapped = [float(v) for v in mapped]
    print('mapped', mapped)
    if not np.allclose(mapped, serial, rtol=0, atol=1e-8):
        raise SystemExit('pool likelihoods differ from serial')
    print('10-core pool likelihoods match serial')
    sys.exit(0)
