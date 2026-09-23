"""A photometric fit leaves the CWD alone and caches filters per user."""
from __future__ import print_function

import os
import shutil
import sys
import tempfile

import numpy as np

if __name__ == '__main__':
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    fixture = os.path.join(repo, 'mosfit', 'tests', 'LSQ12dlf.json')

    work = tempfile.mkdtemp(prefix='mosfit_cwd_')
    cache = tempfile.mkdtemp(prefix='mosfit_cache_')
    os.environ['MOSFIT_CACHE_DIR'] = cache
    os.chdir(work)

    from schwimmbad import SerialPool

    from mosfit.fitter import Fitter, ln_likelihood
    from mosfit.model import Model
    from mosfit.utils import user_cache_dir

    assert user_cache_dir('filters') == os.path.join(cache, 'filters')

    fitter = Fitter(
        test=True, quiet=True, exit_on_prompt=True, prefer_cache=True)
    event = fitter._fetcher.fetch([fixture])[0]
    fitter._event_name = event.get('name', 'LSQ12dlf')
    fitter._event_path = event.get('path', '')
    fitter._event_data = fitter._fetcher.load_data(event)
    pool = SerialPool()
    model = Model(
        model='slsn', data=fitter._event_data, test=True,
        printer=fitter._printer, fitter=fitter, pool=pool)
    assert model.load_data(
        fitter._event_data, event_name=fitter._event_name, pool=pool)
    import mosfit.fitter as ft
    ft.model = model
    ll = float(ln_likelihood(np.full(model._num_free_parameters, 0.5)))
    assert np.isfinite(ll), ll

    left = sorted(os.listdir(work))
    assert 'modules' not in left, 'photometry wrote into the CWD: {}'.format(
        left)

    # A filter whose ``.dat`` is missing everywhere is regenerated from its
    # ``.xml`` into the user cache, not next to the ``.xml`` or into the CWD.
    phot = next(m for m in model._modules.values()
                if type(m).__name__ == 'Photometry')
    pkg_filters = phot._filter_search_paths[0]
    xml_only = tempfile.mkdtemp(prefix='mosfit_xml_')
    for fname in os.listdir(pkg_filters):
        if fname.endswith('.xml'):
            shutil.copy(os.path.join(pkg_filters, fname), xml_only)
    phot._filter_search_paths = [xml_only, phot._filter_cache_path]
    svo = [i for i, b in enumerate(phot._unique_bands) if 'SVO' in b]
    assert svo, 'no SVO bands to exercise'
    phot.load_bands(svo[:1])
    cached = os.listdir(phot._filter_cache_path)
    assert any(f.endswith('.dat') for f in cached), cached
    assert not any(f.endswith('.dat') for f in os.listdir(xml_only))
    assert 'modules' not in os.listdir(work)

    os.chdir(repo)
    for d in (work, cache, xml_only):
        shutil.rmtree(d, ignore_errors=True)
    print('no CWD writes ok')
    sys.exit(0)
