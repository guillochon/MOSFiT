"""The rest-frame SED path must not import the fitting or plotting stacks.

This guards the `lynx` dependency group in `pyproject.toml`: a lightweight
install has no `emcee`, `dynesty` or `numba`, so anything that quietly imports
one of them at module scope would break that install. Run as its own process
so `sys.modules` reflects only what this path actually touched.

`matplotlib`, `seaborn` and `pandas` are deliberately not checked: `astrocats`
imports them itself, so they arrive no matter what MOSFiT does.
"""
from __future__ import print_function

import os
import sys

FORBIDDEN = ('emcee', 'dynesty', 'numba', 'llvmlite')

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

    import numpy as np

    from mosfit.lynx import LynxSource

    leaked = [m for m in FORBIDDEN if m in sys.modules]
    assert not leaked, 'importing mosfit.lynx pulled in {}'.format(leaked)

    # Building and evaluating a model must stay clean too: the module loader
    # walks every module package, so an eager import anywhere shows up here.
    source = LynxSource(model='slsn', phases=np.linspace(0.0, 60.0, 8))
    sed = source.compute_sed()
    assert sed.shape[0] == 8

    leaked = [m for m in FORBIDDEN if m in sys.modules]
    assert not leaked, 'evaluating a model pulled in {}'.format(leaked)

    # h5py is only needed to write the command-line product, not by the
    # in-process API.
    assert 'h5py' not in sys.modules

    print('Lynx import-isolation tests passed.')
