"""MOSFiT: Modular light curve fitting software."""
import os
import re

import astrocats

from . import constants  # noqa: F401
from . import fitter  # noqa: F401
from . import model  # noqa: F401
from . import plotting  # noqa: F401
from . import printer  # noqa: F401
from . import utils  # noqa: F401

authors = []
contributors = []

dir_name = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_name, 'contributors.txt')) as f:
    for cont in f.read().splitlines():
        if '*' in cont:
            authors.append(cont.split('(')[0].strip(' *'))
        else:
            contributors.append(cont.split('(')[0].strip())

__version__ = '2.0.0'
__author__ = ' & '.join([', '.join(authors[:-1]), authors[-1]])
__contributors__ = ' & '.join([', '.join(contributors[:-1]), contributors[-1]])
__license__ = 'MIT'

# Check astrocats version for schema compatibility.
# Keep this floor in sync with the astrocats pin in pyproject.toml.
_ASTROCATS_MIN_VERSION = (0, 5, 0)


def _version_triple(version):
    """Leading X.Y.Z integers from a PEP 440 version string."""
    nums = [int(p) for p in re.findall(r'\d+', version.split('+')[0])[:3]]
    nums.extend([0] * (3 - len(nums)))
    return tuple(nums[:3])


vneed = [str(part) for part in _ASTROCATS_MIN_VERSION]
if _version_triple(astrocats.__version__) < _ASTROCATS_MIN_VERSION:
    raise ImportError(
        'Installed `astrocats` package is out of date for this version of '
        'MOSFiT, please upgrade your `astrocats` install to a version >= `' +
        '.'.join(vneed) + '` with `uv`, `pip`, or `conda`.')
