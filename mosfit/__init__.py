"""MOSFiT: Modular light curve fitting software."""
import astrocats
import os

from . import constants  # noqa: F401
from . import fitter  # noqa: F401
from . import model  # noqa: F401
from . import plotting  # noqa: F401
from . import printer  # noqa: F401
from . import utils  # noqa: F401

__version__ = '0.6.2'
__author__ = 'James Guillochon & Matt Nicholl'
__license__ = 'MIT'

# Check astrocats version for schema compatibility.
right_astrocats = True
vparts = astrocats.__version__.split('.')
req_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')
with open(req_path, 'r') as f:
    for req in f.read().splitlines():
        if 'astrocats' in req:
            vneed = req.split('=')[-1].split('.')
            if int(vparts[0]) < int(vneed[0]):
                right_astrocats = False
            elif int(vparts[1]) < int(vneed[1]):
                right_astrocats = False
            elif int(vparts[2]) < int(vneed[2]):
                right_astrocats = False
if not right_astrocats:
    raise ImportError(
        'Installed `astrocats` package is out of date for this version of '
        'MOSFiT, please upgrade your `astrocats` install to a version >= `' +
        '.'.join(vneed) + '` with either `pip` or `conda`.')
