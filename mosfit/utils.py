# -*- coding: UTF-8 -*-
"""Miscellaneous utility functions."""

import codecs
import hashlib
import json
import os
import re
import sys
import tempfile
from builtins import bytes, str
from collections import OrderedDict
from contextlib import contextmanager
from math import floor, isnan, log10

from dateutil.parser import parse

import numpy as np
import scipy.interpolate
import scipy.ndimage
from six import string_types

syst_syns = {'': 'Vega', 'SDSS': 'AB', 'Standard': 'Vega', 'Landolt': 'Vega'}


def get_url_file_handle(url, timeout=10):
    """Get file handle from urllib request of URL."""
    if sys.version_info[0] >= 3:
        from urllib.request import urlopen
    else:
        from urllib2 import urlopen
    return urlopen(url, timeout=timeout)


def is_date(s):
    """Check if input is a valid date."""
    try:
        parse(s)
        return True
    except ValueError:
        return False


def is_integer(s):
    """Check if input is an integer."""
    if isinstance(s, list) and not isinstance(s, string_types):
        try:
            [int(x) for x in s]
            return True
        except ValueError:
            return False
    else:
        try:
            int(s)
            return True
        except ValueError:
            return False


def is_number(s):
    """Check if input is a number."""
    if isinstance(s, list) and not isinstance(s, string_types):
        try:
            for x in s:
                if isinstance(x, string_types) and ' ' in x:
                    raise ValueError
            [float(x) for x in s]
            return True
        except ValueError:
            return False
    else:
        try:
            if isinstance(s, string_types) and ' ' in s:
                raise ValueError
            float(s)
            return True
        except ValueError:
            return False


def is_coordinate(s):
    """Check if input is a coordinate."""
    matches = re.search(
        '([+-]?([0-9]{1,2}):([0-9]{2})(:[0-9]{2})?\.?([0-9]+)?)', s)

    return False if matches is None else True


def is_datum(s):
    """Check if input is some sort of data."""
    if is_number(s):
        return True
    if is_coordinate(s):
        return True
    return False


def is_bibcode(s):
    """Check if input is a valid bibcode."""
    return not (re.search(
        '[0-9]{4}..........[\.0-9]{4}[A-Za-z]', s) is None)


def pretty_num(x, sig=4):
    """Convert number into string with specified significant digits."""
    if isnan(x) or not np.isfinite(x):
        return str(x)
    return str('%g' % (round_sig(x, sig)))


def round_sig(x, sig=4):
    """Round number with specified significant digits."""
    if x == 0.0:
        return 0.0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def listify(x):
    """Convert item to a `list` of items if it isn't already a `list`."""
    if not isinstance(x, list):
        return [x]
    return x


def entabbed_json_dumps(string, **kwargs):
    """Produce entabbed string for JSON output.

    This is necessary because Python 2 does not allow tabs to be used in its
    JSON dump(s) functions.
    """
    if sys.version_info[:2] >= (3, 3):
        return json.dumps(
            string,
            indent='\t',
            separators=kwargs['separators'],
            ensure_ascii=False)
        return
    newstr = unicode(json.dumps(  # noqa: F821
        string,
        indent=4,
        separators=kwargs['separators'],
        ensure_ascii=False,
        encoding='utf8'))
    newstr = re.sub(
        '\n +',
        lambda match: '\n' + '\t' * (len(match.group().strip('\n')) / 4),
        newstr)
    return newstr


def entabbed_json_dump(dic, f, **kwargs):
    """Write `entabbed_json_dumps` output to file handle."""
    string = entabbed_json_dumps(dic, **kwargs)
    try:
        f.write(string)
    except UnicodeEncodeError:
        f.write(string.encode('ascii', 'replace').decode())


def calculate_WAIC(scores):
    """WAIC from Gelmanp.

    http://www.stat.columbia.edu/~gelman/research/published/waic_understand3
    """
    fscores = [x for y in scores for x in y]
    # Technically needs to be multiplied by -2, but this makes score easily
    # relatable to maximum likelihood score.
    return np.mean(fscores) - np.var(fscores)


def flux_density_unit(unit):
    u"""Return coeffiecent to convert µJy to Jy."""
    if unit == 'µJy':
        return 1.0 / (1.0e-6 * 1.0e-23)
    return 1.0


def frequency_unit(unit):
    """Return coeffiecent to convert GHz to Hz."""
    if unit == 'GHz':
        return 1.0 / 1.0e9
    return 1.0


def hash_bytes(input_string):
    """Return a hash bytestring.

    Necessary to have consistent behavior between Python 2 & 3.
    """
    if sys.version_info[0] < 3:
        return bytes(input_string, 'utf-8')
    return input_string.encode()


def get_model_hash(modeldict, ignore_keys=[]):
    """Return a unique hash for the given model."""
    newdict = OrderedDict()
    for key in modeldict:
        if key not in ignore_keys:
            newdict[key] = modeldict[key]
    string_rep = json.dumps(newdict, sort_keys=True)

    return hashlib.sha512(hash_bytes(string_rep)).hexdigest()[:16]


def get_mosfit_hash(salt=u''):
    """Return a unique hash for the MOSFiT code."""
    import fnmatch
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))

    matches = []
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, '*.py'):
            matches.append(os.path.join(root, filename))

    matches = list(sorted(list(matches)))
    code_str = salt
    for match in matches:
        with codecs.open(match, 'r', 'utf-8') as f:
            code_str += f.read()

    return hashlib.sha512(hash_bytes(code_str)).hexdigest()[:16]


def is_master():
    """Determine if we are the master process."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True


def speak(text, lang='es'):
    """Text to speech. For funp."""
    try:
        from googletrans import Translator
        from gtts import gTTS
        from pygame import mixer
        from tempfile import TemporaryFile

        translator = Translator()
        tts = gTTS(text=translator.translate(text, dest=lang).text, lang=lang)
        mixer.init()

        sf = TemporaryFile()
        tts.write_to_fp(sf)
        sf.seek(0)
        mixer.music.load(sf)
        mixer.music.play()
    except Exception:
        raise


def rebin(a, newshape):
    """Rebin an array to a new shape."""
    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new)
              for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')
    return a[tuple(indices)]


def congrid(a, newdims, method='linear', center=False, minusone=False,
            bounds_error=False):
    """Arbitrary resampling of source array to new dimension sizes.

    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    center:
    True - interpolation points are at the centers of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    """
    if a.dtype not in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](center) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print("[congrid] dimensions error. "
              "This routine currently only support "
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append((old[i] - m1) / (newdims[i] - m1) *
                           (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]
        return newa

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1) *
                           (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method,
                                          bounds_error=bounds_error)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + list(range(ndims - 1))
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method,
                                              bounds_error=bounds_error)
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = list(range(np.rank(newcoords)))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n"
              "Currently only \'neighbour\', \'nearest\',\'linear\',"
              "and \'spline\' are supported.")
        return None


def all_to_list(array):
    """Recursively convert list of numpy arrays to lists."""
    try:
        return ([x.tolist() if type(x).__module__ == 'numpy'
                else all_to_list(x) if type(x) == 'list' else
                x for x in array])
    except TypeError:  # array is not iterable
        return [array]


def replace_multiple(y, xs, rep):
    """Match multiple strings to replace in sequence."""
    for x in xs[1:]:
        y = y.replace(x, rep)
    return y


# Borrowed from astrocats' supernova catalog module.
def name_clean(name):
    """Clean a transient's name."""
    newname = name.strip(' ;,*.')
    if newname.startswith('NAME '):
        newname = newname.replace('NAME ', '', 1)
    if newname.endswith(' SN'):
        newname = newname.replace(' SN', '')
    if newname.endswith(':SN'):
        newname = newname.replace(':SN', '')
    if newname.startswith('MASJ'):
        newname = newname.replace('MASJ', 'MASTER OT J', 1)
    if (newname.startswith('MASTER') and len(newname) > 7 and
            is_number(newname[7])):
        newname = newname.replace('MASTER', 'MASTER OT J', 1)
    if (newname.startswith('MASTER OT') and len(newname) > 10 and
            is_number(newname[10])):
        newname = newname.replace('MASTER OT', 'MASTER OT J', 1)
    if newname.startswith('MASTER OT J '):
        newname = newname.replace('MASTER OT J ', 'MASTER OT J', 1)
    if newname.startswith('OGLE '):
        newname = newname.replace('OGLE ', 'OGLE-', 1)
    if newname.startswith('OGLE-') and len(newname) != 16:
        namesp = newname.split('-')
        if (len(namesp) == 4 and len(namesp[1]) == 4 and
                is_number(namesp[1]) and is_number(namesp[3])):
            newname = 'OGLE-' + namesp[1] + '-SN-' + namesp[3].zfill(3)
        elif (len(namesp) == 2 and is_number(namesp[1][:2]) and
              not is_number(namesp[1][2:])):
            newname = 'OGLE' + namesp[1]
    if newname.startswith('SN SDSS'):
        newname = newname.replace('SN SDSS ', 'SDSS', 1)
    if newname.startswith('SDSS '):
        newname = newname.replace('SDSS ', 'SDSS', 1)
    if newname.startswith('SDSS'):
        namesp = newname.split('-')
        if (len(namesp) == 3 and is_number(namesp[0][4:]) and
                is_number(namesp[1]) and is_number(namesp[2])):
            newname = namesp[0] + '-' + namesp[1] + '-' + namesp[2].zfill(3)
    if newname.startswith('SDSS-II SN'):
        namesp = newname.split()
        if len(namesp) == 3 and is_number(namesp[2]):
            newname = 'SDSS-II SN ' + namesp[2].lstrip('0')
    if newname.startswith('SN CL'):
        newname = newname.replace('SN CL', 'CL', 1)
    if newname.startswith('SN HiTS'):
        newname = newname.replace('SN HiTS', 'SNHiTS', 1)
    if newname.startswith('SNHiTS '):
        newname = newname.replace('SNHiTS ', 'SNHiTS', 1)
    if newname.startswith('GAIA'):
        newname = newname.replace('GAIA', 'Gaia', 1)
    if newname.startswith('KSN-'):
        newname = newname.replace('KSN-', 'KSN', 1)
    if newname.startswith('KSN'):
        newname = 'KSN' + newname[3:].lower()
    if newname.startswith('Gaia '):
        newname = newname.replace('Gaia ', 'Gaia', 1)
    if newname.startswith('Gaia'):
        newname = 'Gaia' + newname[4:].lower()
    if newname.startswith('GRB'):
        newname = newname.replace('GRB', 'GRB ', 1)
    if newname.startswith('GRB ') and is_number(newname[4:].strip()):
        newname = 'GRB ' + newname[4:].strip() + 'A'
    if newname.startswith('ESSENCE '):
        newname = newname.replace('ESSENCE ', 'ESSENCE', 1)
    if newname.startswith('LSQ '):
        newname = newname.replace('LSQ ', 'LSQ', 1)
    if newname.startswith('LSQ') and is_number(newname[3]):
        newname = newname[:3] + newname[3:].lower()
    if newname.startswith('DES') and is_number(newname[3]):
        newname = newname[:7] + newname[7:].lower()
    if newname.startswith('SNSDF '):
        newname = newname.replace(' ', '')
    if newname.startswith('SNSDF'):
        namesp = newname.split('.')
        if len(namesp[0]) == 9:
            newname = namesp[0] + '-' + namesp[1].zfill(2)
    if newname.startswith('HFF '):
        newname = newname.replace(' ', '')
    if newname.startswith('SN HST'):
        newname = newname.replace('SN HST', 'HST', 1)
    if newname.startswith('HST ') and newname[4] != 'J':
        newname = newname.replace('HST ', 'HST J', 1)
    if newname.startswith('SNLS') and newname[4] != '-':
        newname = newname.replace('SNLS', 'SNLS-', 1)
    if newname.startswith('SNLS- '):
        newname = newname.replace('SNLS- ', 'SNLS-', 1)
    if newname.startswith('CRTS CSS'):
        newname = newname.replace('CRTS CSS', 'CSS', 1)
    if newname.startswith('CRTS MLS'):
        newname = newname.replace('CRTS MLS', 'MLS', 1)
    if newname.startswith('CRTS SSS'):
        newname = newname.replace('CRTS SSS', 'SSS', 1)
    if newname.startswith(('CSS', 'MLS', 'SSS')):
        newname = newname.replace(' ', ':').replace('J', '')
    if newname.startswith('SN HFF'):
        newname = newname.replace('SN HFF', 'HFF', 1)
    if newname.startswith('SN GND'):
        newname = newname.replace('SN GND', 'GND', 1)
    if newname.startswith('SN SCP'):
        newname = newname.replace('SN SCP', 'SCP', 1)
    if newname.startswith('SN UDS'):
        newname = newname.replace('SN UDS', 'UDS', 1)
    if newname.startswith('SCP') and newname[3] != '-':
        newname = newname.replace('SCP', 'SCP-', 1)
    if newname.startswith('SCP- '):
        newname = newname.replace('SCP- ', 'SCP-', 1)
    if newname.startswith('SCP-') and is_integer(newname[7:]):
        newname = 'SCP-' + newname[4:7] + str(int(newname[7:]))
    if newname.startswith('PS 1'):
        newname = newname.replace('PS 1', 'PS1', 1)
    if newname.startswith('PS1 SN PS'):
        newname = newname.replace('PS1 SN PS', 'PS', 1)
    if newname.startswith('PS1 SN'):
        newname = newname.replace('PS1 SN', 'PS1', 1)
    if newname.startswith('PS1') and is_number(newname[3]):
        newname = newname[:3] + newname[3:].lower()
    elif newname.startswith('PS1-') and is_number(newname[4]):
        newname = newname[:4] + newname[4:].lower()
    if newname.startswith('PSN K'):
        newname = newname.replace('PSN K', 'K', 1)
    if newname.startswith('K') and is_number(newname[1:5]):
        namesp = newname.split('-')
        if len(namesp[0]) == 5:
            newname = namesp[0] + '-' + namesp[1].zfill(3)
    if newname.startswith('Psn'):
        newname = newname.replace('Psn', 'PSN', 1)
    if newname.startswith('PSNJ'):
        newname = newname.replace('PSNJ', 'PSN J', 1)
    if newname.startswith('TCPJ'):
        newname = newname.replace('TCPJ', 'TCP J', 1)
    if newname.startswith('SMTJ'):
        newname = newname.replace('SMTJ', 'SMT J', 1)
    if newname.startswith('PSN20J'):
        newname = newname.replace('PSN20J', 'PSN J', 1)
    if newname.startswith('SN ASASSN'):
        newname = newname.replace('SN ASASSN', 'ASASSN', 1)
    if newname.startswith('ASASSN-20') and is_number(newname[9]):
        newname = newname.replace('ASASSN-20', 'ASASSN-', 1)
    if newname.startswith('ASASSN '):
        newname = newname.replace('ASASSN ', 'ASASSN-', 1).replace('--', '-')
    if newname.startswith('ASASSN') and newname[6] != '-':
        newname = newname.replace('ASASSN', 'ASASSN-', 1)
    if newname.startswith('ASASSN-') and is_number(newname[7]):
        newname = newname[:7] + newname[7:].lower()
    if newname.startswith('ROTSE3J'):
        newname = newname.replace('ROTSE3J', 'ROTSE3 J', 1)
    if newname.startswith('MACSJ'):
        newname = newname.replace('MACSJ', 'MACS J', 1)
    if newname.startswith('MWSNR'):
        newname = newname.replace('MWSNR', 'MWSNR ', 1)
    if newname.startswith('SN HUNT'):
        newname = newname.replace('SN HUNT', 'SNhunt', 1)
    if newname.startswith('SN Hunt'):
        newname = newname.replace(' ', '')
    if newname.startswith('SNHunt'):
        newname = newname.replace('SNHunt', 'SNhunt', 1)
    if newname.startswith('SNhunt '):
        newname = newname.replace('SNhunt ', 'SNhunt', 1)
    if newname.startswith('ptf'):
        newname = newname.replace('ptf', 'PTF', 1)
    if newname.startswith('SN PTF'):
        newname = newname.replace('SN PTF', 'PTF', 1)
    if newname.startswith('PTF '):
        newname = newname.replace('PTF ', 'PTF', 1)
    if newname.startswith('PTF') and is_number(newname[3]):
        newname = newname[:3] + newname[3:].lower()
    if newname.startswith('IPTF'):
        newname = newname.replace('IPTF', 'iPTF', 1)
    if newname.startswith('iPTF '):
        newname = newname.replace('iPTF ', 'iPTF', 1)
    if newname.startswith('iPTF') and is_number(newname[4]):
        newname = newname[:4] + newname[4:].lower()
    if newname.startswith('PESSTOESO'):
        newname = newname.replace('PESSTOESO', 'PESSTO ESO ', 1)
    if newname.startswith('snf'):
        newname = newname.replace('snf', 'SNF', 1)
    if newname.startswith('SNF '):
        newname = newname.replace('SNF ', 'SNF', 1)
    if (newname.startswith('SNF') and is_number(newname[3:]) and
            len(newname) >= 12):
        newname = 'SNF' + newname[3:11] + '-' + newname[11:]
    if newname.startswith(('MASTER OT J', 'ROTSE3 J')):
        prefix = newname.split('J')[0]
        coords = newname.split('J')[-1].strip()
        decsign = '+' if '+' in coords else '-'
        coordsplit = coords.replace('+', '-').split('-')
        if ('.' not in coordsplit[0] and len(coordsplit[0]) > 6 and
                '.' not in coordsplit[1] and len(coordsplit[1]) > 6):
            newname = (
                prefix + 'J' + coordsplit[0][:6] + '.' + coordsplit[0][6:] +
                decsign + coordsplit[1][:6] + '.' + coordsplit[1][6:])
    if (newname.startswith('Gaia ') and is_number(newname[3:4]) and
            len(newname) > 5):
        newname = newname.replace('Gaia ', 'Gaia', 1)
    if (newname.startswith('AT ') and is_number(newname[3:7]) and
            len(newname) > 7):
        newname = newname.replace('AT ', 'AT', 1)
    if len(newname) <= 4 and is_number(newname):
        newname = 'SN' + newname + 'A'
    if (len(newname) > 4 and is_number(newname[:4]) and
            not is_number(newname[4:])):
        newname = 'SN' + newname
    if (newname.startswith('Sn ') and is_number(newname[3:7]) and
            len(newname) > 7):
        newname = newname.replace('Sn ', 'SN', 1)
    if (newname.startswith('sn') and is_number(newname[2:6]) and
            len(newname) > 6):
        newname = newname.replace('sn', 'SN', 1)
    if (newname.startswith('SN ') and is_number(newname[3:7]) and
            len(newname) > 7):
        newname = newname.replace('SN ', 'SN', 1)
    if (newname.startswith('SN') and is_number(newname[2:6]) and
            len(newname) == 7 and newname[6].islower()):
        newname = 'SN' + newname[2:6] + newname[6].upper()
    elif (newname.startswith('SN') and is_number(newname[2:6]) and
          (len(newname) == 8 or len(newname) == 9) and newname[6:].isupper()):
        newname = 'SN' + newname[2:6] + newname[6:].lower()
    if (newname.startswith('AT') and is_number(newname[2:6]) and
            len(newname) == 7 and newname[6].islower()):
        newname = 'AT' + newname[2:6] + newname[6].upper()
    elif (newname.startswith('AT') and is_number(newname[2:6]) and
          (len(newname) == 8 or len(newname) == 9) and newname[6:].isupper()):
        newname = 'AT' + newname[2:6] + newname[6:].lower()

    newname = (' '.join(newname.split())).strip()
    return newname


# From Django
def slugify(value, allow_unicode=False):
    """Slugify string to make it a valid filename.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Also strip leading and trailing whitespace.
    """
    import unicodedata
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip()
    return re.sub(r'[-\s]+', '-', value)


# Below from
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
@contextmanager
def temp_atomic(suffix='', dir=None):
    """Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving, even in case of an exception.

    Parameters
    ----------
    suffix : string
        optional file suffix
    dir : string
        optional directory to save temporary file in

    """
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir)
    tf.file.close()
    try:
        yield tf.name
    finally:
        try:
            os.remove(tf.name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """Open temporary file object that atomically moves upon exiting.

    Allows reading and writing to and from the same filename.

    The file will not be moved to destination in case of an exception.

    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    *args : mixed
        Any valid arguments for :code:`open`
    **kwargs : mixed
        Any valid keyword arguments for :code:`open`

    """
    fsync = kwargs.get('fsync', False)

    with temp_atomic(
            dir=os.path.dirname(os.path.abspath(filepath))) as tmppath:
        with open(tmppath, *args, **kwargs) as file:
            try:
                yield file
            finally:
                if fsync:
                    file.flush()
                    os.fsync(file.fileno())
        if os.path.isfile(filepath):
            os.remove(filepath)
        os.rename(tmppath, filepath)
