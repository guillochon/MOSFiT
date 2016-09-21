import os

from future import __author__, __license__, __version__
from setuptools import find_packages, setup

dir_path = os.path.dirname(os.path.realpath(__file__))
exec(open(os.path.join(dir_path, 'mosfit', '__init__.py')).read())

setup(
    name='mosfit',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description=('Package that performs maximum likelihood analysis to fit '
                 'semi-analytical model predictions to observed '
                 'astronomical transient data.'),
    license=__license__,
    author=__author__,
    author_email='guillochon@gmail.com',
    url='https://github.com/guillochon/mosfit',
    download_url='https://github.com/guillochon/mosfit/tarball/0.1.1',
    keywords=['astronomy', 'fitting', 'monte carlo', 'modeling'],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3 :: Only',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Scientific/Engineering :: Physics'])
