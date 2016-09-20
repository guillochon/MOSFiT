import os

from setuptools import setup, find_packages

extensions = []

setup(
    name='mosfit',
    packages=find_packages(),
    include_package_data=True,
    version='0.1.0',
    description=('Package that performs maximum likelihood analysis to fit '
                 'semi-analytical model predictions to observed '
                 'astronomical transient data.'),
    author='James Guillochon',
    author_email='guillochon@gmail.com',
    url='https://github.com/guillochon/mosfit',
    download_url='https://github.com/guillochon/mypackage/tarball/0.1',
    keywords=['astronomy', 'fitting', 'monte carlo', 'modeling'],
    classifiers=[])
