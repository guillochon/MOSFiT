from setuptools import setup, find_packages

extensions = []

setup(
    name='mosfit',
    packages=find_packages(),
    include_package_data=True,
    version='0.1.1',
    description=('Package that performs maximum likelihood analysis to fit '
                 'semi-analytical model predictions to observed '
                 'astronomical transient data.'),
    author='James Guillochon',
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
