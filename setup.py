from distutils.core import setup

extensions = []

setup(
    name='mosfit',
    packages=['mosfit'],
    version='0.1',
    description=('Package that performs maximum likelihood analysis to fit '
                 'semi-analytical model predictions to observed '
                 'astronomical transient data.'),
    author='James Guillochon',
    author_email='guillochon@gmail.com',
    url='https://github.com/guillochon/mosfit',
    download_url='https://github.com/guillochon/mypackage/tarball/0.1',
    keywords=['astronomy', 'fitting', 'monte carlo', 'modeling'],
    classifiers=[]
)
