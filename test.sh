#! /bin/bash
set -ev
if [ "$1" = -c ]; then
    RUNNER="coverage run -p --source=mosfit"
    TRUNNER="coverage run -p"
    echo "travis_fold:start:FIT Fitting test data"
else
    RUNNER=python
    TRUNNER=python
fi

mpirun -np 2 $RUNNER -m mosfit -e SN2009do --test -i 1 -f 1 -p 0 -F covariance
$RUNNER -m mosfit -e SN2009do.json --test -i 1 --no-fracking -m magnetar -T 2 -F covariance
$RUNNER -m mosfit -e mosfit/tests/LSQ12dlf.json --test -i 3 --no-fracking -m csm -F n 6.0 -W 120 -M 0.2 --offline
$RUNNER -m mosfit -e SN2008ar --test -i 1 --no-fracking -m ia -F covariance
$RUNNER -m mosfit -e mosfit/tests/LSQ12dlf.json --test -i 2 --no-fracking -m rprocess --variance-for-each band --offline -w products/walkers.json
if [ "$1" = -c ]; then
    $RUNNER -m mosfit -e SN2007bg --test -i 1 --no-fracking -m ic --language ru
else
    $RUNNER -m mosfit -e SN2007bg --test -i 1 --no-fracking -m ic
fi
echo -ne 'y\n\n4\n2\n4\n4\nu\n\n\n\n\n\n13\n\n\n1234567890123456789\ny\n' | $RUNNER -m mosfit -m slsn -e mosfit/tests/PTF10hgi.txt -i 1 --no-write -u --test
$RUNNER -m mosfit -e 09do --test -i 1 --no-fracking -m slsn -S 20 -E 10.0 100.0 -g -c --no-copy-at-launch -x radiusphot
$RUNNER -m mosfit -e mosfit/tests/SN2006le.json --test -i 5 --no-fracking -m csmni --extra-bands u g --extra-instruments LSST -L 55540 55560 --exclude-bands B -s test --quiet -u --offline

if [ "$1" = -c ]; then
    echo "travis_fold:end:FIT Fitting test data done"
    echo "travis_fold:start:GEN Generating random models"
fi

$RUNNER -m mosfit --test -i 0
$RUNNER -m mosfit -i 0 -m default -P parameters_test.json -l 23 0.5
$TRUNNER test.py

if [ "$1" = -c ]; then
    echo "travis_fold:end:GEN Generating random models done"
    echo "travis_fold:start:JUP Testing Jupyter notebooks"
    echo "travis_fold:end:JUP Testing Jupyter notebooks"
    coverage combine
fi
