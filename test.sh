if [ "$1" = -c ] ; then
 RUNNER="coverage run -p --source=mosfit"
 TRUNNER="coverage run -p"
else
 RUNNER=python
 TRUNNER=python
 if [ "$1" = -c ] echo "travis_fold:start:FIT Fitting test data"
 mpirun -np 2 $RUNNER -m mosfit -e SN2009do --travis -i 1 -f 1 -p 0 -F covariance
 mpirun -np 2 $RUNNER -m mosfit -e SN2009do.json --travis -i 1 --no-fracking -m magnetar -T 2 -F covariance
 mpirun -np 2 $RUNNER -m mosfit -e LSQ12dlf --travis -i 100 --no-fracking -m csm -F n 6.0 -W 120 -M 0.2
 mpirun -np 2 $RUNNER -m mosfit -e SN2008ar --travis -i 1 --no-fracking -m ia -F covariance
 mpirun -np 2 $RUNNER -m mosfit -e LSQ12dlf --travis -i 2 --no-fracking -m rprocess --variance-for-each band --offline
 $RUNNER -m mosfit -e SN2007bg --travis -i 1 --no-fracking -m ic
 $RUNNER -m mosfit -e 12dlf --travis -i 1 --no-fracking -m slsn -S 20 -E 10.0 100.0 -g -c --no-copy-at-launch
 $RUNNER -m mosfit -e 2010kd --travis -i 5 --no-fracking -m csmni --extra-bands u g --extra-instruments LSST -L 55540 55560 --exclude-bands B -s test --quiet -u
 if [ "$1" = -c ] echo "travis_fold:end:FIT Fitting test data done"
 if [ "$1" = -c ] echo "travis_fold:start:GEN Generating random models"
 $RUNNER -m mosfit --travis -i 0
 $RUNNER -m mosfit -i 0 -m default -P parameters_test.json
 $TRUNNER test.py
 if [ "$1" = -c ] echo "travis_fold:end:GEN Generating random models done"
 if [ "$1" = -c ] echo "travis_fold:start:JUP Testing Jupyter notebooks"
 if [ "$1" = -c ] echo "travis_fold:end:JUP Testing Jupyter notebooks"
 coverage combine
