"""Run every MOSFiT package test script under coverage.

Discovers ``mosfit/tests/_test_*.py`` and the repo-root generative smoke
``test.py``. This is what GitHub Actions CI invokes.
"""
from __future__ import print_function

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent


def _coverage_available():
    try:
        import coverage  # noqa: F401
        return True
    except ImportError:
        return False


def _clear_coverage_data():
    cov = ROOT / '.coverage'
    if cov.is_file():
        cov.unlink()
    for path in ROOT.glob('.coverage.*'):
        if path.is_file():
            path.unlink()


def main():
    scripts = sorted(TESTS.glob('_test_*.py'))
    root_smoke = ROOT / 'test.py'
    jobs = list(scripts)
    if root_smoke.is_file():
        jobs.append(root_smoke)
    if not jobs:
        print('no test scripts found', file=sys.stderr)
        return 1

    use_cov = _coverage_available()
    if use_cov:
        _clear_coverage_data()
        print('coverage: enabled', flush=True)
    else:
        print('coverage: not installed, running tests only', flush=True)

    failed = []
    env = os.environ.copy()
    env.setdefault('MOSFIT_PHOTOMETRY_DEBUG', '')
    for script in jobs:
        rel = script.relative_to(ROOT)
        print('========', rel, '========', flush=True)
        if use_cov:
            cmd = [
                sys.executable, '-m', 'coverage', 'run', '--parallel-mode',
                str(script),
            ]
        else:
            cmd = [sys.executable, str(script)]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if proc.returncode != 0:
            failed.append(str(rel))
            print('FAILED', rel, 'exit', proc.returncode, flush=True)

    cov_failed = False
    if use_cov:
        combine = subprocess.run(
            [sys.executable, '-m', 'coverage', 'combine'],
            cwd=str(ROOT),
        )
        if combine.returncode != 0:
            print('coverage combine failed', file=sys.stderr)
            cov_failed = True
        else:
            print('======== coverage report ========', flush=True)
            report = subprocess.run(
                [sys.executable, '-m', 'coverage', 'report'],
                cwd=str(ROOT),
            )
            if report.returncode != 0:
                cov_failed = True

    if failed:
        print('failed:', ', '.join(failed), file=sys.stderr)
        return 1
    if cov_failed:
        print('coverage threshold not met', file=sys.stderr)
        return 1
    print('all package tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
