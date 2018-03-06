# -*- encoding: utf-8 -*-
"""The main function."""

import argparse
import codecs
import locale
import os
import shutil
import sys
import time
from operator import attrgetter
from unicodedata import normalize

import numpy as np

from mosfit import __author__, __contributors__, __version__
from mosfit.fitter import Fitter
from mosfit.printer import Printer
from mosfit.utils import get_mosfit_hash, is_master, open_atomic, speak


class SortingHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Sort argparse arguments by argument name."""

    def add_arguments(self, actions):
        """Add sorting action based on `option_strings`."""
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def get_parser(only=None, printer=None):
    """Retrieve MOSFiT's `argparse.ArgumentParser` object."""
    prt = Printer() if printer is None else printer

    parser = argparse.ArgumentParser(
        prog='mosfit',
        description='Fit astrophysical transients.',
        formatter_class=SortingHelpFormatter,
        add_help=only is None)

    parser.add_argument(
        '--language',
        dest='language',
        type=str,
        const='select',
        default='en',
        nargs='?',
        help=("Language for output text."))

    if only == 'language':
        return parser

    parser.add_argument(
        '--events',
        '-e',
        dest='events',
        default=[],
        nargs='+',
        help=prt.text('parser_events'))

    parser.add_argument(
        '--models',
        '-m',
        dest='models',
        default=[],
        nargs='?',
        help=prt.text('parser_models'))

    parser.add_argument(
        '--parameter-paths',
        '-P',
        dest='parameter_paths',
        default=['parameters.json'],
        nargs='+',
        help=prt.text('parser_parameter_paths'))

    parser.add_argument(
        '--walker-paths',
        '-w',
        dest='walker_paths',
        default=[],
        nargs='+',
        help=prt.text('parser_walker_paths'))

    parser.add_argument(
        '--max-time',
        dest='max_time',
        type=float,
        default=1000.,
        help=prt.text('parser_max_time'))

    parser.add_argument(
        '--limiting-magnitude',
        '-l',
        dest='limiting_magnitude',
        default=None,
        nargs='+',
        help=prt.text('parser_limiting_magnitude'))

    parser.add_argument(
        '--band-list',
        '--extra-bands',
        dest='band_list',
        default=[],
        nargs='+',
        help=prt.text('parser_band_list'))

    parser.add_argument(
        '--band-systems',
        '--extra-systems',
        dest='band_systems',
        default=[],
        nargs='+',
        help=prt.text('parser_band_systems'))

    parser.add_argument(
        '--band-instruments',
        '--extra-instruments',
        dest='band_instruments',
        default=[],
        nargs='+',
        help=prt.text('parser_band_instruments'))

    parser.add_argument(
        '--band-bandsets',
        '--extra-bandsets',
        dest='band_bandsets',
        default=[],
        nargs='+',
        help=prt.text('parser_band_bandsets'))

    parser.add_argument(
        '--band-sampling-points',
        dest='band_sampling_points',
        type=int,
        default=17,
        help=prt.text('parser_band_sampling_points'))

    parser.add_argument(
        '--exclude-bands',
        dest='exclude_bands',
        default=[],
        nargs='+',
        help=prt.text('parser_exclude_bands'))

    parser.add_argument(
        '--exclude-instruments',
        dest='exclude_instruments',
        default=[],
        nargs='+',
        help=prt.text('parser_exclude_instruments'))

    parser.add_argument(
        '--exclude-systems',
        dest='exclude_systems',
        default=[],
        nargs='+',
        help=prt.text('parser_exclude_systems'))

    parser.add_argument(
        '--exclude-sources',
        dest='exclude_sources',
        default=[],
        nargs='+',
        help=prt.text('parser_exclude_sources'))

    parser.add_argument(
        '--exclude-kinds',
        dest='exclude_kinds',
        default=[],
        nargs='+',
        help=prt.text('parser_exclude_kinds'))

    parser.add_argument(
        '--fix-parameters',
        '-F',
        dest='user_fixed_parameters',
        default=[],
        nargs='+',
        help=prt.text('parser_user_fixed_parameters'))

    parser.add_argument(
        '--iterations',
        '-i',
        dest='iterations',
        type=int,
        const=0,
        default=-1,
        nargs='?',
        help=prt.text('parser_iterations'))

    parser.add_argument(
        '--smooth-times',
        '--plot-points',
        '-S',
        dest='smooth_times',
        type=int,
        const=0,
        default=20,
        nargs='?',
        action='store',
        help=prt.text('parser_smooth_times'))

    parser.add_argument(
        '--extrapolate-time',
        '-E',
        dest='extrapolate_time',
        type=float,
        default=0.0,
        nargs='*',
        help=prt.text('parser_extrapolate_time'))

    parser.add_argument(
        '--limit-fitting-mjds',
        '-L',
        dest='limit_fitting_mjds',
        type=float,
        default=False,
        nargs=2,
        help=prt.text('parser_limit_fitting_mjds'))

    parser.add_argument(
        '--suffix',
        '-s',
        dest='suffix',
        default='',
        help=prt.text('parser_suffix'))

    parser.add_argument(
        '--num-walkers',
        '-N',
        dest='num_walkers',
        type=int,
        default=None,
        help=prt.text('parser_num_walkers'))

    parser.add_argument(
        '--num-temps',
        '-T',
        dest='num_temps',
        type=int,
        default=1,
        help=prt.text('parser_num_temps'))

    parser.add_argument(
        '--no-fracking',
        dest='fracking',
        default=True,
        action='store_false',
        help=prt.text('parser_fracking'))

    parser.add_argument(
        '--no-write',
        dest='write',
        default=True,
        action='store_false',
        help=prt.text('parser_write'))

    parser.add_argument(
        '--quiet',
        dest='quiet',
        default=False,
        action='store_true',
        help=prt.text('parser_quiet'))

    parser.add_argument(
        '--cuda',
        dest='cuda',
        default=False,
        action='store_true',
        help=prt.text('parser_cuda'))

    parser.add_argument(
        '--no-copy-at-launch',
        dest='copy',
        default=True,
        action='store_false',
        help=prt.text('parser_copy'))

    parser.add_argument(
        '--force-copy-at-launch',
        dest='force_copy',
        default=False,
        action='store_true',
        help=prt.text('parser_force_copy'))

    parser.add_argument(
        '--offline',
        dest='offline',
        default=False,
        action='store_true',
        help=prt.text('parser_offline'))

    parser.add_argument(
        '--prefer-cache',
        dest='prefer_cache',
        default=False,
        action='store_true',
        help=prt.text('parser_prefer_cache'))

    parser.add_argument(
        '--frack-step',
        '-f',
        dest='frack_step',
        type=int,
        default=50,
        help=prt.text('parser_frack_step'))

    parser.add_argument(
        '--burn',
        '-b',
        dest='burn',
        type=int,
        help=prt.text('parser_burn'))

    parser.add_argument(
        '--post-burn',
        '-p',
        dest='post_burn',
        type=int,
        help=prt.text('parser_post_burn'))

    parser.add_argument(
        '--upload',
        '-u',
        dest='upload',
        default=False,
        action='store_true',
        help=prt.text('parser_upload'))

    parser.add_argument(
        '--run-until-converged',
        '-R',
        dest='run_until_converged',
        type=float,
        default=None,
        const=1.1,
        nargs='?',
        help=prt.text('parser_run_until_converged'))

    parser.add_argument(
        '--run-until-uncorrelated',
        '-U',
        dest='run_until_uncorrelated',
        type=int,
        default=None,
        const=5,
        nargs='?',
        help=prt.text('parser_run_until_uncorrelated'))

    parser.add_argument(
        '--maximum-walltime',
        '-W',
        dest='maximum_walltime',
        type=float,
        default=False,
        help=prt.text('parser_maximum_walltime'))

    parser.add_argument(
        '--maximum-memory',
        '-M',
        dest='maximum_memory',
        type=float,
        default=np.inf,
        help=prt.text('parser_maximum_memory'))

    parser.add_argument(
        '--draw-above-likelihood',
        '-d',
        dest='draw_above_likelihood',
        type=float,
        default=False,
        const=True,
        nargs='?',
        help=prt.text('parser_draw_above_likelihood'))

    parser.add_argument(
        '--gibbs',
        '-g',
        dest='gibbs',
        default=False,
        action='store_true',
        help=prt.text('parser_gibbs'))

    parser.add_argument(
        '--save-full-chain',
        '-c',
        dest='save_full_chain',
        default=False,
        action='store_true',
        help=prt.text('parser_save_full_chain'))

    parser.add_argument(
        '--print-trees',
        dest='print_trees',
        default=False,
        action='store_true',
        help=prt.text('parser_print_trees'))

    parser.add_argument(
        '--set-upload-token',
        dest='set_upload_token',
        const=True,
        default=False,
        nargs='?',
        help=prt.text('parser_set_upload_token'))

    parser.add_argument(
        '--ignore-upload-quality',
        dest='check_upload_quality',
        default=True,
        action='store_false',
        help=prt.text('parser_check_upload_quality'))

    parser.add_argument(
        '--test',
        dest='test',
        default=False,
        action='store_true',
        help=prt.text('parser_test'))

    parser.add_argument(
        '--variance-for-each',
        dest='variance_for_each',
        default=[],
        nargs='+',
        help=prt.text('parser_variance_for_each'))

    parser.add_argument(
        '--speak',
        dest='speak',
        const='en',
        default=False,
        nargs='?',
        help=prt.text('parser_speak'))

    parser.add_argument(
        '--version',
        dest='version',
        default=False,
        action='store_true',
        help=prt.text('parser_version'))

    parser.add_argument(
        '--extra-outputs',
        '-x',
        dest='extra_outputs',
        default=[],
        nargs='+',
        help=prt.text('parser_extra_outputs'))

    parser.add_argument(
        '--catalogs',
        '-C',
        dest='catalogs',
        default=[],
        nargs='+',
        help=prt.text('parser_catalogs'))

    parser.add_argument(
        '--open-in-browser',
        '-O',
        dest='open_in_browser',
        default=False,
        action='store_true',
        help=prt.text('parser_open_in_browser'))

    parser.add_argument(
        '--exit-on-prompt',
        dest='exit_on_prompt',
        default=False,
        action='store_true',
        help=prt.text('parser_exit_on_prompt'))

    parser.add_argument(
        '--download-recommended-data',
        dest='download_recommended_data',
        default=False,
        action='store_true',
        help=prt.text('parser_download_recommended_data'))

    parser.add_argument(
        '--local-data-only',
        dest='local_data_only',
        default=False,
        action='store_true',
        help=prt.text('parser_local_data_only'))

    return parser


def main():
    """Run MOSFiT."""
    prt = Printer(
        wrap_length=100, quiet=False, language='en', exit_on_prompt=False)

    parser = get_parser(only='language')
    args, remaining = parser.parse_known_args()

    if args.language == 'en':
        loc = locale.getlocale()
        if loc[0]:
            args.language = loc[0].split('_')[0]

    if args.language != 'en':
        try:
            from googletrans.constants import LANGUAGES
        except Exception:
            raise RuntimeError(
                '`--language` requires `googletrans` package, '
                'install with `pip install googletrans`.')

        if args.language == 'select' or args.language not in LANGUAGES:
            languages = list(
                sorted([LANGUAGES[x].title().replace('_', ' ') +
                        ' (' + x + ')' for x in LANGUAGES]))
            sel = prt.prompt(
                'Select a language:', kind='select', options=languages,
                message=False)
            args.language = sel.split('(')[-1].strip(')')

    prt = Printer(language=args.language)

    language = args.language

    parser = get_parser(printer=prt)
    args = parser.parse_args()

    args.language = language

    prt = Printer(
        wrap_length=100, quiet=args.quiet, language=args.language,
        exit_on_prompt=args.exit_on_prompt)

    if args.version:
        print('MOSFiT v{}'.format(__version__))
        return

    dir_path = os.path.dirname(os.path.realpath(__file__))

    if args.speak:
        speak('Mosfit', args.speak)

    args.start_time = time.time()

    if args.limiting_magnitude == []:
        args.limiting_magnitude = 20.0

    args.return_fits = False

    if (isinstance(args.extrapolate_time, list) and
            len(args.extrapolate_time) == 0):
        args.extrapolate_time = 100.0

    if len(args.band_list) and args.smooth_times == -1:
        prt.message('enabling_s')
        args.smooth_times = 0

    changed_iterations = False
    if args.iterations == -1:
        if len(args.events) == 0:
            changed_iterations = True
            args.iterations = 0
        else:
            args.iterations = 5000

    if args.burn is None and args.post_burn is None:
        args.burn = int(np.floor(args.iterations / 2))

    if args.frack_step == 0:
        args.fracking = False

    if (args.run_until_uncorrelated is not None and
            args.run_until_converged is not None):
        raise ValueError(
            '`-R` and `-U` options are incompatible, please use one or the '
            'other.')
    elif args.run_until_uncorrelated is not None:
        args.convergence_type = 'acor'
        args.convergence_criteria = args.run_until_uncorrelated
    elif args.run_until_converged is not None:
        args.convergence_type = 'psrf'
        args.convergence_criteria = args.run_until_converged

    if is_master():
        # Get hash of ourselves
        mosfit_hash = get_mosfit_hash()

        # Print our amazing ASCII logo.
        if not args.quiet:
            with codecs.open(os.path.join(dir_path, 'logo.txt'),
                             'r', 'utf-8') as f:
                logo = f.read()
                firstline = logo.split('\n')[0]
                # if isinstance(firstline, bytes):
                #     firstline = firstline.decode('utf-8')
                width = len(normalize('NFC', firstline))
            prt.prt(logo, colorify=True)
            prt.message(
                'byline', reps=[
                    __version__, mosfit_hash, __author__, __contributors__],
                center=True, colorify=True, width=width, wrapped=False)

        # Get/set upload token
        upload_token = ''
        get_token_from_user = False
        if args.set_upload_token:
            if args.set_upload_token is not True:
                upload_token = args.set_upload_token
            get_token_from_user = True

        upload_token_path = os.path.join(dir_path, 'cache', 'dropbox.token')

        # Perform a few checks on upload before running (to keep size
        # manageable)
        if args.upload and not args.test and args.smooth_times > 100:
            response = prt.prompt('ul_warning_smooth')
            if response:
                args.upload = False
            else:
                sys.exit()

        if (args.upload and not args.test and
                args.num_walkers is not None and args.num_walkers < 100):
            response = prt.prompt('ul_warning_few_walkers')
            if response:
                args.upload = False
            else:
                sys.exit()

        if (args.upload and not args.test and args.num_walkers and
                args.num_walkers * args.num_temps > 500):
            response = prt.prompt('ul_warning_too_many_walkers')
            if response:
                args.upload = False
            else:
                sys.exit()

        if args.upload:
            if not os.path.isfile(upload_token_path):
                get_token_from_user = True
            else:
                with open(upload_token_path, 'r') as f:
                    upload_token = f.read().splitlines()
                    if len(upload_token) != 1:
                        get_token_from_user = True
                    elif len(upload_token[0]) != 64:
                        get_token_from_user = True
                    else:
                        upload_token = upload_token[0]

        if get_token_from_user:
            if args.test:
                upload_token = ('1234567890abcdefghijklmnopqrstuvwxyz'
                                '1234567890abcdefghijklmnopqr')
            while len(upload_token) != 64:
                prt.message('no_ul_token', ['https://sne.space/mosfit/'],
                            wrapped=True)
                upload_token = prt.prompt('paste_token', kind='string')
                if len(upload_token) != 64:
                    prt.prt(
                        'Error: Token must be exactly 64 characters in '
                        'length.', wrapped=True)
                    continue
                break
            with open_atomic(upload_token_path, 'w') as f:
                f.write(upload_token)

        if args.upload:
            prt.prt(
                "Upload flag set, will upload results after completion.",
                wrapped=True)
            prt.prt("Dropbox token: " + upload_token, wrapped=True)

        args.upload_token = upload_token

        if changed_iterations:
            prt.message('iterations_0', wrapped=True)

        # Create the user directory structure, if it doesn't already exist.
        if args.copy:
            prt.message('copying')
            fc = False
            if args.force_copy:
                fc = prt.prompt('force_copy')
            if not os.path.exists('jupyter'):
                os.mkdir(os.path.join('jupyter'))
            if not os.path.isfile(os.path.join('jupyter',
                                               'mosfit.ipynb')) or fc:
                shutil.copy(
                    os.path.join(dir_path, 'jupyter', 'mosfit.ipynb'),
                    os.path.join(os.getcwd(), 'jupyter', 'mosfit.ipynb'))

            if not os.path.exists('modules'):
                os.mkdir(os.path.join('modules'))
            module_dirs = next(os.walk(os.path.join(dir_path, 'modules')))[1]
            for mdir in module_dirs:
                if mdir.startswith('__'):
                    continue
                full_mdir = os.path.join(dir_path, 'modules', mdir)
                copy_path = os.path.join(full_mdir, '.copy')
                to_copy = []
                if os.path.isfile(copy_path):
                    to_copy = list(filter(None, open(
                        copy_path, 'r').read().split()))

                mdir_path = os.path.join('modules', mdir)
                if not os.path.exists(mdir_path):
                    os.mkdir(mdir_path)
                for tc in to_copy:
                    tc_path = os.path.join(full_mdir, tc)
                    if os.path.isfile(tc_path):
                        shutil.copy(tc_path, os.path.join(mdir_path, tc))
                    elif os.path.isdir(tc_path) and not os.path.exists(
                            os.path.join(mdir_path, tc)):
                        os.mkdir(os.path.join(mdir_path, tc))
                readme_path = os.path.join(mdir_path, 'README')
                if not os.path.exists(readme_path):
                    txt = prt.message('readme-modules', [
                        os.path.join(dir_path, 'modules', 'mdir'),
                        os.path.join(dir_path, 'modules')], prt=False)
                    open(readme_path, 'w').write(txt)

            if not os.path.exists('models'):
                os.mkdir(os.path.join('models'))
            model_dirs = next(os.walk(os.path.join(dir_path, 'models')))[1]
            for mdir in model_dirs:
                if mdir.startswith('__'):
                    continue
                mdir_path = os.path.join('models', mdir)
                if not os.path.exists(mdir_path):
                    os.mkdir(mdir_path)
                model_files = next(
                    os.walk(os.path.join(dir_path, 'models', mdir)))[2]
                readme_path = os.path.join(mdir_path, 'README')
                if not os.path.exists(readme_path):
                    txt = prt.message('readme-models', [
                        os.path.join(dir_path, 'models', mdir),
                        os.path.join(dir_path, 'models')], prt=False)
                    with open(readme_path, 'w') as f:
                        f.write(txt)
                for mfil in model_files:
                    if 'parameters.json' not in mfil:
                        continue
                    fil_path = os.path.join(mdir_path, mfil)
                    if os.path.isfile(fil_path) and not fc:
                        continue
                    shutil.copy(
                        os.path.join(dir_path, 'models', mdir, mfil),
                        os.path.join(fil_path))

    # Then, fit the listed events with the listed models.
    fitargs = vars(args)
    Fitter(**fitargs).fit_events(**fitargs)


if __name__ == "__main__":
    main()
