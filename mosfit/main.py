# -*- encoding: utf-8 -*-
"""The main function."""

import argparse
import codecs
import os
import shutil
import sys
import time
from operator import attrgetter
from unicodedata import normalize

import numpy as np
from astropy.time import Time as astrotime
from six import string_types

from mosfit import __author__, __contributors__, __version__
from mosfit.fitter import Fitter
from mosfit.printer import Printer
from mosfit.utils import get_mosfit_hash, is_master, speak


class SortingHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Sort argparse arguments by argument name."""

    def add_arguments(self, actions):
        """Add sorting action based on `option_strings`."""
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def get_parser(printer=None):
    """Retrieve MOSFiT's `argparse.ArgumentParser` object."""
    prt = Printer() if printer is None else printer

    parser = argparse.ArgumentParser(
        prog='mosfit',
        description='Fit astrophysical transients.',
        formatter_class=SortingHelpFormatter)

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
        '--prefer-fluxes',
        dest='prefer_fluxes',
        default=False,
        action='store_true',
        help=prt.text('parser_prefer_fluxes'))

    parser.add_argument(
        '--time-list',
        '--extra-times',
        dest='time_list',
        default=[],
        nargs='+',
        help=prt.text('parser_time_list'))

    parser.add_argument(
        '--extra-dates',
        dest='date_list',
        default=[],
        nargs='+',
        help=prt.text('parser_time_list'))

    parser.add_argument(
        '--extra-mjds',
        dest='mjd_list',
        default=[],
        nargs='+',
        help=prt.text('parser_time_list'))

    parser.add_argument(
        '--extra-jds',
        dest='jd_list',
        default=[],
        nargs='+',
        help=prt.text('parser_time_list'))

    parser.add_argument(
        '--extra-phases',
        dest='phase_list',
        default=[],
        nargs='+',
        help=prt.text('parser_time_list'))

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
        default=25,
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
        '--release-parameters',
        '-r',
        dest='user_released_parameters',
        default=[],
        nargs='+',
        help=prt.text('parser_user_released_parameters'))

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
        '--generative',
        '-G',
        dest='generative',
        default=False,
        action='store_true',
        help=prt.text('parser_generative'))

    parser.add_argument(
        '--smooth-times',
        '--plot-points',
        '-S',
        dest='smooth_times',
        type=int,
        const=0,
        default=21,
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
        '--output-path',
        '-o',
        dest='output_path',
        default='',
        help=prt.text('parser_output_path'))

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
        '--max-cores',
        dest='max_cores',
        type=int,
        default=1,
        help=prt.text('parser_max_cores'))

    parser.add_argument(
        '--num-temps',
        '-T',
        dest='num_temps',
        type=int,
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
        help=prt.text('parser_frack_step'))

    parser.add_argument(
        '--burn', '-b', dest='burn', type=int, help=prt.text('parser_burn'))

    parser.add_argument(
        '--post-burn',
        '-p',
        dest='post_burn',
        type=int,
        help=prt.text('parser_post_burn'))

    parser.add_argument(
        '--slice-sampler-steps',
        '-SSS',
        dest='slice_sampler_steps',
        type=int,
        default=-1,
        help=prt.text('slice_sampler_steps'))

    parser.add_argument(
        '--run-until-converged',
        '-R',
        dest='run_until_converged',
        type=float,
        default=False,
        const=True,
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
        help=prt.text('parser_maximum_memory'))

    parser.add_argument(
        '--seed', dest='seed', type=int, help=prt.text('parser_seed'))

    parser.add_argument(
        '--draw-above-likelihood',
        '-d',
        dest='draw_above_likelihood',
        type=float,
        const=True,
        nargs='?',
        help=prt.text('parser_draw_above_likelihood'))

    parser.add_argument(
        '--gibbs',
        '-g',
        dest='gibbs',
        action='store_const',
        const=True,
        help=prt.text('parser_gibbs'))

    parser.add_argument(
        '--save-full-chain',
        '-c',
        dest='save_full_chain',
        action='store_const',
        const=True,
        help=prt.text('parser_save_full_chain'))

    parser.add_argument(
        '--quick-save',
        '-qs',
        dest='quick_save',
        action='store_const',
        const=True,
        help=prt.text('parser_quick_save'))

    parser.add_argument(
        '--print-trees',
        dest='print_trees',
        default=False,
        action='store_true',
        help=prt.text('parser_print_trees'))

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
        default=None,
        nargs='*',
        help=prt.text('parser_extra_outputs'))

    parser.add_argument(
        '--no-guessing',
        dest='no_guessing',
        default=False,
        action='store_true',
        help=prt.text('parser_no_guessing'))

    parser.add_argument(
        '--exit-on-prompt',
        dest='exit_on_prompt',
        default=False,
        action='store_true',
        help=prt.text('parser_exit_on_prompt'))

    parser.add_argument(
        '--method',
        '-D',
        dest='method',
        choices=['ensembler', 'ultranest', 'dynesty'],
        default='dynesty',
        help=prt.text('parser_method'))

    return parser


def main():
    """Run MOSFiT."""
    parser = get_parser(printer=Printer(exit_on_prompt=False))
    args = parser.parse_args()

    prt = Printer(
        wrap_length=100,
        quiet=args.quiet,
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

    if (isinstance(args.extrapolate_time, list)
            and not args.extrapolate_time):
        args.extrapolate_time = 100.0

    if args.band_list and args.smooth_times == -1:
        prt.message('enabling_s')
        args.smooth_times = 0

    if is_master():
        if args.method in ('dynesty', 'ultranest'):
            unused_args = [[args.burn, '-b'], [args.post_burn, '-p'],
                           [args.frack_step, '-f'], [args.num_temps, '-T'],
                           [args.run_until_uncorrelated, '-U'],
                           [args.draw_above_likelihood, '-d'],
                           [args.gibbs, '-g'],
                           [args.maximum_memory, '-M']]
            if args.method == 'dynesty':
                unused_args.append([args.save_full_chain, '-c'])
            for ua in unused_args:
                if ua[0] is not None:
                    prt.message(
                        'argument_not_used',
                        reps=[ua[1], '-D dynesty'],
                        warning=True)

    if args.method in ('dynesty', 'ultranest'):
        if args.run_until_converged and args.iterations >= 0:
            raise ValueError(prt.text('R_i_mutually_exclusive'))
        if args.walker_paths is not None:
            raise ValueError(prt.text('w_nester_mutually_exclusive'))

    if args.generative:
        if args.iterations > 0:
            prt.message('generator_supercedes', warning=True)
        args.iterations = 0

    no_events = False
    if args.iterations == -1:
        if not args.events:
            no_events = True
            args.iterations = 0
        else:
            args.iterations = 5000

    if args.time_list:
        if any([any([y in x]) for y in ['-', '/'] for x in args.time_list]):
            try:
                args.time_list = [
                    astrotime(x.replace('/', '-')).mjd for x in args.time_list
                ]
            except ValueError:
                if len(args.time_list) == 1 and isinstance(
                        args.time_list[0], string_types):
                    args.time_list = args.time_list[0].split()
                args.time_list = [float(x) for x in args.time_list]
                args.time_unit = 'phase'
        else:
            if any(['+' in x for x in args.time_list]):
                args.time_unit = 'phase'
            args.time_list = [float(x) for x in args.time_list]

    if args.date_list:
        if no_events:
            prt.message('no_dates_gen', warning=True)
        else:
            args.time_list += [
                str(astrotime(x.replace('/', '-')).mjd) for x in args.date_list
            ]
            args.time_unit = 'mjd'

    if args.mjd_list:
        if no_events:
            prt.message('no_dates_gen', warning=True)
        else:
            args.time_list += [float(x) for x in args.mjd_list]
            args.time_unit = 'mjd'

    if args.jd_list:
        if no_events:
            prt.message('no_dates_gen', warning=True)
        else:
            args.time_list += [
                str(astrotime(float(x), format='jd').mjd) for x in args.jd_list
            ]
            args.time_unit = 'mjd'

    if args.phase_list:
        if no_events:
            prt.message('no_dates_gen', warning=True)
        else:
            args.time_list += [float(x) for x in args.phase_list]
            args.time_unit = 'phase'

    if args.time_list:
        if min(args.time_list) > 2400000:
            prt.message('assuming_jd')
            args.time_list = [x - 2400000.5 for x in args.time_list]
            args.time_unit = 'mjd'
        elif min(args.time_list) > 50000:
            prt.message('assuming_mjd')
            args.time_unit = 'mjd'

    if args.burn is None and args.post_burn is None:
        args.burn = int(np.floor(args.iterations / 2))

    if args.frack_step == 0:
        args.fracking = False

    if (args.run_until_uncorrelated is not None and args.run_until_converged):
        raise ValueError(
            '`-R` and `-U` options are incompatible, please use one or the '
            'other.')
    if args.run_until_uncorrelated is not None:
        args.convergence_type = 'acor'
        args.convergence_criteria = args.run_until_uncorrelated
    elif args.run_until_converged:
        if args.method == 'ensembler':
            args.convergence_type = 'psrf'
            args.convergence_criteria = (1.1
                                         if args.run_until_converged is True
                                         else args.run_until_converged)
        else:
            args.convergence_type = 'dlogz'

    if args.method in ('dynesty', 'ultranest'):
        args.convergence_criteria = (0.02 if args.run_until_converged is True
                                     else args.run_until_converged)

    if is_master():
        # Get hash of ourselves
        mosfit_hash = get_mosfit_hash()

        # Print our amazing ASCII logo.
        if not args.quiet:
            with codecs.open(os.path.join(dir_path, 'logo.txt'), 'r',
                             'utf-8') as f:
                logo = f.read()
                firstline = logo.split('\n')[0]
                # if isinstance(firstline, bytes):
                #     firstline = firstline.decode('utf-8')
                width = len(normalize('NFC', firstline))
            prt.prt(logo, colorify=True)
            prt.message(
                'byline',
                reps=[__version__, mosfit_hash, __author__, __contributors__],
                center=True,
                colorify=True,
                width=width,
                wrapped=False)

        if no_events:
            prt.message('iterations_0', wrapped=True)

        # Create the user directory structure, if it doesn't already exist.
        if args.copy:
            prt.message('copying')
            fc = False
            if args.force_copy:
                fc = prt.prompt('force_copy')
            if not os.path.exists('jupyter'):
                os.mkdir(os.path.join('jupyter'))
            jupyter_src = os.path.join(dir_path, 'jupyter')
            for nb_name in sorted(os.listdir(jupyter_src)):
                if not nb_name.endswith('.ipynb') or nb_name.startswith('.'):
                    continue
                dst_nb = os.path.join(os.getcwd(), 'jupyter', nb_name)
                if not os.path.isfile(dst_nb) or fc:
                    shutil.copy(
                        os.path.join(jupyter_src, nb_name), dst_nb)

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
                    to_copy = list(
                        filter(None,
                               open(copy_path, 'r').read().split()))

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
                    txt = prt.message(
                        'readme-modules', [
                            os.path.join(dir_path, 'modules', 'mdir'),
                            os.path.join(dir_path, 'modules')
                        ],
                        prt=False)
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
                    txt = prt.message(
                        'readme-models', [
                            os.path.join(dir_path, 'models', mdir),
                            os.path.join(dir_path, 'models')
                        ],
                        prt=False)
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

    # Set some default values that we checked above.
    if args.frack_step == 0:
        args.fracking = False
    elif args.frack_step is None:
        args.frack_step = 50
    if args.burn is None and args.post_burn is None:
        args.burn = int(np.floor(args.iterations / 2))
    if args.draw_above_likelihood is None:
        args.draw_above_likelihood = False
    if args.maximum_memory is None:
        args.maximum_memory = np.inf
    if args.gibbs is None:
        args.gibbs = False
    if args.save_full_chain is None:
        args.save_full_chain = False
    if args.num_temps is None:
        args.num_temps = 1
    if args.walker_paths is None:
        args.walker_paths = []
    if args.no_guessing:
        args.guess = False

    # Then, fit the listed events with the listed models.
    fitargs = vars(args)
    Fitter(**fitargs).fit_events(**fitargs)


if __name__ == "__main__":
    main()
