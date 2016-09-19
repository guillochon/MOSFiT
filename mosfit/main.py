import argparse

from mosfit.fitter import Fitter


def main():
    """First, parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        prog='MOSFiT',
        description='Fit astrophysical light curves using AstroCats data.')

    parser.add_argument(
        '--events',
        '-e',
        dest='events',
        default=['mosfit/tests/SN2006le.json'],
        nargs='+')

    parser.add_argument(
        '--models', '-m', dest='models', default=['default'], nargs='+')

    parser.add_argument(
        '--parameter-paths',
        '-P',
        dest='parameter_paths',
        default=[''],
        nargs='+')

    parser.add_argument('--plot-points', dest='plot_points', default=100)

    parser.add_argument(
        '--iterations', '-i', dest='iterations', type=int, default=10)

    parser.add_argument(
        '--num-walkers', '-N', dest='num_walkers', type=int, default=100)

    parser.add_argument(
        '--num-temps', '-T', dest='num_temps', type=int, default=2)

    parser.add_argument(
        '--no-fracking', dest='fracking', default=True, action='store_false')

    parser.add_argument(
        '--frack-step', '-f', dest='frack_step', type=int, default=100)

    parser.add_argument(
        '--post-burn', '-p', dest='post_burn', type=int, default=500)

    parser.add_argument(
        '--travis', dest='travis', default=False, action='store_true')

    args = parser.parse_args()
    """Then, fit the listed events with the listed models.
    """
    Fitter.fit_events(args.events, args.models, args.plot_points,
                      args.iterations, args.num_walkers, args.num_temps,
                      args.parameter_paths, args.fracking, args.frack_step,
                      args.travis, args.post_burn)


if __name__ == "__main__":
    main()
