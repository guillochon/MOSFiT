import argparse

from .fitter import fit_events


def main():
    parser = argparse.ArgumentParser(
        prog='FriendlyFit',
        description='Fit astrophysical light curves using AstroCats data.')

    parser.add_argument(
        '--event-paths', '-p', dest='event_paths', default=[], nargs='+')

    parser.add_argument(
        '--model-paths',
        '-m',
        dest='model_paths',
        default=['example_model.json'],
        nargs='+')

    parser.add_argument('--plot-points', dest='plot_points', default=100)

    parser.add_argument(
        '--iterations', '-i', dest='iterations', type=int, default=10)

    args = parser.parse_args()

    fit_events(args.event_paths, args.model_paths, args.plot_points,
               args.iterations)
