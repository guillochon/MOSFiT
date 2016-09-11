import argparse

from friendlyfit.fitter import Fitter


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

    parser.add_argument(
        '--num-walkers', '-N', dest='num_walkers', type=int, default=100)

    parser.add_argument(
        '--num-temps', '-T', dest='num_temps', type=int, default=2)

    args = parser.parse_args()

    Fitter.fit_events(args.event_paths, args.model_paths, args.plot_points,
                      args.iterations, args.num_walkers, args.num_temps)

if __name__ == "__main__":
    main()
