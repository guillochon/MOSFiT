import argparse
import json

from .model import Model


def main():
    parser = argparse.ArgumentParser(
        prog='FriendlyFit',
        description='Fit astrophysical light curves using AstroCats data.')

    parser.add_argument(
        '--event-paths', '-p', dest='event_paths', default=[], nargs='+')

    parser.add_argument(
        '--model-paths', '-m', dest='model_paths', default=[], nargs='+')

    parser.add_argument(
        '--plot-points', dest='plot_points', default=100)

    args = parser.parse_args()

    for path in args.event_paths:
        with open(path, 'r') as f:
            data = json.loads(f.read())
        for model_path in args.model_paths:
            model = Model(model_path=model_path)

            (fit, full) = model.fit_data(data, plot_points=args.plot_points)
