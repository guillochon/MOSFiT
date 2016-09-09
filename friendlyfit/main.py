import argparse
import json

from .model import Model


def main():
    parser = argparse.ArgumentParser(
        prog='FriendlyFit',
        description='Fit astrophysical light curves using AstroCats data.')

    parser.add_argument('event-paths', '-p', dest='event_paths', default=None)

    parser.parse_args()

    for path in parser.event_paths:
        data = json.loads(path)
        for model_path in parser.model_paths:
            model = Model(model_path)

            (fit, full) = model.fit_data(data, plot_times=parser.plot_times)
