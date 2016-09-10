import json

from .model import Model


class Fitter():
    def fit_events(event_paths=[],
                   model_paths=[],
                   plot_points=100,
                   iterations=10):
        for path in event_paths:
            with open(path, 'r') as f:
                data = json.loads(f.read())
            for model_path in model_paths:
                model = Model(model_path=model_path)

                (fit, full) = model.fit_data(
                    data, plot_points=plot_points, iterations=iterations)
