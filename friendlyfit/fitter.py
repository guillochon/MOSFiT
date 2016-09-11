import json
import warnings

from .model import Model

warnings.filterwarnings("ignore")


class Fitter():
    def fit_events(event_paths=[],
                   model_paths=[],
                   plot_points=100,
                   iterations=10,
                   num_walkers=100,
                   num_temps=2):
        for path in event_paths:
            with open(path, 'r') as f:
                data = json.loads(f.read())
            for model_path in model_paths:
                model = Model(model_path=model_path)

                (walkers, prob) = model.fit_data(
                    data,
                    plot_points=plot_points,
                    iterations=iterations,
                    num_walkers=num_walkers,
                    num_temps=num_temps)
