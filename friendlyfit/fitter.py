import json
import warnings

from .model import Model

warnings.filterwarnings("ignore")


class Fitter():
    """Fit transient events with the provided model.
    """

    def fit_events(event_paths=[],
                   models=[],
                   plot_points=100,
                   iterations=10,
                   num_walkers=100,
                   num_temps=2,
                   parameter_paths=[],
                   fracking=True,
                   frack_step=100):
        for path in event_paths:
            with open(path, 'r') as f:
                data = json.loads(f.read())
            for model in models:
                for parameter_path in parameter_paths:
                    model = Model(
                        model=model, parameter_path=parameter_path)

                    (walkers, prob) = model.fit_data(
                        data,
                        plot_points=plot_points,
                        iterations=iterations,
                        num_walkers=num_walkers,
                        num_temps=num_temps,
                        fracking=fracking,
                        frack_step=frack_step)
