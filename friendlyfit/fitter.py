import json
import os
import warnings

import requests

from .model import Model
from emcee.utils import MPIPool

warnings.filterwarnings("ignore")


class Fitter():
    """Fit transient events with the provided model.
    """

    def fit_events(events=[],
                   models=[],
                   plot_points=100,
                   iterations=10,
                   num_walkers=100,
                   num_temps=2,
                   parameter_paths=[],
                   fracking=True,
                   frack_step=100,
                   travis=False):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        for event in events:
            pool = ''
            try:
                pool = MPIPool(loadbalance=True)
            except ValueError:
                pass
            except:
                raise

            if pool and not pool.is_master():
                pool.wait()

            path = ''
            # If the event name ends in .json, assume a path
            if event.endswith('.json'):
                path = event
                event_name = event.replace('.json', '').split('/')[-1]
            # If not (or the file doesn't exist), download from OSC
            if not path or not os.path.exists(path):
                names_path = os.path.join(
                        dir_path, 'cache', 'names.min.json')
                try:
                    response = requests.get(
                        'https://sne.space/astrocats/astrocats/'
                        'supernovae/output/names.min.json')
                except:
                    print('Warning: Could not download SN names!')
                else:
                    with open(names_path, 'wb') as f:
                        f.write(response.content)
                if os.path.exists(names_path):
                    with open(names_path, 'r') as f:
                        names = json.loads(f.read())
                else:
                    print('Error: Could not read list of SN names!')
                    raise RuntimeError

                event_name = ''
                if event in names:
                    event_name = event
                else:
                    for name in names:
                        if event in names[name]:
                            event_name = name
                            break
                if not event_name:
                    print('Error: Could not find event by that name!')
                    raise RuntimeError
                urlname = event_name + '.json'

                name_path = os.path.join(dir_path, 'cache', urlname)
                try:
                    response = requests.get(
                        'https://sne.space/astrocats/astrocats/'
                        'supernovae/output/json/' + urlname)
                except:
                    print('Warning: Could not download SN names!')
                else:
                    with open(name_path, 'wb') as f:
                        f.write(response.content)
                path = name_path

            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.loads(f.read())
            else:
                print('Error: Could not find supernova data locally or '
                      'on the OSC.')
                raise RuntimeError

            if pool:
                pool.close()

            for model in models:
                for parameter_path in parameter_paths:
                    model = Model(
                        model=model, parameter_path=parameter_path, travis=travis)

                    (walkers, prob) = model.fit_data(
                        data,
                        event_name=event_name,
                        plot_points=plot_points,
                        iterations=iterations,
                        num_walkers=num_walkers,
                        num_temps=num_temps,
                        fracking=fracking,
                        frack_step=frack_step)
