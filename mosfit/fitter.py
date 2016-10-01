import json
import os
import warnings

import numpy as np
import requests
from emcee.utils import MPIPool

from .model import Model
from mosfit.utils import print_inline

warnings.filterwarnings("ignore")


class Fitter():
    """Fit transient events with the provided model.
    """

    def __init__(self):
        pass

    def fit_events(self,
                   events=[''],
                   models=[],
                   plot_points='',
                   max_time='',
                   band_list='',
                   band_systems='',
                   band_instruments='',
                   iterations=1000,
                   num_walkers=50,
                   num_temps=2,
                   parameter_paths=[],
                   fracking=True,
                   frack_step=20,
                   travis=False,
                   post_burn=500):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        for event in events:
            if event:
                pool = ''
                try:
                    pool = MPIPool()
                except ValueError:
                    pass
                except:
                    raise
                event_name = ''
                if not pool or pool.is_master():
                    path = ''
                    # If the event name ends in .json, assume a path
                    if event.endswith('.json'):
                        path = event
                        event_name = event.replace('.json', '').split('/')[-1]
                    # If not (or the file doesn't exist), download from OSC
                    if not path or not os.path.exists(path):
                        names_path = os.path.join(dir_path, 'cache',
                                                  'names.min.json')
                        input_name = event.replace('.json', '')
                        print('Event `{}` interpreted as supernova name, '
                              'downloading list of superova aliases...'.format(
                                  input_name))
                        try:
                            response = requests.get(
                                'https://sne.space/astrocats/astrocats/'
                                'supernovae/output/names.min.json',
                                timeout=10)
                        except:
                            print_inline(
                                'Warning: Could not download SN names (are '
                                'you online?), using cached list.')
                        else:
                            with open(names_path, 'wb') as f:
                                f.write(response.content)
                                f.flush()
                        if os.path.exists(names_path):
                            with open(names_path, 'r') as f:
                                names = json.loads(f.read())
                        else:
                            print('Error: Could not read list of SN names!')
                            raise RuntimeError

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

                        print('Found event by primary name `{}` in the OSC, '
                              'downloading data...'.format(event_name))
                        name_path = os.path.join(dir_path, 'cache', urlname)
                        try:
                            response = requests.get(
                                'https://sne.space/astrocats/astrocats/'
                                'supernovae/output/json/' + urlname,
                                timeout=10)
                        except:
                            print_inline(
                                'Warning: Could not download data for `{}`, '
                                'will attempt to use cached data.'.format(
                                    event_name))
                        else:
                            with open(name_path, 'wb') as f:
                                f.write(response.content)
                                f.flush()
                        path = name_path

                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            data = json.loads(f.read())
                    else:
                        print('Error: Could not find data for `{}` locally or '
                              'on the OSC.'.format(event_name))
                        raise RuntimeError

                if pool:
                    if pool.is_master():
                        for rank in range(1, pool.size + 1):
                            pool.comm.send(event_name, dest=rank, tag=0)
                            pool.comm.send(path, dest=rank, tag=1)
                            pool.comm.send(data, dest=rank, tag=2)
                    else:
                        event_name = pool.comm.recv(source=0, tag=0)
                        path = pool.comm.recv(source=0, tag=1)
                        data = pool.comm.recv(source=0, tag=2)
                        pool.wait()

                if pool:
                    pool.close()

            for mod_name in models:
                for parameter_path in parameter_paths:
                    model = Model(
                        model=mod_name,
                        parameter_path=parameter_path,
                        travis=travis)

                    if not event:
                        print('No event specified, generating dummy data.')
                        event_name = mod_name
                        gen_args = {
                            'name': mod_name,
                            'max_time': max_time,
                            'plot_points': plot_points,
                            'band_list': band_list,
                            'band_systems': band_systems,
                            'band_instruments': band_instruments
                        }
                        data = self.generate_dummy_data(**gen_args)

                    model.fit_data(
                        data,
                        event_name=event_name,
                        plot_points=plot_points,
                        iterations=iterations,
                        num_walkers=num_walkers,
                        num_temps=num_temps,
                        fracking=fracking,
                        frack_step=frack_step,
                        post_burn=post_burn)

    def generate_dummy_data(self,
                            name,
                            max_time=1000.,
                            plot_points=100,
                            band_list=['V'],
                            band_systems=[''],
                            band_instruments=['']):
        time_list = np.linspace(0.0, max_time, plot_points)
        times = np.repeat(time_list, len(band_list))
        bands = [i for s in [band_list for x in time_list] for i in s]
        systs = [i for s in [band_systems for x in time_list] for i in s]
        insts = [i for s in [band_instruments for x in time_list] for i in s]

        data = {name: {'photometry': []}}
        for ti, time in enumerate(times):
            band = bands[ti]
            if isinstance(band, dict):
                band = band['name']

            photodict = {'time': time,
                         'band': band,
                         'magnitude': 0.0,
                         'e_magnitude': 0.0}
            if systs[ti]:
                photodict['system'] = systs[ti]
            if insts[ti]:
                photodict['instrument'] = insts[ti]
            data[name]['photometry'].append(photodict)

        return data
