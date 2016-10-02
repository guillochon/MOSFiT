import datetime
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from math import isnan

import emcee
import numpy as np
from emcee.utils import MPIPool
from mosfit.utils import listify, pretty_num, print_inline
from scipy.optimize import basinhopping


class Model:
    """Define a semi-analytical model to fit transients with.
    """

    MODEL_OUTPUT_DIR = 'products'

    def __init__(self,
                 parameter_path='parameters.json',
                 model='default',
                 wrap_length=100,
                 travis=False):
        self._model_name = model
        self._travis = travis

        self._dir_path = os.path.dirname(os.path.realpath(__file__))

        # Load the model file.
        model = self._model_name
        model_dir = self._model_name

        if '.json' in self._model_name:
            model_dir = self._model_name.split('.json')[0]
        else:
            model = self._model_name + '.json'

        if os.path.isfile(model):
            model_path = model
        else:
            # Look in local hierarchy first
            if os.path.isfile(os.path.join('models', model_dir, model)):
                model_path = os.path.join('models', model_dir, model)
            else:
                model_path = os.path.join(self._dir_path, 'models', model_dir,
                                          model)

        with open(model_path, 'r') as f:
            self._model = json.loads(f.read())

        # Load model parameter file.
        model_pp = os.path.join(
            os.path.split(model_path)[0], 'parameters.json')

        pp = model_pp

        # First try user-specified path
        if parameter_path and os.path.isfile(parameter_path):
            pp = parameter_path
        # Then try directory we are running from
        elif os.path.isfile('parameter.json'):
            pp = 'parameter.json'
        # Finally try model folder
        elif os.path.isfile(model_pp):
            pp = model_pp

        pool = ''
        try:
            pool = MPIPool()
        except ValueError:
            pass
        except:
            raise

        if not pool or pool.is_master():
            print('Model file: ' + model_path)
            print('Parameter file: ' + pp + '\n')

        with open(pp, 'r') as f:
            self._parameters = json.loads(f.read())
        self._num_free_parameters = len(
            [x for x in self._parameters
             if ('min_value' in self._parameters[x] and 'max_value' in
                 self._parameters[x])])
        self._log = logging.getLogger()
        self._modules = {}
        self._bands = []

        # Load the call tree for the model. Work our way in reverse from the
        # observables, first constructing a tree for each observable and then
        # combining trees.
        root_kinds = ['output', 'objective']

        trees = {}
        self.construct_trees(self._model, trees, kinds=root_kinds)

        unsorted_call_stack = {}
        self._max_depth_all = -1
        for tag in self._model:
            cur_model = self._model[tag]
            roots = []
            if cur_model['kind'] in root_kinds:
                max_depth = 0
                roots = [cur_model['kind']]
            else:
                max_depth = -1
                for tag2 in trees:
                    roots.extend(trees[tag2]['roots'])
                    depth = self.get_max_depth(tag, trees[tag2], max_depth)
                    if depth > max_depth:
                        max_depth = depth
                    if depth > self._max_depth_all:
                        self._max_depth_all = depth
            roots = list(set(roots))
            new_entry = cur_model.copy()
            new_entry['roots'] = roots
            if 'children' in new_entry:
                del (new_entry['children'])
            new_entry['depth'] = max_depth
            unsorted_call_stack[tag] = new_entry
        # print(unsorted_call_stack)

        # Currently just have one call stack for all products, can be wasteful
        # if only using some products.
        self._call_stack = OrderedDict()
        for depth in range(self._max_depth_all, -1, -1):
            for task in unsorted_call_stack:
                if unsorted_call_stack[task]['depth'] == depth:
                    self._call_stack[task] = unsorted_call_stack[task]

        for task in self._call_stack:
            cur_task = self._call_stack[task]
            class_name = cur_task.get('class', task)
            mod = importlib.import_module(
                '.' + 'modules.' + cur_task['kind'] + 's.' + class_name,
                package='mosfit')
            mod_class = getattr(mod, mod.CLASS_NAME)
            if cur_task['kind'] == 'parameter' and task in self._parameters:
                cur_task.update(self._parameters[task])
            self._modules[task] = mod_class(name=task, **cur_task)
            if class_name == 'filters':
                self._bands = self._modules[task].band_names()
            # This is currently not functional for MPI
            # cur_task = self._call_stack[task]
            # class_name = cur_task.get('class', task)
            # mod_path = os.path.join('modules', cur_task['kind'] + 's',
            #                         class_name + '.py')
            # if not os.path.isfile(mod_path):
            #     mod_path = os.path.join(self._dir_path, 'modules',
            #                             cur_task['kind'] + 's',
            #                             class_name + '.py')
            # mod_name = ('mosfit.modules.' + cur_task['kind'] + 's.' +
            # class_name)
            # mod = importlib.machinery.SourceFileLoader(mod_name,
            #                                            mod_path).load_module()
            # mod_class = getattr(mod, mod.CLASS_NAME)
            # if cur_task['kind'] == 'parameter' and task in self._parameters:
            #     cur_task.update(self._parameters[task])
            # self._modules[task] = mod_class(name=task, **cur_task)
            # if class_name == 'filters':
            #     self._bands = self._modules[task].band_names()

        for task in reversed(self._call_stack):
            cur_task = self._call_stack[task]
            if 'requests' in cur_task:
                inputs = listify(cur_task.get('inputs', []))
                for i, inp in enumerate(inputs):
                    requests = {}
                    parent = ''
                    for par in self._call_stack:
                        if par == inp:
                            parent = par
                    if not parent:
                        self._log.error(
                            "Couldn't find parent task for  {}!".format(inp))
                        raise ValueError
                    reqs = cur_task['requests'][i]
                    for req in reqs:
                        requests[req] = self._modules[task].request(req)
                    self._modules[parent].handle_requests(**requests)

    def basinhop(self, x):
        """Perform Basin-hopping upon a single walker.
        """
        bh = basinhopping(self.bhprob, x, niter=10)
        return bh

    def construct_trees(self, d, trees, kinds=[], name='', roots=[], depth=0):
        """Construct call trees for each root.
        """
        for tag in d:
            entry = d[tag].copy()
            new_roots = roots
            if entry['kind'] in kinds or tag == name:
                entry['depth'] = depth
                if entry['kind'] in kinds:
                    new_roots.append(entry['kind'])
                entry['roots'] = list(set(new_roots))
                trees[tag] = entry
                inputs = listify(entry.get('inputs', []))
                for inp in inputs:
                    children = {}
                    self.construct_trees(
                        d,
                        children,
                        name=inp,
                        roots=new_roots,
                        depth=depth + 1)
                    trees[tag].setdefault('children', {})
                    trees[tag]['children'].update(children)

    def draw_walker(self, test=True):
        """Draw a walker randomly from the full range of all parameters, reject
        walkers that return invalid scores.
        """
        p = None
        while p is None:
            draw = np.random.uniform(low=0.0, high=1.0, size=self._n_dim)
            if not test:
                p = draw
                break
            score = self.likelihood(draw)
            if not isnan(score) and np.isfinite(score):
                p = draw
        return p

    def fit_data(self,
                 data,
                 event_name='',
                 plot_points=[],
                 iterations=2000,
                 frack_step=100,
                 num_walkers=50,
                 num_temps=2,
                 fracking=True,
                 post_burn=500):
        """Fit the data for a given event with this model using a combination
        of emcee and Basin-hopping.
        """
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            self._modules[task].set_event_name(event_name)
            if cur_task['kind'] == 'data':
                self._modules[task].set_data(
                    data,
                    req_key_values={'band': self._bands},
                    subtract_minimum_keys=['times'])

        ntemps, ndim, nwalkers = (num_temps, self._num_free_parameters,
                                  num_walkers)
        self._event_name = event_name
        self._n_dim = ndim
        self._emcee_est_t = 0.0
        self._bh_est_t = 0.0
        self._fracking = fracking
        self._burn_in = max(iterations - post_burn, 0)

        test_walker = iterations > 0
        lnprob = None
        sampler_args = {}
        serial = False
        pool = ''
        try:
            pool = MPIPool()
        except ValueError:
            psize = 1
            serial = True
        except:
            raise
        else:
            sampler_args = {'pool': pool}
            psize = pool.size

        if serial or pool.is_master():
            print('{} dimensions in problem.\n\n'.format(ndim))
            p0 = [[] for x in range(ntemps)]

            for i, pt in enumerate(p0):
                while len(p0[i]) < nwalkers:
                    self.print_status(
                        desc='Drawing initial walkers',
                        progress=[i * nwalkers + len(p0[i]),
                                  nwalkers * ntemps])
                    if serial:
                        p0[i].append(self.draw_walker(test_walker))
                    else:
                        nmap = nwalkers - len(p0[i])
                        p0[i].extend(
                            pool.map(self.draw_walker, [test_walker] * nmap))
                # p0[i].extend(pool.map(self.draw_walker, range(nwalkers)))
        else:
            pool.wait()
            return

        sampler = emcee.PTSampler(ntemps, nwalkers, ndim, self.likelihood,
                                  self.prior, **sampler_args)

        print_inline('Initial draws completed!')
        print('\n\n')
        p = p0.copy()

        if fracking:
            frack_iters = max(round(iterations / frack_step), 1)
            bmax = int(round(self._burn_in / float(frack_step)))
            loop_step = frack_step
        else:
            frack_iters = 1
            loop_step = iterations

        acort = 1.0
        acorc = 1.0
        for b in range(frack_iters):
            if fracking and b >= bmax:
                loop_step = iterations - self._burn_in
            emi = 0
            st = time.time()
            for p, lnprob, lnlike in sampler.sample(
                    p, iterations=min(loop_step, iterations)):
                emi = emi + 1
                prog = b * frack_step + emi
                try:
                    acorc = max(0.1 * float(prog) / acort, 1.0)
                    acort = max([
                        max(x)
                        for x in sampler.get_autocorr_time(c=min(acorc, 10.0))
                    ])
                except:
                    pass
                acor = [acort, acorc]
                self._emcee_est_t = float(time.time() - st) / emi * (
                    iterations - (b * frack_step + emi))
                self.print_status(
                    desc='Running PTSampler',
                    scores=[max(x) for x in lnprob],
                    progress=[prog, iterations],
                    acor=acor)
            if fracking and b >= bmax:
                break
            if fracking and b < bmax:
                self.print_status(
                    desc='Running Basin-hopping',
                    scores=[max(x) for x in lnprob],
                    progress=[(b + 1) * frack_step, iterations],
                    acor=acor)
                ris, rjs = [0] * psize, np.random.randint(nwalkers, size=psize)

                bhwalkers = [p[i][j] for i, j in zip(ris, rjs)]
                st = time.time()
                if serial:
                    bhs = list(map(self.basinhop, bhwalkers))
                else:
                    bhs = pool.map(self.basinhop, bhwalkers)
                for bhi, bh in enumerate(bhs):
                    p[ris[bhi]][rjs[bhi]] = bh.x
                self._bh_est_t = float(time.time() - st) * (bmax - b - 1)
                scores = [-x.fun for x in bhs]
                self.print_status(
                    desc='Running Basin-hopping',
                    scores=scores,
                    progress=[(b + 1) * frack_step, iterations])

        if pool:
            pool.close()

        walkers_out = OrderedDict()
        for xi, x in enumerate(p[0]):
            walkers_out[xi] = self.run_stack(x, root='output')
            if lnprob is not None:
                walkers_out[xi]['score'] = lnprob[0][xi]
            parameters = {}
            pi = 0
            for ti, task in enumerate(self._call_stack):
                cur_task = self._call_stack[task]
                if (cur_task['kind'] != 'parameter' or
                        'min_value' not in cur_task or
                        'max_value' not in cur_task):
                    continue
                output = self._modules[task].process(**{'fraction': x[pi]})
                value = list(output.values())[0]
                paramdict = {
                    'value': value,
                    'fraction': x[pi],
                    'latex': self._modules[task].latex(),
                    'log': self._modules[task].is_log()
                }
                parameters.update({self._modules[task].name(): paramdict})
                pi = pi + 1
            walkers_out[xi]['parameters'] = parameters

        if not os.path.exists(self.MODEL_OUTPUT_DIR):
            os.makedirs(self.MODEL_OUTPUT_DIR)

        with open(os.path.join(self.MODEL_OUTPUT_DIR, 'walkers.json'),
                  'w') as flast, open(
                      os.path.join(self.MODEL_OUTPUT_DIR,
                                   self._event_name + '.json'), 'w') as f:
            json.dump(walkers_out, flast, indent='\t', separators=(',', ':'))
            json.dump(walkers_out, f, indent='\t', separators=(',', ':'))

        return (p, lnprob)

    def print_status(self,
                     desc='',
                     scores='',
                     progress='',
                     acor='',
                     wrap_length=100):
        """Prints a status message showing the current state of the fitting process.
        """

        class bcolors:
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'

        outarr = [self._event_name]
        if desc:
            outarr.append(desc)
        if isinstance(scores, list):
            scorestring = 'Best scores: [ ' + ', '.join(
                [pretty_num(x) for x in scores]) + ' ]'
            outarr.append(scorestring)
        if isinstance(progress, list):
            progressstring = 'Progress: [ {}/{} ]'.format(*progress)
            outarr.append(progressstring)
        if self._emcee_est_t + self._bh_est_t > 0.0:
            if self._bh_est_t > 0.0 or not self._fracking:
                tott = self._emcee_est_t + self._bh_est_t
            else:
                tott = 2.0 * self._emcee_est_t
            timestring = self.get_timestring(tott)
            outarr.append(timestring)
        if isinstance(acor, list):
            acortstr = pretty_num(acor[0], sig=3)
            acorcstr = pretty_num(acor[1], sig=2)
            if acor[1] < 5.0:
                col = bcolors.FAIL
            elif acor[1] < 10.0:
                col = bcolors.WARNING
            else:
                col = ''
            acorstring = col
            acorstring = acorstring + 'Acor Tau: {} ({}x)'.format(acortstr,
                                                                  acorcstr)
            acorstring = acorstring + bcolors.ENDC if col else ''
            outarr.append(acorstring)

        line = ''
        lines = ''
        li = 0
        for i, item in enumerate(outarr):
            oldline = line
            line = line + (' | ' if li > 0 else '') + item
            li = li + 1
            if len(line) > wrap_length:
                li = 1
                lines = lines + '\n' + oldline
                line = item

        lines = lines + '\n' + line

        print_inline(lines, new_line=self._travis)

    def get_timestring(self, t):
        """Return a string showing the estimated remaining time based upon
        elapsed times for emcee and Basin-hopping.
        """
        td = str(datetime.timedelta(seconds=round(t)))
        return ('Estimated time left: [ ' + td + ' ]')

    def get_max_depth(self, tag, parent, max_depth):
        """Return the maximum depth a given task is found in a tree.
        """
        for child in parent.get('children', []):
            if child == tag:
                new_max = parent['children'][child]['depth']
                if new_max > max_depth:
                    max_depth = new_max
            else:
                new_max = self.get_max_depth(tag, parent['children'][child],
                                             max_depth)
                if new_max > max_depth:
                    max_depth = new_max
        return max_depth

    def likelihood(self, x):
        """Return score related to maximum likelihood.
        """
        outputs = self.run_stack(x, root='objective')
        return outputs['value']

    def prior(self, data):
        """Return score related to paramater priors.
        """
        return 0.0

    def bhprob(self, x):
        """Return score for basinhopping.
        """
        return -self.likelihood(x)

    def run_stack(self, x, root='objective'):
        """Run a stack of modules as defined in the model definition file. Only
        run functions that match the specified root.
        """
        inputs = {}
        outputs = {}
        pos = 0
        cur_depth = self._max_depth_all
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if root not in cur_task['roots']:
                continue
            if cur_task['depth'] != cur_depth:
                inputs = outputs
            cur_depth = cur_task['depth']
            if (cur_task['kind'] == 'parameter' and 'min_value' in cur_task and
                    'max_value' in cur_task):
                inputs.update({'fraction': x[pos]})
                inputs.setdefault('fractions', []).append(x[pos])
                pos = pos + 1
            new_outs = self._modules[task].process(**inputs)
            outputs.update(new_outs)

            if cur_task['kind'] == root:
                return outputs
