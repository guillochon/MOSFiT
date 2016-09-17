import datetime
import importlib
import json
import logging
import sys
import time
from collections import OrderedDict
from math import isnan

import emcee
import numpy as np
from emcee.utils import MPIPool
from scipy.optimize import basinhopping

from .utils import listify, pretty_num, print_inline, round_sig


class Model:
    """Define a semi-analytical model to fit transients with.
    """

    def __init__(self,
                 parameter_path='parameters.json',
                 model_path='example_model.json'):
        self._model_path = model_path
        with open(model_path, 'r') as f:
            self._model = json.loads(f.read())
        with open(parameter_path, 'r') as f:
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
            if 'class' in cur_task:
                class_name = cur_task['class']
            else:
                class_name = task
            mod = importlib.import_module(
                '.' + 'modules.' + cur_task['kind'] + 's.' + class_name,
                package='friendlyfit')
            mod_class = getattr(mod, mod.CLASS_NAME)
            if cur_task['kind'] == 'parameter' and task in self._parameters:
                cur_task.update(self._parameters[task])
            self._modules[task] = mod_class(name=task, **cur_task)
            if class_name == 'filters':
                self._bands = self._modules[task].band_names()
            if 'requests' in cur_task:
                inputs = listify(cur_task.get('inputs', []))
                for i, inp in enumerate(inputs):
                    requests = {}
                    parent = ''
                    for par in self._call_stack:
                        if par == inp:
                            parent = par
                    if not parent:
                        self._log.error("Couldn't find parent task!")
                        raise ValueError
                    reqs = cur_task['requests'][i]
                    for req in reqs:
                        requests[req] = self._modules[task].request(req)
                    self._modules[parent].handle_requests(**requests)

    def basinhop(self, x):
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

    def draw_walker(self, arg):
        """Draw a walker randomly from the full range of all parameters, reject
        walkers that return invalid scores.
        """
        p = None
        while p is None:
            draw = np.random.uniform(low=0.0, high=1.0, size=self._n_dim)
            score = self.likelihood(draw)
            if not isnan(score) and np.isfinite(score):
                p = draw
        return p

    def fit_data(self,
                 data,
                 plot_points=[],
                 iterations=10,
                 frack_step=100,
                 num_walkers=100,
                 num_temps=2,
                 fracking=True):
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['kind'] == 'data':
                self._modules[task].set_data(
                    data,
                    req_key_values={'band': self._bands},
                    subtract_minimum_keys=['times'])

        ntemps, ndim, nwalkers = (num_temps, self._num_free_parameters,
                                  num_walkers)
        self._n_dim = ndim
        # p0 = np.random.uniform(
        #     low=0.0, high=1.0, size=(ntemps, nwalkers, ndim))

        pool = MPIPool(loadbalance=True)

        if pool.is_master():
            print_inline('{} dimensions in problem.'.format(ndim))
            p0 = [[] for x in range(ntemps)]

            for i, pt in enumerate(p0):
                while len(p0[i]) < nwalkers:
                    print_inline(
                        'Drawing initial walkers | Progress: {}/{}'.format(
                            i * nwalkers + len(p0[i]), nwalkers * ntemps))
                    nmap = min(nwalkers - len(p0[i]), 4 * pool.size)
                    p0[i].extend(pool.map(self.draw_walker, range(nmap)))
                # p0[i].extend(pool.map(self.draw_walker, range(nwalkers)))
        else:
            pool.wait()
            sys.exit(0)

        sampler = emcee.PTSampler(
            ntemps, nwalkers, ndim, self.likelihood, self.prior, pool=pool)

        print_inline('Initial draws completed!\n')
        print_inline('Running PTSampler')
        p = p0.copy()
        frack_iters = max(round(iterations / frack_step), 1)
        emcee_est_t = 0.0
        bh_est_t = 0.0
        for b in range(frack_iters):
            emi = 0
            emcee_st = time.time()
            for p, lnprob, lnlike in sampler.sample(
                    p, iterations=min(frack_step, iterations)):
                scorestring = ','.join([pretty_num(max(x)) for x in lnprob])
                timestring = self.get_timestring(emcee_est_t + bh_est_t)
                print_inline('Running PTSampler | Best scores: [ {} ] | '
                             'Progress: {}/{} | '
                             'Estimated time left: {}s'.format(
                                 scorestring, b * frack_step + emi, iterations,
                                 timestring))
                emi = emi + 1
                emcee_est_t = float(time.time() - emcee_st) / emi * (
                    iterations - (b * frack_step + emi))

            if fracking:
                timestring = self.get_timestring(emcee_est_t + bh_est_t)
                print_inline(
                    'Running Basin-hopping | Estimated time left {}s'.format(
                        timestring))
                ris, rjs = [0] * pool.size, np.random.randint(
                    nwalkers, size=pool.size)

                bhs = pool.map(self.basinhop,
                               [p[i][j] for i, j in zip(ris, rjs)])
                bh_st = time.time()
                for bhi, bh in enumerate(bhs):
                    p[ris[bhi]][rjs[bhi]] = bh.x
                bh_est_t = float(time.time() - bh_st) * (frack_iters - b - 1)
                timestring = self.get_timestring(emcee_est_t + bh_est_t)
                scorestring = ','.join([pretty_num(-x.fun) for x in bhs])
                print_inline('Running Basin-hopping | Scores: [ {} ] | '
                             'Estimated time left {}s'.format(scorestring,
                                                              timestring))
        pool.close()

        bestprob = -np.inf
        bestx = lnprob[0][0]
        for i, probs in enumerate(lnprob):
            for j, prob in enumerate(probs):
                if prob > bestprob:
                    bestprob = prob
                    bestx = p[i][j]

        self.run_stack(bestx, root='output')

        return (p, lnprob)

    def get_timestring(self, t):
        return str(datetime.timedelta(seconds=(round_sig(t)))).rstrip('.0')

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
