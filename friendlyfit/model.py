import datetime
import importlib
import json
import logging
import os
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

    def __init__(self, parameter_path='parameters.json', model='default'):
        self._model_name = model
        # Load the model file.
        with open(
                os.path.join('friendlyfit', 'models', model, model + '.json'),
                'r') as f:
            self._model = json.loads(f.read())

        # Load model parameter file.
        model_pp = os.path.join('friendlyfit', 'models', model,
                                'parameters.json')
        # First try user-specified path
        if parameter_path and os.path.isfile(parameter_path):
            pp = parameter_path
        # Then try directory we are running from
        elif os.path.isfile('parameter.json'):
            pp = 'parameter.json'
        # Finally try model folder
        elif os.path.isfile(model_pp):
            pp = model_pp

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
                package='friendlyfit')
            mod_class = getattr(mod, mod.CLASS_NAME)
            if cur_task['kind'] == 'parameter' and task in self._parameters:
                cur_task.update(self._parameters[task])
            self._modules[task] = mod_class(name=task, **cur_task)
            if class_name == 'filters':
                self._bands = self._modules[task].band_names()

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

    def draw_walker(self, arg=''):
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
                 event_name='',
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
        self._event_name = event_name
        self._n_dim = ndim
        self._emcee_est_t = 0.0
        self._bh_est_t = 0.0
        # p0 = np.random.uniform(
        #     low=0.0, high=1.0, size=(ntemps, nwalkers, ndim))

        sampler_args = {}
        try:
            pool = MPIPool(loadbalance=True)
        except ValueError:
            psize = 1
        except:
            raise
        else:
            sampler_args = {'pool': pool}
            psize = pool.size

        if psize == 1 or pool.is_master():
            print_inline('{} dimensions in problem.'.format(ndim))
            p0 = [[] for x in range(ntemps)]

            for i, pt in enumerate(p0):
                while len(p0[i]) < nwalkers:
                    self.print_status(
                        desc='Drawing initial walkers',
                        progress=[i * nwalkers + len(p0[i]), nwalkers * ntemps
                                  ])
                    if psize == 1:
                        p0[i].append(self.draw_walker())
                    else:
                        nmap = min(nwalkers - len(p0[i]), 4 * psize)
                        p0[i].extend(pool.map(self.draw_walker, range(nmap)))
                # p0[i].extend(pool.map(self.draw_walker, range(nwalkers)))
        else:
            pool.wait()
            sys.exit(0)

        sampler = emcee.PTSampler(ntemps, nwalkers, ndim, self.likelihood,
                                  self.prior, **sampler_args)

        print_inline('Initial draws completed!\n')
        p = p0.copy()
        frack_iters = max(round(iterations / frack_step), 1)
        for b in range(frack_iters):
            emi = 0
            st = time.time()
            for p, lnprob, lnlike in sampler.sample(
                    p, iterations=min(frack_step, iterations)):
                emi = emi + 1
                self._emcee_est_t = float(time.time() - st) / emi * (
                    iterations - (b * frack_step + emi))
                self.print_status(
                    desc='Running PTSampler',
                    scores=[max(x) for x in lnprob],
                    progress=[b * frack_step + emi, iterations])

            if fracking:
                self.print_status(
                    desc='Running Basin-hopping',
                    progress=[(b + 1) * frack_step, iterations])
                ris, rjs = [0] * psize, np.random.randint(nwalkers, size=psize)

                bhwalkers = [p[i][j] for i, j in zip(ris, rjs)]
                st = time.time()
                if psize == 1:
                    bhs = list(map(self.basinhop, bhwalkers))
                else:
                    bhs = pool.map(self.basinhop, bhwalkers)
                for bhi, bh in enumerate(bhs):
                    p[ris[bhi]][rjs[bhi]] = bh.x
                self._bh_est_t = float(time.time() - st) * (
                    frack_iters - b - 1)
                scores = [-x.fun for x in bhs]
                self.print_status(
                    desc='Running Basin-hopping',
                    scores=scores,
                    progress=[(b + 1) * frack_step, iterations])
        if psize > 1:
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

    def print_status(self, desc='', scores='', progress=''):
        outarr = [self._event_name]
        if desc:
            outarr.append(desc)
        if isinstance(scores, list):
            scorestring = 'Best scores: [ ' + ','.join(
                [pretty_num(x) for x in scores]) + ' ]'
            outarr.append(scorestring)
        if isinstance(progress, list):
            progressstring = 'Progress: [ {}/{} ]'.format(*progress)
            outarr.append(progressstring)
        if self._emcee_est_t + self._bh_est_t > 0.0:
            timestring = self.get_timestring(self._emcee_est_t +
                                             self._bh_est_t)
            outarr.append(timestring)

        print_inline(' | '.join(outarr))

    def get_timestring(self, t):
        return ('Estimated time left: [ ' + str(
            datetime.timedelta(seconds=(round_sig(t)))).rstrip('0').rstrip('.')
                + ' ]')

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
