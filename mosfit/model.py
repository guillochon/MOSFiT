"""Definitions for the `Model` class."""
import importlib
import inspect
import json
import logging
import os
from collections import OrderedDict
from difflib import get_close_matches
from math import isnan

import numpy as np
# from scipy.optimize import differential_evolution
from scipy.optimize import minimize

# from bayes_opt import BayesianOptimization
from mosfit.constants import LOCAL_LIKELIHOOD_FLOOR
from mosfit.modules.module import Module
from mosfit.printer import Printer
from mosfit.utils import listify


class Model(object):
    """Define a semi-analytical model to fit transients with."""

    MODEL_OUTPUT_DIR = 'products'
    MIN_WAVE_FRAC_DIFF = 0.1

    # class outClass(object):
    #     pass

    def __init__(self,
                 parameter_path='parameters.json',
                 model='default',
                 wrap_length=100,
                 pool=None,
                 fitter=None):
        """Initialize `Model` object."""
        self._model_name = model
        self._pool = pool
        self._is_master = pool.is_master() if pool else False
        self._wrap_length = wrap_length
        self._fitter = fitter

        if self._fitter:
            self._printer = self._fitter._printer
        else:
            self._printer = Printer(pool=pool, wrap_length=wrap_length)

        prt = self._printer

        self._dir_path = os.path.dirname(os.path.realpath(__file__))

        # Load the basic model file.
        if os.path.isfile(os.path.join('models', 'model.json')):
            basic_model_path = os.path.join('models', 'model.json')
        else:
            basic_model_path = os.path.join(self._dir_path, 'models',
                                            'model.json')

        with open(basic_model_path, 'r') as f:
            self._model = json.load(f, object_pairs_hook=OrderedDict)

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
            self._model.update(json.load(f, object_pairs_hook=OrderedDict))

        # Load model parameter file.
        model_pp = os.path.join(
            os.path.split(model_path)[0], 'parameters.json')

        pp = ''

        selected_pp = os.path.join(
            os.path.split(model_path)[0], parameter_path)

        # First try user-specified path
        if parameter_path and os.path.isfile(parameter_path):
            pp = parameter_path
        # Then try directory we are running from
        elif os.path.isfile('parameters.json'):
            pp = 'parameters.json'
        # Then try the model directory, with the user-specified name
        elif os.path.isfile(selected_pp):
            pp = selected_pp
        # Finally try model folder
        elif os.path.isfile(model_pp):
            pp = model_pp
        else:
            raise ValueError('Could not find parameter file!')

        if self._is_master:
            prt.wrapped('Basic model file:', wrap_length)
            prt.wrapped('  ' + basic_model_path, wrap_length)
            prt.wrapped('Model file:', wrap_length)
            prt.wrapped('  ' + model_path, wrap_length)
            prt.wrapped('Parameter file:', wrap_length)
            prt.wrapped('  ' + pp + '\n', wrap_length)

        with open(pp, 'r') as f:
            self._parameter_json = json.load(f, object_pairs_hook=OrderedDict)
        self._log = logging.getLogger()
        self._modules = OrderedDict()
        self._bands = []

        # Load the call tree for the model. Work our way in reverse from the
        # observables, first constructing a tree for each observable and then
        # combining trees.
        root_kinds = ['output', 'objective']

        self._trees = OrderedDict()
        self.construct_trees(self._model, self._trees, kinds=root_kinds)

        unsorted_call_stack = OrderedDict()
        self._max_depth_all = -1
        for tag in self._model:
            model_tag = self._model[tag]
            roots = []
            if model_tag['kind'] in root_kinds:
                max_depth = 0
                roots = [model_tag['kind']]
            else:
                max_depth = -1
                for tag2 in self._trees:
                    roots.extend(self._trees[tag2]['roots'])
                    depth = self.get_max_depth(tag, self._trees[tag2],
                                               max_depth)
                    if depth > max_depth:
                        max_depth = depth
                    if depth > self._max_depth_all:
                        self._max_depth_all = depth
            roots = list(set(roots))
            new_entry = model_tag.copy()
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
            mod_name = cur_task.get('class', task)
            if cur_task[
                    'kind'] == 'parameter' and task in self._parameter_json:
                cur_task.update(self._parameter_json[task])
            self._modules[task] = self._load_task_module(task)
            if mod_name == 'photometry':
                self._bands = self._modules[task].band_names()
            # This is currently not functional for MPI
            # cur_task = self._call_stack[task]
            # mod_name = cur_task.get('class', task)
            # mod_path = os.path.join('modules', cur_task['kind'] + 's',
            #                         mod_name + '.py')
            # if not os.path.isfile(mod_path):
            #     mod_path = os.path.join(self._dir_path, 'modules',
            #                             cur_task['kind'] + 's',
            #                             mod_name + '.py')
            # mod_name = ('mosfit.modules.' + cur_task['kind'] + 's.' +
            # mod_name)
            # mod = importlib.machinery.SourceFileLoader(mod_name,
            #                                            mod_path).load_module()
            # mod_class = getattr(mod, class_name)
            # if (cur_task['kind'] == 'parameter' and task in
            #         self._parameter_json):
            #     cur_task.update(self._parameter_json[task])
            # self._modules[task] = mod_class(name=task, **cur_task)
            # if mod_name == 'photometry':
            #     self._bands = self._modules[task].band_names()

    def _load_task_module(self, task, call_stack=None):
        if not call_stack:
            call_stack = self._call_stack
        cur_task = call_stack[task]
        mod_name = cur_task.get('class', task).lower()
        mod = importlib.import_module(
            '.' + 'modules.' + cur_task['kind'].lower() + 's.' + mod_name,
            package='mosfit')
        class_name = [
            x[0] for x in
            inspect.getmembers(mod, inspect.isclass)
            if issubclass(x[1], Module) and
            x[1].__module__ == mod.__name__][0]
        mod_class = getattr(mod, class_name)
        return mod_class(
            name=task, pool=self._pool, printer=self._printer, **cur_task)

    def create_data_dependent_free_parameters(
            self, variance_for_each=[], output={}):
        """Create free parameters that depend on loaded data."""
        unique_band_indices = list(set(output.get('all_band_indices', [])))
        needs_general_variance = any(
            np.array(output.get('all_band_indices', [])) < 0)

        new_call_stack = OrderedDict()
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if (cur_task.get('class', '') == 'variance' and
                    'band' in listify(variance_for_each)):
                # Find photometry in call stack.
                for ptask in self._call_stack:
                    if ptask == 'photometry':
                        awaves = self._modules[ptask].average_wavelengths(
                            unique_band_indices)
                        abands = self._modules[ptask].band_names(
                            unique_band_indices)
                        band_pairs = list(sorted(zip(awaves, abands)))
                        break
                owav = 0.0
                variance_bands = []
                for bi, (awav, band) in enumerate(band_pairs):
                    wave_frac_diff = abs(awav - owav) / (awav + owav)
                    if wave_frac_diff < self.MIN_WAVE_FRAC_DIFF:
                        continue
                    new_task_name = '-'.join([task, 'band', band])
                    new_task = cur_task.copy()
                    new_call_stack[new_task_name] = new_task
                    if 'latex' in new_task:
                        new_task['latex'] += '_{\\rm ' + band + '}'
                    new_call_stack[new_task_name] = new_task
                    self._modules[new_task_name] = self._load_task_module(
                        new_task_name, call_stack=new_call_stack)
                    owav = awav
                    variance_bands.append([awav, band])
                if needs_general_variance:
                    new_call_stack[task] = cur_task.copy()
                self._printer.wrapped(
                    'Anchoring variances for the following filters '
                    '(interpolating variances for the rest): ' +
                    (', '.join([x[1] for x in variance_bands])),
                    master_only=True)
                self._modules[ptask].set_variance_bands(variance_bands)
            else:
                new_call_stack[task] = cur_task.copy()
        self._call_stack = new_call_stack

    def determine_number_of_measurements(self):
        """Estimate the number of measurements."""
        self._num_measurements = 0
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['kind'] == 'data':
                self._num_measurements += len(
                    self._modules[task]._data['times'])

    def determine_free_parameters(self, extra_fixed_parameters):
        """Generate `_free_parameters` and `_num_free_parameters`."""
        self._free_parameters = []
        self._num_variances = 0
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if (task not in extra_fixed_parameters and
                    cur_task['kind'] == 'parameter' and
                    'min_value' in cur_task and 'max_value' in cur_task and
                    cur_task['min_value'] != cur_task['max_value']):
                self._free_parameters.append(task)
                if cur_task.get('class', '') == 'variance':
                    self._num_variances += 1
        self._num_free_parameters = len(self._free_parameters)

    def exchange_requests(self):
        """Exchange requests between modules."""
        for task in reversed(self._call_stack):
            cur_task = self._call_stack[task]
            if 'requests' in cur_task:
                requests = OrderedDict()
                reqs = cur_task['requests']
                for req in reqs:
                    if reqs[req] not in self._modules:
                        raise RuntimeError(
                            'Request cannot be satisfied because module '
                            '`{}` could not be found.'.format(reqs[req]))
                    requests[req] = self._modules[reqs[req]].send_request(req)
                self._modules[task].receive_requests(**requests)

    def frack(self, arg):
        """Perform fracking upon a single walker.

        Uses a randomly-selected global or local minimization method.
        """
        x = np.array(arg[0])
        step = 0.2
        seed = arg[1]
        np.random.seed(seed)
        my_choice = np.random.choice(range(3))
        # my_choice = 0
        my_method = ['L-BFGS-B', 'TNC', 'SLSQP'][my_choice]
        opt_dict = {'disp': False}
        if my_method in ['TNC', 'SLSQP']:
            opt_dict['maxiter'] = 100
        elif my_method == 'L-BFGS-B':
            opt_dict['maxfun'] = 5000
        # bounds = [(0.0, 1.0) for y in range(self._num_free_parameters)]
        bounds = list(
            zip(np.clip(x - step, 0.0, 1.0), np.clip(x + step, 0.0, 1.0)))

        bh = minimize(
            self.fprob,
            x,
            method=my_method,
            bounds=bounds,
            tol=1.0e-3,
            options=opt_dict)

        # bounds = list(
        #     zip(np.clip(x - step, 0.0, 1.0), np.clip(x + step, 0.0, 1.0)))
        #
        # bh = differential_evolution(
        #     self.fprob, bounds, disp=True, polish=False, maxiter=10)

        # take_step = self.RandomDisplacementBounds(0.0, 1.0, 0.01)
        # bh = basinhopping(
        #     self.fprob,
        #     x,
        #     disp=True,
        #     niter=10,
        #     take_step=take_step,
        #     minimizer_kwargs={'method': "L-BFGS-B",
        #                       'bounds': bounds})

        # bo = BayesianOptimization(self.boprob, dict(
        #     [('x' + str(i),
        #       (np.clip(x[i] - step, 0.0, 1.0),
        #        np.clip(x[i] + step, 0.0, 1.0)))
        #      for i in range(len(x))]))
        #
        # bo.explore(dict([('x' + str(i), [x[i]]) for i in range(len(x))]))
        #
        # bo.maximize(init_points=0, n_iter=20, acq='ei')
        #
        # bh = self.outClass()
        # bh.x = [x[1] for x in sorted(bo.res['max']['max_params'].items())]
        # bh.fun = -bo.res['max']['max_val']

        # m = Minuit(self.fprob)
        # m.migrad()
        return bh

    def construct_trees(self, d, trees, kinds=[], name='', roots=[], depth=0):
        """Construct call trees for each root."""
        leaf = kinds if len(kinds) else name
        if depth > 100:
            raise RuntimeError(
                'Error: Tree depth greater than 100, suggests a recursive '
                'input loop in `{}`.'.format(leaf))
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
                    if inp not in d:
                        suggests = get_close_matches(inp, d, n=1, cutoff=0.8)
                        warn_str = (
                            'Module `{}` for input to `{}` '
                            'not found!'.format(inp, leaf))
                        if len(suggests):
                            warn_str += (
                                ' Did you perhaps mean `{}`?'.
                                format(suggests[0]))
                        raise RuntimeError(warn_str)
                    children = OrderedDict()
                    self.construct_trees(
                        d,
                        children,
                        name=inp,
                        roots=new_roots,
                        depth=depth + 1)
                    trees[tag].setdefault('children', OrderedDict())
                    trees[tag]['children'].update(children)

    def draw_walker(self, test=True):
        """Draw a walker randomly.

        Draw a walker randomly from the full range of all parameters, reject
        walkers that return invalid scores.
        """
        p = None
        while p is None:
            draw = np.random.uniform(
                low=0.0, high=1.0, size=self._num_free_parameters)
            draw = [
                self._modules[self._free_parameters[i]].prior_cdf(x)
                for i, x in enumerate(draw)
            ]
            if not test:
                p = draw
                break
            score = self.likelihood(draw)
            if not isnan(score) and np.isfinite(score):
                p = draw
        return p

    def get_max_depth(self, tag, parent, max_depth):
        """Return the maximum depth a given task is found in a tree."""
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
        """Return score related to maximum likelihood."""
        outputs = self.run_stack(x, root='objective')
        return outputs['value']

    def prior(self, x):
        """Return score related to paramater priors."""
        prior = 0.0
        for pi, par in enumerate(self._free_parameters):
            lprior = self._modules[par].lnprior_pdf(x[pi])
            prior = prior + lprior
        return prior

    def boprob(self, **kwargs):
        """Score for `BayesianOptimization`."""
        x = []
        for key in sorted(kwargs):
            x.append(kwargs[key])

        l = self.likelihood(x) + self.prior(x)
        if not np.isfinite(l):
            return LOCAL_LIKELIHOOD_FLOOR
        return l

    def fprob(self, x):
        """Return score for fracking."""
        l = -(self.likelihood(x) + self.prior(x))
        if not np.isfinite(l):
            return -LOCAL_LIKELIHOOD_FLOOR
        return l

    def run_stack(self, x, root='objective'):
        """Run module stack.

        Run a stack of modules as defined in the model definition file. Only
        run functions that match the specified root.
        """
        inputs = OrderedDict()
        outputs = OrderedDict()
        pos = 0
        cur_depth = self._max_depth_all
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if root not in cur_task['roots']:
                continue
            if cur_task['depth'] != cur_depth:
                inputs = outputs
            inputs.update({'root': root})
            cur_depth = cur_task['depth']
            if task in self._free_parameters:
                inputs.update({'fraction': x[pos]})
                inputs.setdefault('fractions', []).append(x[pos])
                pos = pos + 1
            try:
                new_outs = self._modules[task].process(**inputs)
            except Exception:
                self._printer.wrapped(
                    "Failed to execute module `{}`\'s process().".format(task))
                raise
            outputs.update(new_outs)

            if cur_task['kind'] == root:
                return outputs
