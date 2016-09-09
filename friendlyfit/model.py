import importlib
import json
import logging
from logging import DEBUG, INFO, WARNING

import numpy as np

import emcee


class Model:
    def __init__(self, parameters=[], path='example_model.json'):
        self._path = path
        self._model_dict = json.loads(path)
        self._parameters = parameters
        self._log = logging.getLogger()

        # Load the call tree for the model. Work our way in reverse from the
        # observables, first constructing a tree for each observable and then
        # combining trees.
        trees = []
        trees = self.construct_tree(trees, 'observable')

        print(trees)
        raise (SystemExit)

        # try:
        #     mod = importlib.import_module('.' + mod_name, package='astrocats')
        # except Exception as err:
        #     self._log.error("Import of specified module '{}' failed.".format(mod_name))

    def construct_tree(self, trees, find_tag, depth=0):
        for tag, entry in self._model_dict:
            if entry[tag]['kind'] == find_tag:
                trees.append(entry)
                for inp in entry[tag]['inputs']:
                    trees.append(self.construct_tree(trees, inp, depth + 1))

    def lnprob(self, x, ivar):
        parameters = {}
        for par in self._parameters:
            parameters[par.name] = (
                ivar *
                (par['max_value'] - par['min_value']) + par['min_value'])
        return -0.5 * np.sum(ivar * x**2)

    def fit_data(self, data, plot_times=[]):
        ndim, nwalkers = 10, 100
        ivar = 1. / np.random.rand(ndim)
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.lnprob, args=[ivar])
        sampler.run_mcmc(p0, 1000)
