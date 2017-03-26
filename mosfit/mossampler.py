"""Overridden PTSampler with random Gibbs selection."""
import emcee
import numpy as np
from emcee.ptsampler import PTLikePrior, PTSampler


class MOSSampler(PTSampler):
    """Override PTSampler methods."""

    def get_autocorr_time(self, min_step=0, chain=[], **kwargs):
        """Return a matrix of autocorrelation lengths.

        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.
        Any arguments will be passed to :func:`autocorr.integrate_time`.
        """
        acors = np.zeros((self.ntemps, self.dim))

        for i in range(self.ntemps):
            if len(chain):
                x = np.mean(chain[i, :, min_step:, :], axis=0)
            else:
                x = np.mean(self._chain[i, :, min_step:, :], axis=0)
            acors[i, :] = emcee.autocorr.integrated_time(x, **kwargs)
        return acors

    def sample(
        self, p0, lnprob0=None, lnlike0=None, iterations=1,
            thin=1, storechain=True, gibbs=False):
        """Advance the chains ``iterations`` steps as a generator.

        :param p0:
            The initial positions of the walkers.  Shape should be
            ``(ntemps, nwalkers, dim)``.
        :param lnprob0: (optional)
            The initial posterior values for the ensembles.  Shape
            ``(ntemps, nwalkers)``.
        :param lnlike0: (optional)
            The initial likelihood values for the ensembles.  Shape
            ``(ntemps, nwalkers)``.
        :param iterations: (optional)
            The number of iterations to preform.
        :param thin: (optional)
            The number of iterations to perform between saving the
            state to the internal chain.
        :param storechain: (optional)
            If ``True`` store the iterations in the ``chain``
            property.
        At each iteration, this generator yields
        * ``p``, the current position of the walkers.
        * ``lnprob`` the current posterior values for the walkers.
        * ``lnlike`` the current likelihood values for the walkers.
        """
        if not gibbs:
            super(MOSSampler, self).sample(
                p0, lnprob0, lnlike0, iterations, thin, storechain)

        p = np.copy(np.array(p0))

        # If we have no lnprob or logls compute them
        if lnprob0 is None or lnlike0 is None:
            fn = PTLikePrior(self.logl, self.logp, self.loglargs,
                             self.logpargs, self.loglkwargs, self.logpkwargs)
            if self.pool is None:
                results = list(map(fn, p.reshape((-1, self.dim))))
            else:
                results = list(self.pool.map(fn, p.reshape((-1, self.dim))))

            logls = np.array([r[0] for r in results]).reshape((self.ntemps,
                                                               self.nwalkers))
            logps = np.array([r[1] for r in results]).reshape((self.ntemps,
                                                               self.nwalkers))

            lnlike0 = logls
            lnprob0 = logls * self.betas.reshape((self.ntemps, 1)) + logps

        lnprob = lnprob0
        logl = lnlike0

        # Expand the chain in advance of the iterations
        if storechain:
            nsave = iterations // thin
            if self._chain is None:
                isave = 0
                self._chain = np.zeros((self.ntemps, self.nwalkers, nsave,
                                        self.dim))
                self._lnprob = np.zeros((self.ntemps, self.nwalkers, nsave))
                self._lnlikelihood = np.zeros((self.ntemps, self.nwalkers,
                                               nsave))
            else:
                isave = self._chain.shape[2]
                self._chain = np.concatenate((self._chain,
                                              np.zeros((self.ntemps,
                                                        self.nwalkers,
                                                        nsave, self.dim))),
                                             axis=2)
                self._lnprob = np.concatenate((self._lnprob,
                                               np.zeros((self.ntemps,
                                                         self.nwalkers,
                                                         nsave))),
                                              axis=2)
                self._lnlikelihood = np.concatenate((self._lnlikelihood,
                                                     np.zeros((self.ntemps,
                                                               self.nwalkers,
                                                               nsave))),
                                                    axis=2)

        for i in range(iterations):
            thawed = np.array(sorted(np.random.choice(
                self.dim, np.random.randint(1, self.dim), replace=False)))
            for j in [0, 1]:
                jupdate = j
                jsample = (j + 1) % 2

                pupdate = p[:, jupdate::2, :]
                psample = p[:, jsample::2, :]

                zs = np.exp(np.random.uniform(
                    low=-np.log(self.a), high=np.log(self.a),
                    size=(self.ntemps, self.nwalkers // 2)))

                qs = np.zeros((self.ntemps, self.nwalkers // 2, self.dim))
                for k in range(self.ntemps):
                    js = np.random.randint(0, high=self.nwalkers // 2,
                                           size=self.nwalkers // 2)
                    yrange = np.arange(self.nwalkers // 2).reshape(-1, 1)
                    qs[k, :, :] = psample[k, js, :]
                    qs[k, yrange,
                       thawed] = psample[
                        k, js.reshape(-1, 1), thawed] + zs[k, :].reshape(
                            (self.nwalkers // 2, 1)) * (
                                pupdate[k, yrange, thawed] -
                                psample[k, js.reshape(-1, 1), thawed])

                fn = PTLikePrior(self.logl, self.logp, self.loglargs,
                                 self.logpargs, self.loglkwargs,
                                 self.logpkwargs)
                if self.pool is None:
                    results = list(map(fn, qs.reshape((-1, self.dim))))
                else:
                    results = list(self.pool.map(fn, qs.reshape((-1,
                                                                 self.dim))))

                qslogls = np.array([r[0] for r in results]).reshape(
                    (self.ntemps, self.nwalkers // 2))
                qslogps = np.array([r[1] for r in results]).reshape(
                    (self.ntemps, self.nwalkers // 2))
                qslnprob = qslogls * self.betas.reshape((self.ntemps, 1)) \
                    + qslogps

                logpaccept = self.dim * np.log(zs) + qslnprob \
                    - lnprob[:, jupdate::2]
                logrs = np.log(np.random.uniform(low=0.0, high=1.0,
                                                 size=(self.ntemps,
                                                       self.nwalkers // 2)))

                accepts = logrs < logpaccept
                accepts = accepts.flatten()

                pupdate.reshape((-1, self.dim))[accepts, :] = \
                    qs.reshape((-1, self.dim))[accepts, :]
                lnprob[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslnprob.reshape((-1,))[accepts]
                logl[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslogls.reshape((-1,))[accepts]

                accepts = accepts.reshape((self.ntemps, self.nwalkers // 2))

                self.nprop[:, jupdate::2] += 1.0
                self.nprop_accepted[:, jupdate::2] += accepts

            p, lnprob, logl = self._temperature_swaps(p, lnprob, logl)

            if (i + 1) % thin == 0:
                if storechain:
                    self._chain[:, :, isave, :] = p
                    self._lnprob[:, :, isave, ] = lnprob
                    self._lnlikelihood[:, :, isave] = logl
                    isave += 1

            yield p, lnprob, logl
