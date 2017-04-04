"""Overridden PTSampler with random Gibbs selection, more-reliable acor."""
import numpy as np
from emcee.autocorr import AutocorrError, function
from emcee.ptsampler import PTLikePrior, PTSampler


class MOSSampler(PTSampler):
    """Override PTSampler methods."""

    def get_autocorr_time(
            self, min_step=0, max_walkers=-1, chain=[], **kwargs):
        """Return a matrix of autocorrelation lengths.

        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.
        Any arguments will be passed to :func:`autocorr.integrate_time`.
        """
        acors = np.zeros((self.ntemps, self.dim))

        for i in range(self.ntemps):
            acors[i, :] = 0.0
            if len(chain):
                x = chain[i, :max_walkers, min_step:, :]
            else:
                x = self._chain[i, :max_walkers, min_step:, :]
            for w in x:
                acors[i, :] += self.integrated_time(w, **kwargs)
            acors[i, :] /= len(x)
        return acors

    def integrated_time(
        self, x, low=10, high=None, step=1, c=5, full_output=False,
            axis=0, fast=False):
        """Estimate the integrated autocorrelation time of a time series.

        This estimate uses the iterative procedure described on page 16 of
        `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
        determine a reasonable window size.

        Args:
            x: The time series. If multidimensional, set the time axis using
                the ``axis`` keyword argument and the function will be
                computed for every other axis.
            low (Optional[int]): The minimum window size to test. (default:
                ``10``)
            high (Optional[int]): The maximum window size to test. (default:
                ``x.shape[axis] / (2*c)``)
            step (Optional[int]): The step size for the window search.
                (default: ``1``)
            c (Optional[float]): The minimum number of autocorrelation times
                needed to trust the estimate. (default: ``10``)
            full_output (Optional[bool]): Return the final window size as well
                as the autocorrelation time. (default: ``False``)
            axis (Optional[int]): The time axis of ``x``. Assumed to be the
                first axis if not specified.
            fast (Optional[bool]): If ``True``, only use the first ``2^n`` (for
                the largest power) entries for efficiency. (default: False)

        Returns:
            float or array: An estimate of the integrated autocorrelation time
                of the time series ``x`` computed along the axis ``axis``.
            Optional[int]: The final window size that was used. Only returned
                if ``full_output`` is ``True``.

        Raises
            AutocorrError: If the autocorrelation time can't be reliably
                estimated from the chain. This normally means that the chain
                is too short.

        """
        size = 0.5 * x.shape[axis]
        if int(c * low) >= size:
            raise AutocorrError("The chain is too short")

        # Compute the autocorrelation function.
        f = function(x, axis=axis, fast=fast)

        # Check the dimensions of the array.
        oned = len(f.shape) == 1
        m = [slice(None), ] * len(f.shape)

        # Loop over proposed window sizes until convergence is reached.
        if high is None:
            high = int(size / c)
        for M in np.arange(low, high, step).astype(int):
            # Compute the autocorrelation time with the given window.
            if oned:
                # Special case 1D for simplicity.
                tau = 1 + 2 * np.sum(f[1:M])
            else:
                # N-dimensional case.
                m[axis] = slice(1, M)
                tau = 1 + 2 * np.sum(f[m], axis=axis)

            # Accept the window size if it satisfies the convergence criterion.
            if np.all(tau > 1.0) and M > c * tau.max():
                if full_output:
                    return tau, M
                return tau

            # If the autocorrelation time is too long to be estimated reliably
            # from the chain, it should fail.
            if c * tau.max() >= size:
                break

        raise AutocorrError("The chain is too short to reliably estimate "
                            "the autocorrelation time")

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
            for n in super(MOSSampler, self).sample(
                    p0, lnprob0, lnlike0, iterations, thin, storechain):
                yield n
            return

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
                    qs[k, :, :] = psample[k, :, :]
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
