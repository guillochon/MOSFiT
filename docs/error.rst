.. _error:

===============
Models of error
===============

The choice of error model can affect the score a given physical model receives; an error model that better treats the expected errors (either on the model or observation side) can thus enable a better evaluation of whether a model is a good match to a given set of observations. Commonly, no error modeling is done whatsoever, with a model's fitness being judged solely upon its deviance from the observations and their reported errors (i.e. reduced chi-square).

But what if the model itself has some uncertainty? For semi-analytical approximations of complicated phenomena, most assuredly the models possess some intrinsic error. These errors may be evident in a number of ways: Perhaps a given model cannot produce enough light at a particular frequency, or has an evolution that is not fully captured by the approximation. As all semi-analytical models are prone to such issues, how do we compare two models with different (and unknown) deficiences to a given dataset?

.. _mla:

---------------------------
Maximum likelihood analysis
---------------------------

Maximum likelihood analysis (MLA) is a simple way to include the error directly in the modeling. In MLA, a variance parameter :math:`\sigma` is added to every observation. Because the chi-square metric includes :math:`\sigma` in its denominator, the increase of :math:`\sigma` comes with a cost to the overall score a model receives. As a result, optimizations of such a model will always trend towards solutions where :math:`\chi^2_{\rm red} \rightarrow 1`. The output of MLA thus answers the question of "How much additional error do I need to add to my model/observations to make the model and observations consistent with one another?"

But MLA is rather inflexible, in order to match a model to observations, it must (by construction) increase the variance for *all* observations simultaneously. For most models, this is probably overkill: the models likely deviate in *some* colors, at *some* times. What's more, MLA only allows for the white noise component of the error to expand to accomodate a model, in reality there's likely to be systematic offsets between models and data that leads to *covariant* errors.

.. _gaussian:

------------------
Gaussian processes
------------------

Gaussian processes (GP) provides an error model that addresses these shortcomings of MLA. A white noise component, equivalent to MLA, is still included, but off-diagonal covariance is explicitly modeled by considering the "distance" between observations. ``MOSFiT`` by default uses GP as its error model, with a kernel structure that is described below.

.. _kernel:

Kernel
======

The default kernel is chosen specifically to be ammenable to fitting photometric light curves. The kernel is constructed as a product of two exponential squared kernels, with the distance factors being the time of observation and the average wavelength of the filter used for the observation.
