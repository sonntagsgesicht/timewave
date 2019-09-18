# -*- coding: utf-8 -*-

# timewave
# --------
# timewave, a stochastic process evolution simulation engine in python.
# 
# Author:   sonntagsgesicht, based on a fork of Deutsche Postbank [pbrisk]
# Version:  0.6, copyright Wednesday, 18 September 2019
# Website:  https://github.com/sonntagsgesicht/timewave
# License:  Apache License 2.0 (see LICENSE file)


from math import sqrt, exp, log
from scipy.linalg import cholesky

from .base import MultivariateStochasticProcess
from .gauss import WienerProcess, GeometricBrownianMotion


class SABR(MultivariateStochasticProcess):
    """
    class implementing the Hagan et al SABR model
    """

    def __init__(self, alpha=.1, beta=.2, nu=.3, rho=-.2, start=.05):
        super(SABR, self).__init__((start, alpha))
        self._blend_process = GeometricBrownianMotion(0.0)
        self._vol_process = GeometricBrownianMotion(0.0, nu, alpha)
        self._beta = beta
        self._rho = rho
        self._diffusion_driver = self._blend_process.diffusion_driver + self._vol_process.diffusion_driver

    def __str__(self):
        alpha = self._vol_process.start
        beta = self._beta
        nu = self._vol_process._sigma
        rho = self._rho
        return 'SABR(alpha=%0.4f, beta=%0.4f, nu=%0.4f, rho=%0.4f)' % (alpha, beta, nu, rho)

    def evolve(self, x, s, e, q):
        f, a = x
        if e - s > 0:
            q1, q2 = q
            q2 = self._rho * q1 + sqrt(1. - self._rho ** 2) * q2
            a = self._vol_process.evolve(a, s, e, q2)
            sgn = -1. if f < 0. else 1.
            f += sgn * abs(f) ** self._beta * sqrt(e - s) * a * q1
        return f, a

    def mean(self, t):
        return self.start[0]

    def variance(self, t):
        return self._vol_process.variance(t) * t  # todo give better sabr variance proxy


class MultiGauss(MultivariateStochasticProcess):
    """
    class implementing multi dimensional brownian motion
    """

    def __init__(self, mu=list([0.]), covar=list([[1.]]), start=list([0.])):
        super(MultiGauss, self).__init__(start)
        self._mu = mu
        self._dim = len(start)
        self._cholesky = None if covar is None else cholesky(covar).T
        self._variance = [1.] * self._dim if covar is None else [covar[i][i] for i in range(self._dim)]
        self._diffusion_driver = [WienerProcess(m, sqrt(s)) for m, s in zip(self._mu, self._variance)]

    def __str__(self):
        cov = self._cholesky.T * self._cholesky
        return '%d-MultiGauss(mu=%s, cov=%s)' % (len(self), str(self._mu), str(cov))

    def _drift(self, x, s, e):
        return [m * (e - s) for m in self._mu]

    def evolve(self, x, s, e, q):
        dt = sqrt(e - s)
        q = [qq * dt for qq in q]
        q = list(self._cholesky.dot(q))
        d = self._drift(x, s, e)
        return [xx + dd + qq for xx, dd, qq in zip(x, d, q)]

    def mean(self, t):
        return [s + m * t for s, m in zip(self.start, self._mu)]

    def variance(self, t):
        return [v * t for v in self._variance]
