# coding=utf-8
"""
module containing brownian motion model related classes
"""

from scipy.linalg import cholesky

from math import sqrt, exp, log


class StochasticProcess(object):
    def __init__(self, start):
        self.start = start
        self._diffusion_driver = self

    def __len__(self):
        return len(self.diffusion_driver)

    @property
    def diffusion_driver(self):
        """
            diffusion driver are the underlying `dW` of each process `X` in a SDE like `dX = m dt + s dW`
        :return list(StochasticProcess):
        """
        if self._diffusion_driver is None:
            return self,
        if isinstance(self._diffusion_driver, list):
            return tuple(self._diffusion_driver)
        if isinstance(self._diffusion_driver, tuple):
            return self._diffusion_driver
        return self._diffusion_driver,  # return as a tuple

    def evolve(self, x, s, e, q):
        """
        :param float x: current state value, i.e. value before evolution step
        :param float s: current point in time, i.e. start point of next evolution step
        :param float e: next point in time, i.e. end point of evolution step
        :param float q: standard normal random number to do step
        :return float: next state value, i.e. value after evolution step

        evolves process state `x` from `s` to `e` in time depending of standard normal random variable `q`
        """
        return 0.0


class MultivariateStochasticProcess(StochasticProcess):

    pass


class WienerProcess(StochasticProcess):
    """
    class implementing general Gauss process between grid dates
    """

    def __init__(self, mu=0., sigma=1., start=0.):
        super(WienerProcess, self).__init__(start)
        self._mu = mu
        self._sigma = sigma

    def _drift(self, x, s, e):
        return self._mu * (e - s)

    def _diffusion(self, x, s, e):
        return self._sigma * sqrt(e - s)

    def evolve(self, x, s, e, q):
        return x + self._drift(x, s, e) + self._diffusion(x, s, e) * q

    def mean(self, t):
        return self.start + self._mu * t

    def variance(self, t):
        return self._sigma ** 2 * t


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """
    class implementing Ornstein Uhlenbeck process

    """

    def __init__(self, theta=0.1, mu=0.1, sigma=0.1, start=0.0):
        r"""

        :param flaot theta: mean reversion speed
        :param float mu: drift
        :param float sigma: diffusion
        :param float start: initial value

        .. math::    dx_t = \theta ( \mu - x_t) dt + \sigma dW_t, x_0 = a

        """
        super(OrnsteinUhlenbeckProcess, self).__init__(start)
        self._theta = theta
        self._mu = mu
        self._sigma = sigma

    def _drift(self, x, s, e):
        if self._theta:
            return x * exp(-self._theta * float(e - s)) + self._mu * (1. - exp(-self._theta * float(e - s)))
        else:
            return x

    def _diffusion(self, x, s, e):
        # return self._sigma * (1. - exp(-self._theta * float(e - s)))
        return sqrt(self.variance(float(e-s)))

    def evolve(self, x, s, e, q):
        return self._drift(x, s, e) + self._diffusion(x, s, e) * q

    def mean(self, t):
        return self._drift(self.start, 0., t)

    def variance(self, t):
        if self._theta:
            return self._sigma * self._sigma * (1. - exp(-2. * self._theta * t)) * .5 / self._theta
        else:
            return t


class GeometricBrownianMotion(WienerProcess):
    """
    class implementing general Gauss process between grid dates
    """

    def __init__(self, mu=0., sigma=1., start=1.):
        super(GeometricBrownianMotion, self).__init__(mu, sigma, start)
        self._diffusion_driver = super(GeometricBrownianMotion, self).diffusion_driver

    def evolve(self, x, s, e, q):
        return x * exp(super(GeometricBrownianMotion, self).evolve(0., s, e, q))

    def mean(self, t):
        return self.start * exp(self._mu * t)

    def variance(self, t):
        return self.start ** 2 * exp(2 * self._mu * t) * (exp(self._sigma ** 2 * t) - 1)


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

    def evolve(self, x, s, e, q):
        f, a = x
        if e-s > 0:
            q1, q2 = q
            q2 = self._rho * q1 + sqrt(1.-self._rho ** 2) * q2
            a = self._vol_process.evolve(a, s, e, q2)
            sgn = -1. if f < 0. else 1.
            f += sgn * abs(f) ** self._beta * sqrt(e-s) * a * q1
        return f, a

    def mean(self, t):
        return self.start[0]

    def variance(self, t):
        return self._vol_process.variance(t) * t   # todo give better sabr variance proxy


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
