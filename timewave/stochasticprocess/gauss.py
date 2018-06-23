
from math import sqrt, exp, log

from base import StochasticProcess


class WienerProcess(StochasticProcess):
    """
    class implementing general Gauss process between grid dates
    """

    def __init__(self, mu=0., sigma=1., start=0.):
        super(WienerProcess, self).__init__(start)
        self._mu = mu
        self._sigma = sigma

    def __str__(self):
        return 'N(mu=%0.4f, sigma=%0.4f)' % (self._mu, self._sigma)

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

    def __str__(self):
        return 'OU(theta=%0.4f, mu=%0.4f, sigma=%0.4f)' % (self._theta, self._mu, self._sigma)

    def _drift(self, x, s, e):
        if self._theta:
            return x * exp(-self._theta * float(e - s)) + self._mu * (1. - exp(-self._theta * float(e - s)))
        else:
            return x

    def _diffusion(self, x, s, e):
        # return self._sigma * (1. - exp(-self._theta * float(e - s)))
        return sqrt(self.variance(float(e - s)))

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

    def __str__(self):
        return 'LN(mu=%0.4f, sigma=%0.4f)' % (self._mu, self._sigma)

    def evolve(self, x, s, e, q):
        return x * exp(super(GeometricBrownianMotion, self).evolve(0., s, e, q))

    def mean(self, t):
        return self.start * exp(self._mu * t)

    def variance(self, t):
        return self.start ** 2 * exp(2 * self._mu * t) * (exp(self._sigma ** 2 * t) - 1)


