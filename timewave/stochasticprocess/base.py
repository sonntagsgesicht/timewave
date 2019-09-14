from math import sqrt


class StochasticProcess(object):

    @classmethod
    def random(cls):
        """ initializes stochastic process with some randomly generated parameters

        :rtype: StochasticProcess

        """
        return cls(None)

    @property
    def diffusion_driver(self):
        """ diffusion driver are the underlying `dW` of each process `X` in a SDE like `dX = m dt + s dW`

        :return list(StochasticProcess):

        """
        if self._diffusion_driver is None:
            return self,
        if isinstance(self._diffusion_driver, list):
            return tuple(self._diffusion_driver)
        if isinstance(self._diffusion_driver, tuple):
            return self._diffusion_driver
        return self._diffusion_driver,  # return as a tuple

    def __init__(self, start):
        """
            base class for stochastic process :math:`X`, e.g. Wiener process :math:`W` or Markov chain :math:`M`

        :param start: initial state :math:`X_0`

        """
        super(StochasticProcess, self).__init__()
        self.start = start
        self._diffusion_driver = self

    def __len__(self):
        return len(self.diffusion_driver)

    def __str__(self):
        return self.__class__.__name__ + '()'

    def evolve(self, x, s, e, q):
        """ evolves process state `x` from `s` to `e` in time depending of standard normal random variable `q`

        :param object x: current state value, i.e. value before evolution step
        :param float s: current point in time, i.e. start point of next evolution step
        :param float e: next point in time, i.e. end point of evolution step
        :param float q: standard normal random number to do step

        :return: next state value, i.e. value after evolution step
        :rtype: object

        """
        return 0.0

    def mean(self, t):
        """ expected value of time :math:`t` increment

        :param float t:
        :rtype: float
        :return: expected value of time :math:`t` increment

        """
        return 0.0

    def median(self, t):
        return 0.0

    def variance(self, t):
        """ second central moment of time :math:`t` increment

        :param float t:
        :rtype: float
        :return: variance, i.e. second central moment of time :math:`t` increment

        """
        return 0.0

    def stdev(self, t):
        return sqrt(self.variance(t))

    def skewness(self, t):
        return 0.0

    def kurtosis(self, t):
        return 0.0


class MultivariateStochasticProcess(StochasticProcess):
    pass
