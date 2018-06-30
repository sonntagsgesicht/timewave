

class StochasticProcess(object):

    @classmethod
    def random(cls):
        return cls()

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

    def __init__(self, start):
        self.start = start
        self._diffusion_driver = self

    def __len__(self):
        return len(self.diffusion_driver)

    def __str__(self):
        return self.__class__.__name__ + '()'

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

    def mean(self, t):
        """ expected value of time :math:`t` increment

        :param t:
        :rtype float
        :return:
        """
        return 0.0

    def variance(self, t):
        """ second central moment of time :math:`t` increment

        :param t:
        :rtype float
        :return:
        """
        return 0.0


class MultivariateStochasticProcess(StochasticProcess):
    pass
