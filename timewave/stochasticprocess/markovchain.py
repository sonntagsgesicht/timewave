
import numpy as np

from scipy.stats import norm
from scipy.linalg import expm, logm

from base import StochasticProcess


EPS = 1e-7


class FiniteStateMarkovChain(StochasticProcess):
    @classmethod
    def random(cls, d=5):
        # pick random vector and matrix with positive values
        s = np.random.random((1, d))
        t = np.random.random((d, d))

        # turn them into stochastic ones
        s = s / s.sum(1)
        t = t / t.sum(1).reshape((d, 1))

        # build process instance
        return cls(transition=t.tolist(), start=list(s.flat))

    def __init__(self, transition=None, start=None):
        """

        :param list transition: stochastic matrix of transition probabilities,
                                i.e. np.ndarray with shape=2 and sum of each line equal to 1
                                (optional) default: identity matrix
        :param list start: initial state distribution, i.e. np.ndarray with shape=1 or list, adding up to 1,
                           (optional) default: unique distribution
        """

        # if any argument is missing use defaults
        if transition is None and start is None:
            dim = 2
        else:
            dim = len(transition) if start is None else len(start)

        start = (np.ones(dim) / dim).tolist() if start is None else list(start)
        transition = np.identity(dim) if transition is None else transition
        self._transition_matrix = np.matrix(transition, float)

        super(FiniteStateMarkovChain, self).__init__(start)

        # validate argument shapes in shapes and sum for being stochastic
        assert abs(self._transition_matrix.sum(1) - 1.).sum() < EPS, \
            'transition probabilities do not from distribution.\n' + str(self._transition_matrix)
        assert abs(sum(self.start) - 1.) < EPS, \
            'start does not from distribution.\n' + str(self.start)
        assert (len(self.start), len(self.start)) == self._transition_matrix.shape, \
            'dimension of transition and start argument must meet.\n' \
            + str(self.start) + '\n' + str(self._transition_matrix)

    def __len__(self):
        return len(self.start)

    def _m_pow(self, t, s=0.):
        return self._transition_matrix ** (t - s)

    def evolve(self, x, s, e, q):
        p = norm.cdf(q)
        m = self._m_pow(e, s)
        mm = np.greater(m.cumsum(1), p) * 1.
        mm[0:, 1:] -= mm[0:, :-1]
        return list(np.asarray(x, float).dot(mm).flat)

    def mean(self, t):
        s = np.asarray(self.start, float)
        m = self._m_pow(t)
        return list(s.dot(m).flat)

    def variance(self, t):
        def e_xx(prop, cum_prop):
            b = list()
            for p, c in zip(prop, cum_prop):
                b.append([max(0., min(c, d) - max(c - p, d - q)) for q, d in zip(prop, cum_prop)])
            return np.matrix(b)

        s = np.asarray(self.start, float)
        m = self._m_pow(t)
        v = list()
        for prop, cum_prop, mean in zip(m.T, m.cumsum(1).T, self.mean(t)):
            exx_m = e_xx(list(prop.flat), list(cum_prop.flat))
            exx = s.dot(exx_m).dot(s.T)[0, 0]
            varx = exx - mean ** 2
            varx = max(0., varx) if varx > -EPS else varx  # avoid numerical negatives
            v.append(varx)
        assert min(v) >= 0., 'Got negative variance: ' + str(v)
        return v


class FiniteStateContinuousTimeMarkovChain(FiniteStateMarkovChain):

    def __init__(self, transition=None, start=None):
        super(FiniteStateContinuousTimeMarkovChain, self).__init__(transition, start)
        self._transition_generator = logm(self._transition_matrix)

    def _m_pow(self, t, s=0.):
        return expm(self._transition_generator * (t - s))


class FiniteStateInhomogenuousMarkovChain(FiniteStateMarkovChain):
    @classmethod
    def random(cls, d=5, l=3):
        s = np.random.random((1, d))  # pick random vector or matrix with positive values
        s = s / s.sum(1)  # turn them into stochastic ones

        g = list()
        for _ in range(l):
            t = np.random.random((d, d))  # pick random vector or matrix with positive values
            t = t / t.sum(1).reshape((d, 1))  # turn them into stochastic ones
            g.append(t)

        # build process instance
        return cls(transition=g, start=list(s.flat))

    def __init__(self, transition=[None], start=None):
        super(FiniteStateInhomogenuousMarkovChain, self).__init__(transition.pop(-1), start)
        self._transition_grid = transition

    def _m_pow(self, t, s=0.):
        l = len(self._transition_grid)
        # use transition grid at start
        n = np.identity(len(self), float)
        for i in range(min(s, l), min(t, l)):
            n *= self._transition_grid[i]
        # use super beyond the transition grid
        return n * super(FiniteStateInhomogenuousMarkovChain, self)._m_pow(t, min(t, max(s, l)))

