# -*- coding: utf-8 -*-

# timewave
# --------
# timewave, a stochastic process evolution simulation engine in python.
# 
# Author:   sonntagsgesicht, based on a fork of Deutsche Postbank [pbrisk]
# Version:  0.6, copyright Wednesday, 18 September 2019
# Website:  https://github.com/sonntagsgesicht/timewave
# License:  Apache License 2.0 (see LICENSE file)


"""
module containing stochastic process model producer
"""

from math import sqrt

from scipy.linalg import cholesky

from .engine import Producer, State
from .producers import MultiProducer

from .indexedmatrix import IndexMatrix

# producer


class GaussEvolutionFunctionProducer(Producer):
    """
    class implementing general Gauss process between grid dates
    """

    def __init__(self, func=None, initial_state=None, length=None):
        """
        :param callable func: evolve function, e.g. `lambda x, s, e, q: x + sqrt(e - s) * q` by default
            with `x` current state value, `s` current point in time, i.e. start point of next evolution step,
            `e` next point in time, i.e. end point of evolution step, `q` standard normal random number to do step
        :param initial_state: initial state (value) of evolution,
        :param int or None length: length of `q` as a list of Gauss random numbers,
            if `None` or `0` the evolution function `func` will be invoked with `q`
            not as a list but a float random number.

        class implementing general Gauss process between grid dates and provides state to any evolve style function
        `foo(x, s, e, q)` with `x` last state, `s` last state time, `e` current point in time and
        `q` current Gauss process state
        """
        if func is None:
            func = (lambda x, s, e, q: x + sqrt(e - s) * q)
        self._len = length
        super(GaussEvolutionFunctionProducer, self).__init__(func, initial_state)

    def __len__(self):
        return 0 if self._len is None else self._len

    def evolve(self, new_date):
        """
        evolve to the new process state at the next date

        :param date new_date: date or point in time of the new state
        :return State:
        """
        if self.state.date == new_date and not self.initial_state.date == new_date:
            return self.state
        if self._len:
            q = [self.random.gauss(0., 1.) for _ in range(int(self._len))]
        else:
            q = self.random.gauss(0., 1.)
        self.state.value = self.func(self.state.value, self.state.date, new_date, q)
        self.state.date = new_date
        return self.state


class GaussEvolutionProducer(GaussEvolutionFunctionProducer):
    """
    producer to bring diffusion process to life
    """
    def __init__(self, process):
        """
        :param StochasticProcess process: diffusion process to evolve

        """
        self.process = process
        self.diffusion_driver = process.diffusion_driver
        length = len(process) if len(process) > 1 else None
        super(GaussEvolutionProducer, self).__init__(process.evolve, State(process.start), length)


class _FakeGaussRandom(list):

    def seed(self, *args):
        pass

    def gauss(self, *args):
        return self.pop(0)


class CorrelatedGaussEvolutionProducer(MultiProducer):
    """
    class implementing general correlated Gauss process between grid dates
    """

    def __init__(self, producers, correlation=None, diffusion_driver=None):
        """

        :param list(GaussEvolutionProducer) producers: list of producers to evolve
        :param list(list(float)) or dict((StochasticProcess, StochasticProcess): float) or None correlation:
            correlation matrix of underlying multivariate Gauss process of diffusion drivers.
            If `dict` keys must be pairs of diffusion drivers, diagonal and zero entries can be omitted.
            If not give, all drivers evolve independently.
        :param list(StochasticProcess) or None diffusion_driver: list of diffusion drivers
            indexing the correlation matrix. If not given and `correlation` is not an IndexMatrix,
            e.g. comes already with list of drivers, it is assumed that each process producer has different drivers
            and the correlation is order in the same way.
        """

        # faking random().gauss()
        for p in producers:
            p.random = _FakeGaussRandom()

        super(CorrelatedGaussEvolutionProducer, self).__init__(producers)

        # build correlation from sparse matrix (dict)
        if isinstance(correlation, dict):
            if diffusion_driver is not None:
                raise ValueError("")

            # collect all drives in correlation
            drivers = set()
            for i, j in list(correlation.keys()):
                drivers.add(i)
                drivers.add(j)
            drivers = tuple(drivers)

            list_correlation = [[0.0] * len(drivers) for _ in drivers]
            for i in range(len(list_correlation)):
                list_correlation[i][i] = 1.

            # fill sparse correlation matrix
            for rf_1 in drivers:
                for rf_2 in drivers:
                    if (rf_1, rf_2) in correlation:
                        if (rf_2, rf_1) in correlation:
                            if not correlation[rf_1, rf_2] == correlation[rf_2, rf_1]:
                                _ = rf_1, rf_2
                                raise ValueError("Correlation data must be symmetric. Input at [%d, %d] is not." % _)
                        i, j = drivers.index(rf_1), drivers.index(rf_2)
                        list_correlation[i][j] = correlation[rf_1, rf_2]
                        list_correlation[j][i] = correlation[rf_1, rf_2]
            correlation = list_correlation
            diffusion_driver = drivers

        self._correlation = correlation

        # build valid driver list (according correlation and producers)
        if correlation and not diffusion_driver:
            # no given diffusion_drivers nor keys in correlation
            drivers = list()
            for p in producers:
                for d in p.diffusion_driver:
                    if d not in drivers:
                        drivers.append(d)
            diffusion_driver = tuple(drivers)
            # require enough correlation because here we cannot decide which drivers
            # should be independent or omitted
            if not len(diffusion_driver) == len(correlation):
                raise AssertionError("Correlation dimension must meet number of diffusion drivers.")

        self._diffusion_driver = () if diffusion_driver is None else diffusion_driver

        # build index lists for each producers diffusion drivers
        self._driver_index = dict.fromkeys(producers, [])
        for p in producers:
            dd = [d for d in self._diffusion_driver if d in p.diffusion_driver]
            self._driver_index[p] = [diffusion_driver.index(d) for d in dd]

        # build cholesky
        self._cholesky = cholesky(correlation).T if correlation else None

    def __len__(self):
        l = 0
        for p in self.producers:
            l += 1 if len(p) == 0 else len(p)
        return l

    def evolve(self, new_date):
        """
        evolve to the new process state at the next date

        :param date new_date: date or point in time of the new state
        :return State:
        """
        if all(p.state.date == new_date for p in self.producers):
            return [p.state for p in self.producers]

        if self._cholesky is not None:
            q = [self.random.gauss(0., 1.) for _ in range(len(self._cholesky))]
            q = list(self._cholesky.dot(q))
        else:
            q = list()

        state = list()
        for p in self.producers:
            if len(self._driver_index[p]) == len(p.diffusion_driver):
                qq = [q[i] for i in self._driver_index[p]]
            elif len(self._driver_index[p]) < len(p.diffusion_driver):
                qq = list()
                for d in p.diffusion_driver:
                    qqq = q[self._diffusion_driver.index(d)] if d in self._diffusion_driver else self.random.gauss(0., 1.)
                    qq.append(qqq)
            else:
                qq = [self.random.gauss(0., 1.) for _ in p.diffusion_driver]

            p.random.extend(qq)
            state.append(p.evolve(new_date))

        return state


class MultiGaussEvolutionProducer(CorrelatedGaussEvolutionProducer):
    """
    class implementing multi variant GaussEvolutionProducer
    """

    def __init__(self, process_list, correlation=None, diffusion_driver=None):
        producers = [GaussEvolutionProducer(p) for p in process_list]
        super(MultiGaussEvolutionProducer, self).__init__(producers, correlation, diffusion_driver)
