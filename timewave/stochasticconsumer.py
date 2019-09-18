# -*- coding: utf-8 -*-

# timewave
# --------
# timewave, a stochastic process evolution simulation engine in python.
# 
# Author:   sonntagsgesicht, based on a fork of Deutsche Postbank [pbrisk]
# Version:  0.6, copyright Wednesday, 18 September 2019
# Website:  https://github.com/sonntagsgesicht/timewave
# License:  Apache License 2.0 (see LICENSE file)


from math import sqrt
from random import sample

from .consumers import TransposedConsumer


# statistics and stochastic process consumers


class _Statistics(object):
    """
    calculate basic statistics for a 1 dim empirical sample
    """
    _available = 'count', 'mean', 'stdev', 'variance', 'skewness', 'kurtosis', 'median', 'min', 'max'

    def get(self, item, default=None):
        return getattr(self, item, default)

    def keys(self):
        return self._available

    def values(self):
        return tuple(getattr(self, a, 0.0) for a in list(self.keys()))

    def items(self):
        return list(zip(list(self.keys()), list(self.values())))

    def __init__(self, data, description='', **expected):
        sps = sorted(data)
        l = float(len(sps))
        p = [int(i * l * 0.01) for i in range(100)]
        m1 = self._moment(sps)
        cm2 = self._moment(sps, 2, m1)
        cm3 = self._moment(sps, 3, m1)
        cm4 = self._moment(sps, 4, m1)
        self.description = description
        self.count = len(sps)
        self.mean = m1
        self.variance = 0. if len(set(sps)) == 1 else cm2 * l / (l - 1.)
        self.stdev = sqrt(self.variance)
        self.skewness = 0. if len(set(sps)) == 1 else cm3 / (self.stdev ** 3)
        self.kurtosis = 0. if len(set(sps)) == 1 else cm4 / cm2 ** 2 - 3.
        self.median = sps[p[50]]
        self.min = sps[0]
        self.max = sps[-1]
        self.box = [sps[0], sps[p[25]], sps[p[50]], sps[p[75]], sps[-1]]
        self.percentile = [sps[int(i)] for i in p]
        self.sample = data
        process, time = expected.get('process'), expected.get('time', 1.)
        if process:
            expected.update(dict((k, getattr(process, k)(time)) for k in self._available if hasattr(process, k)))
        self.expected = expected

    @staticmethod
    def _moment(data, degree=1, mean=0.):
        return sum([(rr - mean) ** degree for rr in data]) / float(len(data))

    def __contains__(self, item):
        return item in list(self.keys())

    def __iter__(self):
        return list(self.keys())

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        return '\n' + str(self)

    def __str__(self):
        f = (lambda v: '%0.8f' % v if isinstance(v, (int, float)) else '')
        keys, values = list(self.keys()), list(map(f, list(self.values())))
        mk, mv = max(list(map(len, keys))), max(list(map(len, values)))
        res = [a.ljust(mk) + ' : ' + v.rjust(mv) for a, v in zip(keys, values)]

        if self.expected:
            for l, k in enumerate(self.keys()):
                if k in self.expected:
                    e, v = self.expected[k], getattr(self, k)
                    # diff v - e
                    res[l] += ' - ' + ('%0.8f' % e).rjust(mv) + ' = ' + ('%0.8f' % (v - e)).rjust(mv)
                    if e:  # rel diff if non zero e
                        res[l] += '  (' + ('%+0.3f' % (100. * (v - e) / e)).rjust(mv) + ' %)'

        res = [self.__class__.__name__ + '(' + self.description + ')'] + res
        return '\n'.join(res)


class _MetaStatistics(list):
    _available = 'mean', 'stdev', 'variance', 'skewness', 'kurtosis', 'median'

    def __init__(self, iterable, **expected):
        super(_MetaStatistics, self).__init__(iterable)
        self.expected = expected

    def keys(self):
        return self.__class__._available

    def values(self):
        return list(v for k, v in list(self.items()))

    def items(self):
        keys = list(self.keys())
        data = dict((k, list()) for k in keys)
        for s in self:
            for k in keys:
                data[k].append(getattr(s, k))
        return list((k, data[k]) for k in list(self.keys()))


class _BootstrapStatistics(_MetaStatistics):

    def __init__(self, data, statistics=None, sample_len=0.5, sample_num=100, **expected):
        # Jack knife n*(n-1)
        # bootstrap n*(n-1), (n-1)*(n-2), ...
        statistics = _Statistics if statistics is None else statistics
        if not expected:
            expected = dict(list(statistics(data).items()))
        self.sample = data
        k = int(float(len(data)) * sample_len)
        p = str(expected.get('process', ''))
        iterable = (statistics(sample(data, k), description='%s(%d)' % (p,i), **expected) for i in range(sample_num))
        super(_BootstrapStatistics, self).__init__(iterable, **expected)

    def items(self):
        expected = self.expected
        process, time = expected.get('process'), expected.get('time', 1.)
        if process:
            expected.update(dict((k, getattr(process, k)(time)) for k in self._available if hasattr(process, k)))

        res = list()
        for k, v in list(super(_BootstrapStatistics, self).items()):
            p = str(expected.get('process', ''))
            d = '%s[%s]' % (p, k)
            if k in expected:
                s = _Statistics(v, description=d, mean=expected.get(k))
            else:
                s = _Statistics(v, description=d)
            res.append((k, s))
        return res


class _ConvergenceStatistics(_MetaStatistics):
    def __init__(self, data, statistics=None, sample_num=10, **expected):
        # convergence [:1] -> [:n]
        statistics = _Statistics if statistics is None else statistics
        if not expected:
            expected = dict(list(statistics(data).items()))
        self.sample = data
        k = int(len(data)/sample_num)
        p = str(expected.get('process', ''))
        iterable = ((statistics(data[:i+k], description='%s[0:%d]' % (p,i+k), **expected)) for i in range(0, len(data), k))
        super(_ConvergenceStatistics, self).__init__(iterable, **expected)


# todo class _ValidationStatistics(object): # 60:40 validation test


class _MultiStatistics(object):
    _available = 'count', 'mean', 'variance', 'stdev', 'skewness', 'kurtosis', \
                 'min', 'max', 'median', 'box', 'percentile', 'sample'

    def __init__(self, data):
        self._inner = list(_Statistics(d) for d in zip(*data))

    def __getattr__(self, item):
        if item in _MultiStatistics._available and hasattr(_Statistics(list(range(10))), item):
            return list(getattr(s, item) for s in self._inner)
        else:
            return super(_MultiStatistics, self).__getattribute__(item)


class StatisticsConsumer(TransposedConsumer):
    """
    run basic statistics on storage consumer result per time slice
    """

    def __init__(self, func=None, statistics=None, **kwargs):
        if statistics is None:
            statistics = _Statistics
        self.statistics = statistics
        self.kwargs = kwargs
        super(StatisticsConsumer, self).__init__(func)

    def finalize(self):
        """finalize for StatisticsConsumer"""
        super(StatisticsConsumer, self).finalize()
        # run statistics on timewave slice w at grid point g
        # self.result = [(g, self.statistics(w)) for g, w in zip(self.grid, self.result)]
        if self.kwargs:
            self.result = list(zip(self.grid, (self.statistics(w, **self.kwargs) for w in self.result)))
        else:
            self.result = list(zip(self.grid, list(map(self.statistics, self.result))))


class StochasticProcessStatisticsConsumer(StatisticsConsumer):
    """
    run basic statistics on storage consumer result as a stochastic process
    """

    def finalize(self):
        """finalize for StochasticProcessStatisticsConsumer"""
        super(StochasticProcessStatisticsConsumer, self).finalize()

        class StochasticProcessStatistics(self.statistics):
            """local version to store statistics"""

            def __str__(self):
                s = [k.rjust(12) + str(getattr(self, k)) for k in dir(self) if not k.startswith('_')]
                return '\n'.join(s)

        sps = StochasticProcessStatistics([0, 0])
        keys = list()
        for k in dir(sps):
            if not k.startswith('_'):
                a = getattr(sps, k)
                if isinstance(a, (int, float, str)):
                    keys.append(k)
                #else:
                #    delattr(sps, k)
        for k in keys:
            setattr(sps, k, list())
        grid = list()
        for g, r in self.result:
            grid.append(g)
            for k in keys:
                a = getattr(sps, k)
                a.append(getattr(r, k))
        self.result = grid, sps


class TimeWaveConsumer(TransposedConsumer):
    def finalize(self):
        super(TimeWaveConsumer, self).finalize()

        max_v = max(max(w) for w in self.result)
        min_v = min(min(w) for w in self.result)
        min_l = min(len(w) for w in self.result)
        n = int(sqrt(min_l))  # number of y grid
        y_grid = [min_v + (max_v - min_v) * float(i) / float(n) for i in range(n)]
        y_grid.append(max_v + 1e-12)

        x, y, z = list(), list(), list()  # grid, value, count
        for point, wave in zip(self.grid, self.result):
            for l, u in zip(y_grid[:-1], y_grid[1:]):
                x.append(point)
                y.append(l + (u - l) * .5)
                z.append(float(len([w for w in wave if l <= w < u])) / float(min_l))
        self.result = x, y, z
