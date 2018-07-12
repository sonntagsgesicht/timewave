from math import sqrt

from consumers import TransposedConsumer


# statistics and stochastic process consumers


class _Statistics(object):
    """
    calculate basic statistics for a 1 dim empirical sample
    """

    def __init__(self, data):
        sps = sorted(data)
        l = float(len(sps))
        p = [int(i * l * 0.01) for i in range(100)]
        self.count = len(sps)
        self.mean = sum(sps) / l
        self.variance = 0. if len(set(sps)) == 1 else sum([rr ** 2 for rr in sps]) / (l - 1) - self.mean ** 2
        self.stdev = sqrt(self.variance)
        self.min = sps[0]
        self.max = sps[-1]
        self.median = sps[p[50]]
        self.box = [sps[0], sps[p[25]], sps[p[50]], sps[p[75]], sps[-1]]
        self.percentile = [sps[int(i)] for i in p]
        self.sample = data

    def __str__(self):
        keys = ['count', 'mean', 'stdev', 'variance', 'min', 'median', 'max']
        values = ['%0.8f' % getattr(self, a, 0.0) for a in keys]
        mk = max(map(len, keys))
        mv = max(map(len, values))
        res = [a.ljust(mk) + ' : ' + v.rjust(mv) for a, v in zip(keys, values)]
        return '\n'.join(res)


class _MultiStatistics(object):

    _available = 'count', 'mean', 'variance', 'stdev', 'min', 'max', 'median', 'box', 'percentile', 'sample'

    def __init__(self, data):
        self._inner = list(_Statistics(d) for d in zip(*data))

    def __getattr__(self, item):
        if item in _MultiStatistics._available and hasattr(_Statistics(range(10)), item):
            return list(getattr(s, item) for s in self._inner)
        else:
            return super(_MultiStatistics, self).__getattribute__(item)


class StatisticsConsumer(TransposedConsumer):
    """
    run basic statistics on storage consumer result per time slice
    """

    def __init__(self, func=None, statistics=None):
        if statistics is None:
            statistics = _Statistics
        self.statistics = statistics
        super(StatisticsConsumer, self).__init__(func)

    def finalize(self):
        """finalize for StatisticsConsumer"""
        super(StatisticsConsumer, self).finalize()
        # run statistics on eval slice w at grid point g
        # self.result = [(g, self.statistics(w)) for g, w in zip(self.grid, self.result)]
        # self.result = zip(self.grid, (self.statistics(w) for w in self.result))
        self.result = zip(self.grid, map(self.statistics, self.result))


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
                else:
                    delattr(sps, k)
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
