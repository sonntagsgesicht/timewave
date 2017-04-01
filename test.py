# coding=utf-8
"""
UnitTests for timewave simulation engine
"""
from datetime import datetime
from unittest import TestCase, main
from os import system, getcwd, sep
from math import exp, sqrt

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm
except ImportError:
    print('timewave graphics not available due to ImportError importing matplotlib.pyplot')

from timewave.engine import Engine, Producer, Consumer
from timewave.producers import MultiProducer, DeterministicProducer, StringReaderProducer
from timewave.consumers import TransposedConsumer, ConsumerConsumer, MultiConsumer, StringWriterConsumer
from timewave.stochasticprocess import WienerProcess, OrnsteinUhlenbeckProcess, GeometricBrownianMotion, \
    SABR, MultiGauss
from timewave.stochasticproducer import GaussEvolutionProducer, MultiGaussEvolutionProducer
from timewave.stochasticconsumer import StatisticsConsumer, StochasticProcessStatisticsConsumer, TimeWaveConsumer
from timewave.pyprocess import Wiener_process, GBM_process, OU_process
from timewave.pyprocessproducer import PyProcessProducer, SimplePyProcessProducer
from timewave.plot import plot_consumer_result, plot_timewave_result

DISPLAY_RESULTS = False
PROFILING = False


class WienerProcessProducer(Producer):
    """
    class implementing Brownian motion / Wiener process between grid dates
    """

    def __init__(self, mu=0.0, sigma=1.0, start=0.0):
        super(WienerProcessProducer, self).__init__()
        self.initial_state.value = start
        self._mu = mu
        self._sigma = sigma

    def evolve(self, new_date):
        """
        evolve to the new process state at the next date

        :param date new_date: date or point in time of the new state
        :param float step: random number for step
        :return State:
        """
        if self.state.date == new_date and not self.initial_state.date == new_date:
            return self.state

        dt = float(new_date - self.state.date)
        self.state.value += self._mu * dt + self._sigma * sqrt(dt) * self.random.gauss(0.0, 1.0)
        self.state.date = new_date
        return self.state


class GeometricBrownianMotionProducer(Producer):
    """
    class implementing geometric Brownian motion between grid dates
    """

    def __init__(self, mu=0.0, sigma=0.01, start=1.0):
        super(GeometricBrownianMotionProducer, self).__init__()
        self.initial_state.value = start
        self._mu = mu
        self._sigma = sigma

    def evolve(self, new_date):
        """
        evolve to the new process state at the next date

        :param date new_date: date or point in time of the new state
        :return State:
        """
        if self.state.date == new_date and not self.initial_state.date == new_date:
            return self.state
        dt = float(new_date - self.state.date)
        self.state.value *= exp(self._mu * dt + self._sigma * sqrt(dt) * self.random.gauss(0.0, 1.0))
        self.state.date = new_date
        return self.state


class PlotConsumer(Consumer):
    def __init__(self, title='sample paths'):
        super(PlotConsumer, self).__init__()
        self._title = title

    def finalize(self):
        super(PlotConsumer, self).finalize()
        plot_consumer_result(self.result, self.grid, self._title, '.' + sep + 'pdf')


class PlotTimeWaveConsumer(TimeWaveConsumer):
    def __init__(self, title='timewaves', func=None):
        super(PlotTimeWaveConsumer, self).__init__(func)
        self._title = title

    def finalize(self):
        super(PlotTimeWaveConsumer, self).finalize()
        plot_timewave_result(self.result, self._title, '.' + sep + 'pdf')


class BrownianMotionProducerUnitTests(TestCase):
    def setUp(self):
        self.plot = None

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS:
            if PROFILING:
                try:  # accepted due to external call of snakeviz
                    system('snakeviz worker-0.prof')
                except:
                    pass
            if self.plot:
                self.plot.close()

    def test_brownian_motion_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        grid = range(0, 20)
        self.plot = Engine(WienerProcessProducer(), PlotConsumer('wiener_2d_20_classic')).run(grid, 100)

    def test_brownian_motion_timwave_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        grid = range(0, 100)
        self.plot = Engine(WienerProcessProducer(), PlotTimeWaveConsumer('wiener_3d_100_classic')).run(grid, 1000)

    def test_brownian_motion_statistics(self):
        """
        Monte Carlo simulation of Brownian motion with constant volatility,
        hence initial state should be reached on average
        """

        grid = range(0, 20)
        producer = WienerProcessProducer()
        consumer = ConsumerConsumer(StatisticsConsumer(), StochasticProcessStatisticsConsumer())
        stats, (_, t) = Engine(producer, consumer).run(grid, 5000, profiling=PROFILING)

        for p, s in stats:
            self.assertAlmostEqual(0.0, s.mean, 0)
            self.assertAlmostEqual(0.0, s.median, 0)
            self.assertAlmostEqual(float(p), s.variance, -1)

        self.assertAlmostEqual(0.0, max(t.mean), 0)
        self.assertAlmostEqual(0.0, min(t.mean), 0)

    def test_brownian_motion(self):
        """
        Monte Carlo simulation of Brownian motion with constant volatility,
        hence initial state should be reached on average
        """

        grid = range(0, 20)
        waves = Engine(WienerProcessProducer(), TransposedConsumer()).run(grid, 5000, profiling=PROFILING)
        # check that on average there is no movement
        for g, w in zip(grid, waves):
            mean = sum(w) / len(w)
            vol = sum([x * x for x in w]) / len(w)
            self.assertAlmostEqual(0.0, mean, 0)
            self.assertAlmostEqual(float(g), vol, 0)


class GeometricBrownianMotionProducerUnitTests(TestCase):
    def setUp(self):
        self.plot = None

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS and self.plot:
            self.plot.close()

    def test_geometric_brownian_motion_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        grid = range(0, 20)
        gbm = GeometricBrownianMotionProducer(.05, .05)
        self.plot = Engine(gbm, PlotConsumer('gbm_2d_20_classic')).run(grid, 100)

    def test_geometric_brownian_motion_timwave_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        grid = range(0, 50)
        gbm = GeometricBrownianMotionProducer(.01, .01)
        self.plot = Engine(gbm, PlotTimeWaveConsumer('gbm_3d_50_classic')).run(grid, 5000)

    def test_geometric_brownian_motion_statistics(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        mu = 0.0
        sigma = 0.01
        grid = range(0, 100)

        mean = (lambda t: exp(mu * t))
        variance = (lambda t: exp(2 * mu * t) * (exp(sigma ** 2 * t) - 1))

        producer = GeometricBrownianMotionProducer(mu, sigma)
        consumer = ConsumerConsumer(StatisticsConsumer(), StochasticProcessStatisticsConsumer(), Consumer())
        stats, _, paths = Engine(producer, consumer).run(grid, 500)

        # check that on average there is alright
        for p, s in stats:
            self.assertAlmostEqual(mean(p), s.mean, 0)
            self.assertAlmostEqual(mean(p), s.median, 0)
            self.assertAlmostEqual(variance(p), s.variance, 0)

    def test_geometric_brownian_motion(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        mu = 0.01
        sigma = 0.01
        grid = range(0, 100)

        mean = (lambda t: exp(mu * t))
        variance = (lambda t: exp(2 * mu * t) * (exp(sigma ** 2 * t) - 1))

        waves = Engine(GeometricBrownianMotionProducer(mu, sigma), TransposedConsumer()).run(grid, 5000)
        # check that on average there is no movement
        for g, w in zip(grid, waves):
            s_mean = sum(w) / len(w)
            s_variance = sqrt(sum([x * x for x in w]) / len(w) - s_mean ** 2)
            self.assertAlmostEqual(mean(g), s_mean, 0)
            self.assertAlmostEqual(variance(g), s_variance, 0)


class PyProcessUnitTests(TestCase):
    def setUp(self):
        self.plot = None

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS and self.plot:
            self.plot.close()

    def test_py_process(self):
        """
        Monte Carlo simulation of Brownian motion with constant volatility,
        hence initial state should be reached on average
        """

        grid = range(0, 20)  # fixme: increase num of path to 100
        process_list = list()
        process_list.append(Wiener_process(1, 1))
        process_list.append(GBM_process(.0, .01))
        process_list.append(OU_process(0.1, 0., 0.05))

        for proc in process_list:
            res = Engine(PyProcessProducer(proc), StatisticsConsumer()).run(grid, 5000)

            # check that on average there is no movement
            for p, s in res:
                if p > 0:  # fixme issue at p==0
                    self.assertAlmostEqual(proc.mean(float(p)), s.mean, 0)
                    self.assertAlmostEqual(proc.mean(float(p)), s.median, 0)
                    self.assertAlmostEqual(proc.var(float(p)), s.variance, 0)

    def test_simple_py_process(self):
        """
        Monte Carlo simulation of Brownian motion with constant volatility,
        hence initial state should be reached on average
        """

        grid = range(0, 20)  # fixme: increase num of path to 100
        process_list = list()
        process_list.append(Wiener_process(1, 1))
        process_list.append(GBM_process(.0, .01))
        process_list.append(OU_process(0.1, 0., 0.05))

        for proc in process_list:
            res = Engine(SimplePyProcessProducer(proc), StatisticsConsumer()).run(grid, 5000)

            # check that on average there is no movement
            for p, s in res:
                if p > 0:  # fixme issue at p==0
                    self.assertAlmostEqual(proc.mean(float(p)), s.mean, 0)
                    self.assertAlmostEqual(proc.mean(float(p)), s.median, 0)
                    self.assertAlmostEqual(proc.var(float(p)), s.variance, 0)


class GaussEvolutionProducerUnitTests(TestCase):
    def setUp(self):
        self.plot = None

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS and self.plot:
            self.plot.close()

    def test_wiener_process(self):
        grid = range(20)
        process = WienerProcess(.0, .0001)
        producer = GaussEvolutionProducer(process)
        consumer = ConsumerConsumer(StatisticsConsumer(), PlotTimeWaveConsumer('wiener_3d_20'))
        stats, self.plot = Engine(producer, consumer).run(grid, 5000)

        for p, s in stats:
            self.assertAlmostEqual(process.mean(p), s.mean, 0)
            self.assertAlmostEqual(process.mean(p), s.median, 0)
            self.assertAlmostEqual(process.variance(p), s.variance, 0)

    def test_ou_process(self):
        grid = range(50)
        process = OrnsteinUhlenbeckProcess(.1, -.1, .05, .1)
        producer = GaussEvolutionProducer(process)
        consumer = ConsumerConsumer(StatisticsConsumer(), PlotTimeWaveConsumer('ou_3d_50'))
        stats, self.plot = Engine(producer, consumer).run(grid, 5000)

        for p, s in stats:
            # print p, process.mean(p), s.mean, process.variance(p), s.variance
            self.assertAlmostEqual(process.mean(p), s.mean, 2)
            self.assertAlmostEqual(process.mean(p), s.median, 2)
            if p > 0:
                self.assertAlmostEqual(process.variance(p) / p, s.variance / p, 2)

    def test_ou_2d_process(self):
        grid = range(50)
        process = OrnsteinUhlenbeckProcess(.1, .02, .02, .1)
        producer = GaussEvolutionProducer(process)
        consumer = ConsumerConsumer(StatisticsConsumer(), PlotConsumer('ou_2d_50'))
        stats, self.plot = Engine(producer, consumer).run(grid)

        for p, s in stats:
            # print p, process.mean(p), s.mean, process.variance(p), s.variance
            self.assertAlmostEqual(process.mean(p), s.mean, 2)
            self.assertAlmostEqual(process.mean(p), s.median, 2)
            if p > 0:
                self.assertAlmostEqual(process.variance(p) / p, s.variance / p, 2)

    def test_gbm_process(self):
        grid = range(20)
        process = GeometricBrownianMotion(.1, .01, 0.1)
        producer = GaussEvolutionProducer(process)
        consumer = ConsumerConsumer(StatisticsConsumer(), PlotTimeWaveConsumer('gbm_3d_20'))
        stats, self.plot = Engine(producer, consumer).run(grid)

        for p, s in stats:
            # print p, process.mean(p), s.mean, process.variance(p), s.variance
            self.assertAlmostEqual(process.mean(p), s.mean, 0)
            self.assertAlmostEqual(process.mean(p), s.median, 0)
            self.assertAlmostEqual(process.variance(p), s.variance, 0)

    def test_sabr_process(self):
        grid = range(20)
        process = SABR()
        producer = GaussEvolutionProducer(process)
        consumer = ConsumerConsumer(StatisticsConsumer(lambda s: s.value[0]),
                                    PlotTimeWaveConsumer('sabr_3d_20', lambda s: s.value[0]))
        stats, self.plot = Engine(producer, consumer).run(grid, num_of_workers=None)

        for p, s in stats:
            # print p, process.mean(p), s.mean, process.variance(p), s.variance
            self.assertAlmostEqual(process.mean(p), s.mean, 0)
            self.assertAlmostEqual(process.mean(p), s.median, 0)
            self.assertAlmostEqual(process.variance(p), s.variance, 0)


class MultiProducerUnitTests(TestCase):
    def setUp(self):
        self.plot = None

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS and self.plot:
            self.plot.close()

    def test_multi_producer(self):
        shift = 1.
        grid = range(0, 20)
        producer = MultiProducer(WienerProcessProducer(), WienerProcessProducer(shift))
        consumer = MultiConsumer(TransposedConsumer(), TransposedConsumer())
        first, second = Engine(producer, consumer).run(grid, 500, num_of_workers=None)
        for i in grid:
            for x, y in zip(first[i], second[i]):
                self.assertAlmostEqual(x, y - shift * i)


class MultiGaussEvolutionProducerUnitTests(TestCase):
    def setUp(self):
        self.plot = None

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS and self.plot:
            self.plot.close()

    def test_correlation(self):
        shift = .5
        r = .8
        p = WienerProcess(), WienerProcess(shift)
        pr_1 = MultiGaussEvolutionProducer(p, [[1., r], [r, 1.]])
        self.assertEqual(pr_1._correlation[0][1], r)
        pr_2 = MultiGaussEvolutionProducer(p, {p: r})
        self.assertEqual(set(pr_1._diffusion_driver), set(pr_2._diffusion_driver))
        for r1, r2 in zip(pr_1._correlation, pr_2._correlation):
            for c1, c2 in zip(r1, r2):
                self.assertEqual(c1, c2)
        pr_3 = MultiGaussEvolutionProducer(p, [[1., r], [r, 1.]], pr_1._diffusion_driver)
        for d1, d3 in zip(pr_1._diffusion_driver, pr_3._diffusion_driver):
            self.assertEqual(d1, d3)
        q = WienerProcess(-shift),
        self.assertRaises(AssertionError, MultiGaussEvolutionProducer, p+q, [[1., r], [r, 1.]])
        pr_4 = MultiGaussEvolutionProducer(p+q, {p: r, (q+q): 1.})
        self.assertEqual(len(pr_4._correlation), 3)

        s = SABR()
        d = s.diffusion_driver
        pr_5 = MultiGaussEvolutionProducer(p + (s,), {p: r, d: s._rho})
        self.assertEqual(len(pr_5._correlation), 4)
        self.assertEqual(set(p+d), set(pr_5._diffusion_driver))
        pr_6 = MultiGaussEvolutionProducer(p + (s,), [[1., r], [r, 1.]], p)
        self.assertEqual(pr_6._diffusion_driver, p)
        pr_7 = MultiGaussEvolutionProducer(p + (s,), [[1., r], [r, 1.]], d)
        self.assertEqual(pr_7._diffusion_driver, d)

        q[0]._diffusion_driver = p[0]
        pr_8 = MultiGaussEvolutionProducer(p+q, [[1., r], [r, 1.]])
        self.assertEqual(pr_8._diffusion_driver, p)

    def test_wiener_process(self):
        shift = .5
        grid = range(0, 10)
        r = .8
        producer = MultiGaussEvolutionProducer([WienerProcess(), WienerProcess(shift)], [[1., r], [r, 1.]])
        consumer = MultiConsumer(TransposedConsumer(), TransposedConsumer())
        first, second = Engine(producer, consumer).run(grid, 500, num_of_workers=None)

        t = 'multi wiener 2d scatter'
        fig, ax = plt.subplots()
        ax.scatter(first[1], second[1])
        plt.title(t)
        plt.savefig('.' + sep + 'pdf' + sep + t.replace(' ', '_') + '.pdf')
        plt.close()

        t = 'multi wiener 3d scatter'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in grid:
            ax.scatter([i] * len(first[i]), first[i], second[i])
            for x, y in zip(first[i], second[i]):
                self.assertAlmostEqual(x, y - shift * i, -100)
        plt.title(t)
        plt.savefig('.' + sep + 'pdf' + sep + t.replace(' ', '_') + '.pdf')
        plt.close()

    def test_multi_gauss_process(self):
        grid = range(0, 10)
        r = .9
        producer = GaussEvolutionProducer(MultiGauss([0., 0.], [[1., r], [r, 1.]], [0., 0.]))
        consumer = ConsumerConsumer(TransposedConsumer(lambda s: s.value[0]), TransposedConsumer(lambda s: s.value[1]))
        first, second = Engine(producer, consumer).run(grid, 500, num_of_workers=None)

        t = 'multi gauss 2d scatter'
        fig, ax = plt.subplots()
        ax.scatter(first[1], second[1])
        plt.title(t)
        plt.savefig('.' + sep + 'pdf' + sep + t.replace(' ', '_') + '.pdf')
        plt.close()

        t = 'multi gauss 3d scatter'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in grid:
            ax.scatter([i] * len(first[i]), first[i], second[i])
            for x, y in zip(first[i], second[i]):
                self.assertAlmostEqual(x, y, -100)
        plt.title(t)
        plt.savefig('.' + sep + 'pdf' + sep + t.replace(' ', '_') + '.pdf')
        plt.close()


class DeterministicProducerTests(TestCase):
    def setUp(self):
        self.plot = None

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS and self.plot:
            self.plot.close()

    def test_deterministic_producer(self):
        grid = range(100)
        num_of_paths = 5000
        sample = [[float(i) / float(j + 1) for j in grid] for i in range(num_of_paths)]
        p = DeterministicProducer(sample)
        result = Engine(p, Consumer()).run(p.grid, p.num_of_paths)
        for i, j in zip(sample, result):
            for x, y in zip(i, j):
                self.assertEqual(x, y)

    def test_string_producer(self):
        grid = range(100)
        num_of_paths = 5000
        sample = [[float(i) / float(j + 1) for j in grid] for i in range(num_of_paths)]

        p = DeterministicProducer(sample)
        result_str = Engine(p, StringWriterConsumer()).run(p.grid, p.num_of_paths)
        self.assertTrue(isinstance(result_str, str))

        pp = StringReaderProducer(result_str)
        result = Engine(pp, Consumer()).run(pp.grid, pp.num_of_paths)
        for i, j in zip(sample, result):
            for x, y in zip(i, j):
                self.assertEqual(x, y)


if __name__ == "__main__":
    start_time = datetime.now()

    print('')
    print('======================================================================')
    print('')
    print('run %s' % __file__)
    print('in %s' % getcwd())
    print('started  at %s' % str(start_time))
    print('')
    print('----------------------------------------------------------------------')
    print('')

    main(verbosity=2)

    print('')
    print('======================================================================')
    print('')
    print('ran %s' % __file__)
    print('in %s' % getcwd())
    print('started  at %s' % str(start_time))
    print('finished at %s' % str(datetime.now()))
    print('')
    print('----------------------------------------------------------------------')
    print('')
