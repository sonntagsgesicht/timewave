# coding=utf-8
"""
UnitTests for timewave simulation engine
"""
from datetime import datetime
import unittest
from os import system, getcwd, sep, makedirs, path
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
from timewave.plot import plot_consumer_result, plot_timewave_result

DISPLAY_RESULTS = False
PROFILING = False

p = '.' + sep + 'pdf'
if not path.exists(p):
    makedirs('.' + sep + 'pdf')


# -- ProcessProducer ---

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


# -- PlotConsumer ---

class PlotConsumer(Consumer):
    def __init__(self, title='sample paths', func=None):
        super(PlotConsumer, self).__init__(func)
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


# -- ProcessUnitTests ---

class DeterministicProducerTests(unittest.TestCase):
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


class BrownianMotionProducerUnitTests(unittest.TestCase):
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
        producer = WienerProcessProducer()
        consumer = PlotConsumer('2d-Wiener')
        self.plot = Engine(producer, consumer).run(range(0, 20), 100)

    def test_brownian_motion_timwave_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        producer = WienerProcessProducer()
        consumer = PlotTimeWaveConsumer('3d-Wiener')
        self.plot = Engine(producer, consumer).run(range(0, 100), 1000)

    def test_brownian_motion_statistics(self):
        """
        Monte Carlo simulation of Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        producer = WienerProcessProducer()
        consumer = ConsumerConsumer(StatisticsConsumer(), StochasticProcessStatisticsConsumer())
        stats, (_, t) = Engine(producer, consumer).run(range(0, 20), 5000, profiling=PROFILING)

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
        producer = WienerProcessProducer()
        consumer = TransposedConsumer()
        waves = Engine(producer, consumer).run(range(0, 20), 5000, profiling=PROFILING)

        # check that on average there is no movement
        for g, w in enumerate(waves):
            mean = sum(w) / len(w)
            vol = sum([x * x for x in w]) / len(w)
            self.assertAlmostEqual(0.0, mean, 0)
            self.assertAlmostEqual(float(g), vol, 0)


class GeometricBrownianMotionProducerUnitTests(unittest.TestCase):
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
        producer = GeometricBrownianMotionProducer(.05, .05)
        consumer = PlotConsumer('2d-GBM')
        self.plot = Engine(producer, consumer).run(range(0, 20), 100)

    def test_geometric_brownian_motion_timwave_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        producer = GeometricBrownianMotionProducer(.01, .01)
        consumer = PlotTimeWaveConsumer('3d-GBM')
        self.plot = Engine(producer, consumer).run(range(0, 50), 5000)

    def test_geometric_brownian_motion_statistics(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        mu = 0.0
        sigma = 0.01
        mean = (lambda t: exp(mu * t))
        variance = (lambda t: exp(2 * mu * t) * (exp(sigma ** 2 * t) - 1))

        producer = GeometricBrownianMotionProducer(mu, sigma)
        consumer = ConsumerConsumer(StatisticsConsumer(), StochasticProcessStatisticsConsumer(), Consumer())
        stats, _, paths = Engine(producer, consumer).run(range(0, 100), 500)

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
        mean = (lambda t: exp(mu * t))
        variance = (lambda t: exp(2 * mu * t) * (exp(sigma ** 2 * t) - 1))

        producer = GeometricBrownianMotionProducer(mu, sigma)
        consumer = TransposedConsumer()
        waves = Engine(producer, consumer).run(range(0, 100), 5000)

        # check that on average there is no movement
        for g, w in enumerate(waves):
            s_mean = sum(w) / len(w)
            s_variance = sqrt(sum([x * x for x in w]) / len(w) - s_mean ** 2)
            self.assertAlmostEqual(mean(g), s_mean, 0)
            self.assertAlmostEqual(variance(g), s_variance, 0)


# --- GaussEvolutionProducerUnitTests ---

class GaussEvolutionProducerUnitTests(unittest.TestCase):
    def setUp(self):
        self.plot2d = None
        self.plot3d = None
        self.places = 0
        self.path = 5000
        self.grid = range(20)
        self.process = WienerProcess(.0, .0001)

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS:
            if self.plot2d:
                self.plot2d.close()
            if self.plot3d:
                self.plot3d.close()

    def test_statistics(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = StatisticsConsumer()
        stats = Engine(producer, consumer).run(self.grid, self.path)

        for p, s in stats:
                self.assertAlmostEqual(self.process.mean(p), s.mean, self.places)
                self.assertAlmostEqual(self.process.mean(p), s.median, self.places)
                self.assertAlmostEqual(self.process.variance(p), s.variance, self.places)

    def test_2d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = PlotConsumer('2d-' + str(self.process))
        self.plot2d = Engine(producer, consumer).run(self.grid, 500)

    def test_3d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = PlotTimeWaveConsumer('3d-' + str(self.process))
        self.plot2d = Engine(producer, consumer).run(self.grid, 5000)


class OrnsteinUhlenbeckProcessUnitTests(GaussEvolutionProducerUnitTests):

    def setUp(self):
        super(OrnsteinUhlenbeckProcessUnitTests, self).setUp()
        self.places = 2
        self.grid = range(50)
        self.process = OrnsteinUhlenbeckProcess(.1, .02, .02, .1)
        self.process = OrnsteinUhlenbeckProcess(.1, -.1, .05, .1)


class GeometricBrownianMotionUnitTests(GaussEvolutionProducerUnitTests):

    def setUp(self):
        super(GeometricBrownianMotionUnitTests, self).setUp()
        self.places = 0
        self.grid = range(20)
        self.process = GeometricBrownianMotion(.1, .01, 0.1)


# --- MultiProcessUnitTests ---

class MultiProducerUnitTests(unittest.TestCase):
    def setUp(self):
        self.plot = None

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS and self.plot:
            self.plot.close()

    def test_multi_producer(self):
        shift = 1.
        producer = MultiProducer(WienerProcessProducer(), WienerProcessProducer(shift))
        consumer = MultiConsumer(TransposedConsumer(), TransposedConsumer())
        first, second = Engine(producer, consumer).run(range(0, 20), 500, num_of_workers=None)
        for i in range(len(first)):
            for x, y in zip(first[i], second[i]):
                self.assertAlmostEqual(x, y - shift * i)


# --- GaussEvolutionProducerUnitTests ---

class MultiGaussEvolutionProducerUnitTests(unittest.TestCase):
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
        self.assertRaises(AssertionError, MultiGaussEvolutionProducer, p + q, [[1., r], [r, 1.]])
        pr_4 = MultiGaussEvolutionProducer(p + q, {p: r, (q + q): 1.})
        self.assertEqual(len(pr_4._correlation), 3)

        s = SABR()
        d = s.diffusion_driver
        pr_5 = MultiGaussEvolutionProducer(p + (s,), {p: r, d: s._rho})
        self.assertEqual(len(pr_5._correlation), 4)
        self.assertEqual(set(p + d), set(pr_5._diffusion_driver))
        pr_6 = MultiGaussEvolutionProducer(p + (s,), [[1., r], [r, 1.]], p)
        self.assertEqual(pr_6._diffusion_driver, p)
        pr_7 = MultiGaussEvolutionProducer(p + (s,), [[1., r], [r, 1.]], d)
        self.assertEqual(pr_7._diffusion_driver, d)

        q[0]._diffusion_driver = p[0]
        pr_8 = MultiGaussEvolutionProducer(p + q, [[1., r], [r, 1.]])
        self.assertEqual(pr_8._diffusion_driver, p)

    def test_wiener_process(self):
        shift = .5
        grid = range(0, 10)
        r = .8
        producer = MultiGaussEvolutionProducer([WienerProcess(), WienerProcess(shift)], [[1., r], [r, 1.]])
        consumer = MultiConsumer(TransposedConsumer(), TransposedConsumer())
        first, second = Engine(producer, consumer).run(grid, 500, num_of_workers=None)

        t = '2d-Scatter-MultiWiener'
        fig, ax = plt.subplots()
        ax.scatter(first[1], second[1])
        plt.title(t)
        plt.savefig('.' + sep + 'pdf' + sep + t.replace(' ', '_') + '.pdf')
        plt.close()

        t = '3d-Scatter-MultiWiener'
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

        t = '2d-Scatter-MultiGauss'
        fig, ax = plt.subplots()
        ax.scatter(first[1], second[1])
        plt.title(t)
        plt.savefig('.' + sep + 'pdf' + sep + t.replace(' ', '_') + '.pdf')
        plt.close()

        t = '3d-Scatter-MultiGauss'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in grid:
            ax.scatter([i] * len(first[i]), first[i], second[i])
            for x, y in zip(first[i], second[i]):
                self.assertAlmostEqual(x, y, -100)
        plt.title(t)
        plt.savefig('.' + sep + 'pdf' + sep + t.replace(' ', '_') + '.pdf')
        plt.close()


class SabrUnitTests(unittest.TestCase):

    def setUp(self):
        super(SabrUnitTests, self).setUp()
        self.places = 0
        self.grid = range(20)
        self.process = SABR()

    def tearDown(self):
        if __name__ == '__main__' and DISPLAY_RESULTS:
            if self.plot2d:
                self.plot2d.close()
            if self.plot3d:
                self.plot3d.close()

    def test_statistics(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = StatisticsConsumer(lambda s: s.value[0])
        stats = Engine(producer, consumer).run(self.grid, 5000)

        for p, s in stats:
            self.assertAlmostEqual(self.process.mean(p), s.mean, self.places)
            self.assertAlmostEqual(self.process.mean(p), s.median, self.places)
            self.assertAlmostEqual(self.process.variance(p), s.variance, self.places)

    def test_2d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = PlotConsumer('2d-' + str(self.process), lambda s: s.value[0])
        self.plot2d = Engine(producer, consumer).run(self.grid, 500)

    def test_3d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = PlotTimeWaveConsumer('3d-' + str(self.process), lambda s: s.value[0])
        self.plot2d = Engine(producer, consumer).run(self.grid, 5000)


# --- __main__ ---

if __name__ == "__main__":
    import sys
    import os

    start_time = datetime.now()

    print('')
    print('======================================================================')
    print('')
    print('run %s' % __file__)
    print('in %s' % os.getcwd())
    print('started  at %s' % str(start_time))
    print('')
    print('----------------------------------------------------------------------')
    print('')

    suite = unittest.TestLoader().loadTestsFromModule(__import__("__main__"))
    testrunner = unittest.TextTestRunner(stream=sys.stdout, descriptions=2, verbosity=2)
    testrunner.run(suite)

    print('')
    print('======================================================================')
    print('')
    print('ran %s' % __file__)
    print('in %s' % os.getcwd())
    print('started  at %s' % str(start_time))
    print('finished at %s' % str(datetime.now()))
    print('')
    print('----------------------------------------------------------------------')
    print('')
