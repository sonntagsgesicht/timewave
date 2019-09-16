# -*- coding: utf-8 -*-

# timewave
# --------
# timewave, a stochastic process evolution simulation engine in python.
#
# Author:   sonntagsgesicht, based on a fork of Deutsche Postbank [pbrisk]
# Version:  0.5, copyright Saturday, 14 September 2019
# Website:  https://github.com/sonntagsgesicht/timewave
# License:  Apache License 2.0 (see LICENSE file)

from __future__ import print_function

"""
UnitTests for timewave simulation engine
"""
from datetime import datetime
import unittest
import sys
from os import system, getcwd, sep, makedirs, path
from math import exp, sqrt
from random import Random

import numpy as np

try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm

    def plot_consumer_result(result, grid=None, title='', path=None):
        for l in result:
            plt.plot(grid, l)
        plt.title(title)
        plt.xlabel("time, $t$")
        plt.ylabel("position of process")
        if path:
            plt.savefig(path + sep + title + '.pdf')
            plt.close()


    def plot_timewave_result(result, title='', path=None):
        # Plot a basic wireframe.
        x, y, z = result
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm)
        plt.title(title)
        if path:
            plt.savefig(path + sep + title + '.pdf')
            plt.close()

except ImportError:
    print('timewave graphics not available due to ImportError importing matplotlib.pyplot')

    def plot_consumer_result(*args):
        pass


    def plot_timewave_result(*args):
        pass

    plt = None

sys.path.append("..")

from timewave.engine import Engine, Producer, Consumer
from timewave.producers import MultiProducer, DeterministicProducer, StringReaderProducer
from timewave.consumers import TransposedConsumer, ConsumerConsumer, MultiConsumer, StringWriterConsumer
from timewave.stochasticprocess.gauss import WienerProcess, OrnsteinUhlenbeckProcess, GeometricBrownianMotion, \
    TimeDependentWienerProcess, TimeDependentGeometricBrownianMotion
from timewave.stochasticprocess.multifactor import SABR, MultiGauss
from timewave.stochasticprocess.markovchain import FiniteStateMarkovChain, FiniteStateInhomogeneousMarkovChain, \
    FiniteStateContinuousTimeMarkovChain, AugmentedFiniteStateMarkovChain
from timewave.stochasticproducer import GaussEvolutionProducer, MultiGaussEvolutionProducer
from timewave.stochasticconsumer import StatisticsConsumer, StochasticProcessStatisticsConsumer, TimeWaveConsumer, \
    _MultiStatistics

PROFILING = False

p = '.' + sep + 'pdf'
if not path.exists(p):
    makedirs('.' + sep + 'pdf')

# --- random generator test ---

class RandomGeneratorTestCase(unittest.TestCase):

    def test_random(self):
        places = 0
        num = 20
        path = 50000
        random = Random()
        for x in range(25):
            random.seed()
            sample = list()
            for _ in range(num):
                sample.append(list(random.gauss(0., 1.) for __ in range(path)))
            means = list(np.mean(s) for s in sample)
            cov = list(np.cov(sample).flat)
            res = x, min(means), max(means), min(cov), max(cov), np.mean(np.sum(sample, 0)), np.cov(np.sum(sample, 0))
            print(' '.join('%+2.4f'.ljust(6) % r for r in res))
            self.assertAlmostEqual(0.0, min(means), places)
            self.assertAlmostEqual(0.0, max(means), places)
            self.assertAlmostEqual(0.0, min(cov), places)
            self.assertAlmostEqual(1.0, max(cov), places)
            self.assertAlmostEqual(0.0, np.mean(np.sum(sample, 0)), places)
            self.assertAlmostEqual(num, np.cov(np.sum(sample, 0)), places)


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


# -- DeterministicProducerTests ---

class DeterministicProducerTests(unittest.TestCase):
    def test_deterministic_producer(self):
        grid = list(range(100))
        num_of_paths = 5000
        sample = [[float(i) / float(j + 1) for j in grid] for i in range(num_of_paths)]
        p = DeterministicProducer(sample)
        result = Engine(p, Consumer()).run(p.grid, p.num_of_paths)
        for i, j in zip(sample, result):
            for x, y in zip(i, j):
                self.assertEqual(x, y)

    def test_string_producer(self):
        grid = list(range(100))
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


# -- ProcessUnitTests ---

class BrownianMotionProducerUnitTests(unittest.TestCase):
    def test_brownian_motion_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        producer = WienerProcessProducer()
        consumer = Consumer()
        Engine(producer, consumer).run(list(range(0, 20)), 100)
        plot_consumer_result(consumer.result, consumer.grid, '2d-Wiener', '.' + sep + 'pdf')

    def test_brownian_motion_timwave_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        producer = WienerProcessProducer()
        consumer = TimeWaveConsumer()
        Engine(producer, consumer).run(list(range(0, 100)), 1000)
        plot_timewave_result(consumer.result, '3d-Wiener', '.' + sep + 'pdf')

    def test_brownian_motion_statistics(self):
        """
        Monte Carlo simulation of Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        producer = WienerProcessProducer()
        consumer = ConsumerConsumer(StatisticsConsumer(), StochasticProcessStatisticsConsumer())
        stats, (_, t) = Engine(producer, consumer).run(list(range(0, 20)), 5000, profiling=PROFILING)

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
        waves = Engine(producer, consumer).run(list(range(0, 25)), 25000, profiling=PROFILING)

        # check that on average there is no movement
        for g, w in enumerate(waves):
            mean = sum(w) / len(w)
            vol = sum([x * x for x in w]) / (len(w)-1)
            self.assertAlmostEqual(0.0, mean, 0)
            self.assertAlmostEqual(float(g), vol, 0)


class GeometricBrownianMotionProducerUnitTests(unittest.TestCase):
    def test_geometric_brownian_motion_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        producer = GeometricBrownianMotionProducer(.05, .05)
        consumer = Consumer()
        Engine(producer, consumer).run(list(range(0, 20)), 100)
        plot_consumer_result(consumer.result, consumer.grid, '2d-GBM', '.' + sep + 'pdf')

    def test_geometric_brownian_motion_timwave_plot(self):
        """
        Monte Carlo simulation of geometric Brownian motion with constant volatility,
        hence initial state should be reached on average
        """
        producer = GeometricBrownianMotionProducer(.01, .01)
        consumer = TimeWaveConsumer()
        Engine(producer, consumer).run(list(range(0, 50)), 5000)
        plot_timewave_result(consumer.result, '3d-GBM', '.' + sep + 'pdf')

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
        stats, _, paths = Engine(producer, consumer).run(list(range(0, 100)), 500)

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
        waves = Engine(producer, consumer).run(list(range(0, 100)), 5000)

        # check that on average there is no movement
        for g, w in enumerate(waves):
            s_mean = sum(w) / len(w)
            s_variance = sqrt(sum([x * x for x in w]) / len(w) - s_mean ** 2)
            self.assertAlmostEqual(mean(g), s_mean, 0)
            self.assertAlmostEqual(variance(g), s_variance, 0)


# --- MultiProcessUnitTests ---

class MultiProducerUnitTests(unittest.TestCase):
    def test_multi_producer(self):
        shift = 1.
        producer = MultiProducer(WienerProcessProducer(), WienerProcessProducer(shift))
        consumer = MultiConsumer(TransposedConsumer(), TransposedConsumer())
        first, second = Engine(producer, consumer).run(list(range(0, 20)), 500, num_of_workers=None)
        for i in range(len(first)):
            for x, y in zip(first[i], second[i]):
                self.assertAlmostEqual(x, y - shift * i)


# --- GaussEvolutionProducerUnitTests ---

class GaussEvolutionProducerUnitTests(unittest.TestCase):
    def setUp(self):
        self.places = 3
        self.path = 5000
        self.grid = list(range(20))
        self.process = WienerProcess(.0, .0001)
        self.eval = (lambda s: s.value)

    def test_statistics(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = StatisticsConsumer(func=self.eval)
        stats = Engine(producer, consumer).run(self.grid, self.path)

        for p, s in stats:
            self.assertAlmostEqual(self.process.mean(p), s.mean, self.places)
            # self.assertAlmostEqual(self.process.mean(p), s.median, self.places)
            self.assertAlmostEqual(self.process.variance(p), s.variance, self.places)

    def test_2d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = Consumer()
        Engine(producer, consumer).run(self.grid, 500)
        plot_consumer_result(consumer.result, consumer.grid, '2d-' + str(self.process), '.' + sep + 'pdf')

    def test_3d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = TimeWaveConsumer(self.eval)
        Engine(producer, consumer).run(self.grid, 5000)
        plot_timewave_result(consumer.result, '3d-' + str(self.process), '.' + sep + 'pdf')


class OrnsteinUhlenbeckProcessUnitTests(GaussEvolutionProducerUnitTests):
    def setUp(self):
        super(OrnsteinUhlenbeckProcessUnitTests, self).setUp()
        self.places = 2
        self.grid = list(range(50))
        self.process = OrnsteinUhlenbeckProcess(.1, .02, .02, .1)
        self.process = OrnsteinUhlenbeckProcess(.1, -.1, .05, .1)


class GeometricBrownianMotionUnitTests(GaussEvolutionProducerUnitTests):
    def setUp(self):
        super(GeometricBrownianMotionUnitTests, self).setUp()
        self.places = 2
        self.grid = list(range(20))
        self.process = GeometricBrownianMotion(.1, .01, 0.1)

    def test_mean(self):
        start, drift, vol, time = 1., 0.1, 0.02, 1.
        expected = start * exp((drift + 0.5 * vol ** 2) * time)
        process = GeometricBrownianMotion(drift, vol, start)
        e = Engine(GaussEvolutionProducer(process), StatisticsConsumer())
        mean = list()
        median = list()
        variance = list()
        for seed in range(100):
            d, r = e.run(grid=[0., time], seed=seed, num_of_paths=5000)[-1]
            mean.append(r.mean)
            median.append(r.median)
            variance.append(r.variance)
        self.assertTrue(min(mean) <= expected <= max(mean))
        self.assertTrue(min(median) <= expected <= max(median))
        self.assertTrue(min(mean) <= process.mean(time) <= max(mean))
        self.assertTrue(min(variance) <= process.variance(time) <= max(variance))


class TermWienerProcessUnitTests(GaussEvolutionProducerUnitTests):
    def setUp(self):
        super(TermWienerProcessUnitTests, self).setUp()
        self.places = 0
        self.grid = list(range(10))
        self.process = TimeDependentWienerProcess([0., 0.5, -0.5, 0.], [1., .5, 0.5, 0.3], [0., 3., 5., 7.])

    def test_compare(self):

        process = WienerProcess()
        for g in self.grid:
            self.assertAlmostEqual(process.mean(g), process._mu * g)
            self.assertAlmostEqual(process.variance(g), process._sigma ** 2 * g)

        term_process = TimeDependentWienerProcess()
        for g in self.grid:
            self.assertAlmostEqual(process.mean(g), term_process.mean(g))
            self.assertAlmostEqual(process.variance(g), term_process.variance(g))

        term_process = TimeDependentWienerProcess([0.] * 5, [1.] * 5)
        for g in self.grid:
            self.assertAlmostEqual(process.mean(g), term_process.mean(g))
            self.assertAlmostEqual(process.variance(g), term_process.variance(g))


class TimeDependentGeometricBrownianMotionUnitTests(TermWienerProcessUnitTests):
    def setUp(self):
        super(TimeDependentGeometricBrownianMotionUnitTests, self).setUp()
        self.places = 0
        self.grid = list(range(10))
        self.process = TimeDependentGeometricBrownianMotion([0., 0.05, -0.05, 0.], [0.1, .005, 0.2, 0.12],
                                                            [0., 3., 5., 10.])

    def test_compare(self):
        process = GeometricBrownianMotion(mu=0.01, sigma=0.01)
        for g in self.grid:
            self.assertAlmostEqual(process.mean(g), exp(process._mu * g + 0.5 * process._sigma ** 2 * g))
            self.assertAlmostEqual(process.variance(g), process.mean(g) ** 2 * (exp(process._sigma ** 2 * g) - 1))

        term_process = TimeDependentGeometricBrownianMotion(mu=(0.01,), sigma=(0.01,))
        for g in self.grid:
            self.assertAlmostEqual(process.mean(g), term_process.mean(g))
            self.assertAlmostEqual(process.variance(g), term_process.variance(g))

        term_process = TimeDependentGeometricBrownianMotion([0.01] * 5, [0.01] * 5)
        for g in self.grid:
            self.assertAlmostEqual(process.mean(g), term_process.mean(g))
            self.assertAlmostEqual(process.variance(g), term_process.variance(g))


# --- MarkovChainEvolutionProducerUnitTests


class MarkovChainEvolutionProducerUnitTests(unittest.TestCase):
    def setUp(self):
        self.places = 1
        self.path = 50000
        self.grid = list(range(10))
        s, t = [0.5, 0.5, .0], [[.75, .25, .0], [.25, .5, .25], [.0, .25, .75]]
        self.process = FiniteStateMarkovChain(transition=t, start=s)

    def test_statistics(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = StatisticsConsumer(statistics=_MultiStatistics)
        stats = Engine(producer, consumer).run(self.grid, self.path)

        for p, s in stats:
            for pm, sm in zip(self.process.mean(p), s.mean):
                self.assertAlmostEqual(pm, sm, self.places)
            for pv, sv in zip(self.process.variance(p), s.variance):
                self.assertAlmostEqual(pv, sv, self.places)

    def test_random_statistics(self):
        for d in range(2, 4, 2):
            process = self.process.__class__.random(d)
            producer = GaussEvolutionProducer(process)
            consumer = StatisticsConsumer(statistics=_MultiStatistics)
            stats = Engine(producer, consumer).run(self.grid, self.path)

            msg = '\ntransition matrix:\n' + str(process._transition_matrix)
            msg += '\nstart distribution:\n' + str(process.start)
            for p, s in stats:
                for pm, sm in zip(process.mean(p), s.mean):
                    self.assertAlmostEqual(pm, sm, self.places, 'mean at %d: %f vs. %f' % (p, pm, sm) + msg)
                for pv, sv in zip(process.variance(p), s.variance):
                    self.assertAlmostEqual(pv, sv, self.places, 'variance t %d: %f vs. %f' % (p, pv, sv) + msg)

    def test_2d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = Consumer()
        Engine(producer, consumer).run(self.grid, 500)
        plot_consumer_result(consumer.result, consumer.grid, '2d-' + str(self.process), '.' + sep + 'pdf')

    def _test_3d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = TimeWaveConsumer()
        Engine(producer, consumer).run(self.grid, 5000)
        plot_timewave_result(consumer.result, '3d-' + str(self.process), '.' + sep + 'pdf')


class D5MarkovChainEvolutionProducerUnitTests(MarkovChainEvolutionProducerUnitTests):
    def setUp(self):
        super(D5MarkovChainEvolutionProducerUnitTests, self).setUp()
        self.process = FiniteStateMarkovChain.random(5)


class ContinuousTimeMarkovChainEvolutionProducerUnitTests(MarkovChainEvolutionProducerUnitTests):
    def setUp(self):
        super(ContinuousTimeMarkovChainEvolutionProducerUnitTests, self).setUp()
        s, t = [0.5, 0.5, .0], [[.75, .25, .0], [.25, .5, .25], [.0, .25, .75]]
        self.process = FiniteStateContinuousTimeMarkovChain(transition=t, start=s)
        self.grid = [float(g) * 0.5 for g in self.grid]
        self.places = 0  # todo: improve test resulte to meet at least places=1


class InhomogeneousMarkovChainEvolutionProducerUnitTests(MarkovChainEvolutionProducerUnitTests):
    def setUp(self):
        super(InhomogeneousMarkovChainEvolutionProducerUnitTests, self).setUp()
        s, t = [0.5, 0.5, .0], [[.75, .25, .0], [.25, .5, .25], [.0, .25, .75]]
        self.process = FiniteStateInhomogeneousMarkovChain(transition=[t, t, t, t], start=s)
        self.sample = list(np.random.normal(size=100))

    def test_evolve(self):
        process = FiniteStateMarkovChain(self.process.transition, start=self.process.start)
        x = y = self.process.start
        for i, q in enumerate(self.sample):
            y = process.evolve(y, i, i + 1, q)
            x = self.process.evolve(x, i, i + 1, q)
            for xx, yy in zip(x, y):
                self.assertAlmostEqual(xx, yy)

    def test_evolve_2(self):
        process = FiniteStateMarkovChain(self.process.transition, start=self.process.start)
        n = np.identity(len(self.process.start), float).tolist()
        t_list = [n, n, n, self.process.transition]
        i_process = FiniteStateInhomogeneousMarkovChain(transition=t_list, start=self.process.start)

        x = y = self.process.start

        x = i_process.evolve(x, 0, 1, 1.)
        self.assertEqual(x, y)
        x = i_process.evolve(x, 1, 2, 1.)
        self.assertEqual(x, y)
        x = i_process.evolve(x, 2, 3, 1.)
        self.assertEqual(x, y)

        for i, q in enumerate(self.sample):
            y = process.evolve(y, i + 3, i + 4, q)
            x = self.process.evolve(x, i + 3, i + 4, q)
            for xx, yy in zip(x, y):
                self.assertAlmostEqual(xx, yy)


class AugmentedMarkovChainEvolutionProducerUnitTests(GaussEvolutionProducerUnitTests):
    _underlying_class = FiniteStateMarkovChain

    def setUp(self):
        super(AugmentedMarkovChainEvolutionProducerUnitTests, self).setUp()
        self.places = 1
        self.path = 5000
        self.grid = list(range(10))
        transition = [
            [0.7, 0.2, 0.099, 0.001],
            [0.2, 0.5, 0.29, 0.01],
            [0.1, 0.2, 0.6, 0.1],
            [0.0, 0.0, 0.0, 1.0]]
        r_squared = 1.0
        start = [.3, .2, .5, 0.]
        self.underlying = self._underlying_class(transition, r_squared, start)
        augmentation = (lambda x: 1. if x == 3 else 0.)
        augmentation = [0., 0., 0., 1.]
        self.process = AugmentedFiniteStateMarkovChain(self.underlying, augmentation)
        self.eval = self.process.eval

    def test_start(self):
        self.assertEqual(self.process.start, self.underlying.start)

        start = [1., 1., 1., 1.]
        self.process.start = start
        self.assertEqual(self.process.start, self.underlying.start)
        self.assertEqual(start, self.underlying.start)

        start = [2., 2., 2., 2.]
        self.underlying.start = start
        self.assertEqual(self.process.start, self.underlying.start)
        self.assertEqual(start, self.underlying.start)


class ContinuousAugmentedMarkovChainEvolutionProducerUnitTests(AugmentedMarkovChainEvolutionProducerUnitTests):
    _underlying_class = FiniteStateContinuousTimeMarkovChain


# --- MultiGaussEvolutionProducerUnitTests ---

class MultiGaussEvolutionProducerUnitTests(unittest.TestCase):
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
        grid = list(range(0, 10))
        r = .8
        producer = MultiGaussEvolutionProducer([WienerProcess(), WienerProcess(shift)], [[1., r], [r, 1.]])
        consumer = MultiConsumer(TransposedConsumer(), TransposedConsumer())
        first, second = Engine(producer, consumer).run(grid, 500, num_of_workers=None)

        if plt is not None:
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
        grid = list(range(0, 10))
        r = .9
        producer = GaussEvolutionProducer(MultiGauss([0., 0.], [[1., r], [r, 1.]], [0., 0.]))
        consumer = ConsumerConsumer(TransposedConsumer(lambda s: s.value[0]), TransposedConsumer(lambda s: s.value[1]))
        first, second = Engine(producer, consumer).run(grid, 500, num_of_workers=None)

        if plt is not None:
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
        self.places = 0  # fixme this is not a real test!
        self.grid = list(range(10))
        self.process = SABR()

    def test_statistics(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = StatisticsConsumer(lambda s: s.value[0])
        stats = Engine(producer, consumer).run(self.grid, 50000)

        for p, s in stats:
            # print p, self.process.mean(p), self.process.variance(p), '\n', s
            self.assertAlmostEqual(self.process.mean(p), s.mean, self.places)
            self.assertAlmostEqual(self.process.mean(p), s.median, self.places)
            self.assertAlmostEqual(self.process.variance(p), s.variance, self.places)

    def test_2d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = Consumer(lambda s: s.value[0])
        Engine(producer, consumer).run(self.grid, 500)
        plot_consumer_result(consumer.result, consumer.grid, '2d-' + str(self.process), '.' + sep + 'pdf')

    def test_3d_plot(self):
        producer = GaussEvolutionProducer(self.process)
        consumer = TimeWaveConsumer(lambda s: s.value[0])
        Engine(producer, consumer).run(self.grid, 5000)
        plot_timewave_result(consumer.result, '3d-' + str(self.process), '.' + sep + 'pdf')


# --- __main__ ---

if __name__ == "__main__":
    import sys
    import os

    start_time = datetime.now()

    print('')
    print('======================================================================')
    print('')
    print(('run %s' % __file__))
    print(('in %s' % os.getcwd()))
    print(('started  at %s' % str(start_time)))
    print('')
    print('----------------------------------------------------------------------')
    print('')

    suite = unittest.TestLoader().loadTestsFromModule(__import__("__main__"))
    testrunner = unittest.TextTestRunner(stream=sys.stdout, descriptions=2, verbosity=2)
    testrunner.run(suite)

    print('')
    print('======================================================================')
    print('')
    print(('ran %s' % __file__))
    print(('in %s' % os.getcwd()))
    print(('started  at %s' % str(start_time)))
    print(('finished at %s' % str(datetime.now())))
    print('')
    print('----------------------------------------------------------------------')
    print('')
