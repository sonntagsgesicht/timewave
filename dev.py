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

import sys
import matplotlib

import numpy as np
from math import exp, log, sqrt
from random import Random

sys.path.append('.')
sys.path.append('test')
matplotlib.use('agg')

from unittests import MultiGaussEvolutionProducerUnitTests
from timewave import FiniteStateMarkovChain, AugmentedFiniteStateMarkovChain
from timewave import GaussEvolutionProducer, StatisticsConsumer, Engine
from timewave.stochasticconsumer import _MultiStatistics, _Statistics, _BootstrapStatistics, _ConvergenceStatistics
from timewave import GeometricBrownianMotion, WienerProcess, TimeDependentGeometricBrownianMotion


if True:
    from os import system, getcwd, sep, makedirs, path
    from timewave import TimeWaveConsumer, OrnsteinUhlenbeckProcess

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm

    def plot_timewave_result(result, title='', path=None):
        # Plot a basic wireframe.

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm)
        ax.bar([0], [.1], zs=0., zdir='y', alpha=0.8)
        plt.title(title)
        if path:
            print(('save to', path + sep + title + '.pdf'))
            plt.savefig(path + sep + title + '.pdf')
            plt.close()

    grid = list(range(50))
    process = GeometricBrownianMotion(.05, .05, 0.1)
    process = OrnsteinUhlenbeckProcess(.01, .2, .4, 1.)

    producer = GaussEvolutionProducer(process)
    consumer = TimeWaveConsumer(lambda s: s.value)
    Engine(producer, consumer).run(grid, 5000)

    x, y, z = consumer.result
    z = [min(_, .2) for _ in z]

    title = str(process)
    path='.'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar(grid, [.005]*len(grid), zs=-3., zdir='y', color='r', width=1.)
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm)
    plt.title(title)
    if path:
        print(('save to', path + sep + title + '.pdf'))
        plt.savefig(path + sep + title + '.pdf')
        plt.close()


if False:
    rnd = Random()
    seed = rnd.randint(0, 1000)
    rnd.seed(seed)

    num, start, drift, vol, time = int(1001), 1., .1, .5, 1.
    normal = (lambda qq, q: start + drift * time + vol * sqrt(time) * q)
    log_normal = (lambda qq, q: start * exp(drift * time + vol * sqrt(time) * q))
    func, process = normal, WienerProcess(drift, vol, start)
    func, process = log_normal, GeometricBrownianMotion(drift, vol, start)
    func, process = log_normal, TimeDependentGeometricBrownianMotion(drift, vol, start)

    c = StatisticsConsumer(description=str(process) + '(seed %d)' % seed, process=process, time=time)
    r = Engine(GaussEvolutionProducer(process), c).run(grid=[0., time], num_of_paths=num)[-1][-1]
    print(r)

    r.description = 'engine vs sample (seed %d)' % seed
    r.expected = dict(list(_Statistics([func(rnd.gauss(0., 1.), rnd.gauss(0., 1.)) for _ in range(num)]).items()))
    print(r)

if False:
    start, drift, vol, time = 1., 0.1, .5, 1.
    process = TimeDependentGeometricBrownianMotion(drift, vol, start)
    e = Engine(GaussEvolutionProducer(process), StatisticsConsumer(statistics=_BootstrapStatistics, process=process))
    r = e.run(grid=[0., time], num_of_paths=1000, num_of_workers=None)[-1][-1]
    print((list(r.values())))

if False:
    start, drift, vol, time = 1., 0.1, .5, 1.
    process = TimeDependentGeometricBrownianMotion(drift, vol, start)
    e = Engine(GaussEvolutionProducer(process), StatisticsConsumer(statistics=_ConvergenceStatistics))
    r = e.run(grid=[0., time], num_of_paths=1000, num_of_workers=None)[-1][-1]
    print(r)

if False:
    start, drift, vol, time = 1., 0.1, .5, 1.

    process = TimeDependentGeometricBrownianMotion(drift, vol, start)
    e = Engine(GaussEvolutionProducer(process), StatisticsConsumer())
    mean, median, stdev, variance = list(), list(), list(), list()
    for seed in range(100):
        print(seed, end='')
        r = e.run(grid=[0., time], seed=seed, num_of_paths=1000, num_of_workers=None)[-1][-1]
        mean.append(r.mean)
        median.append(r.median)
        stdev.append(r.stdev)
        variance.append(r.variance)

    print('')
    print(('mean     >\n', _Statistics(mean, mean=process.mean(time))))
    print(('median   >\n', _Statistics(median, mean=process.median(time))))
    print(('stdev    >\n', _Statistics(stdev, mean=sqrt(process.variance(time)))))
    print(('variance >\n', _Statistics(variance, mean=process.variance(time))))

if False:
    n = 10
    grid = list(range(n))
    path = 20000

    # s, t = [0.3427338525545087, 0.6572661474454913], [[0.16046606, 0.83953394], [0.46142568, 0.53857432]]
    # s, t = (0.5, 0.5, .0), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
    # s, t = (1., 0., 0.), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
    # s, t = (0., 1., 0.), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
    # s, t = (0., 0., 1.), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
    # s, t = (0.2, 0.6, 0.2), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
    # s, t = (0.5, 0., .5), ((.5, .5, .0), (.25, .5, .25), (.0, .5, .5))
    # s, t = (.5, .5), ((.8, .2), (.3, .7))
    # s, t = (0., 1.), ((.8, .2), (.3, .7))
    # s, t = (.5, .5), ((.8, .2), (.2, .8))
    f = (.0, .1, .1)

if False:
    p = AugmentedFiniteStateMarkovChain.random(5)
    # p = FiniteStateMarkovChain.random(5)
    print(np.matrix(p._underlying_covariance(1)))
    print(p.variance(1))

if False:
    # process = FiniteStateMarkovChain(transition=t, start=s)
    process = FiniteStateMarkovChain.random(5)

    producer = GaussEvolutionProducer(process)
    consumer = StatisticsConsumer(statistics=_MultiStatistics)
    stats = Engine(producer, consumer).run(grid, path)

    print('')
    for p, s in stats:
        theory = process.mean(p)
        practise = s.mean
        diff = np.asarray(theory) - np.asarray(practise)
        error = max(diff.max(), -diff.min())
        print('')
        print('mean    ', p)
        print('theory  ', theory)
        print('practise', practise)
        print('error   ', error)
        # assert abs(error) < 1e-2

    print('')
    for p, s in []:
        # for p, s in stats:
        theory = process.variance(p)
        practise = s.variance
        diff = np.asarray(theory) - np.asarray(practise)
        error = max(diff.max(), -diff.min())
        print('')
        print('variance', p)
        print('theory  ', theory)
        print('practise', practise)
        print('error   ', error)
        # assert abs(error) < 1e-2

if False:
    augmentation = (lambda x: 1. if x == 3 else 0.)
    transition = [
        [0.7, 0.2, 0.099, 0.001],
        [0.2, 0.5, 0.29, 0.01],
        [0.1, 0.2, 0.6, 0.1],
        [0.0, 0.0, 0.0, 1.0]
    ]
    r_squared = 1.0
    start = [.3, .2, .5, 0.]

    underlying = FiniteStateMarkovChain(transition, r_squared, start)
    process = AugmentedFiniteStateMarkovChain(underlying, augmentation)
    print(process)
    process.start = [1., 1., 1., 0.]
    print(process.start)
    print(underlying.start)

    producer = GaussEvolutionProducer(process)
    consumer = StatisticsConsumer(func=process.eval)
    stats = Engine(producer, consumer).run(grid, path)

    print('')
    for p, s in stats:
        theory = process.mean(p)
        practise = s.mean
        diff = practise - theory
        error = diff
        print('')
        print('mean    ', p)
        print('theory  ', theory)
        print('practise', practise)
        print('error   ', error)
        # assert abs(error) < 1e-2

    print('')
    for p, s in stats:
        theory = process.variance(p)
        practise = s.variance
        diff = practise - theory
        error = diff
        print('')
        print('variance', p)
        print('theory  ', theory)
        print('practise', practise)
        print('error   ', error)
        # assert abs(error) < 1e-2

if False:
    def do_test(t):
        c = MultiGaussEvolutionProducerUnitTests(t)
        c.setUp()
        getattr(c, t)()
        # c.test_multi_gauss_process()
        c.tearDown()


    do_test('test_wiener_process')
    do_test('test_multi_gauss_process')
    do_test('test_correlation')
