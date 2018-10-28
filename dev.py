import numpy as np
from math import exp, log, sqrt
from random import Random

from test import MultiGaussEvolutionProducerUnitTests
from timewave import FiniteStateMarkovChain, AugmentedFiniteStateMarkovChain
from timewave import GaussEvolutionProducer, StatisticsConsumer, Engine
from timewave.stochasticconsumer import _MultiStatistics, _Statistics, _BootstrapStatistics, _ConvergenceStatistics
from timewave import GeometricBrownianMotion, WienerProcess, TimeDependentGeometricBrownianMotion

if False:
    rnd = Random()
    seed = rnd.randint(0, 1000)
    rnd.seed(seed)

    n, start, drift, vol, time = int(1001), 1., .1, .5, 1.
    l, p = (lambda qq, q: start + drift * time + vol * sqrt(time) * q), WienerProcess(drift, vol, start)
    l, p = (lambda qq, q: start * exp(drift * time + vol * sqrt(time) * q)), GeometricBrownianMotion(drift, vol, start)
    l, p = (lambda qq, q: start * exp(drift * time + vol * sqrt(time) * q)), TimeDependentGeometricBrownianMotion(drift, vol, start)

    r = Engine(GaussEvolutionProducer(p), StatisticsConsumer()).run(grid=[0., time], num_of_paths=n)[-1][-1]

    r.description = 'engine vs expected (seed %d)' % seed
    r.expected = p, time
    print r

    r.description = 'engine vs sample (seed %d)' % seed
    r.expected = _Statistics([l(rnd.gauss(0., 1.), rnd.gauss(0., 1.)) for _ in range(n)])
    print r


if True:
    start, drift, vol, time = 1., 0.1, .5, 1.
    statistics = _BootstrapStatistics
    statistics = _ConvergenceStatistics

    process = TimeDependentGeometricBrownianMotion(drift, vol, start)
    e = Engine(GaussEvolutionProducer(process), StatisticsConsumer(statistics=statistics, process=process))
    r = e.run(grid=[0., time], num_of_paths=10000, num_of_workers=None)[-1][-1]
    for b in r:
        print b

if False:
    start, drift, vol, time = 1., 0.1, .5, 1.

    process = TimeDependentGeometricBrownianMotion(drift, vol, start)
    e = Engine(GaussEvolutionProducer(process), StatisticsConsumer())
    mean, median, stdev, variance = list(), list(), list(), list()
    for seed in range(100):
        print seed,
        r = e.run(grid=[0., time], seed=seed, num_of_paths=1000, num_of_workers=None)[-1][-1]
        mean.append(r.mean)
        median.append(r.median)
        stdev.append(r.stdev)
        variance.append(r.variance)

    print ''
    print 'mean     >\n', _Statistics(mean, mean=process.mean(time))
    print 'median   >\n', _Statistics(median, mean=process.median(time))
    print 'stdev    >\n', _Statistics(stdev, mean=sqrt(process.variance(time)))
    print 'variance >\n', _Statistics(variance, mean=process.variance(time))

if False:
    n = 10
    grid = range(n)
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
    print np.matrix(p._underlying_covariance(1))
    print p.variance(1)

if False:
    # process = FiniteStateMarkovChain(transition=t, start=s)
    process = FiniteStateMarkovChain.random(5)

    producer = GaussEvolutionProducer(process)
    consumer = StatisticsConsumer(statistics=_MultiStatistics)
    stats = Engine(producer, consumer).run(grid, path)

    print ''
    for p, s in stats:
        theory = process.mean(p)
        practise = s.mean
        diff = np.asarray(theory) - np.asarray(practise)
        error = max(diff.max(), -diff.min())
        print ''
        print 'mean    ', p
        print 'theory  ', theory
        print 'practise', practise
        print 'error   ', error
        # assert abs(error) < 1e-2

    print ''
    for p, s in []:
        # for p, s in stats:
        theory = process.variance(p)
        practise = s.variance
        diff = np.asarray(theory) - np.asarray(practise)
        error = max(diff.max(), -diff.min())
        print ''
        print 'variance', p
        print 'theory  ', theory
        print 'practise', practise
        print 'error   ', error
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
    print process
    process.start = [1., 1., 1., 0.]
    print process.start
    print underlying.start

    producer = GaussEvolutionProducer(process)
    consumer = StatisticsConsumer(func=process.eval)
    stats = Engine(producer, consumer).run(grid, path)

    print ''
    for p, s in stats:
        theory = process.mean(p)
        practise = s.mean
        diff = practise - theory
        error = diff
        print ''
        print 'mean    ', p
        print 'theory  ', theory
        print 'practise', practise
        print 'error   ', error
        # assert abs(error) < 1e-2

    print ''
    for p, s in stats:
        theory = process.variance(p)
        practise = s.variance
        diff = practise - theory
        error = diff
        print ''
        print 'variance', p
        print 'theory  ', theory
        print 'practise', practise
        print 'error   ', error
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
