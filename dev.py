import numpy as np

from test import MultiGaussEvolutionProducerUnitTests
from timewave import FiniteStateMarkovChain, FiniteStateAffineTimeMarkovChain
from timewave import GaussEvolutionProducer, StatisticsConsumer, Engine
from timewave.stochasticconsumer import _MultiStatistics


def do_test(t):
    c = MultiGaussEvolutionProducerUnitTests(t)
    c.setUp()
    getattr(c, t)()
    # c.test_multi_gauss_process()
    c.tearDown()


# do_test('test_wiener_process')
# do_test('test_multi_gauss_process')
# do_test('test_correlation')


n = 10
grid = range(n)
path = 20000

s, t = [0.3427338525545087, 0.6572661474454913], [[0.16046606, 0.83953394], [0.46142568, 0.53857432]]
s, t = (0.5, 0.5, .0), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
# s, t = (1., 0., 0.), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
# s, t = (0., 1., 0.), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
# s, t = (0., 0., 1.), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
# s, t = (0.2, 0.6, 0.2), ((.75, .25, .0), (.25, .5, .25), (.0, .25, .75))
# s, t = (0.5, 0., .5), ((.5, .5, .0), (.25, .5, .25), (.0, .5, .5))
# s, t = (.5, .5), ((.8, .2), (.3, .7))
# s, t = (0., 1.), ((.8, .2), (.3, .7))
# s, t = (.5, .5), ((.8, .2), (.2, .8))
f = (.0, .1, .1)

# process = FiniteStateMarkovChain(transition=t, start=s)
process = FiniteStateMarkovChain.random(5)
# process = FiniteStateAffineTimeMarkovChain(transition=t, fix=f, start=s)

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
