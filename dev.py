from test import MultiGaussEvolutionProducerUnitTests

def do_test(t):

    c = MultiGaussEvolutionProducerUnitTests(t)
    c.setUp()
    getattr(c, t)()
    #c.test_multi_gauss_process()
    c.tearDown()


#do_test('test_wiener_process')
#do_test('test_multi_gauss_process')
do_test('test_correlation')
