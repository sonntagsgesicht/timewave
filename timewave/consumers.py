# -*- coding: utf-8 -*-

# timewave
# --------
# timewave, a stochastic process evolution simulation engine in python.
# 
# Author:   sonntagsgesicht, based on a fork of Deutsche Postbank [pbrisk]
# Version:  0.5, copyright Saturday, 14 September 2019
# Website:  https://github.com/sonntagsgesicht/timewave
# License:  Apache License 2.0 (see LICENSE file)


from itertools import izip
from json import dumps

from engine import Consumer


class QuietConsumer(Consumer):
    """QuietConsumer returns nothing, since QuietConsumer does simply not populate result in finalize_path()"""

    def finalize_path(self, path_num=None):
        """QuietConsumer does simply not populate result in finalize_path()"""
        pass

    def finalize(self):
        self.result = list()


class StringWriterConsumer(Consumer):

    def __init__(self, str_decoder=None):
        if str_decoder is None:
            str_decoder = dumps
        self.decoder = str_decoder
        super(StringWriterConsumer, self).__init__()

    def finalize(self):
        """
        finalize simulation for consumer
        """
        super(StringWriterConsumer, self).finalize()
        self.result = self.decoder(self.result)


class ResetConsumer(Consumer):
    """
        FunctionConsumer that admits a reset function for each path

    """

    def __init__(self, fixing_func=None, reset_func=None):
        if reset_func is None:
            reset_func = lambda: None
        self.initialize_path_call = reset_func
        super(ResetConsumer, self).__init__(lambda s: fixing_func(s.date))

    def initialize_path(self, path_num=None):
        self.initialize_path_call()
        super(ResetConsumer, self).initialize_path(path_num)

    def finalize(self):
        self.initialize_path_call()
        super(ResetConsumer, self).finalize()


class StackedConsumer(Consumer):
    """stacked version of consumer, i.e. a following consumer is populated with out state of the preceding one"""

    def __init__(self, *consumers):
        super(StackedConsumer, self).__init__()
        self.consumers = list()
        for c in consumers:
            if isinstance(c, (tuple, list)):
                self.consumers.extend(c)
            else:
                self.consumers.append(c)
        for c in self.consumers:
            if isinstance(c, (tuple, list)):
                for cc in c:
                    assert isinstance(cc, Consumer)
            else:
                assert isinstance(c, Consumer)

    def initialize(self, num_of_paths=None, grid=None, seed=None):
        """initialize StackedConsumer"""
        super(StackedConsumer, self).initialize(grid, num_of_paths, seed)
        for c in self.consumers:
            c.initialize(grid, num_of_paths, seed)
        self.state = [c.state for c in self.consumers]

    def initialize_path(self, path_num=None):
        """
        make the consumer_state ready for the next MC path

        :param int path_num:
        """
        for c in self.consumers:
            c.initialize_path(path_num)

    def consume(self, state):
        for c in self.consumers:
            state = c.consume(state)

    def finalize_path(self, path_num=None):
        """finalize path and populate result for ConsumerConsumer"""
        self.consumers[-1].finalize_path(path_num)
        self.result = self.consumers[-1].result

    def finalize(self):
        """finalize for ConsumerConsumer"""
        self.consumers[-1].finalize()
        self.result = self.consumers[-1].result

    def put(self):
        return self.consumers[-1].put()

    def get(self, queue_get):
        self.consumers[-1].get(queue_get)


class ConsumerConsumer(Consumer):
    """
    class implementing the consumer interface
    several consumers can be saved and are executed one after another
    only the result of the last consumer is returned (see finalize_worker)
    """

    def __init__(self, *consumers):
        """
        initializer

        :param list(Consumer):
        """
        super(ConsumerConsumer, self).__init__()

        self.consumers = list()
        for c in consumers:
            if isinstance(c, (tuple, list)):
                self.consumers.extend(c)
            else:
                self.consumers.append(c)
        for c in self.consumers:
            if isinstance(c, (tuple, list)):
                for cc in c:
                    assert isinstance(cc, Consumer)
            else:
                assert isinstance(c, Consumer)
        #: list(Consumer): list of consumers to be used one after another
        self.initial_state = [c.initial_state for c in self.consumers]

    def initialize(self, grid=None, num_of_paths=None, seed=None):
        """initialize ConsumerConsumer"""
        super(ConsumerConsumer, self).initialize(grid, num_of_paths, seed)
        for c in self.consumers:
            c.initialize(grid, num_of_paths, seed)
        self.state = [c.state for c in self.consumers]

    def initialize_path(self, path_num=None):
        """
        make the consumer_state ready for the next MC path

        :param int path_num:
        """
        for c in self.consumers:
            c.initialize_path(path_num)
        self.state = [c.state for c in self.consumers]

    def consume(self, state):
        """
        returns pair containing the result of consumption and consumer state
        the returned state is equal to the state.get_short_rate()
        the returned consume state is None

        :param State state: specific process state
        :return object: the new consumer state
        """
        self.state = [c.consume(state) for c in self.consumers]
        return self.state

    def finalize_path(self, path_num=None):
        """finalize path and populate result for ConsumerConsumer"""
        for c in self.consumers:
            c.finalize_path(path_num)
        self.result = [c.result for c in self.consumers]

    def finalize(self):
        """finalize for ConsumerConsumer"""
        for c in self.consumers:
            c.finalize()
        self.result = [c.result for c in self.consumers]

    def get(self, queue_get):
        """
        get to given consumer states.
        This function is used for merging of results of parallelized MC.
        The first state is used for merging in place. The states must be disjoint.

        :param object queue_get: second consumer state
        """
        for (c, cs) in izip(self.consumers, queue_get):
            c.get(cs)
        self.result = [c.result for c in self.consumers]


class MultiConsumer(ConsumerConsumer):
    def consume(self, state):
        self.state = [c.consume(s) for c, s in zip(self.consumers, state)]
        return self.state


class TransposedConsumer(Consumer):
    """
        TransposedConsumer returns sample distribution per grid point not per sample path
    """

    def finalize(self):
        """finalize for PathConsumer"""
        super(TransposedConsumer, self).finalize()
        self.result = map(list, zip(*self.result))  # transpose result
