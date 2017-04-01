# coding=utf-8
"""
module containing brownian motion model related classes
"""
from json import loads

from engine import Producer


# classical producer


class MultiProducer(Producer):
    def __init__(self, *producers):
        """
        initializer

        :param list(Producer) or produces: list of producers to be used one after another
        """
        super(MultiProducer, self).__init__()

        #: list(Producer): list of consumers to be used one after another
        self.producers = list()

        for p in producers:
            if isinstance(p, (tuple, list)):
                self.producers.extend(p)
            else:
                self.producers.append(p)
        for p in self.producers:
            if isinstance(p, (tuple, list)):
                for pp in p:
                    assert isinstance(pp, Producer)
            else:
                assert isinstance(p, Producer)

        self.initial_state = [p.initial_state for p in self.producers]

    def initialize(self, grid, num_of_paths, seed):
        """ inits producer for a simulation run """
        for p in self.producers:
            p.initialize(grid, num_of_paths, seed)
        self.grid = grid
        self.num_of_paths = num_of_paths
        self.seed = seed
        # self.initial_state.date = grid[0]

    def initialize_worker(self, process_num=None):
        """ inits producer for a simulation run on a single process """
        for p in self.producers:
            p.initialize_worker(process_num)
        # self.initial_state.process = process_num
        self.random.seed(hash(self.seed) + hash(process_num))

    def initialize_path(self, path_num=None):
        """ inits producer for next path, i.e. sets current state to initial state"""
        for p in self.producers:
            p.initialize_path(path_num)
        # self.state = copy(self.initial_state)
        # self.state.path = path_num
        self.random.seed(hash(self.seed) + hash(path_num))

    def evolve(self, new_date):
        """
        evolve to the new process state at the next date, i.e. do one step in the simulation

        :param date new_date: date of the new state
        :return State:
        """
        self.state = [p.evolve(new_date) for p in self.producers]
        return self.state


class DeterministicProducer(Producer):
    def __init__(self, sample_list, func=None, initial_state=None):
        func = (lambda s, i: sample_list[s.path][i])
        super(DeterministicProducer, self).__init__(func, initial_state)
        self.grid = range(len(sample_list[0]))
        self.num_of_paths = len(sample_list)

    def evolve(self, new_date):
        return super(DeterministicProducer, self).evolve(self.grid.index(new_date))


class StringReaderProducer(DeterministicProducer):
    def __init__(self, data_str, str_decoder=None):
        if str_decoder is None:
            str_decoder = loads
        data = str_decoder(data_str)
        super(StringReaderProducer, self).__init__(data)
