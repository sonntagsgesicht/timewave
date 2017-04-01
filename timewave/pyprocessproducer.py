# coding=utf-8
"""
module containing pyprocess related producer classes
"""

from numpy import random

from engine import Producer


class PyProcessProducer(Producer):
    """
    class implementing general stochastic process between grid dates
    """
    def __init__(self, py_process):
        super(PyProcessProducer, self).__init__()
        self.random = random
        self._process = py_process
        self._current_sample = list()
        self._times = list()

    def initialize(self, grid, num_of_paths, seed):
        super(PyProcessProducer, self).initialize(grid, num_of_paths, seed)
        zero = self.grid[0] + self._process.startTime
        self._times = [d - zero for d in self.grid]

    def initialize_path(self, path_num=None):
        super(PyProcessProducer, self).initialize_path(path_num)
        self._current_sample = self._process.sample_path(self._times)[0].tolist()

    def evolve(self, new_date):
        """
        evolve to the new process state at the next date

        :param date new_date: date or point in time of the new state
        :param float step: random number for step
        :return State:
        """
        if self.state.date == new_date and not self.initial_state.date == new_date:
            return self.state

        i = self.grid.index(new_date)
        self.state.value = self._current_sample[i]
        self.state.date = new_date
        return self.state


class SimplePyProcessProducer(Producer):
    """
    class implementing alternative stochastic process between grid dates
    """
    def __init__(self, py_process):
        super(SimplePyProcessProducer, self).__init__()
        self.random = random
        self._process = py_process

    def evolve(self, new_date):
        """
        evolve to the new process state at the next date

        :param date new_date: date or point in time of the new state
        :param float step: random number for step
        :return State:
        """
        if self.state.date == new_date and not self.initial_state.date == new_date:
            return self.state

        dt = float(new_date - self.grid[0])
        self.state.value = self._process.sample_position(dt)
        self.state.date = new_date
        return self.state
