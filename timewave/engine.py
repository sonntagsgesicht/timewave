# coding=utf-8
"""
module containing simulation method related classes incl. multiprocessing support
"""
from cProfile import runctx
from copy import copy
from random import Random

try:  # try accepted due to lack of multiprocessing on iOS Pythonista
    from multiprocessing import cpu_count, current_process, Process, Queue

    CPU_COUNT = cpu_count()
except ImportError:
    cpu_count, current_process, Process, Queue = None, None, None, None
    CPU_COUNT = None


class Producer(object):
    """
    abstract class implementing simple producer for a model between grid dates
    """

    def __init__(self, func=None, initial_state=None):
        if func is None:
            func = (lambda s, d: s.value)
        self.func = func

        if initial_state is None:
            initial_state = State()
        self.initial_state = initial_state

        self.random = Random()
        self.grid = None
        self.num_of_paths = None
        self.seed = None
        self.state = None

    def initialize(self, grid, num_of_paths, seed):
        """ inits producer for a simulation run """
        self.grid = grid
        self.num_of_paths = num_of_paths
        self.seed = seed
        if self.initial_state.date is None:
            self.initial_state.date = grid[0]

    def initialize_worker(self, process_num=None):
        """ inits producer for a simulation run on a single process """
        self.initial_state.process = process_num
        self.random.seed(hash(self.seed) + hash(process_num))

    def initialize_path(self, path_num=None):
        """ inits producer for next path, i.e. sets current state to initial state"""
        self.state = copy(self.initial_state)
        self.state.path = path_num
        self.random.seed(hash(self.seed) + hash(path_num))

    def evolve(self, new_date):
        """
        evolve to the new process state at the next date, i.e. do one step in the simulation

        :param date new_date: date of the new state
        :return State:
        """
        if self.state.date == new_date and not self.initial_state.date == new_date:
            return self.state
        self.state.value = self.func(self.state, new_date)
        self.state.date = new_date
        return self.state


class State(object):
    """
    simulation state
    """

    def __init__(self, value=0.0):
        super(State, self).__init__()
        self.value = value
        self.date = None
        self.process = None
        self.path = None


class Engine(object):
    """
    This class implements Monte Carlo engine
    """

    def __init__(self, producer=None, consumer=None):

        assert isinstance(producer, Producer) and isinstance(consumer, Consumer)
        self.producer = producer
        self.consumer = consumer

        self.grid = None
        self.num_of_paths = None
        self.num_of_workers = None
        self.seed = None

    def run(self, grid=None, num_of_paths=2000, seed=0, num_of_workers=CPU_COUNT, profiling=False):
        """
        implements simulation

        :param list(date) grid: list of Monte Carlo grid dates
        :param int num_of_paths: number of Monte Carlo paths
        :param hashable seed: seed used for rnds initialisation (additional adjustment in place)
        :param int or None num_of_workers: number of parallel workers (default: cpu_count()),
            if None no parallel processing is used
        :param bool profiling: signal whether to use profiling, True means used, else not
        :return object: final consumer state

        It returns a list of lists.
        The list contains per path a list produced by consumer at observation dates
        """
        self.grid = sorted(set(grid))
        self.num_of_paths = num_of_paths
        self.num_of_workers = num_of_workers
        self.seed = seed

        # pre processing
        self.producer.initialize(self.grid, self.num_of_paths, self.seed)
        self.consumer.initialize(self.grid, self.num_of_paths, self.seed)

        if num_of_workers:
            # processing
            workers = list()
            queue = Queue()
            path_per_worker = int(num_of_paths // num_of_workers)
            start_path, stop_path = 0, path_per_worker
            for i in range(num_of_workers):
                if i == num_of_workers - 1:
                    stop_path = num_of_paths  # ensure exact num of path as required
                name = 'worker-%d' % i
                if profiling:
                    # display profile with `snakeviz worker-0.prof`
                    # if not installed `pip install snakeviz`
                    workers.append(Process(target=self._run_parallel_process_with_profiling,
                                           name=name,
                                           args=(start_path, stop_path, queue, name + '.prof')))
                else:
                    workers.append(Process(target=self._run_parallel_process,
                                           name=name,
                                           args=(start_path, stop_path, queue)))
                start_path, stop_path = stop_path, stop_path + path_per_worker
            for worker in workers:
                worker.start()

            # post processing
            for _ in range(num_of_workers):
                self.consumer.get(queue.get())
            for worker in workers:
                worker.join()
        else:
            self._run_process(0, num_of_paths)

        self.consumer.finalize()
        return self.consumer.result

    def _run_parallel_process_with_profiling(self, start_path, stop_path, queue, filename):
        """
        wrapper for usage of profiling
        """
        runctx('Engine._run_parallel_process(self,  start_path, stop_path, queue)', globals(), locals(), filename)

    def _run_parallel_process(self, start_path, stop_path, queue):
        """
        The function calls _run_process and puts results produced by
        consumer at observations of top most consumer in to the queue
        """
        process_num = int(current_process().name.split('-', 2)[1])
        self._run_process(start_path, stop_path, process_num)
        queue.put(self.consumer.put())

    def _run_process(self, start_path, stop_path, process_num=0):
        """
        The function calls _run_path for given set of paths
        """
        # pre processing
        self.producer.initialize_worker(process_num)
        self.consumer.initialize_worker(process_num)

        # processing
        for path in range(start_path, stop_path):
            self._run_path(path)

        # post processing
        self.consumer.finalize_worker(process_num)

    def _run_path(self, path_num):
        """
        standalone function implementing a single loop of Monte Carlo
        It returns list produced by consumer at observation dates

        :param int path_num: path number
        """
        # pre processing
        self.producer.initialize_path(path_num)
        self.consumer.initialize_path(path_num)

        # processing
        for new_date in self.grid:
            state = self.producer.evolve(new_date)
            self.consumer.consume(state)

        # post processing
        self.consumer.finalize_path(path_num)


class Consumer(object):
    """
    base class for simulation consumers
    """

    def __init__(self, func=None):
        """
        initiatlizes consumer by providing a function
        :param func: consumer function with exact 1 argument
        which will consume the producer state. Default will return `state.value`

        :type func: callable
        """
        if func is None:
            func = (lambda s: s.value)
        self.func = func
        self.initial_state = list()
        self.state = list()
        self.result = list()
        self.num_of_paths = None
        self.grid = None
        self.seed = None

    def initialize(self, grid=None, num_of_paths=None, seed=None):
        """
        initialize consumer for simulation
        :param num_of_paths: number of path
        :type num_of_paths: int
        :param grid: list of grid point
        :type grid: list(date)
        :param seed: simulation seed
        :type seed: hashable
        """
        self.num_of_paths = num_of_paths
        self.grid = grid
        self.seed = seed
        self.result = list()
        self.state = self.initial_state

    def initialize_worker(self, process_num=None):
        """
        reinitialize consumer for process in multiprocesing
        """
        self.initialize(self.grid, self.num_of_paths, self.seed)

    def initialize_path(self, path_num=None):
        """
        initialize consumer for next path
        """
        self.state = copy(self.initial_state)
        return self.state

    def consume(self, state):
        """
        consume new producer state
        """
        self.state.append(self.func(state))
        return self.state

    def finalize_path(self, path_num=None):
        """
        finalize last path for consumer
        """
        self.result.append((path_num, self.state))

    def finalize_worker(self, process_num=None):
        """
        finalize process for consumer
        """
        pass

    def finalize(self):
        """
        finalize simulation for consumer
        """
        # todo sort self.result by path_num
        if self.result:
            self.result = sorted(self.result, key=lambda x: x[0])
            p, r = map(list, zip(*self.result))
            self.result = r

    def put(self):
        """
        to put state into multiprocessing.queue
        """
        return self.result

    def get(self, queue_get):
        """
        to get states from multiprocessing.queue
        """
        if isinstance(queue_get, (tuple, list)):
            self.result.extend(queue_get)
