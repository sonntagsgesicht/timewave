
--------
Tutorial
--------

.. toctree::


First step
==========

setup simulation by

::

    engine = Engine(Producer(), Consumer())
    engine.run(range(20))

then run loop starts by

::

    producer/initialize()

setup workers (by default by the number of cpu's) on each worker start
loop by

::

    producer/consumer.initialize_worker()

and invoke loop over paths and start again with

::

    producer/consumer.initialize_path()

then do time evolution of a path

::

    producer.evolve() / consumer.consume()

and finish with last consumer in consumer stack

::

    consumer[-1].finalize_path()

and

::

    consumer[-1].finalize_worker()

put results into queue and take them out by

::

    consumer[-1].put()/get(result)

finish simulation (kind of reduce method)

::

    consumer[-1].finalize()

before returning results from run.
