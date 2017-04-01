timewave simulation engine
==========================

timewave, a classical time evolution simulation engine in python. It
consists of four building blocks.

|Codeship Status for pbrisk/pace|

The State
---------

which evolves over time during a simulation path. It is the nucleus or
node which marks a point of time in a path.

The Producer
------------

is the objects that provides states to the simulation and does the
actual time evolution. Think of the producer building as the constructor
of a stochastic process like a Brownian motion or, less mathematical,
future stock prices or future rain intensities.

The Consumer
------------

is an object that takes a state as a point in time provided by the
producer and consumes it, i.e. does something with it - the actual
calculation if you like.

The Engine
----------

finally, which organizes the creation of states by the producer and the
consumption by the consumer. The engine uses, if present,
multiprocessing, i.e. takes full advantage of multi cpu frameworks.
Therefore the engine splits the simulation into equal but distinct
chunks of path for the number of workers (by default the number of cpu)
and loops over the set of dedicated path in each worker. Each path is
evolved by the producer in states which are at each point in time
consumed directly by consumers. States are, due to limits of resources,
not stored during the simulation. If you like to, use the storage
consumer to save all states.

main frame workflow
-------------------

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

.. |Codeship Status for pbrisk/pace| image:: https://codeship.com/projects/e5f5fcb0-9d66-0134-5a6b-6ae80fc9d0de/status?branch=master
   :target: https://codeship.com/projects/188639
