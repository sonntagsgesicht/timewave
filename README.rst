
Python library *timewave*
-------------------------

.. image:: https://img.shields.io/codeship/f23aa6b0-ba22-0137-5b78-0e0bdbe34106/master.svg
   :target: https://codeship.com//projects/364772
   :alt: CodeShip

.. image:: https://travis-ci.org/sonntagsgesicht/timewave.svg?branch=master
   :target: https://travis-ci.org/sonntagsgesicht/timewave
   :alt: Travis ci

.. image:: https://readthedocs.org/projects/timewave/badge
   :target: http://timewave.readthedocs.io
   :alt: Read the Docs

.. image:: https://img.shields.io/codefactor/grade/github/sonntagsgesicht/timewave/master
   :target: https://www.codefactor.io/repository/github/sonntagsgesicht/timewave
   :alt: CodeFactor Grade

.. image:: https://img.shields.io/codeclimate/maintainability/sonntagsgesicht/timewave
   :target: https://codeclimate.com/github/sonntagsgesicht/timewave/maintainability
   :alt: Code Climate maintainability

.. image:: https://img.shields.io/codecov/c/github/sonntagsgesicht/timewave
   :target: https://codecov.io/gh/sonntagsgesicht/timewave
   :alt: Codecov

.. image:: https://img.shields.io/lgtm/grade/python/g/sonntagsgesicht/timewave.svg
   :target: https://lgtm.com/projects/g/sonntagsgesicht/timewave/context:python/
   :alt: lgtm grade

.. image:: https://img.shields.io/lgtm/alerts/g/sonntagsgesicht/timewave.svg
   :target: https://lgtm.com/projects/g/sonntagsgesicht/timewave/alerts/
   :alt: total lgtm alerts

.. image:: https://img.shields.io/github/license/sonntagsgesicht/timewave
   :target: https://github.com/sonntagsgesicht/timewave/raw/master/LICENSE
   :alt: GitHub

.. image:: https://img.shields.io/github/release/sonntagsgesicht/timewave?label=github
   :target: https://github.com/sonntagsgesicht/timewave/releases
   :alt: GitHub release

.. image:: https://img.shields.io/pypi/v/timewave
   :target: https://pypi.org/project/timewave/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/timewave
   :target: https://pypi.org/project/timewave/
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/dm/timewave
   :target: https://pypi.org/project/timewave/
   :alt: PyPI Downloads


a stochastic process evolution simulation engine in python.

simulation engine
=================

timewave consists of four building blocks.

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

Development Version
-------------------

The latest development version can be installed directly from GitHub:

.. code-block:: bash

    $ pip install --upgrade git+https://github.com/sonntagsgesicht/timewave.git


Contributions
-------------

.. _issues: https://github.com/sonntagsgesicht/timewave/issues
.. __: https://github.com/sonntagsgesicht/timewave/pulls

Issues_ and `Pull Requests`__ are always welcome.


License
-------

.. __: https://github.com/sonntagsgesicht/timewave/raw/master/LICENSE

Code and documentation are available according to the Apache Software License (see LICENSE__).
