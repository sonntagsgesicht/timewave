
.. module:: timewave

-----------------
API Documentation
-----------------

.. toctree::
    :glob:


Timewave Engine
===============

.. py:currentmodule:: timewave.engine

.. autosummary::
    :nosignatures:

    Engine
    State
    Producer
    Consumer

.. automodule:: timewave.engine


Timewave Producer
=================

.. py:currentmodule:: timewave.producers

.. autosummary::
    :nosignatures:

    DeterministicProducer
    StringReaderProducer
    MultiProducer

.. inheritance-diagram:: timewave.producers
    :parts: 1

.. automodule:: timewave.producers


Timewave Consumer
=================

.. py:currentmodule:: timewave.consumers

.. autosummary::
    :nosignatures:

    QuietConsumer
    StringWriterConsumer
    StackedConsumer
    ConsumerConsumer
    MultiConsumer
    ResetConsumer
    TransposedConsumer

.. inheritance-diagram:: timewave.consumers
    :parts: 1

.. automodule:: timewave.consumers


Stochastic Process Simulation
=============================

.. py:currentmodule:: timewave.stochasticproducer

.. inheritance-diagram:: timewave.stochasticproducer
    :parts: 1


.. py:currentmodule:: timewave.stochasticconsumer

.. inheritance-diagram:: timewave.stochasticconsumer
    :parts: 1


.. automodule:: timewave.stochasticproducer

.. automodule:: timewave.stochasticconsumer


Stochastic Process Definition
=============================

.. py:currentmodule:: timewave

.. autosummary::
    :nosignatures:

    stochasticprocess.base.StochasticProcess
    stochasticprocess.base.MultivariateStochasticProcess


.. inheritance-diagram:: timewave.stochasticprocess.gauss
    :parts: 1

.. inheritance-diagram:: timewave.stochasticprocess.markovchain
    :parts: 1

.. automodule:: timewave.stochasticprocess.base
.. automodule:: timewave.stochasticprocess.gauss
.. automodule:: timewave.stochasticprocess.markovchain
.. automodule:: timewave.stochasticprocess.multifactor
