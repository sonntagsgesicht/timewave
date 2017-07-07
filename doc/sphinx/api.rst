
-----------------
API Documentation
-----------------

.. toctree::


Timewave Engine
===============

.. autosummary::
    :nosignatures:

    engine.Engine
    engine.State
    engine.Producer
    engine.Consumer

.. automodule:: engine


Timewave Producer
=================

.. autosummary::
    :nosignatures:

    producers.DeterministicProducer
    producers.StringReaderProducer
    producers.MultiProducer

.. inheritance-diagram:: producers

.. automodule:: producers


Timewave Consumer
=================

.. autosummary::
    :nosignatures:

    consumers.QuietConsumer
    consumers.StringWriterConsumer
    consumers.StackedConsumer
    consumers.ConsumerConsumer
    consumers.MultiConsumer
    consumers.ResetConsumer
    consumers.TransposedConsumer

.. inheritance-diagram:: consumers

.. automodule:: consumers


Stochastic Process
==================

.. inheritance-diagram:: stochasticprocess

.. automodule:: stochasticprocess


Stochastic Process Simulation
=============================

.. inheritance-diagram:: stochasticproducer

.. inheritance-diagram:: stochasticconsumer

.. automodule:: stochasticproducer

.. automodule:: stochasticconsumer

