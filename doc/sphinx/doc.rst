
-----------------
API Documentation
-----------------

.. toctree::
    :glob:


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


Stochastic Process Simulation
=============================

.. inheritance-diagram:: stochasticproducer

.. inheritance-diagram:: stochasticconsumer

.. automodule:: stochasticproducer

.. automodule:: stochasticconsumer


Stochastic Process Definition
=============================

.. autosummary::
    :nosignatures:

    stochasticprocess.base.StochasticProcess
    stochasticprocess.base.MultivariateStochasticProcess


.. inheritance-diagram:: stochasticprocess.gauss
.. inheritance-diagram:: stochasticprocess.markovchain

.. automodule:: stochasticprocess.base
.. automodule:: stochasticprocess.gauss
.. automodule:: stochasticprocess.markovchain
.. automodule:: stochasticprocess.multifactor
