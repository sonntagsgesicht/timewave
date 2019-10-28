# -*- coding: utf-8 -*-

# timewave
# --------
# timewave, a stochastic process evolution simulation engine in python.
# 
# Author:   sonntagsgesicht, based on a fork of Deutsche Postbank [pbrisk]
# Version:  0.6, copyright Wednesday, 18 September 2019
# Website:  https://github.com/sonntagsgesicht/timewave
# License:  Apache License 2.0 (see LICENSE file)


__doc__ = 'timewave, a stochastic process evolution simulation engine in python.'
__license__ = 'Apache License 2.0'

__author__ = 'sonntagsgesicht, based on a fork of Deutsche Postbank [pbrisk]'
__email__ = 'sonntagsgesicht@icloud.com'
__url__ = 'https://github.com/sonntagsgesicht/' + __name__

__date__ = 'Wednesday, 18 September 2019'
__version__ = '0.6.1'
__dev_status__ = '4 - Beta'

__dependencies__ = ('numpy', 'scipy', 'dill', 'multiprocessing_on_dill')
__dependency_links__ = ()
__data__ = ()
__scripts__ = ()


import dill as pickle  # enable proper pickle for lambda expressions
from .engine import *
from .producers import *
from .consumers import *
from .stochasticprocess import *
from .stochasticproducer import *
from .stochasticconsumer import *
