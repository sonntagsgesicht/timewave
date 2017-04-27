# -*- coding: utf-8 -*-

#  timewave
#  ------------
#  timewave, a classical time evolution simulation engine in python.
#
#  Author:  pbrisk <pbrisk@icloud.com>
#  Copyright: 2016, 2017 Deutsche Postbank AG
#  Website: https://github.com/pbrisk/timewave
#  License: APACHE Version 2 License (see LICENSE file)


import codecs
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='timewave',
    description='timewave, a classical time evolution simulation engine in python.',
    version='0.4',
    author='Deutsche Postbank AG [pbrisk]',
    author_email='pbrisk@icloud.com',
    url='https://github.com/pbrisk/timewave',
    bugtrack_url='https://github.com/pbrisk/timewave/issues',
    license='Apache License 2.0',
    packages=['timewave'],
    long_description=codecs.open('README.rst', encoding='utf-8').read(),
    platforms='any',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Education',
        'Topic :: Office/Business',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Utilities',
        'Topic :: Office/Business :: Scheduling',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Localization',
    ], requires=['numpy', 'scipy']
)
