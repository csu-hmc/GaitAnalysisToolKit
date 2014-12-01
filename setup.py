#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

exec(open('gaitanalysis/version.py').read())

description = \
    """Various tools for gait analysis used at the Cleveland State
University Human Motion and Control Lab."""

setup(name='GaitAnalysisToolKit',
      author='Jason K. Moore',
      author_email='moorepants@gmail.com',
      version=__version__,
      url="http://github.com/csu-hmc/GaitAnalysisToolKit",
      description=description,
      license='LICENSE.txt',
      packages=find_packages(),
      install_requires=['numpy>=1.6.1',
                        'scipy>=0.9.0',
                        'matplotlib>=1.1.0',
                        'tables>=2.3.1',
                        'pandas>=0.12.0',
                        'pyyaml>=3.10',
                        'DynamicistToolKit>=0.3.5',
                        'oct2py>=1.2.0'],
      extras_require={'doc': ['sphinx>=1.1.3',
                              'numpydoc>=0.4'],
                      },
      scripts=['bin/dflowdata'],
      tests_require=['nose>1.3.0'],
      test_suite='nose.collector',
      long_description=open('README.rst').read(),
      classifiers=[
                   'Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Physics',
                  ],
      )
