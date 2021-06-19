#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join

from setuptools import setup, find_packages

exec(open('gaitanalysis/version.py').read())

description = ("Various tools for gait analysis used at the Cleveland State "
               "University Human Motion and Control Lab.")

octave_sub_dirs = ['2d_inverse_dynamics',
                   join('2d_inverse_dynamics', 'test'),
                   'inertial_compensation',
                   join('inertial_compensation', 'test'),
                   'mmat',
                   'soder',
                   'time_delay']

file_type_globs = ['*.m', '*.mat', '*.txt']

octave_rel_paths = []
for sub in octave_sub_dirs:
    for glob in file_type_globs:
        octave_rel_paths.append(join('octave', sub, glob))

install_requires = ['numpy>=1.8.2',
                    'scipy>=0.13.3',
                    'matplotlib>=1.3.1',
                    'tables>=3.1.1',
                    'pandas>=0.13.1,<0.24',
                    'pyyaml>=3.10',
                    'DynamicistToolKit>=0.4.0',
                    'oct2py>=2.4.2']

setup(name='GaitAnalysisToolKit',
      author='Jason K. Moore',
      author_email='moorepants@gmail.com',
      version=__version__,
      url="http://github.com/csu-hmc/GaitAnalysisToolKit",
      description=description,
      license='LICENSE.txt',
      packages=find_packages(),
      install_requires=install_requires,
      extras_require={'doc': ['sphinx>=1.2.2',
                              'numpydoc>=0.4',
                              'mock>=1.0.1'],
                      },
      scripts=['bin/dflowdata'],
      # The following ensures that any of these files are installed to the
      # system location.
      package_data={'gaitanalysis': octave_rel_paths,
                    'gaitanalysis.tests': [join('data', glob) for glob in
                                           ['*.txt', '*.csv', '*.yml']]},
      tests_require=['nose>1.3.1'],
      test_suite='nose.collector',
      long_description=open('README.rst').read(),
      classifiers=[
                   'Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.7',
                   'Topic :: Scientific/Engineering :: Physics',
                  ],
      )
