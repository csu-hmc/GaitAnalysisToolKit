Introduction
============

This is a collection of tools that are helpful for gait analysis. Some are
specific to the needs of the Human Motion and Control Lab at Cleveland State
University but other portions may have potential for general use. It is
relatively modular so you can use what you want. It is primarily structured as
a Python distribution but the Octave files are also accessible independently.

.. image:: https://img.shields.io/pypi/v/gaitanalysistoolkit.svg
    :target: https://pypi.python.org/pypi/gaitanalysistoolkit/
    :alt: Latest Version

.. image:: https://zenodo.org/badge/6017/csu-hmc/GaitAnalysisToolKit.svg
   :target: http://dx.doi.org/10.5281/zenodo.13006

.. image:: https://travis-ci.org/csu-hmc/GaitAnalysisToolKit.png?branch=master
   :target: http://travis-ci.org/csu-hmc/GaitAnalysisToolKit

Python Packages
===============

The main Python package is ``gaitanalysis`` and it contains five modules listed
below. ``oct2py`` is used to call Octave routines in the Python code where
needed.

``gait.py``
   General tools for working with gait data such as gait landmark
   identification and 2D inverse dynamics. The main class is ``GaitData``.
``controlid.py``
   Tools for identifying control mechanisms in human locomotion.
``markers.py``
   Routines for processing marker data.
``motek.py``
   Tools for processing and cleaning data from `Motek Medical`_'s products,
   e.g. the D-Flow software outputs.
``utils.py``
   Helper functions for the other modules.

.. _Motek Medical: http://www.motekmedical.com

Each module has a corresponding test module in ``gaitanalysis/tests``
sub-package which contain unit tests for the classes and functions in the
respective module.

Octave Libraries
================

Several Octave routines are included in the ``gaitanalysis/octave`` directory.

``2d_inverse_dynamics``
   Implements joint angle and moment computations of a 2D lower body human.
``inertial_compensation``
   Compensates force plate forces and moments for inertial effects and
   re-expresses the forces and moments in the camera reference frame.
``mmat``
   Fast matrix multiplication.
``soder``
   Computes the rigid body orientation and location of a group of markers.
``time_delay``
   Deals with the analog signal time delays.

Installation
============

You will need Python 2.7 or 3.7+ and setuptools to install the packages. Its
best to install the dependencies first (NumPy, SciPy, matplotlib, Pandas,
PyTables).

Supported versions:

- python >= 2.7 or >= 3.7
- numpy >= 1.8.2
- scipy >= 0.13.3
- matplotlib >= 1.3.1
- tables >= 3.1.1
- pandas >= 0.13.1, <= 0.24.0
- pyyaml >= 3.10
- DynamicistToolKit >= 0.4.0
- oct2py >= 2.4.2
- octave >= 3.8.1

We recommend installing Anaconda_ for users in our lab to get all of the
dependencies.

.. _Anaconda: http://docs.continuum.io/anaconda/

We also utilize Octave code, so an install of Octave with is also required. See
http://octave.sourceforge.net/index.html for installation instructions.

You can install using pip (or easy_install). Pip will theoretically [#]_ get
the dependencies for you (or at least check if you have them)::

   $ pip install https://github.com/csu-hmc/GaitAnalysisToolKit/zipball/master

Or download the source with your preferred method and install manually.

Using Git::

   $ git clone git@github.com:csu-hmc/GaitAnalysisToolKit.git
   $ cd GaitAnalysisToolKit

Or wget::

   $ wget https://github.com/csu-hmc/GaitAnalysisToolKit/archive/master.zip
   $ unzip master.zip
   $ cd GaitAnalysisToolKit-master

Then for basic installation::

   $ python setup.py install

Or install for development purposes::

   $ python setup.py develop

.. [#] You will need all build dependencies and also note that matplotlib
       doesn't play nice with pip.

Dependencies
------------

It is recommended to install the software dependencies as follows:

Octave can be installed from your package manager or from a downloadable
binary, for example on Debian based Linux::

   $ sudo apt-get install octave

For oct2py to work, calling Octave from the command line should work after
Octave is installed. For example,

::

   $ octave
   GNU Octave, version 3.8.1
   Copyright (C) 2014 John W. Eaton and others.
   This is free software; see the source code for copying conditions.
   There is ABSOLUTELY NO WARRANTY; not even for MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  For details, type 'warranty'.

   Octave was configured for "x86_64-pc-linux-gnu".

   Additional information about Octave is available at http://www.octave.org.

   Please contribute if you find this software useful.
   For more information, visit http://www.octave.org/get-involved.html

   Read http://www.octave.org/bugs.html to learn how to submit bug reports.
   For information about changes from previous versions, type 'news'.

   octave:1>

The core dependencies can be installed with conda in a conda environment::

   $ conda create -n gait python=2.7 pip numpy scipy matplotlib pytables pandas pyyaml nose sphinx numpydoc oct2py mock
   $ source activate gait

And the dependencies which do not have conda packages can be installed into the
environment with pip::

   (gait)$ pip install DynamicistToolKit

Tests
=====

When in the repository directory, run the tests with nose::

   $ nosetests

Vagrant
=======

A vagrant file and provisioning script are included to test the code on both a
Ubuntu 12.04 and Ubuntu 13.10 box. To load the box and run the tests simply
type::

   $ cd vagrant
   $ vagrant up

See ``VagrantFile`` and the ``*bootstrap.sh`` files to see what's going on.

Documentation
=============

The documentation is hosted at ReadTheDocs:

http://gait-analysis-toolkit.readthedocs.org

You can build the documentation (currently sparse) if you have Sphinx and
numpydoc::

   $ cd docs
   $ make html
   $ firefox _build/html/index.html

Release Notes
=============

0.2.0
-----

- Support Python 3. [PR `#149`_]
- Minimum dependencies bumped to Ubuntu 14.04 LTS versions and tests run on
  latest conda forge packages as of 2018/08/30. [PR `#140`_]
- The minimum version of the required dependency, DynamicistToolKit, was bumped
  to 0.4.0. [PR `#134`_]
- Reworked the DFlowData class so that interpolation and resampling is based on
  the FrameNumber column in the mocap data instead of the unreliable TimeStamp
  column. [PR `#135`_]
- Added note and setup.py check about higher oct2py versions required for
  Windows.

.. _#149: https://github.com/csu-hmc/GaitAnalysisToolKit/pull/149
.. _#134: https://github.com/csu-hmc/GaitAnalysisToolKit/pull/134
.. _#135: https://github.com/csu-hmc/GaitAnalysisToolKit/pull/135
.. _#140: https://github.com/csu-hmc/GaitAnalysisToolKit/pull/140

0.1.2
-----

- Fixed bug preventing GaitData.plot_grf_landmarks from working.
- Removed inverse_data.mat from the source distribution.

0.1.1
-----

- Fixed installation issue where the octave and data files were not included in
  the installation directory.

0.1.0
-----

- Initial release
- Copied the walk module from DynamicistToolKit @ eecaebd31940179fe25e99a68c91b75d8b8f191f
