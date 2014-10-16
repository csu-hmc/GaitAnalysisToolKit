Introduction
============

This is a collection of tools that are helpful for gait analysis. Some are
specific to the needs of the Human Motion and Control Lab at Cleveland State
University but other portions may have potential for general use. It is
relatively modular so you can use what you want. It is primarily structured as
a Python distribution but the Octave/Matlab files are also accessible
independently.

.. image:: https://travis-ci.org/csu-hmc/Gait-Analysis-Toolkit.png?branch=master
   :target: http://travis-ci.org/csu-hmc/Gait-Analysis-Toolkit

Python Packages
===============

The main Python package is ``gaitanalysis`` and it contains five modules listed
below. ``oct2py`` is used to call Octave routines in the Python code where
needed.

``gait.py``
   General tools for working with gait data such as gait landmark
   identification and 2D inverse dynamics. The main class is ``WalkingData``.
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

Octave/Matlab Librarys
======================

Several Octave/Matlab routines are included in the ``Octave-Matalb-Codes``
directory.

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

You will need Python 2.7 and setuptools to install the packages. Its best to
install the dependencies first (NumPy, SciPy, matplotlib, Pandas, PyTables).
The SciPy Stack instructions are helpful for this:
http://www.scipy.org/stackspec.html.

Supported versions:

- python>=2.7
- numpy>=1.6.1
- scipy>=0.9.0
- matplotlib>=1.1.0
- tables>=2.3.1
- pandas>=0.12.0
- pyyaml>=3.10
- DynamicistToolKit>=0.3.5
- oct2py>=1.2.0
- octave>=3.8.1

We recommend installing Anaconda_ for users in our lab to get all of the
dependencies.

.. _Anaconda: http://docs.continuum.io/anaconda/

We also utilize Octave/Matlab code, so an install of Octave with is also
required. See http://octave.sourceforge.net/index.html for installation
instructions.

You can install using pip (or easy_install). Pip will theoretically [#]_ get
the dependencies for you (or at least check if you have them)::

   $ pip install https://github.com/csu-hmc/Gait-Analysis-Toolkit/zipball/master

Or download the source with your preferred method and install manually.

Using Git::

   $ git clone git@github.com:csu-hmc/Gait-Analysis-Toolkit.git
   $ cd Gait-Analysis-Toolkit

Or wget::

   $ wget https://github.com/csu-hmc/Gait-Analysis-Toolkit/archive/master.zip
   $ unzip master.zip
   $ cd Gait-Analysis-Toolkit-master

Then for basic installation::

   $ python setup.py install

Or install for development purposes::

   $ python setup.py develop

.. [#] You will need all build dependencies and also note that matplotlib
       doesn't play nice with pip.

Tests
=====

Run the tests with nose::

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

Contributing
============

The recommended procedure for contributing code to this repository is detailed
here. It is the standard method of contributing to Github based repositories
(https://help.github.com/articles/fork-a-repo).

If you have don't have access rights to this repository then you should fork
the repository on Github using the Github UI and clone the fork that you just
made to your machine::

   git clone git@github.com:<your-username>/Gait-Analysis-Toolkit.git

Change into the directory::

   cd Gait-Analysis-Toolkit

Now, setup a remote called ``upstream`` that points to the main repository so
that you can keep your local repository up-to-date::

   git remote add upstream git@github.com:csu-hmc/Gait-Analysis-Toolkit.git

Now you have a remote called ``origin`` (the default) which points to **your**
Github account's copy and a remote called ``upstream`` that points to the main
repository on the csu-hmc organization Github account.

It's best to keep your local master branch up-to-date with the upstream master
branch and then branch locally to create new features. To update your local
master branch simply::

   git checkout master
   git pull upstream master

If you have access rights to the main repository simply, clone it and don't
worry about making a fork on your Github account::

   git clone git@github.com:csu-hmc/Gait-Analysis-Toolkit.git

Change into the directory::

   cd Gait-Analysis-Toolkit

Now, to contribute a change to the repository you should create a new branch
off of the local master branch::

   git checkout -b my-branch

Now make changes to the software and be sure to always include tests! Make sure
all tests pass on your machine with::

   nosetests

Once tests pass, add any new files you created::

   git add my_new_file.py

Now commit your changes::

   git commit -am "Added an amazing new feature."

Push your commits to a mirrored branch on the Github repository that you
cloned::

   git push origin my-branch

Now visit the repository on Github (either yours or the main one) and you
should see a "compare and pull button" to make a pull request against the main
repository. Github and Travis-CI will check for merge conflicts and run the
tests again on a cloud machine. You can ask others to review your code at this
point and if all is well, press the "merge" button on the pull request.
Finally, delete the branches on your local machine and on your Github repo::

   git branch -d my-branch && git push origin :my-branch

Git Notes
---------

- The master branch on main repository on Github should always pass all tests
  and we should strive to keep it in a stable state. It is best to not merge
  contributions into master unless tests are passing, and preferably if
  someone else approved your code.
- In general, do not commit changes to your local master branch, always pull in
  the latest changes from the master branch with ``git pull upstream master``
  then checkout a new branch for your changes. This way you keep your local
  master branch up-to-date with the main master branch on Github.
- In general, do not push changes to the main repo master branch directly, use
  branches and push the branches up with a pull request.
- In general, do not commit binary files, files generated from source, or large
  data files to the repository. See
  https://help.github.com/articles/working-with-large-files for some reasons.

Release Notes
=============

0.1.0
-----

- Included Octave/Matlab source for inertial compensation.
- Included Octave/Matlab source to compute inverse 2D dynamics.
- Copied the walk module from DynamicistToolKit @ eecaebd31940179fe25e99a68c91b75d8b8f191f
