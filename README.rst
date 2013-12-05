Introduction
============

This is a collection of tools that are helpful for gait analysis. Some are
specific to the needs of the Human Motion and Control Lab at Cleveland State
University but other portions may have potential for general use. It is
relatively modular so you can use what you want.

Modules
=======

**gait**
   Tools for working with gait data.
**motek**
   Tools for processing data from Motek Medical's products, primarily the
   D-Flow software outputs.
**controlid**
   Tools for identifying control systems in human locomotion.

Installation
============

You will need Python 2.7 and setuptools to install the packages. Its best to
install the dependencies first (NumPy, SciPy, matplotlib, Pandas).  The SciPy
Stack instructions are helpful for this: http://www.scipy.org/stackspec.html.

We recommend installing Anaconda for users in our lab to get all of the
dependencies.

We also utilize Octave/Matlab code, so an install of Octave with the toolkits
is also required.

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

A vagrant file and provisioning script are included to test the code on an
Ubuntu 13.10 box. To load the box and run the tests simply type::

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
in these steps:

1. Fork the repository on Github.
2. Clone your copy on Github to your machine, ``git clone git@github.com:<your-username>/Gait-Analysis-Toolkit.git``.
3. Create a new branch off of the master branch, ``git checkout -b my-branch``.
4. Make changes to the software and be sure to always include tests!
5. Make sure all tests pass on your machine, ``nosetests``.
6. Add any new files to your local git repo, ``git add my_new_file.py``
7. Commit your changes, ``git commit -am "Added an amazing new feature."``.
8. Push your changes to a new branch on your Github repository, ``git push origin my-branch``.
9. Make a pull request on Github to ``csu-hmc/Gait-Analysis-Toolkit``.
10. Make sure all tests pass on Travis and have an author review and merge your
    pull request.
11. Delete the branches on your local machine and on your Github repo, ``git branch -d my-branch && git push origin :my-branch``.

Release Notes
=============

0.1.0
-----

- Included Octave source to compute inverse 2D dynamics.
- Copied the walk module from DynamicistToolKit @ eecaebd31940179fe25e99a68c91b75d8b8f191f
