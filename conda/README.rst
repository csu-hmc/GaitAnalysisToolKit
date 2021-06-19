This directory contains conda recipes for the GaitAnalysisToolKit. There is one
recipe for each version of the software. Note that an octave conda package does
not yet exist so the oct2py install relies on octave being in your path via
another means. You will need to either build an oct2py conda package or make
sure you have a binstar channel with one available because oct2py doesn't exist
in anaconda's main channel.

For example, you can build a package on 64 bit Linux like so::

   $ conda build gaitanalysistoolkit-0.1.2

and upload it to binstart (adjust the path to reflect you anaconda/miniconda
installation)::

   $ binstar upload ~/anaconda/conda-bld/linux-64/gaitanalysistoolkit-0.1.2-0.tar.bz2

and install with::

   $ conda install -c <your-binstar-username> gaitanalysistoolkit
