# Copyright 2017-2018 Tom Eulenfeld, GPLv3
"""
yam Documentation
=================


Motivation
----------

Why another monitoring tool for seismic velocities using ambient noise cross-correlations?

There are several alternatives around, namely `MSNoise <https://github.com/ROBelgium/MSNoise>`_
and MICC `MIIC <https://github.com/miic-sw/miic>`_.
*MSNoise* is especially useful for large datasets and continuous monitoring. Configuration and the state of a project
is managed by sqlite or mysql database. A project can be configured via web interface,
commands are issued via command line interface. Velocity variations are determined with the
Moving Window Cross Spectral technique (MWCS).
*MIIC* is another monitoring library using the time-domain stretching technique.

*Yam*, contrary to MSNoise, is designed for off-line usage, but also includes capabilities to reprocess continuously
growing data. Yam does not rely onto a database, but rather checks on the fly which results already exist and which
results have still to be calculated.
Cross-correlations are written to HDF5 files via the ObsPy plugin obspyh5. Thus, correlation data can be easily
accessed with ObsPy's |read| function after the calculation. It follows a similar processing flow as MSNoise,
but it uses the stretching library from MIIC. (It is of course feasible to implement MWCS.)
One of its strong points is the configuration declared in a simple, but heavily commented JSON file.
It is possible to declare similar configurations.
A possible use case is the reprocessing of the whole dataset in a different frequency band.
Some code was reused from previous project `sito <https://github.com/trichter/sito>`_.


Installation
------------

Dependencies of yam are ``obspy>=1.1 obspyh5>=0.3 h5py``.
Optional dependencies are ``IPython`` and ``cartopy``.
The recommended way to install yam is via `anaconda <https://docs.anaconda.com/anaconda/install.html>`_ and pip::

    conda --add channels conda-forge
    conda create -n yam cartopy h5py IPython matplotlib numpy obspy scipy tqdm
    source activate yam
    pip install yam

After that, you can run the tests with ``yam-runtests`` and check if everything is installed properly.


How to use yam
--------------

The scripts are started with the command line program ``yam``.
``yam -h`` gives an overview over available commands and options. Each command has its own help,
e.g. ``yam correlate -h`` will print help for the ``correlate`` command.

``create`` will create an example configuration file in JSON format.
The processing commands are ``correlate``, ``stack`` and ``stretch``.

``info``, ``print``, ``load`` and ``plot`` commands allow to inspect correlations,
stacks and stretching results as well as preprocessed data and other aspects.
``remove`` removes correlations or stretching results (necessary if configuration changed).

Correlations, corresponding stacks and stretching results are saved in HDF5 files.
The indices inside the HDF5 files are the following (first for correlations, second for stretching results)::

    '{key}/{network1}.{station1}-{network2}.{station2}/{location1}.{channel1}-{location2}.{channel2}/{starttime.datetime:%Y-%m-%dT%H:%M}'
    '{key}/{network1}.{station1}-{network2}.{station2}/{location1}.{channel1}-{location2}.{channel2}'

The strings are expanded with the corresponding metadata.
Several tools are available for analysing the contents of the HDF5 files, e.g. h5ls or hdfview.


About keys and different configurations
***************************************

``key`` in the above indices and as a parameter in the command line interface
is a special parameter which describes the processing chain.
It is best explained with an example: A key could be ``c1_s2d_twow``.
This means data was correlated (``c``) with configuration ``1``, each two days ``2d`` are stacked (``s``) and
finally data was stretched (``t``) using the stretching configuration ``wow``.
The configuration of the keys are defined in the configuration file.
(These ids may not contain ``_``, because ``_`` is used to separate the different processing steps.)
The ``s`` key is special, because it can describe the stacking procedure directly:
For example, ``s5d`` stacks correlations of 5 days,
``s2h`` of 2 hours, ``s5dm2.5d`` is a 2.5 day moving (``m``) stack over 5 days,
with ``d`` corresponding to days and ``h`` corresponding to hours.
But ``s`` can also precede a key which is described in the configuration file.

Valid processing chains could be represented by ``c2`` (data is only correlated),
``c2_t2`` (and directly stretched afterwards),
``c1_s10dm5d_t1`` (correlation, moving stack, stretch),
``c1_s1d_s5dm2d`` (correlation, stack, moving stack) or similar.


Tutorial
********

A small tutorial with an example dataset is included.
It can be loaded into an empty directory with ``yam create --tutorial``.
Now you can try out some of the following commands:

.. code:: bash

    yam info               # plot information about project
    yam info stations      # print inventory info
    yam info data          # plot info about data files
    yam plot stations      # plot station map
    yam print data CX.PATCX..BHZ 2010-02-03       # load data for a specific station and day and print information
    yam load data CX.PATCX..BHZ 2010-02-03        # load data for a specific station and day and start an IPython session
    yam plot data CX.PATCX..BHZ 2010-02-03        # plot a day file
    yam plot prepdata CX.PATCX..BHZ 2010-02-03 1  # plot the preprocessed data of the same day
                                                  # (preprocessing defined in corr config 1)
    yam correlate 1        # correlates data with corr configuration 1
    yam correlate 1        # should finish fast, because everything is already calculated
    yam correlate auto     # correlate data with another configuration suitable for auto-correlations
    yam plot c1_s1d --plottype vs_dist  # plot correlation versus distance
    yam plot cauto --plot-options '{"trim": [0, 10]}'  # plot auto-correlations versus time and change some options
                                                       # ("wiggle" plot also possible)
    yam stack c1_s1d 3dm1d       # stack 1 day correlations with a moving stack of 3 days
    yam stack cauto 2            # stack auto-correlations with stack configuration 2

    yam stretch c1_s1d_s3dm1d 1  # stretch the stacked data with stretch configuration 1
    yam stretch cauto_s2 2       # stretch the stacked auto-correlations with another stretch configuration
    yam info                     # find out about the keys which are already in use
    yam plot cauto_t2            # plot similarity matrices for the given processing chain
    yam plot cauto_s2_t2 --plot-options '{"show_line": true}' --show  # plot all similarity matrices for this processing
                                                                      # chain and display them on screen (zoom etc.)
    yam plot c1_s1d_s3dm1d_t1/CX.PATCX-CX.PB01  # plot similarity matrices, but only for one station combination
                                                # (restricting the group is also possible for stacking and stretching)

Of course, the plots do not look overwhelmingly for such a small dataset.

The readme in the Github repository links to a further advanced tutorial.


Use your own data
*****************

Create the example configuration with ``yam create`` and adapt it to your needs.
A good start is to change the ``inventory`` and ``data`` parameters.


Read correlation and results of stretching procedure in Python for further processing
*************************************************************************************

Use ObsPy's |read| to read correlations and stacks and `~yam.commands.read_dicts()` to read stretching results.

::

    from obspy import read
    from yam import read_dicts

    # read a whole file of correlations
    stream = read('corr.h5', 'H5')
    # to read only part of a file
    stream = read('stack.h5', 'H5', include_only=dict(key='c1_s1d', network1='CX', station1='PATCX',
                                                      network2='CX', station2='PB01'))
    # or specify the group explicitly
    stream = read('stack.h5', 'H5', group='c1_s1d')
    # read the stretching results into a dictionary
    stretch_result = read_dicts('stretch.h5', 'c1_s1d_t1')


Configuration options
*********************

Please see the :ref:`example configuration file <config_label>` configuration file
for an explanation of configuration options.
It follows a table with links to functions which consume the options.
All config options should be documented inside these functions.

========================  ========================================
configuration dictionary      functions consuming the options
========================  ========================================
    io                    Configuration for input and output (needed by most functions in `~yam.commands` module)
    correlate             `~yam.commands.start_correlate` -> `~yam.correlate.correlate`  ->  `~yam.correlate.preprocess`  -> `~yam.correlate.time_norm`, `~yam.correlate.spectral_whitening`
    stack                 `~yam.commands.start_stack` -> `~yam.stack.stack`
    stretch               `~yam.commands.start_stretch` -> `~yam.commands.stretch_wrapper` -> `~yam.stretch.stretch`
    plot_*_options        See corresponding functions in `~yam.imaging` module
========================  ========================================

More information about the different subcommands of yam can be found in the corresponding functions in
`~yam.commands` module.

.. shortcuts e.g. for references to Obspy
.. |read| replace:: :func:`~obspy.core.stream.read`
.. |UTC| replace:: `~obspy.core.utcdatetime.UTCDateTime`
.. |Trace| replace:: `~obspy.core.trace.Trace`
.. |Stream| replace:: `~obspy.core.stream.Stream`
.. |Stream.plot| replace:: `Stream.plot() <obspy.core.stream.Stream.plot>`
.. |Inventory| replace:: `~obspy.core.inventory.inventory.Inventory`
.. |Inventory.plot| replace:: `Inventory.plot() <obspy.core.inventory.inventory.Inventory.plot>`
.. |Inventory.select| replace:: `Inventory.filter() <obspy.core.inventory.inventory.Inventory.select>`
.. |Axes.plot| replace:: `Axes.plot() <matplotlib.axes.Axes.plot>`

.. |io| replace:: io configuration dictionary
"""

__version__ = '0.4.3'

from yam.main import run
from yam.commands import read_dicts
