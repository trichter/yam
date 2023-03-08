---
title: 'yam: Yet another monitoring tool using correlations of ambient noise'
tags:
  - Python
  - geophysics
  - seismology
  - cross-correlation
  - ambient noise
  - monitoring
  - seismic velocity changes
authors:
  - name: Tom Eulenfeld
    orcid: 0000-0002-8378-559X
    affiliation: 1
affiliations:
 - name: Institute of Geosciences, Friedrich Schiller University Jena, Germany
   index: 1
date: 8 March 2023
bibliography: paper.bib

---

# Summary

Monitoring with cross-correlations of ambient noise is a popular technique for investigating relative velocity changes in the local subsurface [@SensSchoenfelder2006].
The method combines two concepts -- Green's function retrieval between two receivers by cross-correlating an isotropic, homogeneous noise field recorded at the two receivers [@Weaver2002; @Shapiro2004] and velocity monitoring using coda wave interferometry [@Snieder2002].
For monitoring, the condition of homogeneity and isotropy of the noise field can be relaxed in in favor of the more convenient condition of constancy of the noise sources.

``yam`` is an ObsPy based [@obspy] Python package for correlating seismic recordings of ambient vibrations and for the monitoring of relative seismic velocity changes.
The package is designed to be run from the command line for off-line use, but also includes capabilities for processing a continuously growing dataset.
``yam`` does not rely on a database, but checks on the fly which results already exist and which results still need to be calculated.
Cross-correlations are written to HDF5 files using the ObsPy plugin obspyh5.
This makes it easy to access the correlation data after computation using ObsPy's ``read()`` function.
Correlations can also be exported to various seismic formats to allow the determination of surface wave dispersion curves.
The analysis of changes in the cross-correlation functions is implemented using the stretching procedure.
One of the strengths of the code is the configuration, which is declared in a simple but heavily commented JSON file.
It is possible to declare similar configurations without explicit repetition.
A possible use case is to reprocess an entire dataset in a different frequency band or to stretch with the same parameters using a different time window.
Parts of this code have been successfully used in @Richter2014 to estimate velocity changes induced by ground shaking and thermal stresses, and in @SensSchoenfelder2019 to estimate velocity changes induced by tidal stresses.

Another popular package for velocity monitoring using correlations of ambient vibrations is MSNoise [@msnoise], which is especially useful for large datasets and continuous monitoring because it uses a sqlite or mysql database. The ``yam`` package does not use this concept in favor of a concise code base.

``yam`` is intended for both researchers and graduate students of seismology.
The package can be installed from PyPI; online documentation and tutorials are available on the project website.

# References
