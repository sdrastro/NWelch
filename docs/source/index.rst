.. NWelch documentation master file, created by
   sphinx-quickstart on Fri Sep 20 22:28:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NWelch documentation
====================

**NWelch** uses Welch's method to estimate the power spectra,
complex cross-spectrum, magnitude-squared coherence, and phase
spectrum of bivariate time series with nonuniform observing
cadence. For univariate time series, users can apply the Welch's
power spectrum estimator or compute a nonuniform fast Fourier
transform-based periodogram. Options include tapering in the
time domain and computing bootstrap false alarm levels. Users
may choose standard 50%-overlapping Welch's segments or design 
their own segmentation scheme. NWelch was designed for Doppler
planet searches but may be applied to any type of time series.
Plotting utilities, Siegel's test for periodicity, and spectral
window visualization are included.

.. image:: https://img.shields.io/badge/GitHub-sdrastro-blue
    :target: https://github.com/sdrastro/NWelch

.. image:: http://img.shields.io/badge/License-GPLv3-blue.svg?style=flat
    :target: https://github.com/sdrastro/NWelch/blob/main/LICENSE


.. toctree::
   :maxdepth: 2

   dependencies
   installation
   quickstart


Contributors
------------

Developed by `Sarah Dodson-Robinson <https://github.com/sdrastro>`_ with contributions from:

* `Charlotte Haley <https://github.com/lootie>`_
* `Victor Ramirez Delgado <https://www.researchgate.net/scientific-contributions/Victor-Ramirez-Delgado-2180522477>`_
* `Justin Harrell <https://udel.academia.edu/justinharrell>`_


License & Citation
------------------

Copyright 2022-2024 Sarah Dodson-Robinson and contributors.

NWelch is being actively developed in `a public GitHub
repository <https://github.com/sdrastro/NWelch/tree/main>`_.
The source code is made available under the terms of the GPL-3.0 license.

If you use NWelch in your research, please cite
`Magnitude-squared Coherence: A Powerful Tool for Disentangling
Doppler Planet Discoveries from Stellar Activity <https://ui.adsabs.harvard.edu/abs/2022AJ....163..169D/abstract>`_:

.. code-block:: tex

   @ARTICLE{2022AJ....163..169D,
          author = {{Dodson-Robinson}, Sarah E. and {Delgado}, Victor Ramirez and {Harrell}, Justin and {Haley}, Charlotte L.},
           title = "{Magnitude-squared Coherence: A Powerful Tool for Disentangling Doppler Planet Discoveries from Stellar Activity}",
         journal = {\aj},
        keywords = {Time series analysis, Period search, Astrostatistics techniques, Radial velocity, Stellar activity, Stellar rotation, 1916, 1955, 1886, 1332, 1580, 1629, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics, Statistics - Methodology},
            year = 2022,
           month = apr,
          volume = {163},
          number = {4},
             eid = {169},
           pages = {169},
             doi = {10.3847/1538-3881/ac52ed},
   archivePrefix = {arXiv},
          eprint = {2201.13342},
    primaryClass = {astro-ph.EP},
          adsurl = {https://ui.adsabs.harvard.edu/abs/2022AJ....163..169D},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }
