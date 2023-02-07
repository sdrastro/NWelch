# NWelch
python implementation of Welch's method for estimating the power spectra, complex cross-spectrum, magnitude-squared coherence, and phase spectrum of unevenly spaced, bivariate time series

Available tasks for univariate time series (TimeSeries class):

- Periodogram of time series (similar to Lomb-Scargle, but nonuniform FFT-based), with optional detrending, windowing, and bootstrap false alarm threshold calculation
- Welch's power spectrum estimate either with 50% overlapping segments (standard Welch's method) or custom user-defined segments (original to this work); optional windowing, segment detrending, and bootstrap false alarm levels 
- Siegel's test for periodicity in a time series (can work with up to 3 periodicities)
- Calculate spectral window resulting from uneven time sampling and optional windowing with minimum 4-term Blackman-Harris or Kaiser-Bessel taper
- Plotting utilities: Welch's power spectrum estimate, Lomb Scargle-like periodogram, complex-valued Fourier transform, spectral window, scatter plot of time series, and histogram of time intervals Delta t
- Write results to file

Available tasks for bivariate time series (Bivariate class):

- Estimates of magnitude-squared coherence, cross-spectrum, and power spectra
- Fisher's z-transformation of magnitude-squared coherence
- Coherence false alarm levels from a theoretical distribution (e.g. Schulz and Stattegger 1997)
- Bootstrap false alarm thresholds for coherence and Welch's periodograms, assuming white noise
- Phase spectrum
- Plotting utilities: magnitude-squared coherence, cross-spectrum, and phase
- Write results to a file

Forthcoming:

- Estimation of persistence time (decay time of time-domain autocorrelation function)
- Red noise models for univariate and bivariate data

Contact: sdr [at] udel [dot] edu

Packages required: numpy, matplotlib, scipy, copy, [resample](https://pypi.org/project/resample/), [finufft](https://finufft.readthedocs.io/en/latest/index.html) 

Recommended reading order for demo notebooks:

1. demo/TimeSeries\_demo.ipynb
2. GJ581/GJ581\_coherence.ipynb
3. aCenB/aCenB\_coherence\_linear.ipynb
4. GJ3998/GJ3998\_coherence.ipynb

The GJ 3998 analysis has example phase plots. An analysis of
alpha Cen B with quadratic detrending is included in the aCenB
folder.

If you use NWelch in your research, please cite [Magnitude-squared Coherence: A Powerful Tool for Disentangling Doppler Planet Discoveries from Stellar Activity](https://ui.adsabs.harvard.edu/abs/2022AJ....163..169D/abstract), Dodson-Robinson, S. E.; Ramirez Delgado, V.; Harrell, J.; Haley, C. L. 2022, Astronomical Journal, Volume 163, Issue 4, id.169

Please feel free to contact me with any questions about NWelch or finufft installation, or NWelch operation.

If you are interested in high-quality power spectrum estimators for astronomical data, check out [Multitaper.jl](https://github.com/lootie/Multitaper.jl), a <tt>Julia</tt> package that we used for all calculations in <em>Optimal frequency-domain analysis for spacecraft time series: Introducing the missing-data multitaper power spectrum estimator</em> [Dodson-Robinson and Haley 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv221116549D/abstract). In that paper, we demonstrate how to use the [Chave (2019)](https://academic.oup.com/gji/article/218/3/2165/5519233?login=false) multitaper power spectrum estimator for spacecraft observations, which have regular observing cadence but are usually missing some data (e.g. IMP/AIMP missions, <em>Kepler</em>, K2, TESS). We also extend the Chave (2019) method to multitaper magnitude-squared coherence and complex demodulation.

---

[![DOI](https://zenodo.org/badge/435631370.svg)](https://zenodo.org/badge/latestdoi/435631370)
