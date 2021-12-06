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

---

## A note about finufft

Installation on a Mac can be tricky, as until very recently Apple disabled multithreading support in the C compiler bundled with Xcode. This [StackOverflow entry](https://stackoverflow.com/questions/58344183/how-can-i-install-openmp-on-my-new-macbook-pro-with-mac-os-catalina) suggests a couple of hacks that will get you a working version of libomp; I went the gcc route. After compiling finufft from source with gcc9, I then needed to invoke 

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

in all code that uses finufft because for some reason python thinks there are multiple openmp libraries on my system. Since setting os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' for multithread computations is an [unsupported, potentially unsafe workaround](https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-when-fitting-models), all calls to the nufft1d3 function (NWelch's workhorse for calculating non-uniform FFTs) force a single-thread computation using the kewyord nthreads=1. The planet-search datasets for which NWelch was designed are rarely large enough for single-thread NFFTs to cause a huge slowdown. But if you have a Linux system with a working openmp library that gives you no problem installing finufft, you can probably speed up NWelch by editing the source code: remove os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' from both Bivariate.py and TimeSeries.py, then remove nthreads=1 from the fft function defined at the top of TimeSeries.py. (For Linux machines or Macs with proper multithreading, there's no problem with leaving the source code alone; you just might have slightly slower NFFTs.) This software has been tested mostly on a Mac, with limited tests on Linux and Windows.

---
