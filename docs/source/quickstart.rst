Quickstart
----------

Be sure to check out the tutorial Jupyter notebooks:

1. `Periodograms and Welch's power spectrum estimates of a single time
series
<https://github.com/sdrastro/NWelch/blob/main/demo/TimeSeries_demo.ipynb>`_

2. `Magnitude-squared coherence and power spectrum estimates from two
simultaneous, unevenly sampled time series
<https://github.com/sdrastro/NWelch/blob/main/GJ581/GJ581_coherence.ipynb>`_
with standard Welch's 50% overlapping segments and Blackman-Harris
tapers

3.  `Magnitude-squared coherence and power spectrum estimates from time
series with large gaps
<https://github.com/sdrastro/NWelch/blob/main/aCenB/aCenB_coherence_linear.ipynb>`_
with demonstrations of detrending and user-defined segments

4. `Magnitude-squared coherence, power spectra, and phase spectra
<https://github.com/sdrastro/NWelch/blob/main/GJ3998/GJ3998_coherence.ipynb>`_
with 50% overlapping segments and Blackman-Harris tapers

**Import NWelch in one of two ways**

If you have the NWelch package installed:

.. code-block:: python

   from NWelch.TimeSeries import TimeSeries

If you weren't able to install the NWelch package and
are relying on the source code:

.. code-block:: python

   import sys
   sys.path.insert(0, 'your/path/to/NWelch/src')
   from TimeSeries import TimeSeries

**Simple periodogram of a univariate time series**

This is like a Lomb-Scargle periodogram, but is computed with
the `non-uniform fast Fourier transform
<https://finufft.readthedocs.io/en/latest/index.html>`_.

.. code-block:: python

   # Create a TimeSeries object
   # Constructor outputs different estimates of a pseudo-Nyqyist
   #   frequency to help you define your frequency grid
   ts = TimeSeries(timestamps, observations)

   # Construct a frequency grid for your periodogram estimate
   #   Choose the maximum frequency that suits your science goals
   ts.frequency_grid(0.5) 

   # Compute the complex-valued Fourier transform, periodogram,
   #   and bootstrap false alarm levels
   # Default is 10,000 bootstrap iterations; specify a different
   #   number with the N_bootstrap keyword
   ts.pow_FT()

   # Plot your periodogram
   ts.powplot()

   # If desired, plot the complex-valued Fourier transform
   ts.Ftplot()



**Welch's power spectrum estimate**

Continued from above

.. code-block:: python

   # Specify the segments, frequency grid, and tapers/windows for
   #   your Welch's power spectrum estimate - this estimator will
   #   use 3 Kaiser-Bessel tapered segments with 50% overlap
   #   and have a maximum frequency of 0.5
   ts.segment_data(3, 0.5, window='KaiserBessel')

   # Estimate the power spectrum
   ts.Welch_powspec()

   # Optionally, compute bootstrap false alarm levels
   #   Default is 10,000 iterations; use N_Welch_bootstrap keyword
   #   to choose a different number
   ts.Welch_powspec_bootstrap()

   # Plot your Welch's power spectrum estimate
   ts.powplot(Welch=True)



**Magnitude-squared estimate from two simultaneous time series**

Phase spectrum and Welch's power spectrum estimates from both time
series are computed simultaneously.

Import the Bivariate class with

.. code-block:: python

   from NWelch.Bivariate import Bivariate

or, if you don't have the NWelch package installed, just

.. code-block:: python

   from Bivariate import Bivariate

   # Create a Bivariate object, which is a combination of two
   #   TimeSeries objects (two_ts)
   two_ts = Bivariate(timestamps, observations1, observations2)

   # Set segmentation scheme, maximum frequency, tapers / windows,
   #   and detrending options
   # plot_windows keyword allows you to visualize the tapers /
   #   windows applied to each segment to make sure they retain
   #   their "bell" shapes
   two_ts.segment_data(4, 0.25, window='BlackmanHarris',
       plot_windows=True)

   # Estimate power spectra and magnitude-squared coherence
   two_ts.Welch_coherence_powspec()

   # Estimate false alarm levels
   #   Specify number of bootstrap iterations with N_coh_bootstrap
   #   keyword - default is 10,000
   two_ts.Welch_coherence_powspec_bootstrap()

   # Plot your power spectra
   two_ts.Welch_pow_plot(x_or_y='x')
   two_ts.Welch_pow_plot(x_or_y='y')

   # Plot coherence estimate
   two_ts.coh_plot()

   # Plot phase spectrum
   two_ts.phase_plot()



**Extended functionality**


*Siegel's test for periodicity* - this is an extension of the
Fisher (1929) test, which is valid only if the time series
traces a single oscillation. Siegel's test works for time series
with up to three oscillations.

.. code-block:: python

   '''For the simple periodogram'''
   # Conservative option, designed for two periodicities
   ts.Siegel_test()

   # Works with up to 3 periodicities, but more prone to false positives
   ts.Siegel_test(tri=True) 

   '''For the Welch power spectrum estimate'''
   ts.Siegel_test(Welch=True)
   ts.Siegel_test(Welch=True, tri=True)



*Estimate and plot the spectral window* - when the time series
sampling is extremely uneven, a sinusoid doesn't create a simple
peak in the periodogram; instead it might create a pitchfork or
zigzag shape, or something even weirder. Visualizing the
spectral window helps you understand the characteristics of your
power spectrum / coherence estimator and interpret your results.

.. code-block:: python

   # For the simple periodogram
   ts.spectral_window()

   # For a Welch's power spectrum estimate
   ts.spectral_window_Welch()



*Quick scatter plot of your time series*

.. code-block:: python

   ts.scatterplot()


*Plot a histogram of log10(timesteps)* - 
This is useful when you're trying to assess the "unevenness" of
your time series - sometimes in astronomy we see timestep
histograms covering six orders of magnitude

.. code-block:: python

   ts.dthist()


Note - all methods in the TimeSeries class can be applied to
either or both series in a Bivariate object:

.. code-block:: python

   two_ts.x_series.scatterplot()
   two_ts.y_series.Siegel_test()


This quickstart guide will get you up and running, but NWelch is more
customizable than is demonstrated here. For example, you can detrend
each Welch's segment with a quadratic, or turn off detrending entirely.
You can add vertical lines to your power spectrum and coherence plots at
frequencies of interest. You can specify the axis labels on your plots,
or change the y-axis scale in power spectrum plots from logarithmic to
linear. Most importantly, you can save your results to an ASCII
file. Each method in the TimeSeries class is thoroughly documented
in `TimeSeries_demo.ipynb
<https://github.com/sdrastro/NWelch/blob/main/demo/TimeSeries_demo.ipynb>`_,
while each method in the Bivariate class is demonstrated in
`GJ581_coherence.ipynb
<https://github.com/sdrastro/NWelch/blob/main/GJ581/GJ581_coherence.ipynb>`_
(for non-astronomers, GJ 581 is an M-dwarf star that my group analyzed
in the `paper accompanying NWelch
<https://ui.adsabs.harvard.edu/abs/2022AJ....163..169D/abstract>`_).

API documentation is on my to-do list.
