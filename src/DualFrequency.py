import numpy as np
import matplotlib.pyplot as plt

# Build on existing NWelch code
from TimeSeries import TimeSeries
from Bivariate import Bivariate

def dual_frequency(data):

    # Check for correct input type and Fourier coefficients
    try:
        if isinstance(data, TimeSeries):
            if (data.Welch_pow is None):
                print("You must calculate Welch's Fourier coefficients and power spectrum using TimeSeries.Welch_powspec() before computing dual-frequency autocoherence. Exiting.")
                return
            else:
                print('Input is single time series; output will be dual-frequency autocoherence')
                output_type = 'autocoherence'
        elif isinstance(data, Bivariate):
            if (data.coh is None):
                print("You must calculate Welch's cross-spectrum and power spectrum using Bivariate.Welch_coherence_powspec() before computing dual-frequency cross-coherence. Exiting.")
                return
            else:
                print('Input contains two contemporaneous time series; output will be dual-frequency cross-coherence.')
                output_type = 'cross-coherence'
        else:
            raise TypeError
    except TypeError:
        print('Input must be a TimeSeries or Bivariate object.  Exiting.')
        return

    # Extract the Welch's segments' Fourier coefficients and autospectra
    # Note: Fourier coefficients have NOT been divided by the sum of the weights
    if output_type == 'autocoherence':
        Fourier_coeffs1 = data.Welch_Fourier_coeffs
        Fourier_coeffs2 = data.Welch_Fourier_coeffs
    else:
        Fourier_coeffs1 = data.x_series.Welch_Fourier_coeffs
        Fourier_coeffs2 = data.y_series.Welch_Fourier_coeffs
                
    dual_cross = []
    for k in range(data.Nseg):
        dual_cross.append(np.outer(np.conj(Fourier_coeffs1[k]), Fourier_coeffs2[k]))
    
    dual_cross_mean = np.mean(np.array(dual_cross), axis=0)
    dual_auto  = np.outer(np.mean(np.abs(Fourier_coeffs1)**2, axis=0), \
                          np.mean(np.abs(Fourier_coeffs2)**2, axis=0))

    return np.abs(dual_cross_mean)**2 / dual_auto