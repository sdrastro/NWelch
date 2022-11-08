import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import trim_mean
from finufft import nufft1d3
from resample.bootstrap import resample
from scipy.special import iv # modified Bessel function of the first kind

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Make readable plots
plt.rcParams.update({"font.size":16, "axes.labelsize":16, "font.family":"sans-serif", "font.sans-serif":"Arial"})

# Utility functions for methods in TimeSeries class
# -------------------------------------------------

# ***Non-uniform FFT***
# time: observation times
# data: observed data
# freqs: desired frequency grid
fft = lambda time, data, freqs: nufft1d3(2*np.pi*time, data, freqs, isign=-1, nthreads=1)

# ***Minimum 4-term Blackman-Harris window***
# time: observation times
def BlackmanHarris(time):
    N = len(time)
    fac = N / (N-1)
    tfrac = time / time[-1] # equivalent to n/N in evenly spaced case
    window = 0.35875 - 0.48829*np.cos(2*np.pi*fac*tfrac) + 0.14128*np.cos(4*np.pi*fac*tfrac) - \
             0.01168*np.cos(6*np.pi*fac*tfrac)
    norm = N / np.sum(window**2) # Normalization from Schulz & Stattegger (1997)
    return np.sqrt(window**2 * norm)

# ***Kaiser-Bessel window (delivers high dynamic range)***
# time: observation times
# alpha: window width parameter (high alpha -> high dynamic range, but wide main lobe)
def KaiserBessel(time, alpha=3):
    tscale = time / time[-1]
    arg_denominator = np.pi*alpha
    arg_numerator = arg_denominator * np.sqrt(1. - (2*tscale - 1.)**2)
    return iv(0, arg_numerator) / iv(0, arg_denominator)
    

# ***TimeSeries class: Methods for handling a single time series***
class TimeSeries:
    

    # *** Constructor***
    def __init__(self, t, obs, display_frequency_info=True):
        # t: vector of observation times
        # obs: vector of observed data (i.e. RV, S-index, H-alpha)
        # display_frequency_info: if true, print out various
        # estimates of the Nyquist-like frequency and Rayleigh resolution
        try:
            test = (len(t) == len(obs))
            if (not test):
                raise ValueError
        except ValueError:
            print("Number of observation times must equal number of data points")
            print("No time series object created")
            return
        ind = np.argsort(t) # sort by time
        self.t = t[ind]
        self.obs = obs[ind].astype(complex) # nufft inputs must be complex
        if (self.t[0] != 0.0): # shift so time series starts at t=0
            self.t = self.t - self.t[0]
        self.N = len(self.t)
        self.dt = np.diff(self.t)
        
        # various possibilities for the Nyquist frequency; Rayleigh resolution
        self.Nyq_meandt = 0.5/np.mean(self.dt)
        self.Nyq_tmeandt_10 = 0.5/trim_mean(self.dt, 0.05)
        self.Nyq_tmeandt_20 = 0.5/trim_mean(self.dt, 0.1)
        self.Nyq_meddt = 0.5/np.median(self.dt)
        self.Rayleigh = 1. / self.t[(self.N-1)]
        
        # Frequency grid, Fourier transform, and power spectra are not yet computed
        self.fgrid = None
        self.ft = None
        self.power = None
        self.Welch_pow = None
        self.segmented = False # No segmenting done in preparation for Welch's algorithm
        self.N_Welch_bootstrap = 0
        
        if display_frequency_info:
            print("Nyquist frequency from mean dt:", f"{self.Nyq_meandt:.5f}")
            print("Nyquist frequency from 10% trimmed mean dt:", f"{self.Nyq_tmeandt_10:.5f}")
            print("Nyquist frequency from 20% trimmed mean dt:", f"{self.Nyq_tmeandt_20:.5f}")
            print("Nyquist frequency from median dt:", f"{self.Nyq_meddt:.5f}")
            print("Rayleigh resolution:", f"{self.Rayleigh:.5f}")
            

    # ***Scatter plot of the time series***
    def scatterplot(self, xlabel="Time", ylabel="Data"): 
        # Use keywords xlabel and ylabel to change the default
        #     axis labels
        plt.figure(figsize=(7,5))
        plt.scatter(self.t, self.obs.real, color='k', edgecolor='k', alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        

    # ***Histogram of timesteps dt = t[i] - t[i-1]***
    def dthist(self, nbins=10):
        # Set nbins keyword to change the number of bins in the histogram
        try:
            n_dt_unique = len(np.unique(self.dt))
            test = (n_dt_unique > nbins)
            if (not test):
                raise ValueError
        except ValueError:
            print("Number of unique time intervals:", n_dt_unique, \
                  "  Number of histogram bins requested:", nbins)
            print("Too few unique time intervals - no histogram")
            return
        plt.figure(figsize=(7,5))
        n, bins, _ = plt.hist(np.log10(self.dt), nbins, color='mediumseagreen', alpha=0.7, rwidth=0.9)
        plt.xlabel(r"$\log_{10} [\Delta t]$")
        plt.ylabel("Number of timesteps")


    # ***Compute frequency grid for Fourier transform and power spectrum***
    def frequency_grid(self, Nyquist, oversample=4): 
        # "Nyquist" frequency is defined by user. It can be one of the four possibilities 
        #     computed when the time series object was constructed, or something else entirely.
        # With the oversample keyword, the user can go above (or below) the Rayleigh resolution:
        #     the frequency grid spacing is self.Rayleigh / oversample
        try:
            if Nyquist < 0:
                raise ValueError
        except ValueError:
            print("Nyquist frequency must be float > 0 - no frequency grid calculated")
            return
        try:
            good_oversample = (((type(oversample) is int) or (type(oversample) is float)) \
                              and (oversample > 0))
            if not good_oversample:
                raise ValueError
        except ValueError:
            print("Oversample must be number > 0 - no frequency grid calculated")
            return
        df = self.Rayleigh / oversample
        self.nf = int(Nyquist // df) # Number of POSITIVE (non-zero) frequencies in the grid
        if (self.nf % 2) != 0: # make sure zero frequency is included
            self.nf += 1
        if (Nyquist > self.Nyq_meddt):
            print("Warning: your requested Nyquist frequency is higher than the Nyquist")
            print("frequency associated with the median timestep. Make sure that makes")
            print("sense for your dataset.")
        self.fgrid = np.linspace(-Nyquist, Nyquist, num=2*self.nf+1, endpoint=True)
        # print(self.fgrid)
        self.powfgrid = self.fgrid[self.nf:]
        
        
    # ***Compute a 1-dimensional, type 3, non-uniform Fourier transform,
    #    plus the associated power spectrum and false alarm thresholds***
    def pow_FT(self, window='None', trend=True, trend_type='linear', N_bootstrap=10000, norm=True, save_noise=False, quiet=False):
        # window = 'BlackmanHarris' or 'KaiserBessel', or 'None': apply the selected non-rectangular taper before doing the Fourier transform; set to 'None' to keep the rectangular spectral window
        # If trend == True, subtract out a trend before doing the Fourier transform
        # trend_type: Type of trend to subtract; options are 'linear' and 'quadratic'
        # N_bootstrap: number of bootstrap iterations used to find the false alarm thresholds
        #     N_bootstrap < 100 will turn off bootstrap
        # If norm == True, normalize the power spectrum using Parseval's theorem
        # set save_noise=True to save the array containing highest peaks from the power 
        #     spectra of the randomly shuffled data (useful if you want to examine the noise properties later)
        # if quiet == True, message about bootstrap being off will not appear
        try:
            not_gridded = (self.fgrid is None)
            if not_gridded:
                raise ValueError
        except ValueError:
            print("You must call frequency_grid() before calling pow_FT()")
            print("No Fourier transform or power spectrum calculated")
            return
        
        try:
            valid_window = ((window == 'BlackmanHarris') or (window == 'KaiserBessel') or (window == 'None'))
            if not valid_window:
                raise ValueError
        except ValueError:
            print("Invalid window type. Choose either window='BlackmanHarris',")
            print("window='KaiserBessel', or window='None'. Default is 'None';")
            print("set to 'None' for standard Lomb-Scargle-type periodogram.")
            return
        
        valid_trend_type = ((trend_type == 'linear') or (trend_type == 'quadratic'))
        if not valid_trend_type:
            print("Trend type not understood. Options are 'linear' or 'quadratic'.")
            print('Defaulting to linear trend')
            trend_type = 'linear'
        
        # Center data by subtracting off mean
        datamean = np.mean(self.obs)
        data = self.obs - datamean
        
        # Subtract trend
        self.trend = trend
        self.trend_type = trend_type
        if self.trend:
            if trend_type == 'linear':
                trendcurve = np.poly1d(np.polyfit(self.t, data.real, 1))
            else:
                trendcurve = np.poly1d(np.polyfit(self.t, data.real, 2))
            data = data - trendcurve(self.t)
        detrended_data = data

        # Apply window
        self.window = window
        if (self.window == 'BlackmanHarris'):
            self.win_coeffs = BlackmanHarris(self.t)
        elif (self.window == 'KaiserBessel'):
            self.win_coeffs = KaiserBessel(self.t)
        else:
            self.win_coeffs = np.ones(self.N)
        data = self.win_coeffs * data

        # Transform (not normalized)
        self.ft = fft(self.t, data, self.fgrid)
        
        # Compute the power spectrum
        self.power = np.abs(self.ft[self.nf:])**2
        
        if norm:
            # Normalize: Sum_i (Power_i * df) = variance(data)
            # powfgrid[1] = df
            norm = np.var(data.real) / np.sum((self.powfgrid[1]) * self.power)
            self.power *= norm
        
        # Bootstrap for false alarm thresholds
        try:
            valid_N_bootstrap = ((type(N_bootstrap) is int) and (N_bootstrap >= 100))
            if not valid_N_bootstrap:
                raise ValueError
        except ValueError:
            if not quiet:
                print("Bootstrap off. To turn on, set integer N_bootstrap >= 100")
            self.N_bootstrap = 0
            return
        self.N_bootstrap = N_bootstrap
        highest_peaks = np.zeros(N_bootstrap)
        # Generator function for resampling
        sampler = resample(detrended_data, size=N_bootstrap)
        for i in range(N_bootstrap):
            if (i % 500 == 0):
                print("Iteration", i)
            sample = next(sampler)
            # Use the same windowing scheme as in the original FT / power spectrum
            sample = sample * self.win_coeffs
            sample_ft = fft(self.t, sample, self.fgrid)
            highest_peaks[i] = np.max(norm * np.abs(sample_ft)**2)
        self.false_alarm_5 = np.quantile(highest_peaks, 0.95)
        self.false_alarm_1 = np.quantile(highest_peaks, 0.99)
        if (self.N_bootstrap >= 10000):
            self.false_alarm_01 = np.quantile(highest_peaks, 0.999)
        if save_noise:
            self.frequency_domain_noise = highest_peaks
        else:
            self.frequency_domain_noise = None
            
            
    # ***Break the time series into segments in preparation for Welch's algorithm***
    def segment_data(self, seg, Nyquist, oversample=6, window='None', plot_windows=False):
        # As above, Nyquist is the maximum frequency desired for the periodogram
        # window is type of window to apply to each segment: 'None' or 'BlackmanHarris' or 'KaiserBessel'
        # seg is either an integer that defines the number of 50% overlapping segments, or
        #    a 2-d array-like giving the beginning and ending indices of each segment, i.e.
        #    seg=5 or seg=np.array([[0, 151], [95, 220], [221, 385], [306, 443]]) 
        #    such a segmenting scheme could be used to leave out one or more big gaps in the time series
        # note: a segment defined as [0, 151] will actually include data points 0 through 150
        #    since python's ranges don't include the final element 
        #    (i.e. arr[0:4] = [arr[0], arr[1], arr[2], arr[3]] but arr[4] is left out)
        try:
            good_Nyquist = (Nyquist > 0)
            if not good_Nyquist:
                raise ValueError
        except ValueError:
            print("Nyquist frequency must be float > 0 - no segmenting performed")
            return
        
        try:
            good_oversample = (((type(oversample) is int) or (type(oversample) is float)) \
                              and (oversample > 0))
            if not good_oversample:
                raise ValueError
        except ValueError:
            print("Oversample must be number > 0 - no segmenting performed")
            return
        
        try:
            valid_window = (((window == 'BlackmanHarris') or (window == 'KaiserBessel')) or (window == 'None'))
            if not valid_window:
                raise ValueError
        except ValueError:
            print("Invalid window type. Choose either window='BlackmanHarris',")
            print("window='KaiserBessel', or window='None'. Default is 'None';")
            print("choose 'None' for standard Lomb-Scargle-type periodogram.")
            return
        
        # Window settings
        self.Welch_window = window
        c50_Harris = 0.038 # window coefficient for 50% overlap
        c50_Kaiser = 0.074
        c50_rectangle = 0.5
        
        # Get the segment boundaries
        if type(seg) is int: # break into 50% overlapping segments
            try:
                valid_seg = (seg >= 2)
                if not valid_seg:
                    raise ValueError
            except ValueError:
                print("Number of segments must be >= 2 - no segmenting performed")
                return
            nperseg = int(2*self.N // (seg+1))
            try:
                enough = (nperseg >= 29)
                if not enough:
                    raise ValueError
            except ValueError:
                print("You must have at least 29 data points per segment -")
                print("not a valid segmentation scheme")
                return
            print("Number of data points per segment:", nperseg)
            segments = []
            for i in range(0,seg-1):
                start = (i*nperseg) // 2
                segments.append([start, start+nperseg])
            segments.append([start+nperseg//2,self.N])
            segments = np.array(segments)
            # Set effective number of segments given windowing scheme
            # window='None' gives rectangular window, c50=0.5
            if (window == 'BlackmanHarris'):
                c50 = c50_Harris # Harris 1978, minimum 4-term Blackman-Harris window
            elif (window == 'KaiserBessel'):
                c50 = c50_Kaiser
            else:
                c50 = c50_rectangle
            self.Nseg_eff = seg / (1 + 2*c50**2 - (2*c50**2 / seg))
        else:
            try:
                valid_seg = (isinstance(seg, np.ndarray) and (seg.ndim == 2) and \
                            (seg.shape[1] == 2) and (np.issubdtype(seg[0,0], int)) \
                             and (seg[-1,1] <= self.N))
                if not valid_seg:
                    raise TypeError
            except TypeError:
                print("Segment input format not understood. seg must be either integer")
                print("number of segments >=2, or 2-d numpy array of integers, with each row")
                print("giving the beginning and (end point + 1) of a segment,")
                print("e.g. np.array([[0, 84], [65, 110], [93, 170]]).")
                print("End of last segment must be <= length of time series.")
                return
            segments = seg
            # Determine whether custom segments are overlapping or not
            overlap = False
            i = 1
            while (i < segments.shape[0]):
                if (segments[i,0] < segments[i-1,1]):
                    overlap=True
                    print("Segment overlap detected - assuming 50% overlap")
                    print("If your segment overlap is far from 50%, consider defining non-overlapping segments")
                    break
                i += 1
            if overlap:
                if (window == 'BlackmanHarris'):
                    c50 = c50_Harris
                elif (window == 'KaiserBessel'):
                    c50 = c50_Kaiser
                else:
                    c50 = c50_rectangle
                self.Nseg_eff = segments.shape[0] / (1 + 2*c50**2 - (2*c50**2 / segments.shape[0]))
            else:
                self.Nseg_eff = segments.shape[0]
        self.segments = segments
        self.Nseg = segments.shape[0]
        
        # Build individual segment time series, set weight of each segment
        Nyq_medians = [] # Median Nyquist frequency from each segment
        Rayleighs = [] # Rayleigh resolutions from each segment
        if (window == 'BlackmanHarris'):
            B_6dB = 2.72 # scaled 6 dB Blackman-Harris window main lobe half width
        elif (window == 'KaiserBessel'):
            B_6dB = 2.39 # scaled Kaiser-Bessel window main lobe half width
        else:
            B_6dB = 1.21 # scaled rectangular window main lobe half width
        bandwidths = [] # Real bandwidth of each segment's taper
        self.s_weights = np.zeros(self.Nseg) # Segments will be weighted by number of data points they cover
        for i in range(self.Nseg):
            s = range(segments[i,0], segments[i,1])
            self.s_weights[i] = len(s)-1
            seg_Nyq_meddt = 0.5/np.median(self.dt[s[0:len(s)-1]])
            seg_Rayleigh = 1. / (self.t[s[-1]] - self.t[s[0]])
            Nyq_medians.append(seg_Nyq_meddt)
            Rayleighs.append(seg_Rayleigh)
            bandwidths.append(B_6dB * seg_Rayleigh) 
        self.Welch_band = np.sum(self.s_weights*bandwidths) / np.sum(self.s_weights)
        self.Welch_Rayleigh = np.min(Rayleighs)

        # Check that user input Nyquist frequency is valid for all segments, adjust if necessary
        max_Nyq_possible = np.min(Nyq_medians)
        if (Nyquist > max_Nyq_possible):
            print("Your Nyquist frequency is too high. Adjusting...")
            Nyquist = max_Nyq_possible
            
        # Build frequency grid for Welch's periodogram calculation
        if (Nyquist > self.Nyq_meddt):
            print("Warning: your requested Nyquist frequency is higher than the Nyquist")
            print("frequency associated with the median timestep. Make sure that makes")
            print("sense for your dataset.")
        self.Welch_nf = oversample * int(Nyquist // self.Welch_Rayleigh)
        if (self.Welch_nf % 2) != 0: # make sure zero frequency is included
            self.Welch_nf += 1
        self.Welch_fgrid = np.linspace(-Nyquist, Nyquist, num=2*self.Welch_nf+1, endpoint=True)
        self.Welch_powgrid = self.Welch_fgrid[self.Welch_nf:]
        
        # Information for user
        print("Number of segments:", self.Nseg)
        print("Segment start and end points:", self.segments)
        print("Effective number of segments:", f"{self.Nseg_eff:.6f}")
        print("Frequency grid spacing:", f"{self.Welch_powgrid[1]-self.Welch_powgrid[0]:.6f}")
        print("Minimum 6-dB main lobe half width:", f"{np.min(bandwidths):.6f}")
        print("Mean 6-dB main lobe half width (1/2 resolution limit):", f"{self.Welch_band:.6f}")
        print("Best achievable Rayleigh limit (1/2 best-case resolution limit):", f"{self.Welch_Rayleigh:.6f}")

        if plot_windows:
            plt.figure(figsize=(8,5))
            for i in range(self.Nseg):
                sg = range(self.segments[i,0], self.segments[i,1])
                if (window == 'BlackmanHarris'):
                    y = BlackmanHarris(self.t[sg]-self.t[sg[0]])
                elif (window == 'KaiserBessel'):
                    y = KaiserBessel(self.t[sg]-self.t[sg[0]])
                else:
                    y = np.ones(len(sg))
                plt.scatter(self.t[sg]-self.t[sg[0]], y)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Segment windows")
        
        # Finish
        self.segmented = True
                            
                            
    # ***Perform a SINGLE Welch's power spectrum estimate***
    #    Use it either on the actual data, or on a scrambled dataset as part
    #      of bootstrap false alarm probability assessment
    def Welch_powspec(self, trend=True, trend_type='linear', norm=True):
        # set trend=True to detrend data is as in TimeSeries.pow_FT(), choose trend type
        # set norm=True to use Parseval's theorem to get the power spectral density normalization
        # set norm=False if normalization doesn't matter, or will take place in a different part
        #    of the code
        if not self.segmented:
            print("You must call segment_data() before computing a Welch's power spectrum estimate")
            return
        
        valid_trend_type = ((trend_type == 'linear') or (trend_type == 'quadratic'))
        if not valid_trend_type:
            print("Trend type not understood. Options are 'linear' or 'quadratic'.")
            print('Defaulting to linear trend')
            trend_type = 'linear'

        # Get power spectrum associated with each segment
        autospec = []
        fft_input_data = []
        for i in range(self.Nseg):
            sg = range(self.segments[i,0], self.segments[i,1])
            time = self.t[sg] - self.t[sg[0]]
            # Center segment data by subtracting off mean
            datamean = np.mean(self.obs[sg])
            data = self.obs[sg] - datamean
                
            # Subtract linear trend
            self.Welch_trend = trend
            self.Welch_trend_type = trend_type
            if self.Welch_trend:
                if self.Welch_trend_type == 'linear':
                    trendcurve = np.poly1d(np.polyfit(time, data.real, 1))
                else:
                    trendcurve = np.poly1d(np.polyfit(time, data.real, 2))
                data = data - trendcurve(time)
        
            # Apply window
            if (self.Welch_window == 'BlackmanHarris'):
                win_coeffs = BlackmanHarris(time)
            elif (self.Welch_window == 'KaiserBessel'):
                win_coeffs = KaiserBessel(time)
            else:
                win_coeffs = np.ones(len(time))
            data = win_coeffs * data
            fft_input_data.append(data)
            # Segment transform (not normalized)
            seg_ft = fft(time, data, self.Welch_fgrid)
            # Compute the power spectrum
            autospec.append(self.s_weights[i] * np.abs(seg_ft[self.Welch_nf:])**2)

        # The Welch's power spectrum estimate
        self.Welch_pow = np.mean(np.array(autospec), axis=0) / np.sum(self.s_weights)
        
        # Normalize the periodogram with Parseval's theorem: Sum(df * power density_i) = time domain variance
        if norm:
            xnorm = np.var(np.concatenate(fft_input_data).ravel()) / np.sum(self.Welch_powgrid[1] * self.Welch_pow)
            self.Welch_pow *= xnorm
        # Done
                            
                            
    # ***Assuming white noise, use bootstrap to calculate
    #    false-alarm thresholds for the Welch's periodogram***
    def Welch_powspec_bootstrap(self, N_Welch_bootstrap=10000, save_noise=False):
        # N_Welch_bootstrap is the number of bootstrap iterations for the false alarm threshold
        #    computation; set to a number <100 to turn off the bootstrap
        # set save_noise=True to save the array containing coherences of the randomly 
        #    shuffled data (useful if you want to examine the noise properties later)
        # White noise bootstrap implemented here is similar to Pardo-Igu'zquiza and 
        #    Rodri'guez-Tovar (2012)
        try:
            Welch_not_calculated = (self.Welch_pow is None)
            if Welch_not_calculated:
                raise ValueError
        except ValueError:
            print("Use Welch_powspec() to calculate a Welch's power spectrum estimate before starting the bootstrap")
            return
        try:
            valid_N_bootstrap = ((type(N_Welch_bootstrap) is int) and (N_Welch_bootstrap >= 100))
            if not valid_N_bootstrap:
                raise ValueError
        except ValueError:
            print("Not a valid bootstrap simulation. To run, set integer N_Welch_bootstrap >= 100")
            return
        
        obs_original = copy.deepcopy(self.obs)
        Welch_pow_original = copy.deepcopy(self.Welch_pow)
        self.N_Welch_bootstrap = N_Welch_bootstrap
        highest_peaks = np.zeros(N_Welch_bootstrap)

        indices = range(self.N)
        sampler = resample(indices, size=N_Welch_bootstrap)

        # Start the loop
        for i in range(N_Welch_bootstrap):
            if (i % 500 == 0):
                print("Iteration", i)
            perm = next(sampler)
            self.obs = obs_original[perm]
            self.Welch_powspec(trend=self.Welch_trend)
            highest_peaks[i] = np.max(self.Welch_pow)
        
        self.Welch_false_alarm_5 = np.quantile(highest_peaks, 0.95)
        self.Welch_false_alarm_1 = np.quantile(highest_peaks, 0.99)
        if (self.N_Welch_bootstrap >= 10000):
            self.Welch_false_alarm_01 = np.quantile(highest_peaks, 0.999)

        if save_noise:
            self.Welch_noise = highest_peaks
        else:
            self.Welch_noise = None
        
        # Put everything back to normal
        self.obs = obs_original
        self.Welch_pow = Welch_pow_original
        
                            
    # *** Perform Siegel's test (1980) for periodicity***
    # By searching for excess power above a test statistic g_crit, the test can
    #    diagnose departures from a white-noise periodogram featuring up to 
    #    3 periodic signals - good for rotation + harmonic(s)
    # Set Welch=False to apply Siegel's test to the standard, single-window
    #    Lomb-Scargle-like periodogram; toggle Welch=True to apply it to Welch's
    #    periodogram
    # Set tri=True to search for up to 3 periodic signals; default is tri=False
    #    which is optimized to search for 2 (i.e. rotation and harmonic)
    #    tri=True has higher risk of mistaking noise for periodicity than tri=False
    def Siegel_test(self, Welch=False, tri=False):
        if not Welch:
            try:
                no_periodogram = (self.power is None)
                if no_periodogram:
                    raise ValueError
            except ValueError:
                print("Use pow_FT() to compute a periodogram before running Siegel's test")
                return
        else:
            try:
                no_Welch = (self.Welch_pow is None)
                if no_Welch: 
                    raise ValueError
            except ValueError:
                print("Use Welch_powspec() to compute a periodogram before running Siegel's test")
                return
        alpha = 0.05 # null hypothesis will be rejected at 5% level
        if not Welch:
            numf = len(self.powfgrid)
            norm_per = self.power / np.sum(self.power)
        else:
            numf = len(self.Welch_powgrid)
            norm_per = self.Welch_pow / np.sum(self.Welch_pow)
        # Fischer (1929) critical g statistic, approximation from Percival & Walden (1993)
        Fischer_g = 1. - (alpha/numf)**(1./(numf-1))
        if not tri:
            lam = 0.6 # conservative choice - grabs 2 periodic signals, not always 3
            acoeff = 1.033
            bcoeff = -0.72356
        else: 
            lam = 0.4 # try it if you think there might be 3 periodic signals, but sometimes picks up noise
            acoeff = 0.9842
            bcoeff = -0.51697
        g_threshold = lam*Fischer_g
        above_g_threshold = np.where(norm_per > g_threshold)[0]
        Tstat = np.sum(norm_per[above_g_threshold] - g_threshold)

        t_threshold = acoeff * numf**bcoeff
        print("T statistic:", f"{Tstat:.5f}")
        print("T threshold for rejecting white noise hypothesis at 5% level:", f"{t_threshold:.5f}")
        if (Tstat > t_threshold):
            print("Null hypothesis rejected: 95% chance this time series has 1 or more periodicities")
        if (Tstat < t_threshold):
            print("Null hypothesis not rejected: This time series could be white noise (or red noise; beware)")
                            
                            
    # ***Calculate the non-Welch spectral window in the frequency domain***
    #    "window function" in astronomer parlance
    def spectral_window(self, plot=True, yscale='log10', outfile="None"):
        # Set plot=False to turn off plotting
        # For y-axis scale, choices are 'log10' or 'linear'
        # To save results, set the outfile keyword to the desired name of the output file
        try:
            no_power = (self.power is None)
            if no_power:
                raise ValueError
        except ValueError:
            print("You must estimate the power spectrum with pow_FT() before computing the spectral window.")
            return
        valid_y = ((yscale == 'log10') or (yscale == 'linear'))
        if not valid_y:      
            print("Invalid setting for plot y-axis scale. Defaulting to log10.")
            yscale='log10'
        winfunc = np.abs(fft(self.t, self.win_coeffs.astype(complex), self.fgrid))**2
        winfunc_norm = np.sum(winfunc * self.powfgrid[1])
        # winfunc /= np.max(winfunc)
        winfunc /= winfunc_norm
        self.window_function = winfunc
        # 6-dB bandwidth
        dB6 = 1/3.981 # Fractional 6-dB power decline
        bw = np.abs(self.fgrid[np.argmin(np.abs(winfunc - dB6*np.max(winfunc)))])
        print('Half bandwidth:', f"{bw:.6f}")
        if plot:
            plt.figure(figsize=(9,5))
            if (yscale == 'log10'):
                plt.semilogy(self.fgrid, winfunc, lw=0.7)
            else:
                plt.plot(self.fgrid, winfunc, lw=0.7)
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            plt.title('Spectral window: ' + self.window)
        if (outfile == "None"):
            print("Single-window results not saved") 
            return
        else:
            try:
                good_filename = (type(outfile) is str)
                if not good_filename:
                    raise TypeError
            except TypeError:
                print("Bad output file name - no results saved")
                return
            header = "Spectral window\nMeasured half main lobe width: {}".format(bw) + "\nfrequency power"
            np.savetxt(outfile, np.column_stack((self.fgrid, winfunc)), header=header)


    # ***Calculate the spectral window of the Welch's estimator***
    def spectral_window_Welch(self, plot=True, yscale='log10', outfile="None"):
        # Compute the average spectral window of the Welch's estimator
        # For y-axis scale, choices are 'log10' or 'linear'
        # If plot=True, you will get a plot of the spectral window
        # To save the spectral window, set outfile keyword to desired filename
        try:
            no_results = (self.Welch_pow is None)
            if no_results:
                raise ValueError
        except ValueError:
            print("No Welch's power spectrum - call Welch_powspec() first")
            return
        valid_y = ((yscale == 'log10') or (yscale == 'linear'))
        if not valid_y:      
            print("Invalid setting for plot y-axis scale. Defaulting to log10.")
            yscale='log10'
        seg_windows = []
        for i in range(self.Nseg):
            sg = range(self.segments[i,0], self.segments[i,1])
            time = self.t[sg] - self.t[sg[0]]
            if (self.Welch_window == 'BlackmanHarris'):
                win_coeffs = BlackmanHarris(time)
            elif (self.Welch_window == 'KaiserBessel'):
                win_coeffs = KaiserBessel(time)
            else:
                win_coeffs = np.ones(len(sg))

            seg_windows.append(self.s_weights[i] * \
                     np.abs(fft(time, win_coeffs.astype(complex), self.Welch_fgrid))**2)
        Welch_window_function = np.mean(np.array(seg_windows), axis=0) / \
                     np.sum(self.s_weights)
        winfunc_norm = np.sum(Welch_window_function * self.Welch_powgrid[1])
        self.Welch_window_function = Welch_window_function / winfunc_norm
        # 6-dB bandwidth
        dB6 = 1/3.981 # Fractional 6-dB power decline
        bw = np.abs(self.Welch_fgrid[np.argmin(np.abs(self.Welch_window_function - \
                                               dB6*np.max(self.Welch_window_function)))])
        print('Half bandwidth:', f"{bw:.6f}")
        if plot:
            plt.figure(figsize=(9,5))
            if (yscale == 'log10'):
                plt.semilogy(self.Welch_fgrid, self.Welch_window_function, lw=0.7)
            else:
                plt.plot(self.Welch_fgrid, self.Welch_window_function, lw=0.7)
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            plt.title('Welch average spectral window') 
        if (outfile == "None"):
            print("Welch average spectral window not saved to file") 
            return
        else:
            try:
                good_filename = (type(outfile) is str)
                if not good_filename:
                    raise TypeError
            except TypeError:
                print("Bad output file name - no results saved")
                return
            header = "Welch's spectral window\nMeasured half main lobe width: {}".format(bw)+ "\nfrequency power"
            np.savetxt(outfile, np.column_stack((self.Welch_fgrid, \
                   self.Welch_window_function)), header=header)
    

    # ***Plot the complex-valued Fourier transform***
    def Ftplot(self, vlines=[], lw=0.8):
        # Use vlines keyword to add vertical lines to the plot
        # Use lw keyword to change linewidth
        try:
            cant_plot = (self.ft is None)
            if cant_plot:
                raise ValueError
        except ValueError:
            print("Can't plot - no Fourier transform computed")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9,6))
        ax1.plot(self.fgrid, self.ft.real, color="mediumblue", lw=lw)
        ax1.set_ylabel(r"$\Re (\mathcal{F}\{x_t\})$")
        ax2.plot(self.fgrid, self.ft.imag, color="mediumblue", lw=lw)
        ax2.set_ylabel(r"$\Im (\mathcal{F}\{x_t\})$")
        ax2.set_xlabel("Frequency")
        for v in vlines:
            ax1.axvline(v, color='k', linestyle=':')
            ax1.axvline(-v, color='k', linestyle=':')
            ax2.axvline(v, color='k', linestyle=':')
            ax2.axvline(-v, color='k', linestyle=':')
                      
        
    # ***Plot the power spectrum***
    def powplot(self, show_thresholds=True, Welch=False, yscale='log10', title=r"$\hat{S}(f)$", vlines=[], lw=0.8):
        # Set show_thresholds=True to show bootstrap false alarm thresholds
        # use title keyword to change the plot title
        # set Welch=True to plot the Welch's power spectrum; default is to plot Lomb-Scargle-like periodogram
        # choices for y-axis scale are 'log10' and 'linear'
        # use vlines keyword to add vertical lines to the plot
        # use lw keyword to change linewidth
        # assumes white noise
        try:
            cant_plot = ((self.power is None) and (self.Welch_pow is None))
            if cant_plot:
                raise ValueError
        except ValueError:
            print("Compute a standard or Welch's periodogram with pow_FT() or Welch_powspec() before plotting")
            return
        valid_y = ((yscale == 'log10') or (yscale == 'linear'))
        if not valid_y:      
            print("Invalid setting for plot y-axis scale. Defaulting to log10.")
            yscale='log10'
        if (Welch and (self.Welch_pow is not None)):
            x = self.Welch_powgrid
            y = self.Welch_pow
            if (show_thresholds and (self.N_Welch_bootstrap >= 100)):
                f5 = self.Welch_false_alarm_5
                f1 = self.Welch_false_alarm_1
                if (self.N_Welch_bootstrap >= 10000):
                    f01 = self.Welch_false_alarm_01
        else:
            x = self.powfgrid
            y = self.power
            if (show_thresholds and (self.N_bootstrap >= 100)):
                f5 = self.false_alarm_5
                f1 = self.false_alarm_1
                if (self.N_bootstrap >= 10000):
                    f01 = self.false_alarm_01
        plt.figure(figsize=(9,5))
        if (yscale == 'log10'):
            plt.semilogy(x[1:], y[1:], color='mediumblue', lw=lw)
        else:
            plt.plot(x[1:], y[1:], color='mediumblue', lw=lw)
        if 'f01' in locals():
            plt.axhline(f01, color='crimson', ls=':', label='bootstrap 0.1% FAP')
        if 'f5' in locals():
            plt.axhline(f1, color='mediumspringgreen', ls=':', label='bootstrap 1% FAP')
            plt.axhline(f5, color='darkorchid', ls=':', label='bootstrap 5% FAP')
            plt.legend(loc="best", fontsize="small")
        for v in vlines:
            plt.axvline(v, color='k', linestyle=':')
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.title(title)
            
            
    # ***Save standard periodogram***
    def save_standard(self, filename):
        # filename: name of results file
        try:
            no_results = (self.ft is None)
            if no_results:
                raise ValueError
        except ValueError:
            print("No results to save - call pow_FT() first")
            return
        try:
            good_filename = (type(filename) is str)
            if not good_filename:
                raise TypeError
        except TypeError:
            print("Bad output file name - no results saved")
            return
        header = "Standard, non-uniform FFT-based periodogram" + \
                 "\nChosen Nyquist frequency: {}".format(self.powfgrid[-1]) + \
                 "\nRayleigh resolution: {}".format(self.Rayleigh) + \
                 "\nDetrended: {}".format(self.trend) + \
                 "\nTrend type: " + self.trend_type + \
                 "\nNumber of bootstrap iterations: {}".format(self.N_bootstrap)
        if (self.N_bootstrap > 0):
            header = header + "\n5% FAP (bootstrap): {}".format(self.false_alarm_5) + \
                              "\n1% FAP (bootstrap): {}".format(self.false_alarm_1)
            if (self.N_bootstrap >= 10000):
                header = header + "\n0.1% FAP (bootstrap): {}".format(self.false_alarm_01)
        header = header + "\nWindow applied: {}".format(self.window)
        header = header + "\nfrequency,power"
        output = np.column_stack((self.powfgrid, self.power))
        np.savetxt(filename, output, fmt='%.8e', delimiter=',', header=header)


    # ***Save Welch's power spectrum estimate***
    def save_Welch(self, filename):
        try:
            no_results = (self.Welch_pow is None)
            if no_results:
                raise ValueError
        except ValueError:
            print("No results to save - call Welch_powspec() first")
            return
        try:
            good_filename = (type(filename) is str)
            if not good_filename:
                raise TypeError
        except TypeError:
            print("Bad output file name - no results saved")
            return
        header = "Welch's power spectrum estimate" + \
                 "\nChosen Nyquist frequency: {}".format(self.Welch_fgrid[-1]) + \
                 "\nRayleigh resolution: {}".format(self.Welch_Rayleigh) + \
                 "\n6-dB main lobe half width (1/2 limiting resolution): {}".format(self.Welch_band) + \
                 "\nActual number of segments: {}".format(self.Nseg) + \
                 "\nEffective number of segments: {}".format(self.Nseg_eff) + \
                 "\nSegment start and end points: {}".format(self.segments.tolist()) + \
                 "\nSegments detrended: {}".format(self.Welch_trend) + \
                 "\nTrend type: " + self.Welch_trend_type + \
                 "\nWindow applied: {}".format(self.Welch_window) + \
                 "\nNumber of bootstrap iterations: {}".format(self.N_Welch_bootstrap)
        if (self.N_Welch_bootstrap > 0):
            header = header + "\n5% FAP (bootstrap): {}".format(self.Welch_false_alarm_5) + \
                              "\n1% FAP (bootstrap): {}".format(self.Welch_false_alarm_1)
            if (self.N_Welch_bootstrap >= 10000):
                header = header + "\n0.1% FAP (bootstrap): {}".format(self.Welch_false_alarm_01)
        header = header + "\nfrequency,power"
        output = np.column_stack((self.Welch_powgrid, self.Welch_pow))
        np.savetxt(filename, output, fmt='%.8e', delimiter=',', header=header)
