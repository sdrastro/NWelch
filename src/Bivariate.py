import numpy as np
import matplotlib.pyplot as plt
import copy
from resample.bootstrap import resample
from scipy import stats

# Bivariate is built on TimeSeries class; import here
import TimeSeries


# ***Lambda function for theoretical coherence false alarm thresholds***
#    equation from Schulz & Stattegger 1997
cthresh = lambda alpha, neff: 1. - alpha**(1./(neff-1))


# ***Fisher's z transformation***
#    inputs: c2 = coherence, nsegeff = effective number of segments
ztrans = lambda c2, nsegeff: np.sqrt(2*nsegeff - 2) * np.arctanh(c2)


# ***Number of crossings of a coherence false alarm threshold***
ncrossings = lambda coh, thresh: (np.diff(np.sign(coh-thresh)) != 0).sum() - (coh-thresh == 0).sum()



# ***Bivariate class for Welch's periodograms and coherence estimates***
class Bivariate:
    
    
    def __init__(self, t, x, y, display_frequency_info=True):
        # t: vector of observation times
        # x: 1st vector of observed data (i.e. RV, S-index, H-alpha)
        # y: 2nd vector of observed data - same times
        # display_frequency_info: if true, print out various estimates of the Nyquist 
        #    frequency and Rayleigh resolution
        try:
            test = (len(t) == len(x) == len(y))
            if (not test):
                raise ValueError
        except ValueError:
            print("Number of observation times must equal number of x and y data points")
            print("No bivariate object created")
            return
        # Create pair of TimeSeries objects
        self.x_series = TimeSeries.TimeSeries(t, x, display_frequency_info=display_frequency_info)
        self.y_series = TimeSeries.TimeSeries(t, y, display_frequency_info=False)
        self.segmented = False # Data aren't segmented yet
        self.coh = None
        self.N_coh_bootstrap = 0
        self.N_red_noise = 0


    def segment_data(self, seg, Nyquist, oversample=4, window='None', plot_windows=False):
        # As above, Nyquist is the maximum frequency desired for the periodogram
        # seg is either an integer that defines the number of 50% overlapping segments, or
        #    a 2-d numpy array giving the beginning and ending indices of each segment, i.e.
        #    seg=5 or seg=np.array([[0, 151], [95, 220], [221, 385], [306, 443]]) 
        #    such a segmenting scheme could be used to leave out one or more big gaps in the time series
        # note: a segment defined as [0, 151] will actually include data points 0 through 150
        #    since python's ranges don't include the final element 
        #    (i.e. arr[0:4] = [arr[0], arr[1], arr[2], arr[3]] but arr[4] is left out)
        self.x_series.segment_data(seg, Nyquist, oversample=oversample, \
             window=window, plot_windows=plot_windows)
        try:
            if not self.x_series.segmented:
                raise ValueError
        except ValueError:
            print("Segmentation failed - try again.")
            return
        self.y_series.segment_data(seg, Nyquist, oversample=oversample, \
             window=window, plot_windows=False)
        self.segmented = True
        self.Nseg = self.x_series.Nseg
        self.Nseg_eff = self.x_series.Nseg_eff
        self.nf = self.x_series.Welch_nf
        self.fgrid = self.x_series.Welch_fgrid
        self.pow_coh_grid = self.x_series.Welch_powgrid
        self.segments = self.x_series.segments
        self.window = window
        

# ***Perform a SINGLE coherence / power spectrum estimate***
#    Use it either on the actual data, or on a scrambled dataset as part
#      of bootstrap false alarm probability assessment
    def Welch_coherence_powspec(self, trend=True, trend_type='linear'):
        # set trend=True to detrend data as in TimeSeries.pow_FT()
        # set trend_type to choose linear or quadratic detrending
        if not self.segmented:
            print("You must call segment_data() before computing coherence")
            return
        
        valid_trend_type = ((trend_type == 'linear') or (trend_type == 'quadratic'))
        if not valid_trend_type:
            print("Trend type not understood. Options are 'linear' or 'quadratic'.")
            print('Defaulting to linear trend')
            trend_type == 'linear'
        
        # Get non-normalized Welch's autospectra for coherence denominator
        self.x_series.Welch_powspec(trend=trend, trend_type=trend_type, norm=False)
        self.y_series.Welch_powspec(trend=trend, trend_type=trend_type, norm=False)

        # Calculate and average cross-spectra associated with each segment
        xycross = []
        for i in range(self.Nseg):
            sg = range(self.segments[i,0], self.segments[i,1])
            x_seg = TimeSeries.TimeSeries(self.x_series.t[sg], self.x_series.obs[sg], display_frequency_info=False)
            y_seg = TimeSeries.TimeSeries(self.y_series.t[sg], self.y_series.obs[sg], display_frequency_info=False)
            x_seg.nf = self.nf
            y_seg.nf = self.nf
            x_seg.fgrid = self.fgrid
            y_seg.fgrid = self.fgrid
            x_seg.powfgrid = self.pow_coh_grid
            y_seg.powfgrid = self.pow_coh_grid
            # Get the segment FFTs
            x_seg.pow_FT(window=self.window, trend=trend, trend_type=trend_type, N_bootstrap=0, quiet=True, norm=False)
            y_seg.pow_FT(window=self.window, trend=trend, trend_type=trend_type, N_bootstrap=0, quiet=True, norm=False)
            # Get the segment cross-spectrum
            xycross.append(self.x_series.s_weights[i] * np.conj(x_seg.ft[x_seg.nf:]) * \
                           y_seg.ft[y_seg.nf:])
                           
        # Order of operations for coherence numerator: mean -> absolute value -> square
        self.cross = np.mean(np.array(xycross), axis=0)
        self.phase = np.arctan2(self.cross.imag, self.cross.real) * 180 / np.pi
        self.cross_mag = np.abs(self.cross) / np.sum(self.x_series.s_weights)
        self.coh_raw = self.cross_mag**2 / (self.x_series.Welch_pow * self.y_series.Welch_pow)
        
        # Compute analytical false alarm levels for coherence
        self.coh_prob_5 = cthresh(0.05, self.Nseg_eff)
        self.coh_prob_1 = cthresh(0.01, self.Nseg_eff)
        self.coh_prob_01 = cthresh(0.001, self.Nseg_eff)
            
        # Debias coherence
        est_bias = (1. - self.coh_raw)**2 / self.Nseg_eff
        self.coh = self.coh_raw - est_bias
        where_neg = np.where(self.coh < 0.)
        self.coh[where_neg] = 0.
        
        # arctanh transformation
        self.coh_transformed =  ztrans(self.coh, self.Nseg_eff)
        
        # Number of threshold crossings per Rayleigh resolution
        self.ncrossperR_5 = ncrossings(self.coh, self.coh_prob_5) / (self.pow_coh_grid[-1] / self.x_series.Welch_Rayleigh)
        self.ncrossperR_1 = ncrossings(self.coh, self.coh_prob_1) / (self.pow_coh_grid[-1] / self.x_series.Welch_Rayleigh)
        self.ncrossperR_01 = ncrossings(self.coh, self.coh_prob_01) / (self.pow_coh_grid[-1] / self.x_series.Welch_Rayleigh)
        
        # Get correctly normalized Welch's autospectra
        self.x_series.Welch_powspec(trend=trend, trend_type=trend_type)
        self.y_series.Welch_powspec(trend=trend, trend_type=trend_type)
        
        # Keep track of detrending
        self.trend = trend
        self.trend_type = trend_type
        # Done
        
        
# ***Perform Siegel's test on the Welch's periodograms***
    def Siegel_Welch(self, tri=False):
    # Set tri=True to search for up to 3 periodic signals; default is tri=False,
    #    which is optimized to search for 2 (i.e. rotation and harmonic)
    #    tri=True has higher risk of mistaking noise for periodicity than tri=False
        try:
            coh_calculated = (self.coh is not None)
            if not coh_calculated:
                raise ValueError
        except ValueError:
            print("You must call Welch_coherence_powspec() before performing Siegel's test")
            return
        print("Siegel's test on Gxx:")
        self.x_series.Siegel_test(Welch=True, tri=tri)
        print("Siegel's test on Gyy:")
        self.y_series.Siegel_test(Welch=True, tri=tri)
        
        
# ***Assuming white noise, use bootstrap to calculate
#    false-alarm thresholds for Welch's periodograms and coherence***
    def Welch_coherence_powspec_bootstrap(self, N_coh_bootstrap=10000, save_noise=False):
        # N_coh_bootstrap is the number of bootstrap iterations for the false alarm threshold
        #    computation; set to a number <100 to turn off the bootstrap
        # set save_noise=True to save the array containing coherences of the randomly 
        #    shuffled data (useful if you want to examine the noise properties later)
        # White noise bootstrap implemented here is similar to Pardo-Igu'zquiza and 
        #    Rodri'guez-Tovar (2012)
        try:
            coh_calculated = (self.coh is not None)
            if not coh_calculated:
                raise ValueError
        except ValueError:
            print("You must calculate a coherence estimate before starting the bootstrap")
            return
        try:
            valid_N_bootstrap = ((type(N_coh_bootstrap) is int) and (N_coh_bootstrap >= 100))
            if not valid_N_bootstrap:
                raise ValueError
        except ValueError:
            print("Not a valid bootstrap simulation. To run, set integer N_bootstrap >= 100")
            return
        
        x_original = copy.deepcopy(self.x_series)
        y_original = copy.deepcopy(self.y_series)
        coh_original = copy.deepcopy(self.coh)
        coh_raw_original = copy.deepcopy(self.coh_raw)
        coh_transformed_original = copy.deepcopy(self.coh_transformed)
        cross_original = copy.deepcopy(self.cross)
        cross_mag_original = copy.deepcopy(self.cross_mag)
        coh_prob_5_original = copy.deepcopy(self.coh_prob_5)
        coh_prob_1_original = copy.deepcopy(self.coh_prob_1)
        coh_prob_01_original = copy.deepcopy(self.coh_prob_01)
        ncrossperR_5_original = copy.deepcopy(self.ncrossperR_5)
        ncrossperR_1_original = copy.deepcopy(self.ncrossperR_1)
        ncrossperR_01_original = copy.deepcopy(self.ncrossperR_01)

        self.N_coh_bootstrap = N_coh_bootstrap
        N_coh_f = self.nf + 1

        coh_noise = np.zeros((N_coh_bootstrap, N_coh_f))
        coh_transformed_noise = np.zeros((N_coh_bootstrap, N_coh_f))
        xpow_highest_peaks = np.zeros(N_coh_bootstrap)
        ypow_highest_peaks = np.zeros(N_coh_bootstrap)
        ncrossingsperR_boot_5 = np.zeros(N_coh_bootstrap)
        ncrossingsperR_boot_1 = np.zeros(N_coh_bootstrap)
        ncrossingsperR_boot_01 = np.zeros(N_coh_bootstrap)

        indices = range(self.x_series.N)
        sampler = resample(indices, size=2*N_coh_bootstrap)

        # Start the loop
        for i in range(N_coh_bootstrap):
            if (i % 500 == 0):
                print("Iteration", i)
            perm = next(sampler)
            self.x_series.obs = x_original.obs[perm]
            perm = next(sampler)
            self.y_series.obs = y_original.obs[perm]
            self.Welch_coherence_powspec(trend=self.trend, trend_type=self.trend_type)
            coh_noise[i,:] = self.coh
            coh_transformed_noise[i,:] = self.coh_transformed
            xpow_highest_peaks[i] = np.max(self.x_series.Welch_pow)
            ypow_highest_peaks[i] = np.max(self.y_series.Welch_pow)
            ncrossingsperR_boot_5[i] = self.ncrossperR_5
            ncrossingsperR_boot_1[i] = self.ncrossperR_1
            ncrossingsperR_boot_01[i] = self.ncrossperR_01
        
        self.xpow_Welch_false_alarm_5 = np.quantile(xpow_highest_peaks, 0.95)
        self.xpow_Welch_false_alarm_1 = np.quantile(xpow_highest_peaks, 0.99)
        self.ypow_Welch_false_alarm_5 = np.quantile(ypow_highest_peaks, 0.95)
        self.ypow_Welch_false_alarm_1 = np.quantile(ypow_highest_peaks, 0.99)
        self.coh_boot_5 = np.quantile(coh_noise, 0.95, axis=0)
        self.coh_boot_1 = np.quantile(coh_noise, 0.99, axis=0)
        self.coh_transformed_boot_5 = np.quantile(coh_transformed_noise, 0.95, axis=0)
        self.coh_transformed_boot_1 = np.quantile(coh_transformed_noise, 0.99, axis=0)
        if (self.N_coh_bootstrap >= 10000):
            self.xpow_Welch_false_alarm_01 = np.quantile(xpow_highest_peaks, 0.999)
            self.ypow_Welch_false_alarm_01 = np.quantile(ypow_highest_peaks, 0.999)
            self.coh_boot_01 = np.quantile(coh_noise, 0.999, axis=0)
            self.coh_transformed_boot_01 = np.quantile(coh_transformed_noise, 0.999, axis=0)

        if save_noise:
            self.coh_noise = coh_noise
            self.xpow_noise = xpow_highest_peaks
            self.ypow_noise = ypow_highest_peaks
        else:
            self.coh_noise = None
            self.xpow_noise = None
            self.ypow_noise = None
        
        # Put everything back to normal
        self.x_series = x_original
        self.x_series.Welch_false_alarm_5 = self.xpow_Welch_false_alarm_5 
        self.x_series.Welch_false_alarm_1 = self.xpow_Welch_false_alarm_1 
        self.y_series = y_original
        self.y_series.Welch_false_alarm_5 = self.ypow_Welch_false_alarm_5 
        self.y_series.Welch_false_alarm_1 = self.ypow_Welch_false_alarm_1 
        if (self.N_coh_bootstrap >= 10000):
            self.x_series.Welch_false_alarm_01 = self.xpow_Welch_false_alarm_01 
            self.y_series.Welch_false_alarm_01 = self.ypow_Welch_false_alarm_01 
        self.x_series.N_Welch_bootstrap = N_coh_bootstrap
        self.y_series.N_Welch_bootstrap = N_coh_bootstrap
        self.coh = coh_original
        self.coh_raw = coh_raw_original
        self.coh_transformed = coh_transformed_original
        self.cross = cross_original
        self.cross_mag = cross_mag_original
        self.coh_prob_5 = coh_prob_5_original
        self.coh_prob_1 = coh_prob_1_original
        self.coh_prob_01 = coh_prob_01_original
        self.ncrossperR_5 = ncrossperR_5_original
        self.ncrossperR_1 = ncrossperR_1_original
        self.ncrossperR_01 = ncrossperR_01_original
        
        # Number of threshold crossings of actual coherence measurement is what percentile
        #    of numbers of threshold crossings from permutations?
        self.ncrossperR_5_percentile = stats.percentileofscore(ncrossingsperR_boot_5, self.ncrossperR_5)
        self.ncrossperR_1_percentile = stats.percentileofscore(ncrossingsperR_boot_1, self.ncrossperR_1)
        self.ncrossperR_01_percentile = stats.percentileofscore(ncrossingsperR_boot_01, self.ncrossperR_01)
        
        # Print out threshold crossing information
        print('\nMean number of false-alarm threshold crossings per Rayleigh resolution from bootstrap simulations:')
        print('5% FAP:', f"{np.mean(ncrossingsperR_boot_5):.3f}")
        print('1% FAP:', f"{np.mean(ncrossingsperR_boot_1):.3f}")
        print('0.1% FAP:', f"{np.mean(ncrossingsperR_boot_01):.3f}")
        print('\nNumber of false-alarm threshold crossings per Rayleigh resolution from actual data:')
        print('5% FAP:', f"{self.ncrossperR_5:.3f}", 'crossings = ', f"{self.ncrossperR_5_percentile:.4f}", '%ile')
        print('1% FAP:', f"{self.ncrossperR_1:.3f}", 'crossings = ', f"{self.ncrossperR_1_percentile:.4f}", '%ile')
        print('0.1% FAP:', f"{self.ncrossperR_01:.3f}", 'crossings = ', f"{self.ncrossperR_5_percentile:.4f}", '%ile\n')
              

# ***Plot coherence***
    def coh_plot(self, transformed=False, show_theoretical_thresholds=False, show_boot_thresholds=True, vlines=[], lw=0.8):
        # set transformed=True to plot the arctanh-transformed version of coherence
        #    transformed=False will plot the non-transformed version
        # set show_theoretical_thresholds=True to put the
        #    analytical white noise false alarm thresholds
        #    (assuming zero coherence) on the plot
        # set show_boot_thresholds=True to put the bootstrap 
        #    coherence thresholds on the plot
        # use vlines keyword to add vertical lines to the plot
        # use lw keyword to change linewidth on plot
        try:
            cant = (self.coh is None)
            if cant:
                raise ValueError
        except ValueError:
            print("Call Welch_coherence_powspec() to calculate coherence before plotting")
            return
        if transformed:
            y = self.coh_transformed
            ylabel = r"$z(f)$"
            prob5 = ztrans(self.coh_prob_5, self.Nseg_eff)
            prob1 = ztrans(self.coh_prob_1, self.Nseg_eff)
            prob01 = ztrans(self.coh_prob_01, self.Nseg_eff)
            if (self.N_coh_bootstrap >= 100):
                boot5 = self.coh_transformed_boot_5
                boot1 = self.coh_transformed_boot_1
                if (self.N_coh_bootstrap >= 10000):
                    boot01 = self.coh_transformed_boot_01
        else:
            y = self.coh
            ylabel = r"$\widehat{C}^2_{xy}(f)$"
            prob5 = self.coh_prob_5
            prob1 = self.coh_prob_1
            prob01 = self.coh_prob_01
            if (self.N_coh_bootstrap >= 100):
                boot5 = self.coh_boot_5
                boot1 = self.coh_boot_1
                if (self.N_coh_bootstrap >= 10000):
                    boot01 = self.coh_boot_01
        plt.figure(figsize=(9,5))
        plt.plot(self.pow_coh_grid, y, color="mediumblue", lw=lw)
        plt.xlabel("Frequency")
        plt.ylabel(ylabel)
        plt.title("Coherence")
        if show_theoretical_thresholds:
            plt.axhline(prob5, color='darkorchid', label="analytical 5%", lw=0.7)
            plt.axhline(prob1, color='mediumspringgreen', label="analytical 1%", lw=0.7)
            plt.axhline(prob01, color='crimson', label="analytical 0.1%", lw=0.7)
        if (self.N_coh_bootstrap >= 100) and show_boot_thresholds:
            plt.plot(self.pow_coh_grid, boot5, color='darkorchid', ls=':', label='bootstrap 5%')
            plt.plot(self.pow_coh_grid, boot1, color='mediumspringgreen', ls=':', label='bootstrap 1%')
            if (self.N_coh_bootstrap >= 10000):
                plt.plot(self.pow_coh_grid, boot01, color='crimson', ls=':', label='bootstrap 0.1%')
        if (show_theoretical_thresholds or show_boot_thresholds):
            plt.legend(loc='best', title='FAPs', fontsize='small')
        for v in vlines:
            plt.axvline(v, color='gray', linestyle=':')
            
            
# ***Plot a Welch's periodogram***
    def Welch_pow_plot(self, x_or_y='x', show_boot_thresholds=True, yscale='log10', vlines=[], lw=0.8):
        # x_or_y='x' plots x(t) Welch's periodogram; x_or_y='y' plots y(t) Welch's periodogram
        # set show_boot_thresholds=True to plot the bootstrap false alarm thresholds
        try:
            cant = self.x_series.Welch_pow is None
            if cant:
                raise ValueError
        except ValueError:
            print("Call Welch_coherence_powspec() to calculate Welch's periodograms before plotting")
            return
        try:
            good_xory = ((x_or_y == 'x') or (x_or_y == 'y'))
            if not good_xory:
                raise ValueError
        except ValueError:
            print("Valid options for keyword x_or_y are 'x' and 'y'")
            return
        if (x_or_y == 'x'):
            self.x_series.powplot(show_thresholds=show_boot_thresholds, \
             Welch=True, title=r"$\widehat{S}_{x}(f)$", yscale=yscale, vlines=vlines, lw=lw)
        else:
            self.y_series.powplot(show_thresholds=show_boot_thresholds, \
             Welch=True, title=r"$\widehat{S}_{y}(f)$", yscale=yscale, vlines=vlines, lw=lw)
        

# ***Plot the cross-spectrum***        
    def cross_plot(self, vlines=[], lw=0.8):
        # vlines: add vertical lines to plot
        # lw: change linewidth
        try:
            cant = self.coh is None
            if cant:
                raise ValueError
        except ValueError:
            print("Call Welch_coherence_powspec() to calculate cross-spectrum before plotting")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9,6))
        ax1.plot(self.fgrid, self.cross.real, color='dodgerblue', lw=lw)
        ax1.set_ylabel(r"$\Re( \mathcal{F}\{ x_t \star y_t \})$")
        ax2.plot(self.fgrid, self.cross.imag, color='firebrick', lw=lw)
        ax2.set_ylabel(r"$\Im( \mathcal{F}\{ x_t \star y_t \})$")
        ax2.set_xlabel("Frequency")
        for v in vlines:
            ax1.axvline(v, color='gray', linestyle=':')
            ax2.axvline(v, color='gray', linestyle=':')

        
# ***Plot the phase spectrum***
    def phase_plot(self, fal='analytical1', vlines=[]):
        # fal keyword sets the false alarm threshold above which the coherence
        #    should sit in order for the phase estimate to be considered meaningful
        #    options are 'analytical5' = analytical 5% FAL, 'analytical1' = analytical 1% FAL,
        #    'analytical01' = analytical 0.1% FAL,
        #    'boot5' = strict bootstrap 5% FAL, 'boot1' = strict bootstrap 1% FAL, and
        #    'boot01' = strict bootstrap 0.1% FAL
        #    
        try:
            cant = self.coh is None
            if cant:
                raise ValueError
        except ValueError:
            print("Call Welch_coherence_powspec() to calculate phase spectrum before plotting")
            return
        valid_fal = ['analytical5', 'analytical1', 'analytical01', 'boot5', 'boot1', 'boot01']
        if not fal in valid_fal:
            print("Warning: invalid false alarm level selected.")
            print("Defaulting to 1% analytical false alarm level.")
            fal = 'analytical1'
        if (fal == 'analytical5'):
            threshold = self.coh_prob_5
        elif (fal == 'analytical01'):
            threshold = self.coh_prob_01
        elif ((fal == 'boot5') and (self.N_coh_bootstrap >= 100)):
            threshold = self.coh_boot_5
        elif ((fal == 'boot1') and (self.N_coh_bootstrap >= 100)):
            threshold = self.coh_boot_1
        elif ((fal == 'boot01') and (self.N_coh_bootstrap >= 10000)):
            threshold = self.coh_boot_01
        else:
            threshold = self.coh_prob_1
        where_meaningful = np.where(self.coh > threshold)[0]
        plt.figure(figsize=(9,5))
        plt.plot(self.pow_coh_grid, self.phase, color='mediumblue', lw=0.8, ls=':')
        plt.scatter(self.pow_coh_grid[where_meaningful], self.phase[where_meaningful], \
                 color='mediumblue', label=r"Significant $\widehat{C}^2_{xy}(f)$", s=5)
        plt.xlabel('Frequency')
        plt.ylabel(r"$\widehat{\phi}_{xy}(f)$")
        plt.legend(loc="best")
        for v in vlines:
            plt.axvline(v, color='gray', linestyle=':')

        
# ***Save all Bivariate results (coherence, Welch's periodograms,
#    false alarm thresholds, bandwidth, resolution, etc.)
    def save_results(self, filename):
        # filename: name of output file
        try:
            no_results = (self.coh is None)
            if no_results:
                raise ValueError
        except ValueError:
            print("No results to save - call Welch_coherence_powspec() first")
            return
        header = "Bivariate results" + \
                 "\nNyquist frequency: {}".format(self.pow_coh_grid[-1]) + \
                 "\nActual number of segments: {}".format(self.Nseg) + \
                 "\nEffective number of segments: {}".format(self.Nseg_eff) + \
                 "\nSegment start and end points: {}".format(self.segments.tolist()) + \
                 "\nSegments detrended: {}".format(self.trend) + \
                 "\nTrend type: " + self.trend_type + \
                 "\nWindow applied: {}".format(self.window)
        # Bandwidth info
        header = header + "\nSegment-averaged -6dB main lobe half width: {}".format(self.x_series.Welch_band)
        # Rayleigh resolution
        header = header + "\nRayleigh resolution (1/2 best-case resolution limit): {}".format(self.x_series.Welch_Rayleigh)
        # Analytical FAPs
        header = header + "\nC2xy 5% FAP (analytical): {}".format(self.coh_prob_5) + \
                          "\nC2xy 1% FAP (analytical): {}".format(self.coh_prob_1) + \
                          "\nC2xy 0.1% FAP (analytical): {}".format(self.coh_prob_01)
        # Bootstrap FAP info for periodograms
        header = header + "\nNumber of bootstrap iterations: {}".format(self.N_coh_bootstrap)
        if (self.N_coh_bootstrap >= 100):
            header = header + "\nGxx 5% FAP (bootstrap): {}".format(self.xpow_Welch_false_alarm_5) + \
                              "\nGxx 1% FAP (bootstrap): {}".format(self.xpow_Welch_false_alarm_1) + \
                              "\nGyy 5% FAP (bootstrap): {}".format(self.ypow_Welch_false_alarm_5) + \
                              "\nGyy 1% FAP (bootstrap): {}".format(self.ypow_Welch_false_alarm_1)
            if (self.N_coh_bootstrap >= 10000):
                header = header + "\nGxx 0.1% FAP (bootstrap): {}".format(self.xpow_Welch_false_alarm_01) + \
                                  "\nGyy 0.1% FAP (bootstrap): {}".format(self.ypow_Welch_false_alarm_01)
        # Write everything out
        header = header + "\nfrequency,Welch_pow_x,Welch_pow_y,cross_real,cross_imag,phase,coh,coh_raw,coh_transformed"
        output = np.column_stack((self.pow_coh_grid, self.x_series.Welch_pow, \
                                  self.y_series.Welch_pow, self.cross.real, \
                                  self.cross.imag, self.phase, self.coh, self.coh_raw, \
                                  self.coh_transformed))
        if (self.N_coh_bootstrap >= 100):
            header = header + ",coh_boot_FAP_5,coh_transformed_boot_FAP_5,coh_boot_FAP_1,coh_transformed_boot_FAP_1"
            output = np.column_stack((output, self.coh_boot_5, self.coh_transformed_boot_5, self.coh_boot_1, self.coh_transformed_boot_1))
            if (self.N_coh_bootstrap >= 10000):
                header = header + ",coh_boot_FAP_01,coh_transformed_boot_FAP_01"
                output = np.column_stack((output, self.coh_boot_01, self.coh_transformed_boot_01))

        np.savetxt(filename, output, fmt='%.8e', delimiter=',', header=header)
