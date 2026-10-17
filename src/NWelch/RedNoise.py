import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from NWelch.input_checks import *

plt.rcParams.update({"font.size":16, "axes.labelsize":16, 
                     "font.family":"sans-serif", "font.sans-serif":"Arial"})

ar1_param_guesses = np.array([0.7, 1])
pl_param_guesses = np.array([-1.2, 0.1])
"""default model parameter guesses."""


# Workhorse functions to generate spectrum models and time-domain noise realizations

def ar1spec(frequency, phi, sigma):
    """calculate AR(1) power spectrum from model parameters."""
    
    return (sigma**2) / (1 - 2*phi*np.cos(2*np.pi*frequency) + phi**2)
    

def plspec(frequency, exponent, logcoeff):
    """calculate power law spectrum from model parameters."""
    
    return np.exp(exponent * np.log(frequency) + logcoeff)
    

def gen_ar1(time, phi, sigma):
    """generate a time-domain ar1 realization from model parameters."""

    dt = np.diff(time)
    N = len(time)
    zeta = np.random.randn(N)
    ar1real = np.zeros(N)
    ar1real[0] = zeta[0]
    for i in range(N-1):
        ar1real[i+1] = phi**dt[i] * ar1real[i] + zeta[i]
    return ar1real


def gen_pl(time, expt, logcoeff, fgrid):
    """generate a time-domain realization from a power law spectrum."""
    
    N = len(time)
    Delta_f = fgrid[1] - fgrid[0]
    specmodel = plspec(fgrid, expt, logcoeff)
    randphases = 2*np.pi * np.random.random(N)
    plreal = np.zeros(N)
    for i in range(N):
        plreal[i] = np.sum(np.sqrt(specmodel * 2*np.pi*Delta_f) * \
           np.cos(2*np.pi * fgrid * time[i] + randphases[i]))
    return plreal
    

# Objective functions

def wnll_ar1(params, frequency, specest):
    """compute Whittle negative log likelihood for AR(1) model."""

    specmod = ar1spec(frequency, params[0], params[1])
    return sum((np.log(specmod) + (specest/specmod)))


def wnll_pl(params, frequency, specest):
    """compute Whittle negative log likelihood for power law model."""

    specmod = plspec(frequency, params[0], params[1])
    return sum((np.log(specmod) + (specest/specmod)))
    

# Class begins here

class RedNoise:
"""methods for calculating red noise-based false alarm levels."""

    def __init__(self, t, obs, err, fgrid, specest, Welch=False, segs=None, 
                 red_noise_type='ar1', method='Nelder-Mead', tol=1e-8, 
                 trend=True, trend_type='linear', start_gridpoint=1, 
                 oplot_limits=np.array([[0.001, 1], [0.5, 1.5]])
                 window='None'):

        """set time series, power spectrum estimate, and fit parameters for red noise FALS.

        If RedNoise functions are called from NWelch, this RedNoise object is created
        automatically, and the data and most of the fitting parameters will be pegged 
        to NWelch class variables. If the RedNoise module is used standalone, you 
        will need to instantiate the RedNoise object yourself and use the keywords to 
        set fitting parameters. Defaults are provided.

        Args:
            t (ndarray): array of observation time stamps.
            obs (ndarray): array containing observed data.
            err (ndarray): array containing error bars.
            fgrid (ndarray): frequency grid for power spectrum estimate.
            specest (ndarray): power spectrum estimate. Can come from NWelch or another module.

        Kwargs:
            Welch (bool): if True, the power spectrum was computed with Welch's method.
            segs (None/ndarray): 2d ndarray containing segment start and end indices.
                                 Ignored if Welch=False.
            red_noise_type (str): type of model to fit; options are 'ar1' or 'powerlaw'.
            method (str): Whittle NLL minimization method used by scipy.optimize.minimize.
            tol (float): tolerance for fit convergence.
            trend (bool): if True, subtract a long-term trend before calculating FALs.
            trend_type (str): type of trend to subtract. Options are 'linear' or 'quadratic'.
            start_gridpoint (int): lowest gridpoint index for fit; set >=1 to exclude f=0.
            oplot_limits (ndarray): axis limits for pseudocolor plot of NLL(param1, param2).
            window (str): type of window to apply when calculating FALs. Options are
                          'None', 'BlackmanHarris', 'KaiserBessel'.

        Attributes:
            t (ndarray): array of observation time stamps.
            obs (ndarray): array containing observed data.
            err (ndarray): array containing error bars.
            fgrid (ndarray): frequency grid for power spectrum estimate.
            specest (ndarray): power spectrum estimate.
            Welch (bool): whether power spectrum was computed with Welch's method.
            segs (None/ndarray): 2d ndarray containing Welch segment start and end indices.
            red_noise_type (str): type of red noise model to fit.
            method (str): Whittle NLL minimization method used by scipy.optimize.minimize.
            tol (float): tolerance for fit convergence.
            trend (bool): whether to subtract a long-term trend before calculating FALs.
            trend_type (str): type of trend to subtract.
            start_gridpoint (int): lowest gridpoint index for fit.
            oplot_limits (ndarray): axis limits for pseudocolor plot of NLL(param1, param2).
            window (str): type of window to apply when calculating FALs.

        Raises:
            ValueError: if t, obs, and err OR fgrid and specest don't have same lengths.
            ValueError: if seg does not define statistically reasonable segments.
        """
        try:
            if not array_check(t, obs, domain='time'):
                raise ValueError
        except ValueError:
            return
        try:
            if not array_check(t, err, domain='time'):
                raise ValueError
        except ValueError:
            return
        try:
            if not array_check(fgrid, specest, domain='frequency'):
                raise ValueError
        except ValueError:
            return

        self.t = t
        self.obs = obs
        self.err = err
        self.fgrid = fgrid
        self.specest = specest
            
        self.Welch = check_Bool(Welch, False)
        if self.Welch:
            if not check_custom_segments(seg):
                return
        self.segs = segs
        self.window = window_check(window)
                
        self.red_noise_type = check_red_model_type(red_noise_type)
        self.method = minimize_method_check(method)
        self.tol = check_tol(tol)
        self.oplot_limits = check_oplot_limits(oplot_limits)


    def model_fit(self, guess_pars, red_noise_type='ar1', plot_fit=True, 
                  plot_objective=True, print_result=True):
        """Find best-fit parameters for red noise model.
    
        Args:
            guess_pars (ndarray/list): initial guesses for parameters.
    
        Kwargs:
            red_noise_type (str): type of red noise model to fit, ar1 or powerlaw.
            plot_fit (bool): if True, display estimated spectrum with best-fit model.
            plot_objective (bool): if True, display objective function plots from fitting.
            print_result (bool): if True, print best-fit parameters and convergence info.
    
        Returns:
            ndarray: best-fit model parameters.
            float: Whittle NLL of best-fit model.
        """
    
        # Keyword checks
        plot_fit = check_Bool(plot_fit, True)
        plot_objective = check_Bool(plot_objective, True)
        print_result = check_Bool(display_result, True)
        if not check_param_guesses(guess_pars):
            if red_noise_type == 'ar1':
                guess_pars = ar1_param_guesses
            else:
                guess_pars = pl_param_guesses
    
        # Define minimization bounds
        bnds_ar1 = ((-1, 1), (0, None))
        bnds_pl = ((-7, 3), (None, None))
    
        if red_noise_type == 'ar1':
            wnll = wnll_ar1
            bounds = bnds_ar1
        else:
            wnll = wnll_powerlaw
            bounds = bnds_pl
    
        #Callback function to store Whittle NLLs of each minimize iteration
        objective_values = []
        def keep_NLL(x):
            objective_values.append(wnll(x))

        fr = self.fgrid[self.start_gridpoint:]
        sp = self.specest[self.start_gridpoint:]
    
        # Minimization             
        estspec_wnll = minimize(wnll, guess_pars, args=(fr, sp),
                                method=self.method, tol=self.tol, 
                                bounds=bounds, callback=keep_NLL)
    
        # Printing results
        if print_result:
            if self.model_type == 'ar1':
                print("----------- AR(1) FITTING RESULTS ----------")
                print("Phi = %0.2f"%(estspec_wnll.x[0]))
                print("Sigma = %0.2f"%(estspec_wnll.x[1]))
            else:
                print("----------- Power law FITTING RESULTS ----------")
                print("Exponent = %0.2f"%(estspec_wnll.x[0]))
                print("Coefficient = %0.2f"%(np.exp(estspec_wnll.x[1])))
            print("Whittle NLL = %0.2f"%(estspec_wnll.fun))

        if red_noise_type == 'ar1':
            loglog=False
        else:
            loglog=True
            
        # Plot best-fit model
        if plot_fit:
            plot_bestfit(estspec_wnll.x, red_noise_type=red_noise_type, loglog=loglog)
    
        # Plot objective function diagnostics
        if plot_objective:
            plot_objective(objective_values, self.oplot_limits, red_noise_type=red_noise_type)
    
        return estspec_wnll.x, estspec_wnll.fun


    def plot_bestfit(model_pars, red_noise_type='ar1', loglog=False)
        """Plot estimated power spectrum with best-fit model.
    
        Args:
            model_pars (ndarray): parameters of best-fit model.
    
        Kwargs:
            red_noise_type (str): name of red noise model, ar1 or powerlaw.
            loglog (bool): if True, make a loglog plot, otherwise it will be semilog-y.
        """
        red_noise_type = check_red_noise_type(red_noise_type)
        loglog = check_bool(loglog, False)
        sg = self.start_gridpoint
    
        plt.figure(figsize=(10,6))
        if loglog:
            plt.loglog(self.fgrid[sg:], self.specest[sg:], label=r"$\hat{S}(f)$", 
                       color="green", alpha=0.6)
        else:
            plt.semilogy(self.fgrid[sg:], self.specest[sg:], label=r"$\hat{S}(f)$",
                         color="green", alpha=0.6)
        if red_noise_type == 'ar1':
            plt.plot(self.fgrid[sg:], ar1spec(self.fgrid[sg:], *model_pars), 
                     color="purple", label="AR(1) fit")
        else:
            plt.plot(self.fgrid[sg:], plspec(self.fgrid[sg:], *model_pars), 
                     color='purple', label='Power-law fit')
        plt.grid(axis="both")
        plt.xlabel('Frequency')
        plt.ylabel('Power spectral density')
        plt.legend(loc='lower left', fontsize='small', ncol=2,
                   facecolor='white', framealpha=1)
    
    
    # Plot objective function and minimization check
    def plot_objective(objective_values, oplot_limits, red_noise_type):
        """Diagnostic plots of fit results.
        
        Four panels: objective function vs. number ofiterations,
        pseudocolor plot of objective function as a function of parameter guesses,
        line plots of objective function marginalized over each parameter.
    
        Args:
            objective_values (list): Whittle NLL values at each iteration returned by model_fit.
            oplot_limits (ndarray): parameter space limits for plots.
            red_noise_type (str): type of red noise model fitted (ar1 or powerlaw).
        """
    
        x = np.linspace(oplot_limits[0][0], oplot_limits[0][1], 100)
        y = np.linspace(oplot_limits[1][0], oplot_limits[1][1], 100)
        X, Y = np.meshgrid(x, y)
        sg = self.start_gridpoint
    
        ofunc = np.zeros((len(x),len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                pars = np.array([X[i,j], Y[i,j]]) 
                if red_noise_type == 'ar1':
                    ofunc[i,j] = wnll_ar1(pars, self.fgrid[sg:], self.specest[sg:])
                else:
                    ofunc[i,j] = wnll_powerlaw(pars, self.fgrid[sg:], self.specest[sg:])
    
        min_idx = np.unravel_index(np.argmin(ofunc), ofunc.shape)
        min_x = X[min_idx]
        min_y = Y[min_idx]
    
        #1D objective function and parameters
        param2 = min_y
        obj_func1 = np.zeros(len(x))
        for i in range(len(x)):
            obj_func1[i]= wnll_ar1([x[i], param2])
    
        param1 = min_x
        obj_func2 = np.zeros(len(y))
        for i in range(len(y)):
            obj_func2[i]= wnll_ar1([param1, y[i]])
    
        #Plotting the 4 plots together
    
        if red_noise_type='ar1':
            xlabel = r"$\phi$"
            ylabel = r"$\sigma$"
        else:
            xlabel = 'Exponent'
            ylabel = 'Coefficient'
        whitnllstr = 'Whittle NLL'
    
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,7))
        fig.suptitle("Minimization check with objective function plots")
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, 
                            wspace=0.4, hspace=0.4)
    
        # Plot 1: Objective function vs. iterations
        ax1.plot(objective_values)
        ax1.set_xlabel("Number of iterations")
        ax1.set_ylabel(whitnllstr)
    
        # Plot 2: 2D objective function vs. parameters
        levels = np.logspace(np.log10(np.min(ofunc)),           
                             np.log10(np.max(ofunc)), 20)
        norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)
        ax2.pcolormesh(X, Y, ofunc, norm = norm)
        c0 = ax2.scatter(min_x, min_y, color='red', marker='o', s=10)
        fig.colorbar(c0, ax=ax2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
    
        # Plot 3: Objective function vs. parameter 1
        ax3.plot(x, obj_func1)
        ax3.scatter(x[np.argmin(obj_func1)], np.min(obj_func1), color="black")
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(whitnllstr)
    
        # Plot 4: Objective function vs. parameter 2
        ax4.plot(y,obj_func2)
        ax4.scatter(y[np.argmin(obj_func2)], np.min(obj_func2), color="black")
        ax4.set_xlabel(ylabel)
        ax4.set_ylabel(whitnllstr)


    def choose_noise_model(plot=True, print_result=True, oplot_limits_ar1=None,
                           oplot_limits_pl=None):
        """Automatically determine most appropriate noise model.
    
        For automatic noise model selection, we will
        simply fit both models and pick the one with the lower
        Whittle NLL. Result is stored in self.red_noise_type.
        We won't compute NLL distributions here.       
        Use nll_distributions() to generate full distributions 
        and visually examine them.
    
        Kwargs:
            plot (bool): if True, plot the best-fit models and objective functions.
            print_result (bool): if True, print fitting results.
            oplot_limits_ar1 (bool): objective function plots for AR(1) model.
            oplot-limits_pl (bool): objevtive function plots for power law model.
        """
    
        plot = check_bool(plot, True)
        print_result = check_bool(print_result, True)
        oplot_limits_ar1 = check_oplot_limits(oplot_limits_ar1, 'ar1')
        oplot_limits_pl = check_oplot_limits(oplot_limits_pl, 'powerlaw')
    
        ar1_nll, ar1_pars, ar1_estspec = 
                model_fit(self.fgrid, self.specest, ar1_param_guesses, 
                          plot_fit=plot, plot_objective=plot,
                          red_noise_type='ar1', print_result=print_result)
    
        pl_nll, pl_pars, pl_estspec = 
                model_fit(self.fgrid, self.specest, pl_param_guesses, 
                          plot_fit=plot, plot_objective=plot,
                          red_noise_type='powerlaw', print_result=print_result)
    
        if plot:
            # Plot best-fit model of each type
            plot_bestfit(self.fgrid, self.specest, ar1_estspec.x, 
                         red_noise_type='ar1')
            plot_objective(self.fgrid, self.specest, 'ar1', 
                            ar1_nll, oplot_limits_ar1)
            plot_bestfit(fgrid, specest, pl_estspec.x, 
                         red_noise_type='powerlaw', loglog=True)
            plot_objective(fgrid, specest, 'powerlaw', 
                            pl_nll, oplot_limits_pl)
    
        if ar1_nll <= pl_nll:
            self.red_noise_type = 'ar1'
        else:
            self.red_noise_type = 'powerlaw'
    
        if print_result:
            print('Best-fit model: ' + model_type)


    def FALs(num=1000, red_noise_type='ar1', combine_mode='mean', 
             guess_pars=ar1_param_guesses, plot_distribution=True,
             plot_FALs=True, return_WNLLs=False, return_pars=False):
        """compute FALs using Monte Carlo realizations of the time series. 

        Minimum number of Monte Carlo realizations for Monte Carlo is 100.
        For best results, set num >= 1000 to get 1% false alarm levels. 
        Set num >= 1000 to get 0.1% false alarm levels. You must have at least
        num iterations to get a (1/num) false alarm level.
    
        Kwargs:
            num (int): number of Monte Carlo iterations.
            red_noise_type (str): name of red noise model type, ar1 or powerlaw.
            combine_mode (str): method to combine Welch's spectrum estimates, 'median or 'mean'.
            guess_pars (ndarray): guess parameters for model fitting.
            plot_distribution (bool): if True, plot the resulting NLL distribution.
            plot_FALs (bool): if True, plot power spectrum estimate with FALs.
            return_WNLLs (bool): if True, return Whittle NLLs
            return_pars (bool): if True, return fitted params.

        Returns:
            dict: dictionary of frequency-dependent 5%, 1% and (if computed) 0.1% FALs.
                Optionally, also return Monte Carlo Whittle NLLs and best-fit parameters.

        Raises:
            ValueError: if num < 100.
        """
    
        # Input checks
        num = check_bootstrap(num)
        if (num < 100):
            print('Must set num >= 100 to run Monte Carlo')
            return
    
        if not check_param_guesses(guess_pars):
            if red_noise_type == 'ar1':
                guess_pars = ar1_param_guesses
            else:
                guess_pars = pl_param_guesses
    
        # Make a TimeSeries object with the actual observations in order to verify
        #   valid segmentation scheme (if Welch) and successful power spectrum estimate
        base_ts = TimeSeries(self.time, self.obs)
        if self.Welch:
            base_ts.segment_data(self.segs, self.Nyquist, oversample=self.oversample, 
                                 window=self.window, plot_windows=False, quiet=True)
            if not check_segmented(base_ts.segmented):
                return
            base_ts.Welch_powspec(trend=self.trend, trend_type=self.trend_type,
                                  combine_mode=combine_mode)
            if not check_Welch_power(base_ts.Welch_pow):
                return
        else:
            base_ts.frequency_grid(self.Nyquist, oversample=self.oversample)
            base_ts.pow_FT(N_bootstrap=0, quiet=True, window=self.window, 
                           trend=self.trend, trend_type=self.trend_type,
                           combine_mode=combine_mode)
            if not check_power(base_ts.power):
                return
    
        # Now perform the Monte Carlo
        N = len(self.time)
        Nf = len(self.fgrid)
        speciters = np.array((num, Nf))
        bfpars = np.array((num, 2))
        wnlls = np.array(num)
        for i in range(num):
            obsMC = self.obs + np.random.randn(N) * self.err
            ts = TimeSeries(self.time, self.obsMC)
            if Welch:
                ts.segment_data(segs, self.Nyquist, oversample=self.oversample, 
                                window=self.window, plot_windows=False, quiet=True)
                ts.Welch_powspec(trend=self.trend, trend_type=self.trend_type, mode=combine_mode)
                spec = ts.get_Welch()
                speciters[i,:] = spec['Welch_power']
            else:
                ts.frequency_grid(self.Nyquist, oversample=self.oversample)
                ts.pow_FT(window=self.window, trend=self.trend, trend_type=self.trend_type,
                          N_bootstrap=0)
                spec = ts.get_periodogram()
                speciters[i,:] = spec['periodogram']
            fr = spec['frequency']
            speciters[i, :] = pwr
            wnll, pars, _ = model_fit(fr[self.start_gridpoint:], pwr[self.start_gridpoint:], 
                                      guess_pars, plot_fit=False, red_noise_type=red_noise_type,
                                      plot_objective=False, print_result=False)
            wnlls[i] = wnll
            bfpars[i, :] = pars
            # Finished with Monte Carlo
    
        # Plot the NLL distribution
        if plot_distribution:
            if red_noise_type == 'ar1':
                titlestr = 'AR(1)'
            else:
                titlestr = 'Power law'
            plt.hist(wnlls, color='mediumseagreen', alpha=0.7)
            plt.xlabel('Whittle NLL')
            plt.ylabel('Number of trials')
            plt.title(titlestr + ' model')
            plt.grid(color='0.85')

        # Calculate FALs
        fal5 = np.quantile(speciters, 0.95, axis=0)
        fal1 = np.quantile(speciters, 0.99, axis=0)
        fal_dict = {'frequency':fgrid, 'fal5pct':fal5, 'fal1pct':fal1}
        if num >= 1000:
            fal01 = np.quantile(speciters, 0.999, axis=0)
            fal_dict['fal0.1pct'] = fal01

        if return_WNLLs:
            fal_dict['Monte Carlo NLLs'] = wnlls

        if return_pars:
            fal_dict['Monte Carlo pars'] = bfpars
    
        return fal_dict