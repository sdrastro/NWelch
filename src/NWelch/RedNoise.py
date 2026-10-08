import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from NWelch.input_checks import *

plt.rcParams.update({"font.size":16, "axes.labelsize":16, 
                     "font.family":"sans-serif", "font.sans-serif":"Arial"})

# Default model parameter guesses
ar1_param_guesses = np.array([0.7, 1])
pl_param_guesses = np.array([-1.2, 0.1])


def ar1spec(frequency, phi, sigma):
    """Calculate AR(1) power spectrum from model parameters."""
    
    return (sigma**2) / (1 - 2*phi*np.cos(2*np.pi*frequency) + phi**2)
    

def plspec(frequency, exponent, logcoeff):
    """Calculate power law spectrum from model parameters."""
    
    return np.exp(exponent * np.log(frequency) + logcoeff)
    

def gen_ar1(time, phi, sigma):
    """Generate a time-domain ar1 realization from model parameters."""

    dt = np.diff(time)
    N = len(time)
    zeta = np.random.randn(N)
    ar1real = np.zeros(N)
    ar1real[0] = zeta[0]
    for i in range(N-1):
        ar1real[i+1] = phi**dt[i] * ar1real[i] + zeta[i]
    return ar1real


def gen_pl(time, expt, logcoeff, fgrid):
    """Generate a time-domain realization from a power law spectrum."""
    N = len(time)
    Delta_f = fgrid[1] - fgrid[0]
    specmodel = plspec(fgrid, expt, logcoeff)
    randphases = 2*np.pi * np.random.random(N)
    plreal = np.zeros(N)
    for i in range(N):
        plreal[i] = np.sum(np.sqrt(specmodel * 2*np.pi*Delta_f) * \
           np.cos(2*np.pi * fgrid * time[i] + randphases[i]))
    return plreal
    

def wnll_ar1(params, frequency, specest):
    """Compute Whittle likelihood for AR(1) model."""
    
    specmod = ar1spec(frequency, params[0], params[1])
    return sum((np.log(specmod) + (specest/specmod)))
    

def wnll_powerlaw(params, frequency, specest):
    """Compute Whittle likelihood for power law model."""
    
    specmod = plspec(frequency, params[0], params[1])
    return sum((np.log(specmod) + (specest/specmod)))
    

def model_fit(fgrid, specest, guess_pars, plot_fit=True, 
              red_noise_type='ar1', plot_objective=True,
              oplot_limits=np.array([[0.001, 1], [0.5, 1.5]]),
              print_result=True, method='Nelder-Mead', tol=1e-8,
              start_gridpoint=1):
    """Find best-fit parameters for red noise model.

    Args:
        fgrid (ndarray): frequency grid of power spectrum estimate.
        specest (ndarray): power spectrum estimate.
        guess_pars (ndarray/list): initial guesses for parameters.

    Kwargs:
        plot_fit (Bool): if True, display estimated spectrum with best-fit model.
        red_noise_type (str): type of model to fit; options are 'ar1' or 'powerlaw'.
        plot_objective (Bool): if True, display objective function plots from fitting.
        oplot_limits (ndarray): axis limits for pseudocolor plot of NLL(param1, param2).
        print_result (Bool): if True, print best-fit parameters and convergence info.
        method (str): Whittle NLL minimization method used by scipy.optimize.minimize.
        tol (float): tolerance for fit convergence.
        start_gridpoint (int): lowest gridpoint index for fit; set >=1 to exclude f=0.

    Returns:
        ndarray: best-fit model parameters.
        float: Whittle NLL of best-fit model.
    """

    # Keyword checks
    red_noise_type = check_red_model_type(red_noise_type)
    plot_fit = check_Bool(plot_fit, True)
    plot_objective = check_Bool(plot_objective, True)
    print_result = check_Bool(display_result, True)
    method = minimize_method_check(method)
    tol = check_tol(tol)
    oplot_limits = check_oplot_limits(oplot_limits, model_type)
    if not check_param_guesses(guess_pars):
        if red_noise_type == 'ar1':
            guess_pars = ar1_param_guesses
        else:
            guess_pars = pl_param_guesses

    # Define minimization bounds
    bnds_ar1 = ((-1, 1), (0, None))
    bnds_pl = ((-7, 3), (None, None))

    if model_type == 'ar1':
        wnll = wnll_ar1
        bounds = bnds_ar1
    else:
        wnll = wnll_powerlaw
        bounds = bnds_pl

    #Callback function to store Whittle NLLs of each minimize iteration
    objective_values = []
    def keep_NLL(x):
        objective_values.append(wnll(x))

    # Minimization             
    estspec_wnll = minimize(wnll, guess_pars, method=method, tol=tol, 
                            bounds=bounds, callback=keep_NLL)

    # Printing results
    if print_result:
        if model_type == 'ar1':
            print("----------- AR(1) FITTING RESULTS ----------")
            print("Phi = %0.2f"%(estspec_wnll.x[0]))
            print("Sigma = %0.2f"%(estspec_wnll.x[1]))
        else:
            print("----------- Power law FITTING RESULTS ----------")
            print("Exponent = %0.2f"%(estspec_wnll.x[0]))
            print("Coefficient = %0.2f"%(np.exp(estspec_wnll.x[1])))
        print("Whittle NLL = %0.2f"%(estspec_wnll.fun))

    # Plot best-fit model
    if plot_fit:
        plot_bestfit(fgrid, specest, estspec_wnll.x)

    # Plot objective function diagnostics
    if plot_objective:
        plot_objective(fgrid, specest, model_type, objective_values, oplot_limits)

    return estspec_wnll.x, estspec_wnll.fun


def plot_bestfit(fgrid, specest, model_pars, red_noise_type='ar1', loglog=False)
    """Plot estimated power spectrum with best-fit model.

    Args:
        fgrid (ndarray): frequency grid.
        specest (ndarray): power spectrum estimate.
        model_pars (ndarray): parameters of best-fit model.

    Kwargs:
        red_noise_type (str): red noise model type; options are 'ar1' or 'powerlaw'.
        loglog (Bool): if True, make a loglog plot, otherwise it will be semilog-y.
    """
    
    loglog = check_Bool(loglog, False)
    red_noise_type = check_red_noise_type(red_noise_type)

    plt.figure(figsize=(10,6))
    if loglog:
        plt.loglog(fgrid, specest, label=r"$\hat{S}(f)$", color="green",
                   alpha=0.6)
    else:
        plt.semilogy(fgrid, specest, label=r"$\hat{S}(f)$", color="green",
                     alpha=0.6)
    if red_noise_type == 'ar1':
        plt.plot(fgrid, ar1spec(fgrid, *model_pars), color="purple",
                 label="AR(1) fit")
    else:
        plt.plot(fgrid, plspec(fgrid, *model_pars), color='purple',
                 label='Power-law fit')
    plt.grid(axis="both")
    plt.xlabel('Frequency')
    plt.ylabel('Power spectral density')
    plt.legend(loc='lower left', fontsize='small', ncol=2,
               facecolor='white', framealpha=1)


# Plot objective function and minimization check
def plot_objective(fgrid, specest, red_noise_type, objective_values, oplot_limits):
    """Diagnostic plots of fit results.
    
    Four panels: objective function vs. number ofiterations,
    pseudocolor plot of objective function as a function of parameter guesses,
    line plots of objective function marginalized over each parameter.

    Args:
        fgrid (ndarray): frequency grid.
        specest (ndarray): power spectrum estimate.
        red_noise_type (str): type of red noise model, 'ar1' or 'powerlaw'.
        objective_values (list): Whittle NLL values at each iteration returned by model_fit.
        oplot_limits (ndarray): parameter space limits for plots.
    """

    x = np.linspace(oplot_limits[0][0], oplot_limits[0][1], 100)
    y = np.linspace(oplot_limits[1][0], oplot_limits[1][1], 100)
    X, Y = np.meshgrid(x, y)

    ofunc = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            pars = np.array([X[i,j], Y[i,j]]) 
            if red_noise_type == 'ar1':
                ofunc[i,j] = wnll_ar1(pars, fgrid, specest)
            else:
                ofunc[i,j] = wnll_powerlaw(pars, fgrid, specest)

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


def choose_noise_model(fgrid, specest, tol=1e-8,
                       minimize_method='Nelder-Mead', 
                       plot=True, print_result=True,
                       oplot_limits_ar1=None,
                       oplot_limits_pl=None):
    """Automatically determine most appropriate noise model.

    For automatic noise model selection, we will
    simply fit both models and pick the one with the lower
    Whittle NLL. We won't compute NLL distributions here.
    Use nll_distributions() to generate full distributions 
    and visually examine them.

    Args:
        fgrid (ndarray): frequency grid.
        specest (ndarray): estimated power spectrum.

    Kwargs:
        tol (float): relative tolerance for NLL minimization.
        minimize_method (str): numerical method for NLL minimization.
        plot (Bool): if True, plot the best-fit models and objective functions.
        print_result (Bool): if True, print fitting results.
        oplot_limits_ar1 (Bool): objective function plots for AR(1) model.
        oplot-limits_pl (Bool): objevtive function plots for power law model.
    """

    plot = check_Bool(plot, True)
    print_result = check_Bool(print_result, True)
    minimize_method = minimize_method_check(minimize_method)
    tol = check_tol(tol)
    oplot_limits_ar1 = check_oplot_limits(oplot_limits_ar1, 'ar1')
    oplot_limits_pl = check_oplot_limits(oplot_limits_pl, 'powerlaw')

    ar1_nll, ar1_pars, ar1_estspec = 
            model_fit(fgrid, specest, ar1_param_guesses, 
                      plot_fit=plot, plot_objective=plot,
                      print_result=print_result,
                      red_noise_type='ar1', method=method, 
                      tol=tol)

    pl_nll, pl_pars, pl_estspec = 
            model_fit(fgrid, specest, pl_param_guesses, 
                      plot_fit=plot, plot_objective=plot,
                      print_result=print_result, 
                      red_noise_type='powerlaw', method=method, 
                      tol=tol)

    if plot:
        # Plot best-fit model
        plot_bestfit(fgrid, specest, ar1_estspec.x, 
                     red_noise_type='ar1')
        plot_objective(fgrid, specest, 'ar1', 
                        ar1_nll, oplot_limits_ar1)
        plot_bestfit(fgrid, specest, pl_estspec.x, 
                     red_noise_type='powerlaw', loglog=True)
        plot_objective(fgrid, specest, 'powerlaw', 
                        pl_nll, oplot_limits_pl)

    if ar1_nll <= pl_nll:
        model_type = 'ar1'
    else:
        model_type = 'powerlaw'

    if print_result:
        print('Best-fit model: ' + model_type)

    return model_type # Done


def nll_distribution(time, obs, err, segs=None, num=1000, Nyquist=0.5, 
                     window='None', oversample=4, trend=True, 
                     trend_type='linear', combine_mode='mean', 
                     red_noise_type='ar1', minimize_method='Nelder-Mead', 
                     tol=1e-8, guess_pars=ar1_param_guesses, 
                     start_gridpoint=1, plot_distribution=True):
    """compute NLL distribution for realizations of the time series tweaked within error bars.
    
    Args:
        time (ndarray):
        obs (ndarray):
        err (ndarray):

    Kwargs:
        segs (int / ndarray / None):
        num (int):
        Nyquist (float):
        window (str):
        oversample (int / float):
        trend (Bool):
        trend_type (str):
        combine_mode (str):
        red_noise_type (str):
        minimize_method (str):
        tol (float):
        guess_pars (ndarray):
        start_gridpoint (int):
        plot_distribution (Bool):
    """

    # Input checks
    num = check_bootstrap(num)
    if (num < 100):
        print('Must set num >= 100 to run Monte Carlo')
        return
    red_noise_type = check_red_noise_type(red_noise_type)
    detrend = check_Bool(detrend, True)
    trend_type = trend_check(trend_type)
    minimize_method = minimize_method_check(minimize_method)
    tol = check_tol(tol)
    valid_fgrid = check_nyquist(Nyquist) and check_oversample(oversample)
    if not valid_fgrid:
        return

    if not check_param_guesses(guess_pars):
        if red_noise_type == 'ar1':
            guess_pars = ar1_param_guesses
        else:
            guess_pars = pl_param_guesses

    if segs == None:
        Welch = False
    else:
        Welch = True

    if (not isinstance(start_gridpoint, int)) or (start_gridpoint < 0) \
           or (start_gridpoint >= 500):
        print('Invalid start gridpoint. Must be integer >=0 and <= 500. Defaulting to 1.')
        start_gridpoint = 1

    # Make a TimeSeries object with the actual observations in order to check
    #   for valid segmentation scheme (if Welch) and successful power spectrum estimate
    base_ts = TimeSeries(time, obs)
    if Welch:
        base_ts.segment_data(segs, Nyquist, oversample=oversample, window=window,
                             plot_windows=False, quiet=True)
        if not check_segmented(base_ts.segmented):
            return
        base_ts.Welch_powspec(trend=trend, trend_type=trend_type, combine_mode=combine_mode)
        if not check_Welch_power(base_ts.Welch_pow):
            return
    else:
        base_ts.frequency_grid(Nyquist, oversample=oversample)
        base_ts.pow_FT(N_bootstrap=0, quiet=True, window=window, trend=trend,
                       trend_type=trend_type, combine_mode=combine_mode)
        if not check_power(base_ts.power):
            return

    # Now perform the Monte Carlo
    N = len(time)
    Nf = len(fgrid)
    speciters = np.array((num, Nf))
    bfpars = np.array((num, 2))
    wnlls = np.array(num)
    for i in range(num):
        obsMC = obs + np.random.randn(N) * err
        ts = TimeSeries(time, obsMC)
        if Welch:
            ts.segment_data(segs, Nyquist, oversample=oversample, window=window,
                            plot_windows=False, quiet=True)
            ts.Welch_powspec(trend=trend, trend_type=trend_type, mode=combine_mode)
            spec = ts.get_Welch()
            pw = spec['Welch_power']
        else:
            ts.frequency_grid(Nyquist, oversample=oversample)
            ts.pow_FT(window=window, trend=trend, trend_type=trend_type,
                      N_bootstrap=0)
            spec = ts.get_periodogram()
            pw = spec['periodogram']
        fr = spec['frequency']
        speciters[i, :] = pw
        wnll, pars, _ = model_fit(fr[start_gridpoint:], pw[start_gridpoint:], 
                                  guess_pars, plot_fit=False, 
                                  red_noise_type=red_noise_type, plot_objective=False,
                                  print_result=False, minimize_method=minimize_method, 
                                  tol=tol)
        wnlls[i] = wnll
        bfpars[i, :] = pars
        # Finished with Monte Carlo

    # Plot the NLL distribution
    if plot_distribution:
        plt.hist(wnlls, color='mediumseagreen', alpha=0.7)
        plt.xlabel('Whittle NLL')
        plt.ylabel('Number of trials')
        plt.title(red_noise_type + ' model')
        plt.grid(color='0.85')

    return wnlls, bfpars

# So far I haven't thought about how to apply this same code to a simple periodogram (non-Welch)!! -> Can do this with a simple arg or kwarg: if segs == None, simple periodogram
# if segs arg is in NWelch format, do the segmentation

# Should I make red_noise_type and start_gridpoint sticky somehow? Or maybe just wrap all of this
#   code into an object so that every parameter is sticky throughout the analysis?

# Calculate FALs
def FALs(time, nlldist, segs, Nyquist=0.5, red_noise_type='ar1', 
         window='None', oversample=4, trend=True, 
         trend_type='linear', combine_mode='mean')
    
