import numpy as np
from scipy.optimize import minimize
from NWelch.input_checks import *

# AR(1) power spectrum
def ar1spec(frequency, phi, sigma):
    return (sigma**2) / (1 - 2*phi*np.cos(2*np.pi*frequency) + phi**2)
    

# Power law spectrum
def plspec(frequency, exponent, coeff):
    return np.exp(exponent * np.log(frequency) + coeff)
    

# Whittle likelihood for AR(1)
def wnll_ar1(params, frequency, specest):
    specmod = ar1spec(frequency, params[0], params[1])
    return sum((np.log(specmod) + (specest/specmod)))
    

# Whittle likelihood for power law
def wnll_powerlaw(params, frequency, specest):
    specmod = plspec(frequency, params[0], params[1])
    return sum((np.log(specmod) + (specest/specmod)))
    

# Find best-fit noise model 
def model_fit(fgrid, specest, guess_pars, plot_fit=True, 
              model_type='ar1', plot_objective=True,
              oplot_limits=np.array([[0.001, 1], [0.5, 1.5]]),
              print_result=True, method='Nelder-Mead', tol=1e-8):

    # Keyword checks
    model_type = check_red_model_type(model_type)
    plot_fit = check_Bool(plot_fit, True)
    plot_objective = check_Bool(plot_objective, True)
    print_result = check_Bool(display_result, True)
    method = minimize_method_check(method)
    tol = check_tol(tol)
    oplot_limits = _check_oplot_limits(oplot_limits, model_type)

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
    estspec_wnll = minimize(wnll, x0, method=method, tol=tol, 
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
            print("Coefficient = %0.2f"%(estspec_wnll.x[1]))
        print("Whittle NLL = %0.2f"%(estspec_wnll.fun))

    # Plot best-fit model
    if plot_fit:
        plot_bestfit(fgrid, specest, estspec_wnll.x)

    # Plot objective function diagnostics
    if plot_objective:
        _plot_objective(fgrid, specest, model_type, objective_values, oplot_limits)

    return whittle_nll, estspec_wnll.x, estspec_wnll.fun


# Plot estimated power spectrum with best-fit model
def plot_bestfit(fgrid, specest, model_pars, red_noise_type='ar1', loglog=False)
    
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
def _plot_objective(fgrid, specest, red_noise_type, objective_values, oplot_limits):

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


# Find most appropriate noise model
#   For automatic noise model selection, we will
#     simply fit both models and pick the one with the lower
#     Whittle NLL. We won't compute NLL distributions here.
#   Users can also generate full distributions, visually examine them,
#     and choose their favorite model
def choose_noise_model(fgrid, specest,
                       ar1_param_guesses=[0.7, 1],
                       pl_param_guesses=[-1.2, 0.1],
                       method='Nelder-Mead', tol=1e-8,
                       plot=True, print_result=True,
                       oplot_limits_ar1=None,
                       oplot_limits_pl=None):

    plot = check_Bool(plot, True)
    print_result = check_Bool(print_result, True)
    method = minimize_method_check(method)
    tol = check_tol(tol)
    oplot_limits_ar1 = _check_oplot_limits(oplot_limits_ar1, 'ar1')
    oplot_limits_pl = _check_oplot_limits(oplot_limits_pl, 'powerlaw')

    ar1_nll, ar1_pars, ar1_estspec = 
            model_fit(fgrid, specest, ar1_param_guesses, 
                      plot_fit=plot, plot_objective=plot,
                      print_result=print_result,
                      model_type='ar1', method=method, 
                      tol=tol)

    pl_nll, pl_pars, pl_estspec = 
            model_fit(fgrid, specest, pl_param_guesses, 
                      plot_fit=plot, plot_objective=plot,
                      print_result=print_result, 
                      model_type='powerlaw', method=method, 
                      tol=tol)

    if plot:
        # Plot best-fit model
        plot_bestfit(fgrid, specest, ar1_estspec.x, 
                     red_noise_type='ar1')
        _plot_objective(fgrid, specest, 'ar1', 
                        ar1_nll, oplot_limits_ar1)
        plot_bestfit(fgrid, specest, pl_estspec.x, 
                     red_noise_type='powerlaw', loglog=True)
        _plot_objective(fgrid, specest, 'powerlaw', 
                        pl_nll, oplot_limits_pl)

    if ar1_nll <= pl_nll:
        model_type = 'ar1'
    else:
        model_type = 'powerlaw'

    if print_result:
        print('Best-fit model: ' + model_type)

    return model_type


def gen_spectrum_realizations():


# Generate an ar1 realization in the time domain
def gen_ar1():


# Generate a realization from a known power-law spectrum
#   in the time domain
def gen_pl():
    
