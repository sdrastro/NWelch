import numpy as np
from scipy.optimize import minimize
from NWelch.input_checks import *

# AR(1) power spectrum
def _ar1spec(frequency, phi, sigma):
    return (sigma**2) / (1 - 2*phi*np.cos(2*np.pi*frequency) + phi**2)

# Power law spectrum
def _plspec(frequency, exponent, coeff):
    return np.exp(exponent * np.log(frequency) + coeff)

# Whittle likelihood for AR(1)
def wnll_ar1(params, frequency, specest):
    specmod = _ar1spec(frequency, params[0], params[1])
    return sum((np.log(specmod) + (specest/specmod)))

# Whittle likelihood for power law
def wnll_powerlaw(params, frequency, specest):
    specmod = _plspec(frequency, params[0], params[1])
    return sum((np.log(specmod) + (specest/specmod)))

# Find best-fit AR(1) 
def ar1_fit(fgrid, specest, guess_pars, plot_fit=True, plot_objective=True,
            oplot_limits=np.array([[0.001, 1], [0.5, 1.5]]),
            print_result=True, method='Nelder-Mead', tol=1e-8):

    # Keyword checks
    plot_fit = check_Bool(plot_fit, True)
    plot_objective = check_Bool(plot_objective, True)
    print_result = check_Bool(display_result, True)
    method = minimize_method_check(method)
    tol = check_tol(tol)
    oplot_limits = check_oplot_limits(oplot_limits)

    #Callback function to store Whittle NLLs of each minimize iteration
    objective_values = []
    def keep_NLL(x):
        objective_values.append(wnll_ar1(x))

    # Minimization             
    bnds = ((-1, 1), (0, None))
    estspec_wnll = minimize(wnll_ar1, x0, method=method, tol=tol, 
                            bounds=bnds, callback=keep_NLL)

    # Best-fit parameter estimates
    whittle_nll = estspec_wnll.fun
    phi = estspec_wnll.x[0]
    sigma = estspec_wnll.x[1]

    # Printing results
    if print_result:
        print("------------------- AR(1) FITTING RESULTS ---------------")
        print("Phi = %0.2f"%(phi))
        print("Sigma = %0.2f"%(sigma))
        print("Whittle NLL = %0.2f"%(whittle_nll))

    # Plot best-fit model
    if plot_fit:
        plot_bestfit(fgrid, specest, [phi, sigma])


# Plot estimated power spectrum with best-fit model
def plot_bestfit(fgrid, specest, model_pars, model_type='ar1', loglog=False)
    loglog = check_Bool(loglog, False)
    plt.figure(figsize=(10,6))
    if loglog:
        plt.loglog(fgrid, specest, label=r"$\hat{S}(f)$", color="green",
                   alpha=0.6)
    else:
        plt.semilogy(fgrid, specest, label=r"$\hat{S}(f)$", color="green",
                     alpha=0.6)
    if model_type == 'ar1':
        plt.plot(fgrid, _ar1spec(fgrid, *model_pars), color="purple",
                 label="AR(1) fit")
    else:
        plt.plot(fgrid, _plspec(fgrid, *model_pars), color='purple',
                 label='Power-law fit')
    plt.grid(axis="both")
    plt.xlabel('Frequency')
    plt.ylabel('Power spectral density')
    plt.legend(loc='lower left', fontsize='small', ncol=2,
               facecolor='white', framealpha=1)


# Find most appropriate noise model
#   Unlike in the full RedNoiseFALs software package, we will
#   simply fit both models and pick the one with the lower
#   Whittle NLL. We won't compute NLL distributions here.
def choose_noise_model(frequency, specest,
                       ar1_param_guesses=[0.7, 1],
                       pl_param_guesses=[-1.2, 0.1],
                       method='Nelder-Mead', tol=1e-8):
