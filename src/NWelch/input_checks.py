# Check for appropriate inputs to various functions - if the
#    check is repeated in multiple functions, put it here to
#    clean up the quantitative code


# Check that a keyword is a string
check_string = lambda keyword: isinstance(keyword, str)


# Check for gradient-free red noise model fitting method
def minimize_method_check(minimize_method):
    if not check_string(minimize_method):
        print('minimize_method must be a string.')
        print('Defaulting to Nelder-Mead.')
        minimize_method = 'Nelder-Mead'
    else:
        if not ((minimize_method == 'Nelder-Mead') or \
             (minimize_method == 'TNC') or (minimize_method == 'Powell')):
            print('Choose gradient-free method: Nelder-Mead, TNC, or Powell.')
            print('Defaulting to Nelder-Mead.')
            minimize_method = 'Nelder-Mead'
    return minimize_method


# Check for valid window type: Must be 'KaiserBessel',
#   BlackmanHarris', or 'None'
def window_check(window):
    try:
        valid_window = ((window == 'BlackmanHarris') or \
                        (window == 'KaiserBessel') or (window == 'None'))
        if not valid_window:
            raise ValueError
    except ValueError:
        print("Invalid window type. Choose 'BlackmanHarris',") 
        print("'KaiserBessel', or 'None'. Defaulting to 'None'.")  
        window = 'None'
    return window


# Check for valid trend type: Must be 'linear' or 'quadratic'.
def trend_check(trend_type):
    valid_trend_type = ((trend_type == 'linear') or \
                        (trend_type == 'quadratic'))
    if not valid_trend_type:
         print("trend_type not understood. Options: 'linear' or 'quadratic'.")
         print('Defaulting to linear trend')
         trend_type = 'linear'
    return trend_type


# Check for Boolean keyword
def check_Bool(keyword, default): 
    if not isinstance(keyword, bool):
        keyword = default
    return keyword


# Check for valid number of bootstrap iterations
def check_bootstrap(Nboot):
     try:
         valid_N_bootstrap = ((type(Nboot) is int) and (Nboot >= 100))
         if not valid_N_bootstrap:
              raise ValueError
     except ValueError:
         print("Bootstrap off. To turn on, set integer >= 100")
         Nboot = 0
     return Nboot


# Check if data have been segmented
def check_segmented(segmented):
    if not segmented:
        print("You must call segment_data() first.")
    return segmented


# Check whether Welch's power spectrum estimate has been computed
def check_Welch_power(power):
    computed = True
    try:
        if (power is None):
            raise ValueError
    except ValueError:
        print("Error!")
        print("Welch's power spectrum estimate not computed.")
        print("Use Welch_powspec() first.")
        computed = False
    return computed


# Check whether periodogram has been computed
def check_power(power):
    computed = True
    try:
        if (power is None):
            raise ValueError
    except ValueError:
        print("Error!")
        print("Window, Fourier coefficients, and periodogram not computed.")
        print("Use pow_FT() to compute the above.")
        computed = False
    return computed


# Check oversample keyword
def check_oversample(oversample):
     try:
         good_oversample = (((type(oversample) is int) or \
                             (type(oversample) is float)) \
                             and (oversample > 0))
         if not good_oversample:
              raise ValueError
     except ValueError:
         print("Oversample must be number > 0: returning")      
     return good_oversample


# Check for valid Nyquist frequency
def check_Nyquist(Nyquist):
    try:
        valid_Nyquist = (((type(Nyquist) is int) or \
                           type(Nyquist is float)) \
                          and (Nyquist > 0))
        if not valid_Nyquist:
            raise ValueError 
    except ValueError:
        print("Nyquist must be float > 0 - returning")
    return valid_Nyquist


# Check validity of linear or log plot scale keyword
def check_plot_scale(scale):
    valid_y = ((scale == 'log10') or (scale == 'linear'))
        if not valid_y:     
            print("Invalid setting for y-axis scale. Defaulting to log10.")
            scale = 'log10'
    return scale


# Check valid vlines keyword
def check_vlines(vlines):
    valid_vlines = True
    if not isinstance(vlines, list):
        print('vlines keyword must be list')
        valid_vlines = False
    for vl in vlines:
        if not ((isinstance(vlines, int) or isinstance(vlines, float)):
            print('All entries in vlines list must be int or float')
            valid_vlines = False
    return valid_vlines
    

# Check that coherence has been computed
def check_coherence(coh):
    if coh is None:
        print('No coherence computed. Use Welch_coherence_powspec() first.')
        return False
    else:
        return True


# Check for valid linewidth keyword
def check_linewidth(lw):
    if ((not isinstance(lw, float)) and (not isinstance(lw, int))) \
         or (lw < 0.1):
         print('Invalid linewidth. Defaulting to 0.8.')
         lw = 0.8 
    return lw


# Check for valid tolerance
def check_tol(tol):
    if (not isinstance(tol, float)) or (tol >= 1) or (tol <= 0):
        print('Invalid tolerance. Defaulting to 1e-8.')
        tol = 1e-8
    return tol


# Check for valid 2-d objective plot limits (red noise fit)
def check_oplot_limits(oplot_limits):
    defaults = np.array([[0.001, 1], [0.5, 1.5]])
    if not isinstance(oplot_limits, ndarray):
        print('oplot_limits must be ndarray. Setting defaults.')
        oplot_limits = defaults
    elif oplot_limits.shape != (2,2):
        print('oplot_limits must be ndarray of [[xlow, xhigh], [ylow, yhigh]]'.)
        print('Setting defaults.')
        oplot_limits = defaults
    else:
        pass
    return oplot_limits
