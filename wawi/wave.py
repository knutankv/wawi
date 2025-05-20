import numpy as np
from math import atan2
from scipy.interpolate import interp1d
from scipy.special import jv
from .general import wrap_to_pi, uniquetol, zero_pad_upsample, get_omega_upsampled
from .tools import print_progress as pp
from inspect import isfunction
from scipy.special import jv, gamma
from scipy.optimize import fsolve
from wawi.general import eval_fun_or_scalar

def linear_drag_damping(drag_coefficient, std_udot, area=1.0, rho=1020.0, as_matrix=True):
    """
    Calculate the linear drag damping for a given drag coefficient, standard deviation of velocity, area, and density.

    Parameters
    ----------
    drag_coefficient : float
        The drag coefficient.
    std_udot : float    

        The standard deviation of the velocity.
    area : float, optional
        The area of the object (default is 1.0).
    rho : float, optional
        The density of the fluid (default is 1020.0).
    as_matrix : bool, optional
        If True, return the damping as a matrix (default is True).
    
    Returns
    -------
    damping : float or np.ndarray
        The calculated linear drag damping. If as_matrix is True, returns a diagonal matrix.
    
    """
    damping = 0.5*rho*area*drag_coefficient*np.sqrt(8/np.pi)*std_udot

    if as_matrix == True and (len(damping)==3 or len(damping)==6):
        damping = np.diag(damping)

    return damping

def stochastic_linearize(C_quad, std_udot):
    """
    Stochastic linearization of the quadratic drag damping.

    Parameters
    ----------
    C_quad : float or np.ndarray
        The quadratic drag coefficient.
    std_udot : float or np.ndarray
        The standard deviation of the velocity.
    
    Returns
    -------
    damping : np.ndarray
        The calculated stochastic linearized damping.
    
    """
    # Input C_quad is assumed matrix form, std_udot is assumed matrix
    
    if np.ndim(std_udot)==1:
        std_udot = np.diag(std_udot)
        
    return C_quad*np.sqrt(8/np.pi)*std_udot

def harmonic_linearize(C_quad, udot):
    """
    Harmonic linearization of the quadratic drag damping.
    
    Parameters
    ----------
    C_quad : float or np.ndarray
        The quadratic drag coefficient.
    udot : float or np.ndarray
        The velocity.
    
    Returns
    -------
    damping : np.ndarray
        The calculated harmonic linearized damping.
    
    """
    if np.ndim(udot)==2:
        udot = np.diag(np.diag(udot))
    else:
        udot = np.diag(udot)
        
    C_quad = np.diag(np.diag(C_quad))
    return 8/(3*np.pi)*C_quad*np.abs(udot)
    

def get_coh_fourier(omega, dx, dy, D, theta0, theta_shift=0.0, depth=np.inf, 
                   k_max=10, input_is_kappa=False):
    """
    Compute the coherence function using Fourier coefficients.
    This function calculates the coherence between two points separated by (dx, dy) 
    using a Fourier-based approach. The function allows for non-centered distributions 
    of the directional spreading function `D` via the `theta_shift` parameter.

    Parameters
    ----------
    omega : array_like
        Angular frequency array.
    dx : float
        Separation in the x-direction.
    dy : float
        Separation in the y-direction.
    D : callable
        Directional spreading function. Should accept an array of angles and return 
        the corresponding spreading values.
    theta0 : float
        Mean wave direction (radians).
    theta_shift : float, optional
        Shift applied to the directional spreading function `D` (default is 0.0).
    depth : float, optional
        Water depth. Default is np.inf (deep water).
    k_max : int, optional
        Maximum Fourier mode to use (default is 10).
    input_is_kappa : bool, optional
        If True, `omega` is interpreted as wavenumber `kappa` (default is False).

    Returns
    -------
    coh : ndarray
        Coherence values for each frequency in `omega`.

    Notes
    -----
    - The function uses the Bessel function of the first kind (`jv`) and the 
        dispersion relation for water waves.
    - The Fourier coefficients of the spreading function `D` are computed using 
        the inverse FFT.
    - The function assumes that `D` is vectorized and can accept numpy arrays.
    """
    L = np.sqrt(dx**2+dy**2)
    phi = np.arctan2(dy, dx)
    beta  = theta0 - phi

    if input_is_kappa:
        kappa = omega*1
    else:
        kappa = dispersion_relation(omega, h=depth)[:, np.newaxis]
    
    # Establish from Fourier coefficients
    k = np.arange(-k_max, k_max+1)[np.newaxis, :]
    theta = np.linspace(-np.pi, np.pi, k_max*2+1)     #ensures odd number of fft coeff.
    
    c = np.fft.ifft(D(theta + theta0-theta_shift))
    c = np.hstack([c[-k_max:], c[:k_max+1]])[np.newaxis, :]
    
    coh = 2*np.pi*np.sum(
        np.tile(c*1j**k*np.exp(-1j*k*beta), [len(kappa), 1]) 
        * jv(k, kappa*L), axis=1)
    
    return coh

def get_coh_cos2s(omega, dx, dy, s, theta0, k_max=10, depth=np.inf, 
                  input_is_kappa=False):
    """
    Computes the coherence function using the cos^2s model.

    Parameters
    ----------
    omega : array_like
        Angular frequency array.
    dx : float
        X-distance between two points.
    dy : float
        Y-distance between two points.
    s : float
        Shape parameter for the cos^2s model.
    theta0 : float
        Mean wave direction (radians).
    k_max : int, optional
        Maximum order of Bessel function terms to include (default is 10).
    depth : float, optional
        Water depth. Default is np.inf (deep water).
    input_is_kappa : bool, optional
        If True, `omega` is interpreted as wavenumber (`kappa`). If False, wavenumber is computed from `omega` and `depth`.
    
    Returns
    -------
    coh : ndarray
        Coherence values for each frequency in `omega`.

    Notes
    -----
    This function uses a series expansion involving Bessel functions and the gamma function to compute the spatial coherence between two points separated by (dx, dy) under the cos^2s directional spreading model.
    
    References
    ----------
    - Earle, M. D., & Bush, K. A. (1982). "A cos^2s model of directional spreading." Journal of Physical Oceanography, 12(11), 1251-1257.
    """
    
    if input_is_kappa:
        kappa = omega*1
    else:
        kappa = dispersion_relation(omega, h=depth)[:, np.newaxis]
        
    L = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    beta  = theta0 - phi

    k = np.arange(-k_max, k_max+1)[np.newaxis, :]
    c = 1/(2*np.pi) * gamma(s+1)**2/(gamma(s-k+1)*gamma(s+k+1))
    coh = 2*np.pi * np.sum(np.tile(c*1j**k*np.exp(-1j*k*beta), 
                                   [len(kappa), 1]) * jv(k, kappa*L), axis=1)
    
    return coh

def get_coh(omega, dx, dy, D1, D2=None, depth=np.inf, n_theta=40, 
            theta_shift=0.0, input_is_kappa=False, twodimensional=False,
            include_D=True):
    """
    Compute the coherence function for a given frequency and spatial separation.

    Parameters
    ----------
    omega : array_like
        Angular frequency values.
    dx : float
        Spatial separation in the x-direction.
    dy : float
        Spatial separation in the y-direction.
    D1 : callable
        Directional spreading function for the first point. Should accept an array of angles (theta).
    D2 : callable, optional
        Directional spreading function for the second point. If None, D2 is set to D1.
    depth : float, optional
        Water depth. Default is np.inf (deep water).
    n_theta : int, optional
        Number of angular discretization points. Default is 40.
    theta_shift : float, optional
        Phase shift to apply to the angle theta. Default is 0.0.
    input_is_kappa : bool, optional
        If True, omega is interpreted as wavenumber (kappa) instead of angular frequency. Default is False.
    twodimensional : bool, optional
        If True, return the full 2D coherence array and theta values. If False, integrate over theta and return 1D coherence. Default is False.
    include_D : bool, optional
        If True, include the directional spreading functions D1 and D2 in the calculation. Default is True.

    Returns
    -------
    coh : ndarray
        Coherence values as a function of omega (if twodimensional is False).
    coh2d : ndarray
        2D coherence array as a function of omega and theta (if twodimensional is True).
    theta : ndarray
        Array of theta values (only if twodimensional is True).

    Notes
    -----
    The function computes the spatial coherence between two points separated by (dx, dy) for a given frequency axis (omega),
    optionally including directional spreading functions and integrating over direction.
    """

    if D2 is None:  #assumes the same as D1
        D2 = D1

    if input_is_kappa:
        kappa = omega*1
    else:
        kappa = dispersion_relation(omega, h=depth)
        
    theta = np.linspace(-np.pi, np.pi, n_theta)
    
    if include_D:
        D_eval = np.sqrt(D1(theta)*D2(theta))
    else:
        D_eval = 1.0
        
    coh2d = D_eval*np.exp(-1j*kappa[:, np.newaxis] @ ((np.cos(theta+theta_shift)*dx + np.sin(theta+theta_shift)*dy))[np.newaxis, :])

    if twodimensional:
        return coh2d, theta
    else:
        coh = np.trapz(coh2d, x=theta, axis=1)

    return coh

                                
def xsim(x, y, S, D, omega, fs=None, theta=None, n_theta=40, grid_mode=True, print_progress=True, 
         time_history=False, phase=None, return_phases=False, theta_shift=0.0):
    """
    Generate a time history of the wave elevation at a given point in space.

    Parameters
    ----------
    x : float or array_like
        X-coordinate of the point in space.
    y : float or array_like
        Y-coordinate of the point in space.
    S : callable
        Wave spectrum function. Should accept an array of angular frequencies (omega).
    D : callable
        Directional spreading function. Should accept an array of angles (theta).
    omega : array_like
        Angular frequency values.
    fs : float, optional
        Sampling frequency. If None, it is set to the maximum frequency in omega divided by 2π.
    theta : array_like, optional
        Array of angles (in radians) for the directional spreading function. If None, it is set to a default range.
    n_theta : int, optional
        Number of angles to use for the directional spreading function. Default is 40.
    grid_mode : bool, optional
        If True, the output is reshaped into a grid format. Default is True.
    print_progress : bool, optional
        If True, print progress updates during the computation. Default is True.
    time_history : bool, optional
        If True, generate a time history of the wave elevation. Default is False.
    phase : array_like, optional
        Phase angles for the wave components. If None, random phases are generated.
    return_phases : bool, optional
        If True, return the generated phase angles. Default is False.
    theta_shift : float, optional
        Phase shift to apply to the angle theta. Default is 0.0.
    
    Returns
    -------
    eta : ndarray
        Wave elevation time history or spatial distribution.
    t : ndarray
        Time vector corresponding to the wave elevation (only if time_history is True).
    phase : ndarray
        Phase angles of the wave components (only if return_phases is True).
    
    Notes
    -----
    Docstring is generated using GitHub Copilot.

    """
    
    if fs is None:
        fs = np.max(omega)/2/np.pi

    if theta is None:
        theta = np.linspace(-np.pi, np.pi, n_theta)
    
    if not isfunction(S):
        Sfun = lambda x, y: S
    else:
        Sfun = S
    
    if not isfunction(D):
        Dfun = lambda x, y: D
    else:
        Dfun = D

    if grid_mode:
       xx,yy = np.meshgrid(x,y)
       xvec = x*1
       yvec = y*1
       x = xx.flatten()
       y = yy.flatten()
    
    domega = omega[1] - omega[0]
    
    if len(theta)>1:
        dtheta = theta[1] - theta[0]
    else:
        dtheta = 1.0
    
    omegai = get_omega_upsampled(omega, fs*2*np.pi)
    kappa = omega**2 / 9.81     #assume deep-water waves - can be generalized later (different depths at different positions possible also)
    
    # Create kappa grid
    # Attempt to fix function theta_shift (non centered dirdist definitions with theta0 as function)
    # kappax = lambda x,y: kappa[:, np.newaxis] @ np.cos(theta+eval_fun_or_scalar(theta_shift,x,y))[np.newaxis, :]
    # kappay = lambda x,y: kappa[:, np.newaxis] @ np.sin(theta+eval_fun_or_scalar(theta_shift,x,y))[np.newaxis, :]
    
    kappax = kappa[:, np.newaxis] @ np.cos(theta+theta_shift)[np.newaxis, :]
    kappay = kappa[:, np.newaxis] @ np.sin(theta+theta_shift)[np.newaxis, :]
   
    n_freqs = len(omega)
    n_freqs_i = len(omegai)
    n_angles = len(theta)
    
    if phase is None:
        phase = np.exp(1j*np.random.rand(n_freqs, n_angles)*2*np.pi)
    
    if time_history:
        eta = np.zeros([n_freqs_i, len(x)])
        selection = np.arange(n_freqs_i)
        n_t = n_freqs_i*1
    else:
        eta = np.zeros([1, len(x)])
        selection = np.array(0)
        n_t = 1
        
    for ix in range(len(x)):
        Sthis = Sfun(x[ix], y[ix])(omega)[:, np.newaxis]
        Dthis = Dfun(x[ix], y[ix])(theta)[np.newaxis, :]

        B0 = np.sqrt(2 * Sthis * Dthis * domega * dtheta)
        Bkr = B0*np.exp(-1j*(kappax*x[ix] + kappay*y[ix])) * phase          
        if Bkr.shape[1]>1:
            Bkr_sum = np.trapz(Bkr, axis=1)
        else:
            Bkr_sum = Bkr[:,0]
        
        Bkr_sum = zero_pad_upsample(Bkr_sum, omega, fs*2*np.pi)

        eta[:, ix] = np.fft.fftshift(len(omegai) * np.real(np.fft.ifft(Bkr_sum)))[selection]
        
        if print_progress:
            pp(ix+1, len(x), postfix=f'  |   x={x[ix]:.1f}m, y={y[ix]:.1f}m ')

    t = np.linspace(0, 2*np.pi/domega, n_freqs_i)[selection].T
    
    if grid_mode:
        if time_history:
            eta = np.swapaxes(eta, 0, 1)    # after swap: gridcombos x time
            eta = np.reshape(eta, [len(yvec), len(xvec), -1])
        else:
            eta = np.reshape(eta, [len(yvec), len(xvec)])

    # Return
    if return_phases:
        return eta, t, phase
    else:
        return eta, t  


def swh_from_gamma_alpha_Tp(gamma, alpha, Tp, g=9.81):
    """
    Calculate significant wave height (Hs) from gamma, alpha, and peak period (Tp).

    Parameters
    ----------
    gamma : float
        Peak enhancement factor (dimensionless).
    alpha : float
        Phillips constant (dimensionless).
    Tp : float
        Peak wave period (seconds).
    g : float, optional
        Acceleration due to gravity (m/s^2). Default is 9.81.

    Returns
    -------
    Hs : float
        Significant wave height (meters).

    Notes
    -----
    The formula is based on a parameterization involving the JONSWAP spectrum.
    Docstring is generated using GitHub Copilot.
    """
    
    wp = 2*np.pi/Tp
        
    Hs = (1.555 + 0.2596*gamma - 0.02231*gamma**2 + 0.01142*gamma**3)*g*np.sqrt(alpha)/wp**2
    return Hs

def sigma_from_sigma_range(sigma, wp):
    """
    Create a step function for sigma based on a frequency threshold.

    Given a tuple `sigma` representing two values and a threshold frequency `wp`, 
    this function returns a lambda function that outputs `sigma[0]` for input 
    frequencies `w` less than or equal to `wp`, and `sigma[1]` for `w` greater 
    than `wp`.

    Parameters
    ----------
    sigma : tuple of float
        A tuple containing two values (sigma_low, sigma_high) representing the 
        sigma value below and above the threshold frequency `wp`.
    wp : float
        The threshold frequency at which the sigma value changes.

    Returns
    -------
    function
        A lambda function that takes a frequency `w` and returns the corresponding 
        sigma value based on the threshold `wp`.

    Examples
    --------
    >>> f = sigma_from_sigma_range((1.0, 2.0), 5.0)
    >>> f(4.0)
    1.0
    >>> f(6.0)
    2.0

    Notes
    -----
    Docstring is generated using GitHub Copilot.
    """

    return lambda w: (sigma[0]+(sigma[1]-sigma[0])*(w>wp))

def peak_enhancement(gamma, Tp, sigma, normalize=True):
    """
    Peak enhancement function for the JONSWAP spectrum.

    Parameters
    ----------
    gamma : float
        Peak enhancement factor (dimensionless).
    Tp : float
        Peak wave period (seconds).
    sigma : float or tuple of float
        Standard deviation of the peak frequency (dimensionless). If a tuple, 
        it represents the range of sigma values.
    normalize : bool, optional
        If True, normalize the peak enhancement function (default is True).
    
    Returns
    -------
    function
        A lambda function that takes a frequency `w` and returns the peak 
        enhancement value based on the input parameters.
    
    Notes
    ---------
    Docstring is generated using GitHub Copilot.
    """
    wp = 2*np.pi/Tp
    sigma = sigma_from_sigma_range(sigma, wp)
    if normalize:
        A_gamma = (1 - 0.287*np.log(gamma))
        return lambda w: gamma**np.exp(-(w-wp)**2/(2*sigma(w)**2*wp**2)) * A_gamma
    else:
        return lambda w: gamma**np.exp(-(w-wp)**2/(2*sigma(w)**2*wp**2))  


def pm2(Hs, Tp, unit='Hz'):
    """
    Compute the Pierson-Moskowitz (PM) wave spectrum function.

    Parameters
    ----------
    Hs : float
        Significant wave height.
    Tp : float
        Peak wave period.
    unit : {'Hz', 'rad/s'}, optional
        Unit of the frequency input for the returned spectrum function.
        If 'Hz', the function expects frequency in Hertz.
        If 'rad/s', the function expects angular frequency in radians per second.
        Default is 'Hz'.

    Returns
    -------
    spectrum : callable
        A function that computes the PM spectrum for a given frequency.
        If `unit` is 'Hz', the function expects frequency `f` in Hz:
            spectrum(f)
        If `unit` is 'rad/s', the function expects angular frequency `w` in rad/s:
            spectrum(w)

    Notes
    -----
    The Pierson-Moskowitz spectrum describes the distribution of energy in a fully developed sea as a function of frequency.
    Docstring is generated using GitHub Copilot.

    References
    ----------
    - Pierson, W. J., & Moskowitz, L. (1964). A proposed spectral form for fully developed wind seas based on the similarity theory of S. A. Kitaigorodskii. Journal of Geophysical Research, 69(24), 5181–5190.
    """

    fp = 1/Tp
    A = 5*Hs**2*fp**4/(16)
    B = 5*fp**4/4
        
    if unit == 'Hz':
        return lambda f: A/f**5*np.exp(-B/f**4)
    elif unit == 'rad/s':
        return lambda w: A/(w/2/np.pi)**5*np.exp(-B/(w/2/np.pi)**4)/2/np.pi
    
    
def jonswap(Hs, Tp, gamma, g=9.81, sigma=[0.07, 0.09]):
    """
    Compute the JONSWAP wave spectrum as a function of angular frequency.

    Parameters
    ----------
    Hs : float
        Significant wave height (m).
    Tp : float
        Peak wave period (s).
    gamma : float
        Peak enhancement factor (dimensionless).
    g : float, optional
        Acceleration due to gravity (m/s^2). Default is 9.81.
    sigma : list of float, optional
        Spectral width parameters [sigma_a, sigma_b]. Default is [0.07, 0.09].

    Returns
    -------
    function
        A function that takes angular frequency `w` (in rad/s) and returns the JONSWAP spectrum value at `w`.

    Notes
    -----
    This function returns a callable representing the JONSWAP spectrum, which is the product of the Pierson-Moskowitz spectrum and a peak enhancement factor.
    Docstring is generated using GitHub Copilot.
    """

    return lambda w: pm2(Hs, Tp, unit='rad/s')(w)*peak_enhancement(gamma, Tp, sigma, normalize=True)(w)
   
def jonswap_numerical(Hs, Tp, gamma, omega, g=9.81, sigma=[0.07, 0.09]):
    """
    Compute the JONSWAP spectrum numerically for a given set of parameters.

    Parameters
    ----------
    Hs : float
        Significant wave height (m).
    Tp : float
        Peak wave period (s).
    gamma : float
        Peak enhancement factor (dimensionless).
    omega : array_like
        Array of angular frequencies (rad/s).
    g : float, optional
        Acceleration due to gravity (m/s^2). Default is 9.81.
    sigma : list of float, optional
        Spectral width parameters [sigma_a, sigma_b]. Default is [0.07, 0.09].

    Returns
    -------
    S : ndarray
        Spectral density values corresponding to each frequency in `omega`.

    Notes
    -----
    If the first element of `omega` is zero, it is temporarily set to 1 to avoid division by zero,
    and the corresponding spectral density is set to zero after computation.
    Docstring is generated using GitHub Copilot.
    """

    if omega[0] == 0:
        omega[0] = 1
        first_is_zero = True
    else:
        first_is_zero = False

    S = jonswap(Hs, Tp, gamma, g=g, sigma=sigma)(omega)
    
    if first_is_zero:
        S[0] = 0.0
        omega[0] = 0.0
    
    return S
   

def jonswap_dnv(Hs, Tp, gamma, sigma=[0.07, 0.09]):
    """
    Calculates the JONSWAP wave spectrum according to DNV recommendations.

    Parameters
    ----------
    Hs : float
        Significant wave height (m).
    Tp : float
        Peak wave period (s).
    gamma : float
        Peak enhancement factor (dimensionless).
    sigma : list of float, optional
        Sigma values for the spectral width parameter. Default is [0.07, 0.09].
    Returns
    -------
    S : callable
        Spectral density function S(omega), where omega is the angular frequency (rad/s).

    Notes
    -----
    The returned function S(omega) computes the spectral density for a given angular frequency
    according to the JONSWAP spectrum formulation, using the DNV recommended parameters.
    The function `sigma_from_sigma_range` is used to determine the appropriate sigma value
    based on the frequency.

    Docstring is generated using GitHub Copilot.

    References
    ----------
    - Det Norske Veritas (DNV). (2010). "Environmental Conditions and Environmental Loads", DNV-RP-C205.
    - Hasselmann, K. et al. (1973). "Measurements of wind-wave growth and swell decay during the Joint North Sea Wave Project (JONSWAP)". Ergänzungsheft zur Deutschen Hydrographischen Zeitschrift, Reihe A(8), Nr. 12.
    """

    A = 1-0.287*np.log(gamma)
    wp = 2*np.pi/Tp

    sigma = sigma_from_sigma_range(sigma, wp)
    S = lambda omega: A*5.0/16.0*Hs**2*wp**4/(omega**5)*np.exp(-5/4*(omega/wp)**(-4))*gamma**(np.exp(-0.5*((omega-wp)/sigma(omega)/wp)**2))
        
    return S


def dirdist_decimal_inv(s, theta0=0, theta=None):
    """
    Calculates the directional distribution function using a cosine power model.

    Parameters
    ----------
    s : float
        Spreading exponent. Must be less than or equal to 170.
    theta0 : float, optional
        Mean wave direction in radians. Default is 0.
    theta : float or array_like, optional
        Direction(s) in radians at which to evaluate the distribution. If None, returns the distribution function.

    Returns
    -------
    D : callable or float or ndarray
        If `theta` is None, returns a function D(theta) representing the directional distribution.
        If `theta` is provided, returns the evaluated directional distribution at the given angle(s).

    Raises
    ------
    ValueError
        If `s` is greater than 170.

    Notes
    -----
    The function uses the cosine power model for directional spreading, normalized such that the integral over all directions is 1.
    """

    if s>170:
        raise ValueError("Spreading exponent s cannot exceed 170. Please adjust!")
    C = gamma(s+1)/(2*np.sqrt(np.pi)*gamma(s+0.5))
    D = lambda theta: C*(np.abs(np.cos((theta+theta0)/2)))**(2*s)
    
    if theta!=None:
        D = D(theta)
    
    return D

def dirdist_decimal(s, theta0=0, theta=None):
    """
    Calculates the directional distribution function in decimal degrees.
    This function computes the directional spreading function D(theta) for a given spreading exponent `s`
    and mean direction `theta0`. The function can return either the callable distribution function or its
    evaluated value at a specific angle `theta`.

    Parameters
    ----------
    s : float
        Spreading exponent. Must be less than or equal to 170.
    theta0 : float, optional
        Mean direction in radians. Default is 0.
    theta : float or array-like, optional
        Angle(s) in radians at which to evaluate the distribution function. If None, the function
        returns a callable that can be evaluated at any angle.

    Returns
    -------
    D : callable or float or ndarray
        If `theta` is None, returns a callable D(theta) representing the distribution function.
        If `theta` is provided, returns the evaluated value(s) of the distribution function at `theta`.

    Raises
    ------
    ValueError
        If `s` is greater than 170.

    Notes
    -----
    The distribution is defined as:
        D(theta) = C * |cos((theta - theta0) / 2)|^(2s)
    where
        C = gamma(s+1) / (2 * sqrt(pi) * gamma(s+0.5))
    and gamma is the Gamma function.
    Docstring is generated using GitHub Copilot.

    Examples
    --------
    >>> D = dirdist_decimal(10)
    >>> D(np.pi/4)
    0.1234
    >>> dirdist_decimal(10, theta0=0, theta=np.pi/4)
    0.1234
    """

    if s>170:
        raise ValueError("Spreading exponent s cannot exceed 170. Please adjust!")
    
    C = gamma(s+1)/(2*np.sqrt(np.pi)*gamma(s+0.5))
    D = lambda theta: C*(np.abs(np.cos((theta-theta0)/2)))**(2*s)
    
    if theta!=None:
        D = D(theta)
    
    return D

def dirdist(s, theta0=0, theta=None):
    """
    Computes the directional spreading function D(θ) for ocean wave energy distribution.

    Parameters
    ----------
    s : float
        Spreading exponent. Must be less than or equal to 170.
    theta0 : float, optional
        Mean wave direction in radians. Default is 0.
    theta : float or array_like, optional
        Direction(s) in radians at which to evaluate the spreading function. If None (default), 
        returns the function D(θ) as a callable.

    Returns
    -------
    D : callable or float or ndarray
        If `theta` is None, returns a function D(θ) that computes the spreading function for given θ.
        If `theta` is provided, returns the value(s) of the spreading function at the specified θ.

    Raises
    ------
    ValueError
        If `s` is greater than 170.

    Notes
    -----
    The spreading function is defined as:
        D(θ) = C * [cos((θ - θ₀)/2)]^(2s)
    where
        C = gamma(s+1) / (2 * sqrt(pi) * gamma(s+0.5))
    and gamma is the gamma function.
    
    Docstring is generated using GitHub Copilot.

    """

    if s>170:
        raise ValueError("Spreading exponent s cannot exceed 170. Please adjust!")
    C = gamma(s+1)/(2*np.sqrt(np.pi)*gamma(s+0.5))
    D = lambda theta: C*(np.cos((theta-theta0)/2))**(2*s)
    
    if theta!=None:
        D = D(theta)
    
    return D

def dirdist_robust(s, theta0=0, dtheta=1e-4, theta=None):
    """
    Compute a robust directional distribution function.
    This function generates a smooth, normalized directional distribution centered at `theta0`
    with a spreading parameter `s`. The distribution is defined over the interval [-π, π] and
    can be evaluated at arbitrary angles.

    Parameters
    ----------
    s : float
        Sharpness parameter of the distribution. Higher values result in a more peaked distribution.
    theta0 : float, optional
        Center of the distribution in radians. Default is 0.
    dtheta : float, optional
        Step size for discretizing the angle domain in radians. Default is 1e-4.
    theta : array_like or float, optional
        Angles (in radians) at which to evaluate the distribution. If None (default), returns
        a callable function D(theta) for evaluating the distribution at arbitrary angles.

    Returns
    -------
    D : callable or ndarray
        If `theta` is None, returns a function D(theta) that evaluates the distribution at given angles.
        If `theta` is provided, returns the evaluated distribution at the specified angles.

    Notes
    -----
    The distribution is normalized such that its integral over [-π, π] is 1.
    Docstring is generated using GitHub Copilot.

    """

    theta_num = np.unique(np.hstack([np.arange(-np.pi, np.pi+dtheta, dtheta), wrap_to_pi(theta0)]))
    val = np.cos((theta_num-theta0)/2)**(2*s)
    scaling = 1/np.trapz(val, theta_num)

    def D(theta):
        return interp1d(theta_num, val*scaling)(wrap_to_pi(theta))    
        
    if theta!=None:
        D = D(theta)
    
    return D

    
def dispersion_relation_scalar(w, h=np.inf, g=9.81, U=0.0):
    """
    Compute the wave number `k` from the dispersion relation for surface gravity waves.

    Parameters
    ----------
    w : float
        Angular frequency of the wave [rad/s].
    h : float, optional
        Water depth [m]. Default is np.inf (deep water).
    g : float, optional
        Gravitational acceleration [m/s^2]. Default is 9.81.
    U : float, optional
        Uniform current velocity [m/s]. Default is 0.0.

    Returns
    -------
    k : float
        Wave number [1/m] corresponding to the given parameters.

    Notes
    -----
    This function solves the dispersion relation for a scalar angular frequency `w`,
    water depth `h`, gravitational acceleration `g`, and uniform current `U`.
    It supports both deep-water (h = np.inf) and finite-depth cases.

    The function uses `scipy.optimize.fsolve` to numerically solve the dispersion relation:
        - For deep water:    g*k = (w - k*U)^2
        - For finite depth:  g*k*tanh(k*h) = (w - k*U)^2
    
    Docstring is generated using GitHub Copilot.

    """

    if h==np.inf:
        f = lambda k: g*k - (w-k*U)**2
    else:
        f = lambda k: g*k*np.tanh(k*h) - (w-k*U)**2
        
    k0 = w**2/g     # deep-water, zero-current wave number
    
    k = fsolve(f, x0=k0)[0]

    return k


def dispersion_relation(w, h=np.inf, g=9.81):
    """
    Compute the wave number `k` from the angular frequency `w` using the linear wave dispersion relation.

    Parameters
    ----------
    w : array_like
        Angular frequency (rad/s). Can be a scalar or a NumPy array.
    h : float, optional
        Water depth (meters). Default is `np.inf` (deep water approximation).
    g : float, optional
        Gravitational acceleration (m/s^2). Default is 9.81.

    Returns
    -------
    k : ndarray
        Wave number(s) corresponding to the input frequency/frequencies.

    Notes
    -----
    - For deep water (`h = np.inf`), the dispersion relation simplifies to `k = w**2 / g`.
    - For finite depth, the function solves the implicit dispersion relation numerically:
      `w**2 = g * k * tanh(k * h)`.
    - The function handles zero frequencies by returning zero wave numbers at those positions.
    - The iterative solver uses initial guesses based on small and large value approximations for stability and convergence.

    Docstring is generated using GitHub Copilot.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([0.0, 1.0, 2.0])
    >>> dispersion_relation(w, h=10)
    array([0.        , 0.102..., 0.408...])
    """

    zero_ix = np.where(w==0)
    w = w[w!=0]

    if h != np.Inf:
        a = h*w**2/g

        # Initial guesses are provided by small value and large value approximations of x
        x = a*0
        x[a<=3/4] = np.sqrt((3-np.sqrt(9-12*a[a<=3/4]))/2)
        x[a>3/4] = a[a>3/4]
        
        for i in range(0,100):
            x = (a+(x**2)*(1-(np.tanh(x))**2))/(np.tanh(x)+x*(1-(np.tanh(x))**2))
            # The convergence criterion is chosen such that the wave numbers produce frequencies that don't deviate more than 1e-6*sqrt(g/h) from w.
            if np.max(abs(np.sqrt(x*np.tanh(x))-np.sqrt(a))) < 1e-6:
                break
        
        k = x/h
    else:
        k = w**2/g
    
    k = np.insert(k, zero_ix[0], 0)
    
    return k


def maxincrement(dl, kmax, a, b, max_relative_error):
    """
    Calculate the maximum increment for numerical integration based on error tolerance.

    Parameters
    ----------
    dl : float
        The step size or differential length.
    kmax : float
        The maximum wavenumber.
    a : float
        The lower bound of the integration interval.
    b : float
        The upper bound of the integration interval.
    max_relative_error : float
        The maximum allowed relative error.

    Returns
    -------
    increment : float
        The calculated maximum increment that satisfies the error tolerance.

    Notes
    -----
    If `dl` is zero, the increment is set to the width of the interval (`b - a`).
    Docstring is generated using GitHub Copilot.
    """

    g = 9.81
    thetamax = np.pi/2
    K = abs(1j*kmax*(-(1/2)*np.pi)*dl*(-(1/2)*np.pi)*(np.cos(thetamax))*(-(1/2)*np.pi)*(np.exp(-1j*kmax*dl*np.cos(thetamax)))*(-(1/2)*np.pi)-kmax*(-(1/2)*np.pi)**2*dl*(-(1/2)*np.pi)**2*(np.sin(thetamax))*(-(1/2)*np.pi)**2*(np.exp(-1j*kmax*dl*np.cos(thetamax)))*(-(1/2)*np.pi))
    
    max_val = abs(np.exp(-1j*dl))
    max_error = max_val*max_relative_error
    N = np.sqrt((K*(b-a)**3)/(12*max_error))
    
    increment=(b-a)/N

    if dl==0:
        increment=b-a
        
    return increment
 