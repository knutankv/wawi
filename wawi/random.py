import numpy as np
def zero_crossing_period(S, omega):
    """
    Estimate the zero-crossing period from a spectrum.

    Parameters
    ----------
    S : array_like
        Spectral density values.
    omega : array_like
        Angular frequency values.

    Returns
    -------
    float
        Estimated zero-crossing period.
    
    Notes
    -----
    Docstring is generated using Github Copilot.
    """
    return 2*np.pi*np.sqrt(np.trapz(S, x=omega)/np.trapz(omega**2*S, x=omega))

def stoch_mom(S, omega, n=0):
    """
    Compute the n-th spectral moment.

    Parameters
    ----------
    S : array_like
        Spectral density values.
    omega : array_like
        Angular frequency values.
    n : int, optional
        Order of the moment (default is 0).

    Returns
    -------
    float
        n-th spectral moment.
    
    Notes
    -----
    Docstring is generated using Github Copilot.
    """
    return np.trapz(S*omega**n, x=omega)

def m0(S, omega):
    """
    Compute the zeroth spectral moment.

    Parameters
    ----------
    S : array_like
        Spectral density values.
    omega : array_like
        Angular frequency values.

    Returns
    -------
    float
        Zeroth spectral moment.
    
    Notes
    -----
    Docstring is generated using Github Copilot.
    """
    return stoch_mom(S, omega, n=0)

def m2(S, omega):
    """
    Compute the second spectral moment.

    Parameters
    ----------
    S : array_like
        Spectral density values.
    omega : array_like
        Angular frequency values.

    Returns
    -------
    float
        Second spectral moment.
    
    Notes
    -----
    Docstring is generated using Github Copilot.
    """
    return stoch_mom(S, omega, n=2)

def v0_from_spectrum(S, omega):
    """
    Estimate zero-crossing frequency from a spectrum.

    Parameters
    ----------
    S : array_like
        Spectral density values.
    omega : array_like
        Angular frequency values.

    Returns
    -------
    float
        Estimated zero-crossing frequency.
    
    Notes
    -----
    Docstring is generated using Github Copilot.
    """
    return 1/(2*np.pi) * np.sqrt(m2(S, omega)/m0(S, omega))

def v0(m0, m2):
    """
    Calculate zero-crossing frequency from spectral moments.

    Parameters
    ----------
    m0 : float
        Zeroth spectral moment.
    m2 : float
        Second spectral moment.

    Returns
    -------
    float
        Zero-crossing frequency.

    Notes
    -----
    Docstring is generated using Github Copilot.
    """
    return 1/(2*np.pi) * np.sqrt(m2/m0)

def peakfactor(T, v0):
    """
    Calculate the peak factor for a given time period and zero-crossing frequency.

    Parameters
    ----------
    T : float
        The time period over which the peak factor is calculated.
    v0 : float
        The frequency parameter.

    Returns
    -------
    kp : float
        The calculated peak factor.

    Notes
    -----
    The peak factor is computed using the formula:
        c = sqrt(2 * log(v0 * T))
        kp = c + Euler-Mascheroni constant / c
    where `np.euler_gamma` is the Euler-Mascheroni constant.

    Docstring is generated using Github Copilot.

    """

    c = np.sqrt(2*np.log(v0*T))
    kp = c + np.euler_gamma/c
    return kp

def expmax(T, v0, std):
    """
    Calculate the expected maximum value based on temperature, initial value, and standard deviation.

    Parameters
    ----------
    T : float or array-like
        Temperature or an array of temperature values.
    v0 : float
        Initial value parameter.
    std : float
        Standard deviation.

    Returns
    -------
    float or array-like
        The expected maximum value computed as peakfactor(T, v0) multiplied by std.

    Notes
    -----
    This function relies on an external function `peakfactor` which should be defined elsewhere.
    Docstring is generated using Github Copilot.

    Examples
    --------
    >>> expmax(300, 1.0, 0.5)
    1.23  # Example output, depends on peakfactor implementation
    """

    return peakfactor(T,v0)*std

def expmax_from_spectrum(S, omega, T):
    """
    Estimate the expected maximum value from a given spectrum.

    Parameters
    ----------
    S : array_like
        Spectral density values.
    omega : array_like
        Angular frequency values corresponding to the spectrum.
    T : float
        Duration or time interval over which the expected maximum is calculated.

    Returns
    -------
    float
        The expected maximum value estimated from the spectrum.

    Notes
    -----
    This function relies on auxiliary functions `m0`, `m2`, `v0`, and `expmax` to compute
    spectral moments and the expected maximum.

    Docstring is generated using Github Copilot.
    """
    m0_val = m0(S, omega)
    std = np.sqrt(m0_val)

    v0_val = v0(m0_val, m2(S, omega))

    return expmax(T, v0_val, std)

def peakfactor_from_spectrum(S, omega, T):
    """
    Calculate the peak factor from a given spectrum.

    Parameters
    ----------
    S : array_like
        Spectral density values.
    omega : array_like
        Angular frequency values corresponding to the spectrum.
    T : float
        Duration or period over which the peak factor is calculated.

    Returns
    -------
    float
        The calculated peak factor based on the input spectrum and period.

    Notes
    -----
    This function uses the `peakfactor` and `v0` helper functions to compute the result.
    Docstring is generated using Github Copilot.
    """

    return peakfactor(T, v0(S, omega))