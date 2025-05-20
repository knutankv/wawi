import numpy as np

def gumbel_log(umax):
    """
    Compute the Gumbel reduced variate for a given array of maxima.

    Parameters
    ----------
    umax : array_like
        Array of maxima values.

    Returns
    -------
    umax_ordered : ndarray
        The input maxima sorted in descending order.
    loglogF : ndarray
        The Gumbel reduced variate corresponding to each sorted maxima.

    Examples
    --------
    >>> import numpy as np
    >>> umax = np.array([2.3, 3.1, 1.8, 2.9])
    >>> umax_ordered, loglogF = gumbel_log(umax)

    Notes
    -------
    This function sorts the input array of maxima in descending order and computes the
    Gumbel reduced variate (log-log of the empirical cumulative distribution function)
    for each value.
    """
    
    umax_ordered = np.sort(umax)[::-1]
    N_stat = len(umax)
    F = 1-np.arange(1, N_stat+1)/(N_stat+1)
    loglogF = -np.log(-np.log(F))

    return umax_ordered, loglogF