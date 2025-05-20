import numpy as np
from scipy.optimize import fsolve

def get_combinations(n_p, n_q=None):
    '''
    Establish all combinations of n indices.

    Parameters
    ----------
    n_p : int
        Number of indices.
    n_q : None, optional
        If integer is given, this is used - otherwise standard value None makes n_q = n_p.

    Returns
    -------
    combos : list of list of int
        List of all unique index combinations (2-by-n_comb).

    Notes
    -----
    Docstring generated using GitHub Copilot.
    '''
    p = np.arange(n_p)
    if n_q is None:
        q = p
        
    combos = []
    for pi in p:
        for qi in q:
            combo = list(np.sort([pi, qi]))
            if (pi != qi) and (combo not in combos):
                combos.append(combo)
                
    return combos

def wave_number_id(Sx, x, k0, Sref=None):
    '''
    Experimental function to establish wave number nonparametrically from cross spectral density.

    Parameters
    ----------
    Sx : ndarray
        Cross spectral density matrix, shape (n_channels, n_channels, n_freq).
    x : ndarray
        Array of spatial positions, shape (n_channels,).
    k0 : float
        Initial guess for wave number (not used in current implementation).
    Sref : None, optional
        Reference spectral density (not used in current implementation).

    Returns
    -------
    k : ndarray
        Estimated wave numbers for each frequency, shape (n_freq,).

    Notes
    -----
    Docstring generated using GitHub Copilot.
    '''
    k = np.zeros(Sx.shape[2])
    n_freq = Sx.shape[2]
    combos = get_combinations(len(x))
             
    b = np.zeros(len(combos)).astype(complex)

    for ix, combo in enumerate(combos):
        dx = x[combo[1]] - x[combo[0]]
        b[ix] = -dx*1j       
        
    for n in range(n_freq):
        lnGamma = np.zeros(len(combos)).astype(complex)
        k_all = [None]*len(combos)
        for ix,combo in enumerate(combos):
            dof1,dof2 = combo
            S = np.sqrt(Sx[dof1,dof1, n]*Sx[dof2,dof2, n])
            
            if S==0:
                lnGamma[ix] = np.nan
            else:
                lnGamma[ix] = np.log(Sx[dof1, dof2, n]/S)
            k_all[ix] = 1j/dx * lnGamma[ix]
            
        k[n] = np.mean(k_all)

        

    return k