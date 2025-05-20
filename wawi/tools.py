import numpy as np

def print_progress(t, tmax, length=20, sym='=', postfix='', startstop_sym=' '):
    """
    Print a progress bar to the console.

    Parameters
    ----------
    t : int or float
        Current progress value.
    tmax : int or float
        Maximum progress value.
    length : int, optional
        Length of the progress bar (default is 20).
    sym : str, optional
        Symbol used to represent progress (default is '=').
    postfix : str, optional
        String to append at the end of the progress bar (default is '').
    startstop_sym : str, optional
        Symbol to use at the start and end of the progress bar (default is ' ').

    Returns
    -------
    None
        This function prints the progress bar to the console.
    """
    progress = t/tmax
    n_syms = np.floor(progress*length).astype(int)
    string = "\r[%s%-"+ str(length*len(sym)) +"s%s] %3.0f%%" + postfix
    print(string % (startstop_sym,sym*int(n_syms), startstop_sym, progress*100), end='')