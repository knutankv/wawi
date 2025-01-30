import numpy as np

def print_progress(t, tmax, length=20, sym='=', postfix='', startstop_sym=' '):
    progress = t/tmax
    n_syms = np.floor(progress*length).astype(int)
    string = "\r[%s%-"+ str(length*len(sym)) +"s%s] %3.0f%%" + postfix
    print(string % (startstop_sym,sym*int(n_syms), startstop_sym, progress*100), end='')