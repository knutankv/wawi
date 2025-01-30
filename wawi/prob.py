import numpy as np

def gumbel_log(umax):
    umax_ordered = np.sort(umax)[::-1]
    N_stat = len(umax)
    F = 1-np.arange(1, N_stat+1)/(N_stat+1)
    loglogF = -np.log(-np.log(F))

    return umax_ordered, loglogF