from .wave import jonswap
import numpy as np
from scipy.optimize import curve_fit

def fit_jonswap(S, w, sigma=[0.07, 0.09], initial_values=None):
    if initial_values is None:
        initial_values = {}
        
    Hs0 = np.sqrt(np.trapz(S, w))*4

    p0 = {'Hs': Hs0, 'Tp': 1, 'gamma': 1}     #standard values of curve_fit are 1
    p0.update(**initial_values)

    if w[0]==0:
        w = w[1:]
        S = S[1:]

    fun = lambda om, Hs, Tp, gamma: jonswap(Hs, Tp, gamma, sigma=sigma)(om)
    popt,__ = curve_fit(fun, w, S)
    out = dict(Hs=popt[0], Tp=popt[1], gamma=popt[2], p0=[p0['Hs'], p0['Tp'], p0['gamma']])

    return out
    
def onesided_to_twosided(omega, S, axis=-1):
    S2 = 0.5*np.concatenate([np.flip(S, axis=axis), S], axis=axis)
    omega2 = np.hstack([np.flip(-omega), omega])
    
    return omega2, S2


def twosided_to_onesided(omega, S):
    n_samples = len(omega)    
    return omega[:n_samples//2], S[:,:,:n_samples//2]


def ramp_up(Nramp, Ntot):
    t_scale = np.ones(Ntot)
    t_scale[:Nramp] = np.linspace(0, 1, Nramp)
    return t_scale

def ramp_up_t(t, t0):
    Nramp = np.sum(t<t0)
    Ntot = len(t)
    
    return ramp_up(Nramp, Ntot)