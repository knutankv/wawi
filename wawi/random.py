import numpy as np

def zero_crossing_period(S, omega):
    return 2*np.pi*np.sqrt(np.trapz(S, x=omega)/np.trapz(omega**2*S, x=omega))

def stoch_mom(S, omega, n=0):
    return np.trapz(S*omega**n, x=omega)

def m0(S, omega):
    return stoch_mom(S, omega, n=0)

def m2(S, omega):
    return stoch_mom(S, omega, n=2)

def v0_from_spectrum(S, omega):
    return 1/(2*np.pi) * np.sqrt(m2(S, omega)/m0(S, omega))

def v0(m0,m2):
    return 1/(2*np.pi) * np.sqrt(m2/m0)

def peakfactor(T, v0):
    c = np.sqrt(2*np.log(v0*T))
    kp = c + np.euler_gamma/c
    return kp

def expmax(T, v0, std):
    return peakfactor(T,v0)*std

def expmax_from_spectrum(S, omega, T):
    m0_val = m0(S, omega)
    std = np.sqrt(m0_val)

    v0_val = v0(m0_val, m2(S, omega))

    return expmax(T, v0_val, std)

def peakfactor_from_spectrum(S, omega, T):
    return peakfactor(T, v0(S, omega))