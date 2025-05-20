import sys
import numpy as np
from . import wind
from .general import rodrot, blkdiag, correct_matrix_size, transform_3dmat
from .tools import print_progress as pp
from .random import peakfactor

from scipy.interpolate import interp1d

#%% General
def dry_modalmats(f, m, rayleigh={'stiffness':0, 'mass':0}, xi0=0):
    """
    Construct dry modal mass, damping, and stiffness matrices.

    Parameters
    ----------
    f : array_like
        Natural frequencies (Hz).
    m : array_like
        Modal masses (kg).
    rayleigh : dict, optional
        Dictionary with keys 'stiffness' and 'mass' for Rayleigh damping coefficients.
    xi0 : float, optional
        Constant modal critical damping ratio value (added on top of Rayleigh damping).

    Returns
    -------
    Mdry : ndarray
        Modal mass matrix.
    Cdry : ndarray
        Modal damping matrix.
    Kdry : ndarray
        Modal stiffness matrix.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    w = (f*2*np.pi)
    k = np.multiply(w**2, m)
    Kdry = np.diag(k)
    Mdry = np.diag(m)
    
    c = k*rayleigh['stiffness'] + m*rayleigh['mass'] + xi0*2*np.sqrt(k*m)
    Cdry = np.diag(c)  

    return Mdry, Cdry, Kdry


def wet_physmat(pontoon_types, angles, mat):
    """
    Construct frequency dependent physical matrix for pontoons.

    Parameters
    ----------
    pontoon_types : list of int
        List with one element per pontoon, indicating the pontoon type (index of Mh and Ch).
    angles : list of float
        List of angles of pontoons (in radians).
    mat : list of ndarray
        List of 3D numpy matrices (6 x 6 x Nfreq), with Npontoons entries.

    Returns
    -------
    mat_tot : ndarray
        Frequency dependent modal matrix (Nmod x Nmod x Nfreq).

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    Nponts = len(angles)

    if len(np.shape(mat[0])) == 3:
        Nfreqs = np.shape(mat[0])[2]
    else:
        Nfreqs = 1
        mat = np.reshape(mat, [len(mat), 6, 6, 1])

    mat_global = np.empty([6*Nponts, 6*Nponts, Nfreqs], dtype=mat[0].dtype)

    T = np.zeros([6, 6])

    for pont in range(0, Nponts):
        pt = pontoon_types[pont]
        T0 = rodrot(angles[pont])
        T[0:3, 0:3], T[3:6, 3:6] = T0, T0

        for k in range(0, Nfreqs):    # Loop through discrete freqs
            mat_global[pont*6:pont*6+6, pont*6:pont*6+6, k] = np.dot(np.dot(T.T, mat[pt][:, :, k]), T)

    if Nfreqs == 1:
        mat_global = mat_global[:, :, 0]

    return mat_global

def frf_fun(M, C, K, inverse=False):
    """
    Return a function that computes the frequency response function (FRF) or its inverse.

    Parameters
    ----------
    M : callable
        Function returning mass matrix for a given frequency.
    C : callable
        Function returning damping matrix for a given frequency.
    K : callable
        Function returning stiffness matrix for a given frequency.
    inverse : bool, optional
        If True, return the inverse FRF (default is False).

    Returns
    -------
    function
        Function that computes the FRF or its inverse for a given frequency.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    if inverse:
        return lambda omega_k: -omega_k**2*M(omega_k) + omega_k*1j*C(omega_k) + K(omega_k)
    else:
        return lambda omega_k: np.linalg.inv(-omega_k**2*M(omega_k) + omega_k*1j*C(omega_k) + K(omega_k))
    
def frf(M, C, K, w, inverse=False):
    """
    Establish frequency response function from M, C and K matrices (all may be frequency dependent).

    Parameters
    ----------
    M : ndarray
        Mass matrix (Ndofs x Ndofs x Nfreq or Ndofs x Ndofs).
    C : ndarray
        Damping matrix (Ndofs x Ndofs x Nfreq or Ndofs x Ndofs).
    K : ndarray
        Stiffness matrix (Ndofs x Ndofs x Nfreq or Ndofs x Ndofs).
    w : array_like
        Frequency axis.
    inverse : bool, optional
        If True, return the inverse FRF (default is False).

    Returns
    -------
    H : ndarray
        Frequency response function matrix (Ndofs x Ndofs x Nfreq).

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    n_dofs = np.shape(K)[0]
    n_freqs = len(w)

    if len(np.shape(M)) == 2:
        M = np.tile(np.reshape(M, [n_dofs, n_dofs, 1]), [1, 1, n_freqs])
    if len(np.shape(C)) == 2:
        C = np.tile(np.reshape(C, [n_dofs, n_dofs, 1]), [1, 1, n_freqs])
    if len(np.shape(K)) == 2:
        K = np.tile(np.reshape(K, [n_dofs, n_dofs, 1]), [1, 1, n_freqs])

    if inverse is True:
        wmat = np.tile(w, [n_dofs, n_dofs, 1])
        H = -wmat**2*M + wmat*1j*C + K
    else:
        H = np.empty([n_dofs, n_dofs, n_freqs], dtype=complex)    # Memory allocation

        for k, wk in enumerate(w):
            Mk = mat3d_sel(M, k)
            Ck = mat3d_sel(C, k)
            Kk = mat3d_sel(K, k)
            H[:, :, k] = np.linalg.inv(-wk**2*Mk + 1j*wk*Ck + Kk)

    return H


def sum_frfs(*args):
    """
    Sum frequency response function matrices by summing the inverses and reinverting.

    Parameters
    ----------
    *args : ndarray
        Frequency response function matrices (Ndofs x Ndofs x Nfreq).

    Returns
    -------
    H : ndarray
        Frequency response function matrix (Ndofs x Ndofs x Nfreq).

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    Hinv = np.zeros(np.shape(args[0]))

    for Hi in args:
        Hinv = Hinv + np.inv(Hi)

    H = np.inv(Hinv)

    return H


def mat3d_sel(mat, k):  
    """
    Select the k-th slice from a 3D matrix, or return the matrix if 2D.

    Parameters
    ----------
    mat : ndarray
        2D or 3D matrix.
    k : int
        Index of the slice to select.

    Returns
    -------
    matsel : ndarray
        Selected matrix slice.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    if len(np.shape(mat)) == 3:
        matsel = mat[:, :, k]
    else:
        matsel = mat

    return matsel


def phys2modal(mat_global, phi_pontoons, inverse=False):
    """
    Transform frequency dependent physical matrix to modal matrix or vice versa.

    Parameters
    ----------
    mat_global : ndarray
        Global system matrix (6*Nponts x 6*Nponts x Nfreq or 6*Nponts x 6*Nponts).
    phi_pontoons : ndarray
        Modal transformation matrix (DOFs referring to pontoons only).
    inverse : bool, optional
        If True, transform from modal to physical (default is False).

    Returns
    -------
    mat_modal : ndarray
        Frequency dependent modal matrix (Nmod x Nmod x Nfreq).

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    if inverse is True:
        phi_pontoons = np.transpose(phi_pontoons)   # Transpose phi matrix if inverse transformation

    mat_shape = np.shape(mat_global)
    Nmodes = np.shape(phi_pontoons)[1]

    if len(mat_shape) == 3:     # 3D matrix (frequency dependent)
        mat_modal = np.empty([Nmodes, Nmodes,  mat_shape[2]])

        for k in range(0, mat_shape[2]):
            mat_modal[:, :, k] = np.dot(np.dot(phi_pontoons.T, mat_global[:, :, k]), phi_pontoons)
    else:                       # 2D matrix (no frequency dependency)
        mat_modal = np.dot(np.dot(phi_pontoons.T, mat_global), phi_pontoons)

    return mat_modal

#%% Assembly
def assemble_hydro_matrices_full(pontoons, omega):
    """
    Assemble full hydrodynamic mass, damping, and stiffness matrices for all pontoons.

    Parameters
    ----------
    pontoons : list
        List of pontoon objects.
    omega : array_like
        Frequency axis.

    Returns
    -------
    Mh : ndarray
        Hydrodynamic mass matrix.
    Ch : ndarray
        Hydrodynamic damping matrix.
    Kh : ndarray
        Hydrodynamic stiffness matrix.
    node_labels : list
        List of node labels for each pontoon.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    node_labels = [pontoon.node for pontoon in pontoons]
    n_dofs = len(pontoons)*6
    n_freqs = len(omega)

    Mh = np.zeros([n_dofs, n_dofs, n_freqs])
    Ch = np.zeros([n_dofs, n_dofs, n_freqs])
    Kh = np.zeros([n_dofs, n_dofs, n_freqs])
    
    for ix, pontoon in enumerate(pontoons):
        if max(omega)>max(pontoon.pontoon_type.original_omega) or min(omega)<min(pontoon.pontoon_type.original_omega):
            print(f'WARNING: frequency outside range for {pontoon.label} --> extrapolated')
        
        for k, omega_k in enumerate(omega):
            Mh[ix*6:ix*6+6, ix*6:ix*6+6, k] = pontoon.get_M(omega_k)
            Ch[ix*6:ix*6+6, ix*6:ix*6+6, k] = pontoon.get_C(omega_k)
            Kh[ix*6:ix*6+6, ix*6:ix*6+6, k] = pontoon.get_K(omega_k)

    return Mh, Ch, Kh, node_labels


#%% General, model set up
def rayleigh(alpha, beta, omega):
    """
    Compute Rayleigh damping ratio for a given frequency axis.

    Parameters
    ----------
    alpha : float
        Mass proportional damping coefficient.
    beta : float
        Stiffness proportional damping coefficient.
    omega : array_like
        Frequency axis.

    Returns
    -------
    xi : ndarray
        Damping ratio for each frequency.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    ix_zero = np.where(omega==0)
    
    xi = alpha * (1/(2*omega)) + beta*(omega/2)
    xi[ix_zero] = np.nan
    
    return xi

def rayleigh_damping_fit(xi, omega_1, omega_2):
    """
    Fit Rayleigh damping coefficients for given target damping and frequencies.

    Parameters
    ----------
    xi : float
        Target damping ratio.
    omega_1 : float
        First frequency (rad/s).
    omega_2 : float
        Second frequency (rad/s).

    Returns
    -------
    rayleigh_coeff : dict
        Dictionary with 'mass' and 'stiffness' Rayleigh coefficients.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    rayleigh_coeff = dict()
    rayleigh_coeff['mass'] = 2*xi*(omega_1*omega_2)/(omega_1+omega_2)
    rayleigh_coeff['stiffness'] = 2*xi/(omega_1+omega_2)
   
    return rayleigh_coeff

#%% Simulation
def freqsim_fun(Sqq, H):
    """
    Return a function that computes the response spectral density matrix.

    Parameters
    ----------
    Sqq : callable
        Function returning input spectral density matrix for a given frequency.
    H : callable
        Function returning frequency response matrix for a given frequency.

    Returns
    -------
    function
        Function that computes the response spectral density matrix for a given frequency.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    def response(omega):
        return H(omega) @ Sqq(omega) @ H(omega).conj().T

    return response
    

def freqsim(Sqq, H):
    """
    Compute the response spectral density matrix for all frequencies.

    Parameters
    ----------
    Sqq : ndarray
        Input spectral density matrix (Ndofs x Ndofs x Nfreq).
    H : ndarray
        Frequency response matrix (Ndofs x Ndofs x Nfreq).

    Returns
    -------
    Srr : ndarray
        Response spectral density matrix (Ndofs x Ndofs x Nfreq).

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    n_freqs = np.shape(Sqq)[2]
    Srr = np.zeros(np.shape(Sqq)).astype('complex')
    
    for k in range(0, n_freqs):
        Srr[:,:,k] = H[:,:,k] @ Sqq[:,:,k] @ H[:,:,k].conj().T

    return Srr


def var_from_modal(omega, S, phi, only_diagonal=True):
    """
    Compute variance from modal spectral density.

    Parameters
    ----------
    omega : array_like
        Frequency axis.
    S : ndarray
        Modal spectral density matrix (Nmod x Nmod x Nfreq).
    phi : ndarray
        Modal transformation matrix.
    only_diagonal : bool, optional
        If True, return only the diagonal elements (default is True).

    Returns
    -------
    var : ndarray
        Variance matrix or its diagonal.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    var = phi @ np.real(np.trapz(S, omega, axis=2)) @ phi.T

    if only_diagonal==True:
        var = np.diag(var)
        
    return var

def peakfactor_from_modal(omega, S, phi, T, only_diagonal=True):
    """
    Compute peak factor from modal spectral density.

    Parameters
    ----------
    omega : array_like
        Frequency axis.
    S : ndarray
        Modal spectral density matrix (Nmod x Nmod x Nfreq).
    phi : ndarray
        Modal transformation matrix.
    T : float
        Duration for peak factor calculation.
    only_diagonal : bool, optional
        If True, return only the diagonal elements (default is True).

    Returns
    -------
    kp : ndarray
        Peak factor matrix or its diagonal.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    m0 = phi @ np.real(np.trapz(S, omega, axis=2)) @ phi.T
    m2 = phi @ np.real(np.trapz(S*omega**2, omega, axis=2)) @ phi.T
    v0 = 1/(2*np.pi) * np.sqrt(m2/m0)

    kp = peakfactor(T, v0)
    if only_diagonal==True:
        kp = np.diag(kp)
        
    return kp

def expmax_from_modal(omega, S, phi, T, only_diagonal=True):
    """
    Compute expected maximum from modal spectral density.

    Parameters
    ----------
    omega : array_like
        Frequency axis.
    S : ndarray
        Modal spectral density matrix (Nmod x Nmod x Nfreq).
    phi : ndarray
        Modal transformation matrix.
    T : float
        Duration for expected maximum calculation.
    only_diagonal : bool, optional
        If True, return only the diagonal elements (default is True).

    Returns
    -------
    expmax : ndarray
        Expected maximum matrix or its diagonal.

    Notes
    -----
    Docstring is generated or modified using GitHub Copilot.
    """
    m0 = phi @ np.real(np.trapz(S, omega, axis=2)) @ phi.T
    m2 = phi @ np.real(np.trapz(S*omega**2, omega, axis=2)) @ phi.T
    v0 = 1/(2*np.pi) * np.sqrt(m2/m0)
    
    expmax = peakfactor(T, v0) * np.sqrt(m0)
    expmax[m0==0] = 0.0  # avoid nans when 0.0 response

    if only_diagonal==True:
        expmax = np.diag(expmax)
        
    return expmax

