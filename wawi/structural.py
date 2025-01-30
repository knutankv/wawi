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
    Construct dry modal matrices.

    Args:
        f: natural frequencies (Hz)
        m: modal masses (kg)
        rayleigh: dictionary with keys ('stiffness' and 'mass') characterizing damping proportional to stiffness and mass
        xi0: constant modal critical damping ratio value (added on top of Rayleigh damping)

    Returns:
        Mdry: mass matrix
        Cdry: damping matrix
        Kdry: stiffness matrix

    Knut Andreas Kvaale, 2017
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
    Construct frequency dependent physical matrix.

    Args:
        pontoon_types: list with one element per pontoon, indicating the pontoon type (referred to the index of Mh and Ch)
        angles: list of angles of pontoons (in radians)
        mat: list of 3D numpy matrices (6 x 6 x Nfreq), with Npontoons entries

    Returns:
        mat_tot: frequency dependent modal matrix (Nmod x Nmod x Nfreq)

    Knut Andreas Kvaale, 2017
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
    if inverse:
        return lambda omega_k: -omega_k**2*M(omega_k) + omega_k*1j*C(omega_k) + K(omega_k)
    else:
        return lambda omega_k: np.linalg.inv(-omega_k**2*M(omega_k) + omega_k*1j*C(omega_k) + K(omega_k))
    
def frf(M, C, K, w, inverse=False):
    """
    Establish frequency response function from M, C and K matrices (all may be frequency dependent).

    Args:
        M: mass matrix (Ndofs x Ndofs x Nfreq or Ndofs x Ndofs)
        C: damping matrix (Ndofs x Ndofs x Nfreq or Ndofs x Ndofs)
        K: stiffness matrix (Ndofs x Ndofs x Nfreq or Ndofs x Ndofs)
        w: frequency axis
    Optional keywords:
        inverse: state if the inverse of the frf should be returned instead of the frf (standard = False)
    Returns:
        H: frequency response function matrix (Ndofs x Ndofs x Nfreq)

    Knut Andreas Kvaale, 2017
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
    Sum frequency response function matrices, by summing the inverses and reinverting.

    Optional args:
        first argument: first FRF (Ndofs x Ndofs x Nfreq)
        second argument: second ...
        etc..
    Returns:
        H: frequency response function matrix (Ndofs x Ndofs x Nfreq)

    Knut Andreas Kvaale, 2017
    """

    Hinv = np.zeros(np.shape(args[0]))

    for Hi in args:
        Hinv = Hinv + np.inv(Hi)

    H = np.inv(Hinv)

    return H


def mat3d_sel(mat, k):  
    
    if len(np.shape(mat)) == 3:
        matsel = mat[:, :, k]
    else:
        matsel = mat

    return matsel


def phys2modal(mat_global, phi_pontoons, inverse=False):
    """
    Transform frequency dependent physical matrix to modal matrix.

    Args:
        mat_global: global system matrix (6*Nponts x 6*Nponts x Nfreq or 6*Nponts x 6*Nponts)
        phi_pontoons: modal transformation matrix (DOFs referring to pontoons only)
        [inverse]: if True, the transform is from modal to physical, i.e., phi * mat * phi^T.  (default = False)
    Returns:
        mat_modal: frequency dependent modal matrix (Nmod x Nmod x Nfreq)

    Knut Andreas Kvaale, 2017
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
    ix_zero = np.where(omega==0)
    
    xi = alpha * (1/(2*omega)) + beta*(omega/2)
    xi[ix_zero] = np.nan
    
    return xi

def rayleigh_damping_fit(xi, omega_1, omega_2):
    rayleigh_coeff = dict()
    rayleigh_coeff['mass'] = 2*xi*(omega_1*omega_2)/(omega_1+omega_2)
    rayleigh_coeff['stiffness'] = 2*xi/(omega_1+omega_2)
   
    return rayleigh_coeff

#%% Simulation
def freqsim_fun(Sqq, H):
    def response(omega):
        return H(omega) @ Sqq(omega) @ H(omega).conj().T

    return response
    

def freqsim(Sqq, H):
    n_freqs = np.shape(Sqq)[2]
    Srr = np.zeros(np.shape(Sqq)).astype('complex')
    
    for k in range(0, n_freqs):
        Srr[:,:,k] = H[:,:,k] @ Sqq[:,:,k] @ H[:,:,k].conj().T

    return Srr


def var_from_modal(omega, S, phi, only_diagonal=True):
    var = phi @ np.real(np.trapz(S, omega, axis=2)) @ phi.T

    if only_diagonal==True:
        var = np.diag(var)
        
    return var

def peakfactor_from_modal(omega, S, phi, T, only_diagonal=True):
    m0 = phi @ np.real(np.trapz(S, omega, axis=2)) @ phi.T
    m2 = phi @ np.real(np.trapz(S*omega**2, omega, axis=2)) @ phi.T
    v0 = 1/(2*np.pi) * np.sqrt(m2/m0)

    kp = peakfactor(T, v0)
    if only_diagonal==True:
        kp = np.diag(kp)
        
    return kp

def expmax_from_modal(omega, S, phi, T, only_diagonal=True):
    m0 = phi @ np.real(np.trapz(S, omega, axis=2)) @ phi.T
    m2 = phi @ np.real(np.trapz(S*omega**2, omega, axis=2)) @ phi.T
    v0 = 1/(2*np.pi) * np.sqrt(m2/m0)
    
    expmax = peakfactor(T, v0) * np.sqrt(m0)
    expmax[m0==0] = 0.0  # avoid nans when 0.0 response

    if only_diagonal==True:
        expmax = np.diag(expmax)
        
    return expmax

