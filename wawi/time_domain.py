import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import block_diag, cholesky

def band_truncate_3d(M3d, n):
    for k in range(M3d.shape[2]):
        M3d[:,:,k] = band_truncate(M3d[:,:,k], n=n)
    
    return M3d


def band_truncate(M, n=1):
    if n is None:
        n = M.shape[0]
        
    Mt = M*0

    for ix in range(M.shape[0]):
        n1 = ix-n if (ix-n)>=0 else 0
        n2 = ix+n+1 if (ix+n)<=M.shape[0] else M.shape[0]

        Mt[ix, n1:n2] = M[ix, n1:n2]
    
    return Mt

def fft_time(omega, t0=0, n_fft=None):
    if n_fft is None:
        n_fft = len(omega)
    domega = omega[1] - omega[0] 
    t = np.linspace(t0, np.pi*2/domega, n_fft)

    return t   

def spectrum_to_process(S, reg_factor=None, zero_limit=None):
    B = S*0     # copy S to chol factor
    norm_pre = np.linalg.norm(S)
    if reg_factor is not None:
        for k in range(S.shape[2]):
            S[:,:,k] = S[:,:,k] + np.linalg.norm(S[:,:,k]) * np.eye(S.shape[0]) * reg_factor
        
    if (np.linalg.norm(S)-norm_pre)/norm_pre>0.1:
        raise ValueError('Regularization causes significant increase in norm of S (10%). Please adjust.')
    
    # Cholesky decomposition
    for k in range(B.shape[2]):
        if zero_limit is None or np.max(np.max(np.abs(S[:,:,k])))>zero_limit:
            B[:,:,k] = cholesky(S[:, :, k], lower=True)
    return B
        

def simulate_mdof(S, omega, fs=None, tmax=None, reg_factor=None, zero_limit=1e-12, 
                  phase_angles=None, return_phase_angles=False, interpolation_kind='linear', 
                  component_scaling=None, print_status=False):
    
    if omega[0] !=0:
        omega = np.insert(omega, 0, 0)
        S = np.dstack([0*S[:,:,0], S])
    
    if component_scaling is not None:
        c = np.tile(np.array(component_scaling), int(np.round(S.shape[0]/len(component_scaling))))
        scale_mat = c[np.newaxis,:].T @ c[np.newaxis, :]
        for k in range(S.shape[2]):
            S[:,:,k] = scale_mat * S[:,:,k]

    B = spectrum_to_process(S, reg_factor=reg_factor, zero_limit=zero_limit)  
    B = interp1d(omega, B, axis=2, fill_value=0, kind=interpolation_kind, bounds_error=False)

    n_dofs = S.shape[0]

    if fs is None:
        omega_max = omega[-1]
    else:
        omega_max = fs*(np.pi*2)
        if omega_max>omega[-1] and print_status:
            print('fs too high (larger than largest omega value in input) - zero padding automatically enforced')
        
    if tmax is None:
        domega = omega[1]-omega[0]
    else:
        domega = np.pi*2/tmax

    omega = np.arange(omega[0], omega_max+domega, domega)
    
    # Summation
    p = np.zeros([n_dofs, len(omega)])    
    Bi = B(omega)

    if phase_angles is None:
        alpha = 2*np.pi * np.random.rand(n_dofs, len(omega))
    else:
        alpha = phase_angles*1
        
    for j in range(n_dofs):   
        for m in range(j+1):
            p[j, :] = p[j, :] + (np.sqrt(2*domega) * 
                     np.real(np.fft.fft(Bi[j, m, :] * np.exp(1j*alpha[m, :]))))
        
    if component_scaling is not None:
        p = p/np.tile(c[:, np.newaxis], reps=p.shape[1])
        
    t = fft_time(omega)
    
    if return_phase_angles:
        return p, t, alpha
    else:
        return p, t


def simulate_mdof_direct(S, omega, reg_factor=None):
    B = spectrum_to_process(S, reg_factor)
    
    # Summation
    domega = omega[1] - omega[0]
    
    t = fft_time(omega)
    p = np.zeros([B.shape[0], len(t)])
    for j in range(B.shape[0]): 
        
        # Eq. 19 computed below for each j
        for m in range(j+1):
            for k, omega_k in enumerate(omega):
                phi = 2*np.pi * np.random.rand()
                p[j, :] = p[j, :] + np.sqrt(2*domega) * np.real(B[j, m, k] * np.exp(1j*(omega_k*t + phi)))
    
    return p, t
    