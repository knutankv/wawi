import numpy as np
from .tools import print_progress as pp
from scipy.interpolate import interp1d

def maxreal(phi):
    """
    Rotate complex vectors (stacked column-wise) such that the absolute values of the real parts are maximized.

    Arguments
    ---------------------------
    phi : double
        complex-valued modal transformation matrix (column-wise stacked mode shapes)

    Returns
    ---------------------------
    phi_max_real : boolean
        complex-valued modal transformation matrix, with vectors rotated to have maximum real parts
    """   

    angles = np.expand_dims(np.arange(0,np.pi, 0.01), axis=0)
    phi_max_real = np.zeros(np.shape(phi)).astype('complex')
    for mode in range(0,np.shape(phi)[1]):
        rot_mode = np.dot(np.expand_dims(phi[:, mode], axis=1), np.exp(angles*1j))
        max_angle_ix = np.argmax(np.sum(np.real(rot_mode)**2,axis=0), axis=0)

        phi_max_real[:, mode] = phi[:, mode] * np.exp(angles[0, max_angle_ix]*1j)*np.sign(sum(np.real(phi[:, mode])))

    return phi_max_real

def get_mode_sort(lambd, order_fun_dict=None, return_as_dict=True, remove_conjugates=['dynamic'], sort=['dynamic']):
    """
    Sorts and classifies eigenvalues (modes) according to specified criteria.

    Parameters
    ----------
    lambd : array_like
        Array of eigenvalues to be classified and sorted.
    order_fun_dict : dict or list, optional
        Dictionary mapping mode types (str) to functions that return boolean masks for selecting eigenvalues.
        If a list is provided, only the specified keys from the standard dictionary are used.
        If None (default), a standard set of mode types is used: 'dynamic', 'undamped', 'unstable', 'overdamped'.
    return_as_dict : bool, optional
        If True (default), returns a dictionary mapping mode types to indices of eigenvalues.
        If False, returns a concatenated array of indices.
    remove_conjugates : list or bool, optional
        List of mode types for which to remove every second index (to remove conjugate pairs).
        If True, applies to all mode types. If False or empty list, does not remove conjugates.
        Default is ['dynamic'].
    sort : list or bool, optional
        List of mode types for which to sort indices by the absolute value of eigenvalues.
        If True, applies to all mode types. If False or empty list, does not sort.
        Default is ['dynamic'].

    Returns
    -------
    ix_dict : dict
        Dictionary mapping mode types to arrays of indices of eigenvalues, if `return_as_dict` is True.
    indices : ndarray
        Concatenated array of indices, if `return_as_dict` is False.

    Notes
    -----
    This function is useful for modal analysis, where eigenvalues are classified into categories such as
    'dynamic', 'undamped', 'unstable', and 'overdamped' based on their real and imaginary parts. 

    Docstring is generated using GitHub Copilot.
    """

    order_fun_dict_std = {
        'dynamic': lambda l: np.logical_and(np.imag(l)!=0, np.real(l)<0),
        'undamped':lambda l: np.real(l)==0,
        'unstable': lambda l: np.logical_and(np.imag(l)!=0, np.real(l)>0),
        'overdamped': lambda l: np.logical_and(np.imag(l)==0, np.real(l)<0)
        }
        
    if order_fun_dict is None:
        order_fun_dict = order_fun_dict_std  
    elif type(order_fun_dict) is list:  #specify certain terms from standard dict defined
        order_fun_dict = {key:order_fun_dict_std[key] for key in order_fun_dict}
        
    ix_dict = dict()    
    
    for key, fun in order_fun_dict.items():     
        ix = np.where(fun(lambd))[0]
        
        if key in sort:
            ix2 = np.argsort(np.abs(lambd[ix]))
            ix = ix[ix2]
            
        if remove_conjugates:
            ix = ix[::2]
            
        ix_dict[key] = ix
        
        
    if return_as_dict:
        return ix_dict
    else:
        return np.hstack(list(ix_dict.values()))
       
def sort_modes(lambd, phi=None, order_fun_dict=None, return_as_dict=True, remove_conjugates=['dynamic'], sort=['dynamic']):
    """
    Sorts and organizes modal data (eigenvalues and optionally eigenvectors) according to specified criteria.

    Parameters
    ----------
    lambd : np.ndarray
        Array of eigenvalues or modal parameters to be sorted.
    phi : np.ndarray, optional
        Array of eigenvectors or mode shapes corresponding to `lambd`. If provided, will be sorted in the same order as `lambd`.
    order_fun_dict : dict, optional
        Dictionary of functions specifying custom sorting orders for the modes.
    return_as_dict : bool, default=True
        If True, returns results as dictionaries keyed by mode type. If False, returns concatenated arrays.
    remove_conjugates : list of str, default=['dynamic']
        List of mode types for which conjugate pairs should be removed.
    sort : list of str, default=['dynamic']
        List of mode types to be sorted.

    Returns
    -------
    lambd_dict : dict or np.ndarray
        Dictionary of sorted eigenvalues by mode type if `return_as_dict` is True, otherwise a concatenated array of sorted eigenvalues.
    phi_dict : dict or np.ndarray or None
        Dictionary of sorted eigenvectors by mode type if `return_as_dict` is True and `phi` is provided, otherwise a concatenated array of sorted eigenvectors. Returns None if `phi` is not provided.
   
    Notes
    -----
    This function relies on `get_mode_sort` to determine the sorting and organization of the modes. Docstring is generated using GitHub Copilot.

    See also
    --------    
    get_mode_sort : Function to classify and sort eigenvalues based on specified criteria.
    
    """

    ix_dict = get_mode_sort(lambd, order_fun_dict=order_fun_dict, return_as_dict=True, remove_conjugates=['dynamic'], sort=['dynamic'])
    lambd_dict = dict()
    
    if phi is not None:
        phi_dict = dict()

    for key in ix_dict:
        ix = ix_dict[key]
        lambd_dict[key] = lambd[ix]
        if phi is not None:
            phi_dict[key] = phi[:, ix]       

    if not return_as_dict:
        lambd_list = [lambdi for lambdi in list(lambd_dict.values()) if lambdi.size!=0]
        lambd_sorted = np.hstack([lambd_list])
        
        if phi is not None:
            phi_list = [phii for phii in list(phi_dict.values()) if phii.size!=0]
            phi_sorted = np.hstack(phi_list)
        else:
            phi_sorted = None
        return lambd_sorted, phi_sorted
    else:
        return lambd_dict, phi_dict
        

def statespace(K, C, M):
    """
    Constructs the state-space matrix A for a second-order system defined by stiffness (K), damping (C), and mass (M) matrices.

    Parameters
    ----------
    K : ndarray
        Stiffness matrix of shape (n, n).
    C : ndarray
        Damping matrix of shape (n, n).
    M : ndarray
        Mass matrix of shape (n, n).

    Returns
    -------
    A : ndarray
        State-space matrix of shape (2n, 2n) representing the system in first-order form.

    Notes
    -----
    The state-space matrix A is constructed for the system:
        M * x'' + C * x' + K * x = 0
    which is converted to first-order form as:
        [x']   = [ 0      I ] [x ]
        [x'']    [-M⁻¹K -M⁻¹C] [x']

    Docstring is generated using GitHub Copilot.
    """

    ndofs = np.shape(K)[0]
    A = np.zeros([2*ndofs, 2*ndofs])
    A[0:ndofs, ndofs:2*ndofs] = np.eye(ndofs)
    A[ndofs:2*ndofs, 0:ndofs] = -np.linalg.inv(M) @ K
    A[ndofs:2*ndofs, ndofs:2*ndofs] = -np.linalg.inv(M) @ C

    return A


def iteig(K, C, M, omega=None, omega_ref=0, input_functions=True, itmax=None, tol=None, keep_full=False,
          mac_min=0.9, w_initial=None, normalize=False, print_progress=False, print_warnings=True,
          track_by_psi=True, remove_velocity=True, divergence_protection=True):  
    """
    Iterative eigenvalue solver for frequency-dependent state-space systems.
  
    Parameters
    ----------
    K : callable or array_like
        Stiffness matrix or function K(omega) returning the stiffness matrix at frequency omega.
    C : callable or array_like
        Damping matrix or function C(omega) returning the damping matrix at frequency omega.
    M : callable or array_like
        Mass matrix or function M(omega) returning the mass matrix at frequency omega.
    omega : array_like or None, optional
        Frequency vector for interpolation if K, C, M are arrays (default is None).
    omega_ref : float, optional
        Reference frequency for initial eigenvalue estimation (default is 0).
    input_functions : bool, optional
        If True, K, C, M are assumed to be functions of omega. If False, they are interpolated from arrays (default is True).
    itmax : int or None, optional
        Maximum number of iterations per mode (default is 15).
    tol : float or None, optional
        Convergence tolerance for frequency (default is 1e-4).
    keep_full : bool, optional
        If True, keep all modes (including velocity DOFs); otherwise, keep only displacement modes (default is False).
    mac_min : float, optional
        Minimum Modal Assurance Criterion (MAC) value for convergence (default is 0.9).
    w_initial : array_like or None, optional
        Initial guess for modal frequencies (default is sorted imaginary parts of reference eigenvalues).
    normalize : bool, optional
        If True, normalize mode shapes to have maximum absolute value of 1 (default is False).
    print_progress : bool, optional
        If True, print progress during iterations (default is False).
    print_warnings : bool, optional
        If True, print warnings for non-convergence or divergence (default is True).
    track_by_psi : bool, optional
        If True, track modes by MAC with reference mode shapes; otherwise, by index (default is True).
    remove_velocity : bool, optional
        If True, remove velocity DOFs from output mode shapes (default is True).
    divergence_protection : bool, optional
        If True, activate divergence protection by averaging frequencies on oscillatory divergence (default is True).

    Returns
    -------
    lambd : ndarray
        Array of converged eigenvalues (complex).
    q : ndarray
        Array of corresponding eigenvectors (mode shapes).
    not_converged : list of int
        List of mode indices that did not converge within the maximum number of iterations.

    Notes
    -----
    This function computes the eigenvalues and eigenvectors (modes) of a system with frequency-dependent stiffness (K), 
    damping (C), and mass (M) matrices using an iterative approach. It is particularly useful for systems where these matrices
    are functions of frequency (omega)
    
    - The function assumes that the state-space matrix is constructed by the `statespace` function.
    - The function uses the Modal Assurance Criterion (MAC) to check convergence of mode shapes.
    - If `input_functions` is False, K, C, and M are interpolated using quadratic interpolation.
    - The function can print warnings and progress information if enabled.

    Docstring is generated using GitHub Copilot.

    See Also
    --------
    statespace : Function to construct the state-space matrix.
    xmacmat : Function to compute the MAC matrix.
    mac : Function to compute the MAC value between two vectors.
    """   
     
    mean_w = False
    
    if itmax is None:
        itmax = 15
    
    if tol is None:
        tol = 1e-4
        
    
    if not input_functions:
        K = interp1d(omega, K, kind='quadratic', fill_value='extrapolate')
        C = interp1d(omega, C, kind='quadratic', fill_value='extrapolate')
        M = interp1d(omega, M, kind='quadratic', fill_value='extrapolate')
     
    ndofs = K(0).shape[0]
    lambd = np.zeros([2*ndofs], dtype=complex)
    q = np.zeros([2*ndofs, 2*ndofs], dtype=complex)
       
    # Reference phi
    A = statespace(K(omega_ref), C(omega_ref), M(omega_ref))
    lambd_ref, q_ref = np.linalg.eig(A)
    not_converged = []
    omega_ref = np.abs(np.imag(lambd_ref))
    
    if w_initial is None:
        w_initial = np.sort(np.abs(np.imag(lambd_ref)))

    for m in range(2*ndofs):
        w = w_initial[m]
        
        wprev = np.inf
        wprev2 = -np.inf
        
        phi = q[:,0]*0   
        phiprev = 0*phi
        phiprev2 = 0*phi
        
        q_ref_m = q_ref[:, m:m+1] 

        for i in range(0, itmax):     
            A = statespace(K(w), C(w), M(w))
            lambdai, qi = np.linalg.eig(A)
            
            if track_by_psi:
                macs = xmacmat(qi, phi2=q_ref_m, conjugates=False)
                m_ix = np.argmax(macs, axis=0)[0]
                
                phi = qi[:, m_ix:m_ix+1]
                lambdai = lambdai[m_ix]
            else:
                phi = qi[:, m:m+1]
                lambdai = lambdai[m]

            w = abs(np.imag(lambdai))

            if mean_w:
                w = (wprev+w)/2
            
            if w==0:
                mean_w = False
                break
            elif np.abs(w - wprev2)/w <= tol and np.abs(w - wprev)/w <= tol and mac(phi, phiprev)>=mac_min and mac(phi, phiprev2)>=mac_min:    # Converged!
                mean_w = False    
                break
            elif np.abs(w - wprev2)/w <= tol and np.abs(w - wprev)/w > tol and mac(phi, phiprev)<mac_min and mac(phi, phiprev2)>=mac_min:  
                if divergence_protection:
                    mean_w = True
                    extra_message = 'Divergence protection active - attempting to continue with average values'
                else:
                    extra_message = ''
                if print_warnings:
                    print(f'Warning: Suspecting oscillatory divergence on iteration of mode {int(np.ceil(m/2))}. Please conduct checks.')
                    print(f'Re(lambd) = {np.real(lambdai):.2f}, Im(lambd) = {np.imag(lambdai):.2f}%')
                    print(extra_message)
                
            wprev2 = wprev*1
            wprev = w*1
            
            phiprev2 = phiprev*1
            phiprev = phi*1
    
            if i is itmax-1:
                if print_warnings:
                    print(f'Warning: Maximum number of iterations ({i+1}) performed on mode {int(np.ceil(m/2))}. Storing last value.')
                    print(f'Last three damped natural frequencies: {wprev2:.2f}, {wprev:.2f}, {w:.2f} rad/s')
                    print(f'Corresponding automacs (3,1) and (3,2): {xmacmat(phi, phiprev2):.1f}, {xmacmat(phi, phiprev):.1f}')
                not_converged.append(int(m/2))
        
        lambd[m] = lambdai
        q[:, m] = phi[:,0]

        if print_progress:
            pp(m+2,2*ndofs+1, sym='>', postfix=' finished with iterative modal analysis.')

    if print_progress:
        print(' ')
    
    if remove_velocity:
        q = q[:ndofs, :]  
        
    if not keep_full:
        lambd = lambd[::2]
        q = q[:, ::2]
    
    if normalize==True:
        for mode in range(0, np.shape(q)[1]):
            q[:, mode] = q[:, mode]/max(abs(q[:, mode]))
            
    # Sort 
    ix = np.argsort(np.abs(lambd))
    lambd = lambd[ix]
    q = q[:,ix]
    
    return lambd, q, not_converged


def iteig_naive(K, C, M, itmax=None, tol=1e-4):  
    """
    Compute eigenvalues and eigenvectors for a parameter-dependent state-space system using an iterative naive approach.
    This function iteratively solves for the eigenvalues and eigenvectors of a system defined by parameter-dependent stiffness (K), damping (C), and mass (M) matrices. The iteration is performed for each mode, updating the frequency parameter until convergence or a maximum number of iterations is reached.
    
    Parameters
    ----------
    K : callable
        Function returning the stiffness matrix for a given frequency parameter `w`.
    C : callable
        Function returning the damping matrix for a given frequency parameter `w`.
    M : callable
        Function returning the mass matrix for a given frequency parameter `w`.
    itmax : int, optional
        Maximum number of iterations for convergence (default is 15).
    tol : float, optional
        Convergence tolerance for the frequency parameter (default is 1e-4).
    
    Returns
    -------
    lambd : ndarray of complex
        Array of computed eigenvalues of shape (2 * ndofs,).
    q : ndarray of complex
        Array of computed eigenvectors of shape (2 * ndofs, 2 * ndofs).
    
    Notes
    -----
    - The function assumes that `K(w)`, `C(w)`, and `M(w)` return square matrices of the same size for any input `w`.
    - The function uses a state-space formulation via the `statespace` function (not defined here).
    - Eigenvalues and eigenvectors are computed for each mode using `numpy.linalg.eig`.

    Docstring is generated using GitHub Copilot.
    """
    
    if itmax is None:
        itmax = 15
    
    if tol is None:
        tol = 1e-4
            
    ndofs = K(0).shape[0]
    lambd = np.zeros([2*ndofs], dtype=complex)
    q = np.zeros([2*ndofs, 2*ndofs], dtype=complex)
       
    for m in range(0, 2*ndofs, 2):
        w = 0.0
        wprev = np.inf

        for i in range(0, itmax):     
            A = statespace(K(w), C(w), M(w))
            lambdai, qi = np.linalg.eig(A)
            ix = np.argsort(np.abs(lambdai))
            lambdai = lambdai[ix][m]
            qi = qi[:, ix][:, m]
 
            w = abs(-np.imag(lambdai))

            if np.abs(w - wprev) <= tol:
                break
                
            wprev = w*1
        
        lambd[m] = lambdai
        q[:, m] = qi

    return lambd, q


def iteig_freq(K, C, M, omega=None, itmax=15, reference_omega=0, input_functions=True, 
               tol=1e-4, keep_full=False, mac_min=0.98, w_initial=None, 
                normalize=False, print_progress=False, print_warnings=True, divergence_protection=True):  
    """
    Iterative eigenvalue analysis for frequency-dependent state-space systems.

    Parameters
    ----------
    K : callable or ndarray
        Stiffness matrix or function returning the stiffness matrix as a function of frequency.
    C : callable or ndarray
        Damping matrix or function returning the damping matrix as a function of frequency.
    M : callable or ndarray
        Mass matrix or function returning the mass matrix as a function of frequency.
    omega : array_like, optional
        Frequency vector for interpolation if input matrices are not functions. Default is None.
    itmax : int, optional
        Maximum number of iterations per mode. Default is 15.
    reference_omega : float, optional
        Reference frequency for initial guess or interpolation. Default is 0.
    input_functions : bool, optional
        If True, K, C, and M are assumed to be functions of frequency. If False, they are interpolated.
        Default is True.
    tol : float, optional
        Convergence tolerance for frequency and mode shape. Default is 1e-4.
    keep_full : bool, optional
        If True, keep full state-space eigenvectors and eigenvalues. Default is False.
    mac_min : float, optional
        Minimum Modal Assurance Criterion (MAC) for convergence of mode shapes. Default is 0.98.
    w_initial : array_like, optional
        Initial guess for modal frequencies. If None, zeros are used. Default is None.
    normalize : bool, optional
        If True, normalize mode shapes to unit maximum absolute value. Default is False.
    print_progress : bool, optional
        If True, print progress during computation. Default is False.
    print_warnings : bool, optional
        If True, print warnings for non-convergence or divergence. Default is True.
    divergence_protection : bool, optional
        If True, apply protection against oscillatory divergence. Default is True.

    Returns
    -------
    lambd : ndarray
        Array of converged eigenvalues (complex), sorted by absolute value.
    q : ndarray
        Array of corresponding eigenvectors (modes), sorted to match `lambd`.
    not_converged : list of int
        List of mode indices that did not converge within the maximum number of iterations.

    Notes
    -----
    This function computes the eigenvalues and eigenvectors (modes) of a system with frequency-dependent
    stiffness (K), damping (C), and mass (M) matrices using an iterative approach. It supports both
    direct matrix input and interpolated functions for frequency-dependent matrices.

    - The function uses an iterative approach to solve for eigenvalues and eigenvectors of
        frequency-dependent systems, which may not always converge for all modes.
    - If `input_functions` is False, the matrices are interpolated using quadratic interpolation.
    - The Modal Assurance Criterion (MAC) is used to check convergence of mode shapes.
    - Divergence protection can be enabled to handle oscillatory divergence in the iterative process.

    Docstring is generated using GitHub Copilot.
    """
    if not input_functions:
        K = interp1d(omega, K, kind='quadratic', fill_value='extrapolate')
        C = interp1d(omega, C, kind='quadratic', fill_value='extrapolate')
        M = interp1d(omega, M, kind='quadratic', fill_value='extrapolate')
        
    if divergence_protection:
        extra_message = '[Divergence protection active]'
    else:
        extra_message = ''
        
    ndofs = K(0).shape[0]

    lambd = np.zeros([2*ndofs], dtype=complex)
    q = np.zeros([2*ndofs, 2*ndofs], dtype=complex)
     
    not_converged = []
    
    if not w_initial:
        w_initial = np.zeros([ndofs])
    
    for m in range(0, 2*ndofs):
        w = w_initial[int(m/2)]
        
        wprev = np.inf
        wprev2 = -np.inf
        
        phi = q[:,0]*0   
        phiprev = 0*phi
        phiprev2 = 0*phi

        for i in range(0, itmax):          
            A = statespace(K(w), C(w), M(w))
            
            lambdai, qi = np.linalg.eig(A)
            sortix = np.argsort(np.imag(lambdai))
            
            qi = qi[:, sortix]
            lambdai = lambdai[sortix]
            
            w = abs(np.imag(lambdai[m]))
            phi = qi[:, m]

            if np.abs(w - wprev2) <= tol and np.abs(w - wprev) <= tol and mac(phi, phiprev)>=mac_min and mac(phi, phiprev2)>=mac_min:    # Converged!
                break
            elif np.abs(w - wprev2) <= tol and np.abs(w - wprev) > tol and mac(phi, phiprev)<mac_min and mac(phi, phiprev2)>=mac_min:  
                if divergence_protection:
                    w = (w+wprev)/2
                if print_warnings:
                    print(f'Oscillatory divergence, mode {np.ceil(m/2):.0f} omega = {w:.2f}, xi = {100*-np.real(lambdai[m])/np.abs(lambdai[m]):.1f} % {extra_message}')
                    
            wprev2 = wprev*1
            wprev = w*1
            
            phiprev2 = phiprev*1
            phiprev = phi*1
    
            if i is itmax-1:
                if print_warnings:
                    print('** Maximum number of iterations (%i) performed on mode %i. Storing last value.' % (i+1, np.ceil(m/2)))
                    # print('Last three damped natural frequencies: %f, %f, %f rad/s' % (wprev2, wprev, w))
                    # print('Corresponding automacs (3,1) and (3,2): %f, %f' % (xmacmat(phi, phiprev2), xmacmat(phi, phiprev)))
                not_converged.append(int(m/2))
                
        lambd[m] = lambdai[m]
        q[:, m] = qi[:, m] 

        if print_progress:
            pp(m+2 ,2*ndofs+1, sym='>', postfix=' finished with iterative modal analysis.')

    if print_progress:
        print(' ')
        
    if not keep_full:
        q = q[0:ndofs, :]    
        lambd = lambd[0::2]
        q = q[:, 0::2]
    else:
        lambd[1::2] = np.conj(lambd[0::2])
        q[:, 1::2] = np.conj(q[:,0::2])
    
    if normalize==True:
        for mode in range(0, np.shape(q)[1]):
            q[:, mode] = q[:, mode]/max(abs(q[:, mode]))

    # Sort 
    ix = np.argsort(np.abs(lambd))
    lambd = lambd[ix]
    q = q[:,ix]

    return lambd, q, not_converged

    

def xmacmat(phi1, phi2=None, conjugates=True):
    """
    Modal assurance criterion numbers, cross-matrix between two modal transformation matrices (modes stacked as columns).

    Arguments
    ---------------------------
    phi1 : double
        reference modes
    phi2 : double, optional
        modes to compare with, if not given (i.e., equal default value None), phi1 vs phi1 is assumed
    conjugates : True, optional
        check the complex conjugates of all modes as well (should normally be True)

    Returns
    ---------------------------
    macs : double
        matrix of MAC numbers
    """
    # If no phi2 is given, assign value of phi1
    if phi2 is None:
        phi2 = 1.0*phi1
        
    if len(np.shape(phi1))==1:
        phi1 = np.expand_dims(phi1, axis=0).T
        
    if len(np.shape(phi2))==1:
        phi2 = np.expand_dims(phi2, axis=0).T

    # norms1 = np.dot(np.expand_dims(np.sum(phi1.T * phi1.T.conj(), axis=1), axis=0), np.expand_dims(np.sum(phi2.T * phi2.T.conj(),axis=1), axis=1))
    norms = np.real(np.sum(phi1.T * np.conj(phi1.T), axis=1))[:,np.newaxis] @ np.real(np.sum(phi2.T * np.conj(phi2.T),axis=1))[np.newaxis,:]


    if conjugates:
        macs1 = np.divide(abs(np.dot(phi1.T, phi2))**2, norms)
        macs2 = np.divide(abs(np.dot(phi1.T, phi2.conj()))**2, norms)     
        macs = np.maximum(macs1, macs2)
    else:
        macs = np.divide(abs(np.dot(phi1.T, phi2))**2, norms)

    macs = np.real(macs)
    
    if np.size(macs) == 1:
        macs = macs[0,0]
    
    return macs

def mac(phi1, phi2):
    """
    Calculate the Modal Assurance Criterion (MAC) between two vectors.

    Parameters
    ----------
    phi1 : array_like
        First mode shape vector.
    phi2 : array_like
        Second mode shape vector.

    Returns
    -------
    mac_value : float
        The MAC value between `phi1` and `phi2`, ranging from 0 (no correlation) to 1 (perfect correlation).

    Notes
    -----
    The MAC is a statistical indicator used to quantify the similarity between two mode shapes (vectors).
    It is commonly used in modal analysis to compare experimental and analytical mode shapes. Both `phi1` and `phi2` should be 1-D arrays of the same length.    
    
    """

    mac_value = np.real(np.abs(np.dot(phi1.T,phi2))**2 / np.abs((np.dot(phi1.T, phi1) * np.dot(phi2.T, phi2))))
    return mac_value



def mcf(phi):
    """
    Calculate the modal complexity factor (MCF) for a set of mode shapes.

    Parameters
    ----------
    phi : ndarray
        Array of mode shapes. Can be a 1D array of length n (single mode) or a 2D array
        of shape (n, m), where n is the number of degrees of freedom and m is the number
        of modes. Complex values are expected.

    Returns
    -------
    modal_complexity_factor : ndarray
        1D array of modal complexity factors, one for each mode.

    Notes
    -----
    The modal complexity factor is a measure of the coupling between the real and imaginary
    parts of complex mode shapes. It is commonly used in modal analysis to quantify the
    degree of non-proportional damping or mode complexity.

    Docstring is generated using GitHub Copilot.
    """

    # Ensure on matrix format
    if phi.ndim == 1:
        phi = phi[:,np.newaxis]


    n_modes = np.shape(phi)[1]

    X = np.real(phi)
    Y = np.imag(phi)

    modal_complexity_factor = [None]*n_modes
    for mode in range(0,n_modes):
        modal_complexity_factor[mode] = np.abs(np.dot(X[:,mode], Y[:,mode]))**2 / (np.abs(np.dot(X[:,mode], X[:,mode])) * np.abs(np.dot(Y[:,mode], Y[:,mode])))

    modal_complexity_factor = np.array(modal_complexity_factor)
    return modal_complexity_factor


def mpc(phi):
    """
    Calculates the Modal Phase Collinearity (MPC) indicator for a set of mode shapes.
   
    Parameters
    ----------
    phi : np.ndarray
        Array of mode shapes. Can be a 1D array (single mode) or a 2D array (each column is a mode shape).

    Returns
    -------
    mpc_val : np.ndarray
        Array of MPC values, one for each mode.

    Notes
    -----
    The MPC is a measure used to assess the consistency of complex mode shapes, as described in:
    Pappa, R. S., Elliott, K. B., & Schenk, A. (1993). Consistent-mode indicator for the eigensystem realization algorithm. 
    Journal of Guidance, Control, and Dynamics, 16(5), 852–858.

    Docstring is generated using GitHub Copilot.
    """

    # Ensure on matrix format
    if phi.ndim == 1:
        phi = phi[:,np.newaxis]

    n_modes = np.shape(phi)[1]
    mpc_val = [None]*n_modes

    for mode in range(0,n_modes):
        phin = phi[:, mode]
        Sxx = np.dot(np.real(phin), np.real(phin))
        Syy = np.dot(np.imag(phin), np.imag(phin))
        Sxy = np.dot(np.real(phin), np.imag(phin))

        eta = (Syy-Sxx)/(2*Sxy)

        lambda1 = (Sxx+Syy)/2 + Sxy*np.sqrt(eta**2+1)
        lambda2 = (Sxx+Syy)/2 - Sxy*np.sqrt(eta**2+1)

        mpc_val[mode] = ((lambda1-lambda2)/(lambda1+lambda2))**2

    mpc_val = np.array(mpc_val)
    return mpc_val


def scale_phi(phi, scaling):
    """
    Scales each mode (column) of the input matrix `phi` by the corresponding value in `scaling`.

    Parameters
    ----------
    phi : numpy.ndarray
        A 2D array where each column represents a mode to be scaled.
    scaling : array_like
        A 1D array or list of scaling factors, one for each mode (column) in `phi`.

    Returns
    -------
    phi_scaled : numpy.ndarray
        A 2D array of the same shape as `phi`, where each column has been multiplied by the corresponding scaling factor.

    Examples
    --------
    >>> import numpy as np
    >>> phi = np.array([[1, 2], [3, 4]])
    >>> scaling = [10, 0.5]
    >>> scale_phi(phi, scaling)
    array([[10.,  1.],
           [30.,  2.]])
    
    Notes
    -------
    Docstring is generated using GitHub Copilot.
    """
    phi_scaled = phi*1
    for mode in range(phi.shape[1]):
        phi_scaled[:,mode] = phi[:,mode] * scaling[mode]
        
    return phi_scaled

def normalize_phi(phi, include_dofs=[0,1,2,3,4,5,6], n_dofs=6):
    """
    Normalizes the columns of the mode shape matrix `phi` based on the maximum absolute value 
    of selected degrees of freedom (DOFs).
    
    Parameters
    ----------
    phi : np.ndarray
        The mode shape matrix to be normalized. Each column represents a mode.
    include_dofs : list of int, optional
        List of DOF indices to consider when determining the normalization scaling for each mode.
        Default is [0, 1, 2, 3, 4, 5, 6].
    n_dofs : int, optional
        The total number of DOFs per node or element. Default is 6.

    Returns
    -------
    phi_n : np.ndarray
        The normalized mode shape matrix, with the same shape as `phi`.
    mode_scaling : np.ndarray
        The scaling factors used for normalization, one per mode (column of `phi`).

    Notes
    -----
    - The normalization is performed such that the maximum absolute value among the selected DOFs
      for each mode is 1, preserving the sign of the maximum value.
    - If the maximum value for a mode is zero, the scaling factor is set to 1 to avoid division by zero.
    
    Docstring is generated using GitHub Copilot.
    """
    phi_n = phi*0

    phi_for_scaling = np.vstack([phi[dof::n_dofs, :] for dof in include_dofs])
    mode_scaling = np.max(np.abs(phi_for_scaling), axis=0)
    ix_max = np.argmax(np.abs(phi_for_scaling), axis=0)
    signs = np.sign(phi_for_scaling[ix_max, range(0, len(ix_max))])
    signs[signs==0] = 1
    mode_scaling[mode_scaling==0] = 1

    phi_n = phi/np.tile(mode_scaling[np.newaxis,:]/signs[np.newaxis,:], [phi.shape[0], 1])

    return phi_n, mode_scaling


def restructure_as_ref(phi_ref, phi, min_mac=0.0, ensure_unique=True, 
                       return_all=True, accept_conjugates=True):
    """
    Restructure modes based on reference modal transformation matrix phi_ref.

    Arguments
    ---------------------------
    phi_ref : double
        reference modes
    phi : double
        modes to compare with
    min_mac : double, 0.0
        minimum MAC value for mode to be placed in index arrays

    Returns
    ---------------------------
    ixs : integers
        array of indices, describing which elements to copy to reference 
        numbering (indices indicate position in original numbering, 
                   as in phi - position relates to reference numbering)
    ref_ixs : integers
        array of indices, describing which elements to assign values to 
        (using reference numbering) - equal to 0:n_modes if min_mac=0.0 (all are populated).
    """
    xmacs = xmacmat(phi_ref, phi2=phi, conjugates=accept_conjugates)  # n_modes_ref-by-n_modes
    ixs = np.argmax(xmacs, axis=1)      # indices of what mode in phi_ref each mode in phi2 are closest to, i.e, phi_rs = phi[:, ix]
    ref_ixs = np.arange(len(ixs))
    xmacs_out = xmacs*1

    if ensure_unique:
        for row in range(xmacs.shape[0]):
            xmac_row = xmacs[row, :]
            xmacs[row, xmac_row<np.max(xmac_row)] = 0
            
        for col in range(xmacs.shape[1]):
            xmac_col = xmacs[:, col]
            xmacs[xmac_col<np.max(xmac_col), col] = 0

    ok = np.diag(xmacs[:, ixs])>min_mac

    ixs = ixs[ok]
    ref_ixs = ref_ixs[ok]        
    rejected_ixs = np.where(~ok)[0]
    xmacs_out = xmacs_out[ref_ixs, ixs]

    if return_all:
        return ixs, ref_ixs, rejected_ixs, xmacs_out
    else:
        return ixs


def robust_phi_restructure(phi_ref, phi, accept_conjugates=False):
    """
    Restructure modes based on reference modal transformation matrix phi_ref.

    Arguments
    ---------------------------
    phi_ref : double
        reference modes
    phi : double
        modes to compare with

    Returns
    ---------------------------
    ixs : integers
        array of indices, describing which elements to copy to reference numbering (indices indicate position in original numbering, as in phi - position relates to reference numbering)
    discarded_ixs : integers
        rest of modes (if not the best match to any mode in reference mode)
   """
   
    xmacs = xmacmat(phi_ref, phi2=phi, conjugates=accept_conjugates)  # n_modes_ref-by-n_modes
    ixs = np.argmax(xmacs, axis=0)   

    # Make unique based on selected columns in ixs
    for ix in np.unique(ixs):
        all_this_ix_ix = np.where(ixs==ix)[0]
        ix_max = np.argmax(xmacs[:, ixs==ix], axis=0)
        
        n = len(all_this_ix_ix)
        discards = np.array([i for i in range(n) if i != ix_max])

        if len(discards)>=0:
            ixs[all_this_ix_ix[discards]] = np.nan
            
    all_ixs = np.arange(0, phi_ref.shape[1])
    discarded_ixs = np.sort([i for i in all_ixs if i not in ixs])

    return ixs, discarded_ixs



def freq_eig(K, C, M, omega, omega_ref=None, phi_ref=None, min_mac=0.0, input_functions=True, keep_full=False):
    if not input_functions:
        K = interp1d(omega, K, kind='quadratic', fill_value='extrapolate')
        C = interp1d(omega, C, kind='quadratic', fill_value='extrapolate')
        M = interp1d(omega, M, kind='quadratic', fill_value='extrapolate')
        
    if phi_ref is None:
        if omega_ref is None:
            prev_compare = True
            omega_ref = omega[0]
        else:
            prev_compare = False
            
        A_ref = statespace(K(omega_ref), C(omega_ref), M(omega_ref))
        lambda_ref, phi_ref = np.linalg.eig(A_ref)
        sortix = np.argsort(abs(lambda_ref))
        phi_ref = phi_ref[:, sortix]
    else:
        prev_compare = False            
    
    phi = [None]*len(omega)
    lambd = [None]*len(omega)

    for k, omega_k in enumerate(omega):
        phi[k] = np.nan*phi_ref
        lambd[k] = np.nan*np.ones(phi_ref.shape[1]).astype('complex')
        
        A = statespace(K(omega_k), C(omega_k), M(omega_k))
        lambd_k, phi_k = np.linalg.eig(A)
        mode_ix, ref_ix, __, __ = restructure_as_ref(phi_ref, phi_k, min_mac=min_mac, ensure_unique=False)
        lambd[k][ref_ix] = lambd_k[mode_ix]
        phi[k][:, ref_ix] = phi_k[:, mode_ix]
        
        if prev_compare:
            phi_ref = phi[k]
            
    if not keep_full:
        lambd = [lambd_k[::2] for lambd_k in lambd]
        phi = [phi_k[::2,::2] for phi_k in phi]
        
    return lambd, phi