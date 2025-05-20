import csv
from itertools import chain, count
import numpy as np
from inspect import isfunction

def block_truncate(M, p, n_dofs=6):
    """
    Truncate a block matrix to a specified size.

    Parameters
    ----------
    M : ndarray
        Input matrix (2D or 3D).
    p : int
        Number of blocks to keep.
    n_dofs : int, optional
        Number of degrees of freedom per block (default is 6).

    Returns
    -------
    Mtr : ndarray
        Truncated matrix.

    Notes
    -----
    Docstring generated using GitHub Copilot.
    """
    Mtr = M*1
    if np.ndim(M)==2:
        is_2d = True
        Mtr = Mtr[:,:,np.newaxis]
    else:
        is_2d = False

    N = int(Mtr.shape[0]/n_dofs)
    if N*n_dofs != Mtr.shape[0]:
        raise ValueError('Incompatible integers defined')

    mask = Mtr[:,:,0]*False
    for i in range(N-p+1):
        mask[i*6:(i+p)*6, i*6:(i+p)*6] = True

    for k in range(Mtr.shape[2]):
        Mtr[:,:,k] = Mtr[:,:,k]*mask

    if is_2d:
        Mtr = Mtr[:,:,0]

    return Mtr

def eval_fun_or_scalar(par, x, y, x0=0, y0=0):
    """
    Evaluate a function or return a scalar.

    Parameters
    ----------
    par : callable or scalar
        Function or scalar to evaluate.
    x : float
        x value.
    y : float
        y value.
    x0 : float, optional
        x offset (default is 0).
    y0 : float, optional
        y offset (default is 0).

    Returns
    -------
    result : float
        Evaluated result.

    Notes
    -----
    Docstring generated using GitHub Copilot.
    """
    if isfunction(par):
        if 'x' in par.__code__.co_varnames and 'y' in par.__code__.co_varnames:
            return par(x - x0, y - y0)
        elif 'x' in par.__code__.co_varnames:
            return par(x - x0)
        elif 'y' in par.__code__.co_varnames:
            return par(y - y0)      
    else:
        return par

def zero_pad_upsample(B, omega, omega_max):
    """
    Zero-pad and upsample a vector to a new maximum frequency.

    Parameters
    ----------
    B : ndarray
        Input vector.
    omega : ndarray
        Frequency vector.
    omega_max : float
        Maximum frequency after upsampling.

    Returns
    -------
    Bi : ndarray
        Zero-padded and upsampled vector.
    """
    dw = omega[1]-omega[0]
    omega_max0 = np.max(omega)
    n_add = int(np.floor((omega_max-omega_max0)/dw))
    Bi = np.hstack([B, np.zeros(n_add)])
    return Bi

def get_omega_upsampled(omega, omega_max):
    """
    Get upsampled frequency vector.

    Parameters
    ----------
    omega : ndarray
        Original frequency vector.
    omega_max : float
        Maximum frequency after upsampling.

    Returns
    -------
    omegai : ndarray
        Upsampled frequency vector.
    """
    dw = omega[1]-omega[0]
    omega_max0 = np.max(omega)
    omega_add = np.arange(omega_max0+dw, omega_max, dw)
    omegai = np.hstack([omega, omega_add])
    return omegai

def equal_energy_omega(S, omega_min, omega_max, n, dmax=np.inf, domega_ref=1e-4):  
    """
    Generate frequency vector with equal energy intervals.

    Parameters
    ----------
    S : callable
        Spectral density function.
    omega_min : float
        Minimum frequency.
    omega_max : float
        Maximum frequency.
    n : int
        Number of intervals.
    dmax : float, optional
        Maximum interval width (default is np.inf).
    domega_ref : float, optional
        Reference frequency step (default is 1e-4).

    Returns
    -------
    omegas : ndarray
        Frequency vector with equal energy intervals.
    """
    from scipy.integrate import cumtrapz

    omega_ref = np.arange(omega_min, omega_max, domega_ref)
    E = np.trapz(S(omega_ref), x=omega_ref)
    dE = E/n
    E_cum = cumtrapz(S(omega_ref), x=omega_ref)  
    relE = (E_cum/dE-0.5)
    indices = np.abs(np.round(relE))
    __, jump_ix = np.unique(indices, return_index=True)

    omegas = np.hstack([omega_min,(omega_ref[jump_ix[:-1]] + omega_ref[jump_ix[1:]])/2, omega_max])
    add_omegas = []
    for range_ix in range(len(omegas)-1):
        om0, om1 = omegas[range_ix:range_ix+2]
        domega = om1 - om0
        
        n_sub = domega/dmax
        if n_sub>1.0:
            add_omegas.append(np.linspace(om0, om1, int(np.ceil(n_sub)))[1:-1])
    
    if len(add_omegas)>0:
        omegas = np.unique(np.hstack([omegas, np.hstack(add_omegas)]))

    return omegas

def blkdiag(mat, n):
    """
    Create a block diagonal matrix.

    Parameters
    ----------
    mat : ndarray
        Matrix to repeat along the diagonal.
    n : int
        Number of blocks.

    Returns
    -------
    blk : ndarray
        Block diagonal matrix.
    """
    return np.kron(np.eye(n), mat)

def fun_sum(*functions):
    """
    Sum multiple functions.

    Parameters
    ----------
    *functions : callable
        Functions to sum.

    Returns
    -------
    fun : callable
        Function representing the sum.
    """
    def fun(x):
        return sum([f(x) for f in functions])
    
    return fun

def fun_scale(function, scaling):
    """
    Scale a function by a constant.

    Parameters
    ----------
    function : callable
        Function to scale.
    scaling : float
        Scaling factor.

    Returns
    -------
    fun : callable
        Scaled function.
    """
    def fun(x):
        return function(x)*scaling
    
    return fun

def fun_const_sum(function, constant):
    """
    Add a constant to a function.

    Parameters
    ----------
    function : callable
        Function to add to.
    constant : float
        Constant to add.

    Returns
    -------
    fun : callable
        Function plus constant.
    """
    def fun(x):
        return function(x) + constant
    return fun

def eval_3d_fun(fun, z):
    """
    Evaluate a function over a 1D array and stack results into a 3D array.

    Parameters
    ----------
    fun : callable
        Function to evaluate.
    z : ndarray
        1D array of input values.

    Returns
    -------
    result : ndarray
        Stacked results.
    """
    return np.stack([fun(z_k) for z_k in z], axis=2)

def correct_matrix_size(mat, n_freqs, n_dofs):
    """
    Extend matrix to specified dimensions (frequencies and DOFs).

    Parameters
    ----------
    mat : ndarray
        Matrix to be checked and possibly modified.
    n_freqs : int
        Number of frequencies (depth of final matrix).
    n_dofs : int
        Number of degrees of freedom (height and width of final matrix).

    Returns
    -------
    mat_extended : ndarray
        Corrected/verified matrix.
    """
    mat_shape = np.array(np.shape(mat))

    for ix in range(0, 3-len(mat_shape)):
        mat_shape = np.append(mat_shape, 1)

    if mat_shape[0] == 1 and mat_shape[2] == 1: 	#scalar
        mat_extended = np.ones([n_dofs, n_dofs, n_freqs])*mat
    elif mat_shape[2] == 1 and mat_shape[0] == n_dofs:          # If constant matrix (nDofs-by-nDofs)
        mat_extended = np.tile(mat[:,:,np.newaxis], [1, 1, n_freqs])
    elif mat_shape[2] == n_freqs and mat_shape[0] == n_dofs:     # If correct
        mat_extended = mat
    else:
        raise ValueError('Input dimensions are not valid!')

    return mat_extended

def wrap_to_pi(angle):
    """
    Wrap angle to [-pi, pi].

    Parameters
    ----------
    angle : float or ndarray
        Input angle(s).

    Returns
    -------
    wrapped : float or ndarray
        Wrapped angle(s).
    """
    return (angle + np.pi) % (2*np.pi) - np.pi

def wrap_to_circular(x, rng=[-np.pi, np.pi]):
    """
    Wrap values to a circular range.

    Parameters
    ----------
    x : float or ndarray
        Input value(s).
    rng : list or ndarray, optional
        Range to wrap to (default is [-pi, pi]).

    Returns
    -------
    wrapped : float or ndarray
        Wrapped value(s).
    """
    x0 = np.mean(rng)
    dx_sym = (np.max(rng) - np.min(rng))/2
    x_centered = (x + dx_sym) % (2*dx_sym) - dx_sym

    return x_centered+x0

def merge_tr_phi(phi_trans, phi_rot, thread_stack=True):
    """
    Merge matrices of phi for translational and rotational DOFs.

    Parameters
    ----------
    phi_trans : ndarray
        Phi matrix with only translational DOFs.
    phi_rot : ndarray
        Phi matrix with only rotational DOFs.
    thread_stack : bool, optional
        If True, stack DOFs in thread order (default is True).

    Returns
    -------
    phi_combined : ndarray
        Phi matrix with all DOFs.
    """
    Ndofs = phi_trans.shape[0]*2

    if thread_stack is True:
        trans_dofs = np.array([np.array([0, 1, 2]) + n*6 for n in range(0, Ndofs/6)]).reshape(-1)
        rot_dofs = trans_dofs+3
    else:
        trans_dofs = np.array(range(0, Ndofs/2)).reshape(-1)
        rot_dofs = trans_dofs + Ndofs/2

    phi_combined = np.empty([phi_trans.shape[0]*2, phi_trans.shape[1]])
    phi_combined[trans_dofs, :] = phi_trans
    phi_combined[rot_dofs, :] = phi_rot

    return phi_combined

def mat3d_sel(mat, k):
    """
    Select a 2D slice from a 3D matrix.

    Parameters
    ----------
    mat : ndarray
        3D matrix.
    k : int
        Index of the slice.

    Returns
    -------
    matsel : ndarray
        Selected 2D matrix.
    """
    if len(np.shape(mat)) is 3:
        matsel = mat[:, :, k]
    else:
        matsel = mat

    return matsel

def interp1z(z,mat,znew):
    """
    Interpolate 3D matrix along z-component.

    Parameters
    ----------
    z : ndarray
        z axis.
    mat : ndarray
        3D matrix.
    znew : ndarray
        New z axis.

    Returns
    -------
    matnew : ndarray
        Interpolated 3D matrix.
    """
    matnew = np.zeros([1,len(mat[0]),len(mat[0][0])])
    for dof1 in range(0,len(mat[0])):
        for dof2 in range(0,len(mat[0][0])):
            matnew[:,dof1,dof2]=np.interp(znew,z,mat[:,dof1,dof2])
    return matnew

def interpolate_3d(z, mat, zi):
    """
    Interpolates 3D NumPy array.

    Parameters
    ----------
    z : ndarray
        Original z axis.
    mat : ndarray
        3D matrix (n_x-by-n_y-by-n_z).
    zi : ndarray
        Interpolated z axis.

    Returns
    -------
    mati : ndarray
        Interpolated matrix.
    """
    mat_shape = np.shape(mat)
    mati = np.zeros([mat_shape[0], mat_shape[1], len(zi)])

    for dof1 in range(0, mat_shape[0]):
        for dof2 in range(0, mat_shape[1]):
            mati[dof1, dof2, :] = np.interp(zi, z, mat[dof1, dof2, :])

    return mati

def fast_interpolation(x, y, new_x):
    """
    Fast cubic spline interpolation for multidimensional arrays.

    Parameters
    ----------
    x : ndarray
        Original x values.
    y : ndarray
        Original y values.
    new_x : ndarray
        New x values for interpolation.

    Returns
    -------
    result : ndarray
        Interpolated values.
    """
    from scipy.interpolate import interp1d
    from scipy.interpolate._fitpack import _bspleval
    f = interp1d(x, y, axis=-1, kind=3)
    xj,cvals,k = f._spline
    result = np.empty_like(new_x)
    for (i, j), value in np.ndenumerate(new_x):
        result[i, j] = _bspleval(value, x, cvals[:, i, j], k, 0)
    return result

def transformation_matrix(angles, axis):
    """
    Generate transformation matrices for a set of angles and a given axis.

    Parameters
    ----------
    angles : ndarray
        Array of angles.
    axis : int
        Axis of rotation (0, 1, or 2).

    Returns
    -------
    T : ndarray
        Transformation matrices (len(angles), 6, 6).
    """
    T = np.empty([len(angles),6,6])
    for idx,angle in enumerate(angles):
        c = np.cos(angle)
        s = np.sin(angle)

        if axis==0:
            T0 = np.matrix([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis==1:
            T0 = np.matrix([[c,0,s],[0,1,0],[-s,0,c]])
        elif axis==2:
            T0 = np.matrix([[c,-s,0],[s,c,0],[0,0,1]])
        T[idx]=np.bmat([[T0,np.zeros([3,3])],[np.zeros([3,3]),T0]])

    return T

def rodrot(theta, rotaxis=[0, 0, 1], style='row'):
    """
    Establishes 3D rotation matrix based on Euler-Rodrigues formula.

    Parameters
    ----------
    theta : float
        Rotation angle (in radians).
    rotaxis : array_like, optional
        Vector defining rotation axis (default [0, 0, 1]).
    style : str, optional
        Output style ('row' or other, default 'row').

    Returns
    -------
    T : ndarray
        Transformation matrix.
    """
    axis = np.asarray(rotaxis)
    axis = rotaxis/np.sqrt(np.dot(rotaxis, rotaxis))    # Normalize
    a = np.cos(theta/2.0)
    b, c, d = axis*np.sin(theta/2.0)
    a2, b2, c2, d2 = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    T = np.array([[a2+b2-c2-d2, 2*(bc-ad), 2*(bd+ac)],
                  [2*(bc+ad), a2+c2-b2-d2, 2*(cd-ab)],
                  [2*(bd-ac), 2*(cd+ab), a2+d2-b2-c2]])
    
    if style=='row':
        T = T.T
    
    return T

def multi_rodrot(angles, rotaxis=[0,0,1], style='row'):
    """
    Generate multiple 6x6 rotation matrices for a set of angles.

    Parameters
    ----------
    angles : ndarray
        Array of angles.
    rotaxis : array_like, optional
        Rotation axis (default [0, 0, 1]).
    style : str, optional
        Output style (default 'row').

    Returns
    -------
    T : list of ndarray
        List of 6x6 rotation matrices.
    """
    T = [None]*len(angles)
    for ix, angle in enumerate(angles):
        T[ix] = blkdiag(rodrot(angle, rotaxis=rotaxis, style=style), 2)

    return T

def stack_T(T_multi):
    """
    Stack multiple transformation matrices into a block diagonal matrix.

    Parameters
    ----------
    T_multi : list of ndarray
        List of transformation matrices.

    Returns
    -------
    T : ndarray
        Block diagonal transformation matrix.
    """
    n_dofs = np.shape(T_multi[0])[0]
    N = len(T_multi)

    T = np.zeros([n_dofs*N, n_dofs*N])

    for n, Tn in enumerate(T_multi):
        T[n_dofs*n:n_dofs*n+n_dofs, n_dofs*n:n_dofs*n+n_dofs] = Tn

    return T

def transform_unit(e1, e2p):
    """
    Generate a transformation matrix from two unit vectors.

    Parameters
    ----------
    e1 : array_like
        First unit vector.
    e2p : array_like
        Second vector (not necessarily unit).

    Returns
    -------
    T : ndarray
        Transformation matrix (3x3).
    """
    e1 = np.array(e1).flatten()
    e2p = np.array(e2p).flatten()

    e3 = np.cross(e1, e2p)         # Direction of the third unit vector
    e2 = np.cross(e3, e1)          # Direction of the second unit vector

    e1 = e1/np.linalg.norm(e1)     # Normalize the direction vectors to become unit vectors
    e2 = e2/np.linalg.norm(e2)
    e3 = np.cross(e1,e2)

    T = np.vstack([e1,e2,e3])

    return T

def transform_3dmat(mat, T):
    """
    Transform a 3D matrix using a transformation matrix.

    Parameters
    ----------
    mat : ndarray
        3D matrix (n_dofs, n_dofs, n_freqs).
    T : ndarray
        Transformation matrix.

    Returns
    -------
    mat_transformed : ndarray
        Transformed 3D matrix.
    """
    M, N = np.shape(T)[0:2]
    [n_dofs, __, n_freqs] = np.shape(mat)

    if n_dofs != M:
        raise ValueError('The dimensions of T and mat must match.')

    mat_transformed = np.zeros([N, N, n_freqs])

    if np.iscomplexobj(T) or np.iscomplexobj(mat):
        mat_transformed = mat_transformed.astype('complex')

    for k in range(0, n_freqs):
        mat_transformed[:, :, k] = np.dot(np.dot(T.conj().T, mat[:, :, k]), T)

    return mat_transformed

def assess_diagonality_fro(M):
    """
    Assess diagonality of a Hermitian positive-definite matrix using Frobenius norm.

    Parameters
    ----------
    M : ndarray
        Input matrix (2D, square).

    Returns
    -------
    diagonality : float
        Diagonality measure.
    """
    if M.ndim != 2:
        raise ValueError('Input must be 2d array.')
    if np.shape(M)[0] != np.shape(M)[1]:
        raise ValueError('Matrix must be square.')

    N = np.shape(M)[0]
    M_m05 = np.diag(np.diag(M)**-0.5)
    M_hat = np.dot(np.dot(M_m05, M), M_m05)
    M_o = M_hat-np.eye(N)

    diagonality = 0.5*np.linalg.norm(M_o, 'fro')**2

    return diagonality

def assess_diagonality(M):
    """
    Assess diagonality of a Hermitian positive-definite matrix.

    Parameters
    ----------
    M : ndarray
        Input matrix (2D, square).

    Returns
    -------
    diagonality : float
        Diagonality measure.
    """
    if M.ndim != 2:
        raise ValueError('Input must be 2d array.')
    if np.shape(M)[0] != np.shape(M)[1]:
        raise ValueError('Matrix must be square.')

    N = np.shape(M)[0]
    Mdiag = np.diag(np.diag(M))
    Moffdiag = M-Mdiag

    diagonality = np.linalg.norm(Mdiag)/np.linalg.norm(M)

    return diagonality

def modify_stiffness(stiffness, submerged_vol, water_dens, g, z_cog, z_cog_mod):
    """
    Modify stiffness matrix for change in center of gravity.

    Parameters
    ----------
    stiffness : ndarray
        Original stiffness matrix.
    submerged_vol : float
        Submerged volume.
    water_dens : float
        Water density.
    g : float
        Gravitational acceleration.
    z_cog : float
        Original center of gravity (z).
    z_cog_mod : float
        Modified center of gravity (z).

    Returns
    -------
    stiffness_mod : ndarray
        Modified stiffness matrix.
    """
    stiffness_mod = stiffness
    stiffness_mod[3, 3] = stiffness[3, 3]  + submerged_vol * water_dens * g * (z_cog - z_cog_mod)
    stiffness_mod[4, 4] = stiffness[4, 4]+ submerged_vol * water_dens * g * (z_cog - z_cog_mod)

    return stiffness_mod

def maxreal(phi, preserve_conjugates=False):
    """
    Rotate complex mode shapes to maximize real part.

    Parameters
    ----------
    phi : ndarray
        Mode shape matrix.
    preserve_conjugates : bool, optional
        If True, preserve conjugate pairs (default False).

    Returns
    -------
    phi_max_real : ndarray
        Rotated mode shapes with maximized real part.
    """
    angles = np.expand_dims(np.arange(0,np.pi/2, 0.01), axis=0)
    
    if phi.ndim==1:
        phi = np.array([phi]).T

    phi_max_real = np.zeros(np.shape(phi)).astype('complex')        

    for mode in range(np.shape(phi)[1]):
        rot_mode = np.dot(np.expand_dims(phi[:, mode], axis=1), np.exp(angles*1j))
        max_angle_ix = np.argmax(np.sum(np.real(rot_mode)**2, axis=0), axis=0)

        phi_max_real[:, mode] = phi[:, mode] * np.exp(angles[0, max_angle_ix]*1j)*np.sign(sum(np.real(phi[:, mode])))
  
    return phi_max_real

def create_circular_x(x, tol=1e-5, rng=np.array([-np.pi, np.pi])):
    """
    Create a circularly wrapped and sorted version of x.

    Parameters
    ----------
    x : ndarray
        Input array.
    tol : float, optional
        Tolerance for uniqueness (default 1e-5).
    rng : ndarray, optional
        Range for wrapping (default [-pi, pi]).

    Returns
    -------
    x_ang : ndarray
        Wrapped and sorted array.
    sort_ix : ndarray
        Indices of unique values.
    """
    xrng = np.max(x)-np.min(x)

    wrapped_x = np.sort(wrap_to_circular(x, rng))
    wrapped_x_mod = wrap_to_circular(wrapped_x, rng + np.mean(rng))
    dx = np.abs(wrapped_x_mod[-1]-wrapped_x_mod[0])

    x_ang = wrap_to_circular(x, rng)
    x_ang, sort_ix = uniquetol(x_ang, 1e-10)
    
    x_ang = np.hstack([x_ang[0]-dx, x_ang, x_ang[-1]+dx])
    sort_ix = np.hstack([sort_ix[-1], sort_ix, sort_ix[0]])
    
    return x_ang, sort_ix

def interp1d_angular(x, mat, rng=[-np.pi, np.pi], axis=-1, **kwargs):
    """
    Interpolate data defined on a circular domain.

    Parameters
    ----------
    x : ndarray
        Input angular values.
    mat : ndarray
        Data to interpolate.
    rng : list or ndarray, optional
        Range for wrapping (default [-pi, pi]).
    axis : int, optional
        Axis along which to interpolate (default -1).
    **kwargs
        Additional arguments to scipy.interpolate.interp1d.

    Returns
    -------
    interp_fun : callable
        Interpolation function.
    """
    from scipy.interpolate import interp1d
    x_ang, sort_ix = create_circular_x(x, rng=rng)
    mat = np.take(mat, sort_ix, axis=axis)
    return lambda x: interp1d(x_ang, mat, axis=axis, **kwargs)(wrap_to_circular(x, rng=rng))

def interp_hydro_transfer(omega, theta, Q, rng=[-np.pi, np.pi], theta_axis=1, omega_axis=2, interpolation_kind='linear', theta_settings=dict(), omega_settings=dict()):
    """
    Interpolate hydrodynamic transfer function in both frequency and angle.

    Parameters
    ----------
    omega : ndarray
        Frequency vector.
    theta : ndarray
        Angle vector.
    Q : ndarray
        Transfer function data.
    rng : list or ndarray, optional
        Range for wrapping (default [-pi, pi]).
    theta_axis : int, optional
        Axis for theta (default 1).
    omega_axis : int, optional
        Axis for omega (default 2).
    interpolation_kind : str, optional
        Interpolation kind (default 'linear').
    theta_settings : dict, optional
        Additional settings for theta interpolation.
    omega_settings : dict, optional
        Additional settings for omega interpolation.

    Returns
    -------
    Qi : callable
        Interpolated transfer function.
    """
    from scipy.interpolate import interp1d
    Qi = lambda om, th: interp1d(omega, interp1d(theta, Q, axis=theta_axis, fill_value='extrapolate', **theta_settings)(wrap_to_circular(th, rng=rng)), 
                                   fill_value='extrapolate', kind=interpolation_kind, axis=omega_axis, **omega_settings)(om)
    
    return Qi

def interp2d_angular(x, theta, mat, rng=[-np.pi, np.pi], method='linear', **kwargs):
    """
    2D interpolation on a circular domain.

    Parameters
    ----------
    x : ndarray
        First coordinate array.
    theta : ndarray
        Angular coordinate array.
    mat : ndarray
        Data to interpolate.
    rng : list or ndarray, optional
        Range for wrapping (default [-pi, pi]).
    method : str, optional
        Interpolation method (default 'linear').
    **kwargs
        Additional arguments to scipy.interpolate.griddata.

    Returns
    -------
    funi : callable
        Interpolation function.
    """
    from scipy.interpolate import griddata

    def funi(xi, thetai):
        xi = np.array([xi]).flatten()
        thetai = np.array([thetai]).flatten()
        
        Xi, Yi = np.meshgrid(xi, wrap_to_circular(thetai, rng=[-np.pi, np.pi]))
        X, Y = np.meshgrid(x, wrap_to_circular(theta, rng=[-np.pi, np.pi]))
        mati = np.zeros([mat.shape[0], len(xi), len(thetai)]).astype(complex)
        
        for comp in range(mati.shape[0]):
            mati[comp,:,:] = griddata((X.flatten(),Y.flatten()), mat[comp,:,:].flatten(), (Xi, Yi), method='linear')
     
        return mati
        
    return funi

def uniquetol(a, tol):
    """
    Return unique values of an array within a tolerance.

    Parameters
    ----------
    a : ndarray
        Input array.
    tol : float
        Tolerance for uniqueness.

    Returns
    -------
    result : ndarray
        Unique values.
    i : ndarray
        Indices of unique values.
    """
    import numpy as np
    i = np.argsort(a.flat)
    d = np.append(True, np.diff(a.flat[i]))
    result = a.flat[i[d>tol]]
    i = i[d>tol]
    
    return result, i