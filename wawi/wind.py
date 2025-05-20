import numpy as np
from scipy.interpolate import interp1d
from .modal import statespace, iteig, restructure_as_ref, iteig_naive
from .tools import print_progress as pp
from scipy.special import jv as besselj, yv as bessely
from .general import rodrot, blkdiag
from .plot import plot_ads

conv_text = r''' 
-----------------------------------------------------
|                                                   |
| ~ ~ ~~~ ~  ~~ ~ /^^^^^^^^^^^^\ 88ooo...  .  .  .  |
| ~ ~ ~ ~~ ~ ~   ~\____________/  88ooo¨¨¨¨  ¨¨     |
|                   CONVERGED!                      |
-----------------------------------------------------
'''

beaufort_dict = {
    'calm': [0, 0.5],
    'light air': [0.5, 1.5],
    'light breeze': [1.6, 3.3],
    'gentle breeze': [3.4, 5.5],
    'moderate breeze': [5.6, 7.9],
    'fresh breeze': [8, 10.7],
    'strong breeze': [10.8, 13.8],
    'moderate gale': [13.9, 17.1],
    'gale': [17.2, 20.7],
    'strong gale': [20.8, 24.4],
    'storm': [24.5, 28.4],
    'violent storm': [28.5, 32.6],
    'hurricane': [32.7, np.inf]
}

def get_beaufort(U0):
    """Get the Beaufort scale for a given wind speed U0."""

    return [key for key in beaufort_dict if inrange(U0, beaufort_dict[key])][0]

def inrange(num, rng):
    """Check if a number is in a given range."""
    return num<=np.max(rng) and num>=np.min(rng)

class LoadCoefficients:
    """
    Class to hold the load coefficients for a aerodynamic section
    
    Attributes
    ----------
    Cd : float
        Drag coefficient
    dCd : float
        Derivative of drag coefficient with respect to reduced velocity
    Cl : float
        Lift coefficient
    dCl : float
        Derivative of lift coefficient with respect to reduced velocity
    Cm : float
        Moment coefficient
    dCm : float 
        Derivative of moment coefficient with respect to reduced velocity
    fill_empty : bool
        If True, fill empty attributes with zeros
    
    Methods
    -------
    fill_empty_with_zeros : None
        Fills empty attributes with zeros
    to_dict : dict
        Converts the LoadCoefficients object to a dictionary
        
    """
    keys = ['Cd', 'Cm', 'Cl', 'dCd', 'dCm', 'dCl']
    
    def __repr__(self):
        return 'LoadCoefficients (Cd, Cl, Cm, dCd, dCl, dCm)'

    def __str__(self):
        return f'Cd:{self.Cd}, dCd:{self.dCd}, Cl:{self.Cl}, dCl:{self.dCl}, Cm:{self.Cm}, dCm:{self.dCm}'
    
    def __init__(self, Cd=None, dCd=None, Cl=None, dCl=None, Cm=None, dCm=None, fill_empty=True):
        self.Cd = Cd
        self.dCd = dCd
        self.Cl = Cl
        self.dCl = dCl
        self.Cm = Cm
        self.dCm = dCm
        
        if fill_empty:
            self.fill_empty_with_zeros()
        
    def fill_empty_with_zeros(self):
        """
        Fills empty attributes with zeros.
        """
        # Fill empty attributes with zeros
        for key in self.keys:
            if getattr(self, key) is None:
                setattr(self, key, 0)
                
    def to_dict(self):
        """
        Converts the LoadCoefficients object to a dictionary.

        Returns
        -------
        dict
            A dictionary with the keys and values of the LoadCoefficients object.
        """

        return {key: getattr(self, key) for key in self.keys}

class ADs:
    """
    Class to hold the aerodynamic derivatives (ADs) for a given aerodynamic section.

    Attributes
    ----------
    type : str
        Type of the aerodynamic section
    P1, P2, P3, P4, P5, P6 : function
        Functions representing the aerodynamic derivatives for the section, related to horizontal motion of the section.
    H1, H2, H3, H4, H5, H6 : function
        Functions representing the aerodynamic derivatives for the section, related to vertical motion of the section.
    A1, A2, A3, A4, A5, A6 : function
        Functions representing the aerodynamic derivatives for the section, related to twisting motion of the section.

    Methods
    -------
    plot : None
        Plots the aerodynamic derivatives for the section
    to_dict : dict
        Converts the ADs object to a dictionary
    evaluate_all : dict
        Evaluates all aerodynamic derivatives for a given reduced velocity
    evaluate : dict
        Evaluates a specific aerodynamic derivative for a given reduced velocity

    """

    ad_keys = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 
              'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
              'A1', 'A2', 'A3', 'A4', 'A5', 'A6']

    P1, P2, P3, P4, P5, P6 = None, None, None, None, None, None
    H1, H2, H3, H4, H5, H6 = None, None, None, None, None, None
    A1, A2, A3, A4, A5, A6 = None, None, None, None, None, None

    def __init__(self, ad_type='not specified', 
                 P1=None, P2=None, P3=None, P4=None, P5=None, P6=None,
                 H1=None, H2=None, H3=None, H4=None, H5=None, H6=None,
                 A1=None, A2=None, A3=None, A4=None, A5=None, A6=None):
        
        self.type = ad_type
        
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        self.P4 = P4
        self.P5 = P5
        self.P6 = P6
        
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.H4 = H4
        self.H5 = H5
        self.H6 = H6
        
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4
        self.A5 = A5
        self.A6 = A6
            
    def plot(self, v=np.arange(0,5,0.01), **kwargs):
        """
        Plots the aerodynamic derivatives for the section.
        
        Parameters
        ----------
        v : array-like
            Reduced velocity values for which to plot the aerodynamic derivatives.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to the `plot_ads` function.
        """

        return plot_ads(self.to_dict(), v, **kwargs)
        
        
    def to_dict(self):
        """
        Converts the ADs object to a dictionary.

        Returns
        -------
        dict
            A dictionary with the keys and values of the ADs object.
        """

        return {key: getattr(self, key) for key in self.ad_keys}
    
    def evaluate_all(self, v):
        """
        Evaluates all aerodynamic derivatives for a given reduced velocity.

        Parameters
        ----------
        v : float
            Reduced velocity for which to evaluate the aerodynamic derivatives.
        
        Returns
        -------
        dict
            A dictionary with the evaluated aerodynamic derivatives.
        """

        AD_evaluated = dict()
        for key in self.ad_keys:
            AD_evaluated[key] = getattr(self, key)(v)

        return AD_evaluated
    
    
    def evaluate(self, key, v):
        """
        Evaluates a specific aerodynamic derivative for a given reduced velocity.

        Parameters
        ----------
        key : str

            The key of the aerodynamic derivative to evaluate.
        v : float
            Reduced velocity for which to evaluate the aerodynamic derivative.

        Returns
        -------
        float
            The evaluated aerodynamic derivative.
        """

        AD_evaluated = getattr(self, key)(v)

        return AD_evaluated
    
def flatplate_ads():
    """Flat plate aerodynamic derivatives.
    
    Returns
    -------
    dict
        A dictionary with the aerodynamic derivatives for a flat plate.
        
    """
    ad_dict = dict()  
    
    def F(v):
        J1 = besselj(1, 0.5/v)
        Y1 = bessely(1, 0.5/v)
        J0 = besselj(0, 0.5/v)
        Y0 = bessely(0, 0.5/v)
        
        a = J1 + Y0
        b = Y1 - J0
        c = a**2 + b**2
        
        return (J1*a + Y1*b)/c
        
    def G(v):
        J1 = besselj(1, 0.5/v)
        Y1 = bessely(1, 0.5/v)
        J0 = besselj(0, 0.5/v)
        Y0 = bessely(0, 0.5/v)
        
        a = J1 + Y0
        b = Y1 - J0
        c = a**2 + b**2       
        return -(J1*J0 + Y1*Y0)/c
    
    ad_dict['H1'] = lambda v: -2*np.pi*F(v)*v
    ad_dict['H2'] = lambda v: np.pi/2*(1+F(v)+4*G(v)*v)*v
    ad_dict['H3'] = lambda v: 2*np.pi*(F(v)*v-G(v)/4)*v
    ad_dict['H4'] = lambda v: np.pi/2*(1+4*G(v)*v)
    ad_dict['H5'] = lambda v: 0*v
    ad_dict['H6'] = lambda v: 0*v

    ad_dict['A1'] = lambda v: -np.pi/2*F(v)*v
    ad_dict['A2'] = lambda v: -np.pi/8*(1-F(v)-4*G(v)*v)*v
    ad_dict['A3'] = lambda v: np.pi/2*(F(v)*v-G(v)/4)*v
    ad_dict['A4'] = lambda v: np.pi/2*G(v)*v   
    ad_dict['A5'] = lambda v: 0*v
    ad_dict['A6'] = lambda v: 0*v

    ad_dict['P1'] = lambda v: 0*v
    ad_dict['P2'] = lambda v: 0*v
    ad_dict['P3'] = lambda v: 0*v
    ad_dict['P4'] = lambda v: 0*v
    ad_dict['P4'] = lambda v: 0*v
    ad_dict['P5'] = lambda v: 0*v
    ad_dict['P6'] = lambda v: 0*v
    
    return ad_dict


def quasisteady_ads(D, B, load_coefficients):
    """Quasi-steady aerodynamic derivatives.

    Parameters
    ----------
    D : float
        Diameter of the section
    B : float
        Width of the section
    load_coefficients : LoadCoefficients or dict
        Load coefficients for the section. If a dictionary is provided, it should contain
        the keys 'Cd', 'dCd', 'Cl', 'dCl', 'Cm', and 'dCm' with corresponding values.
        If a LoadCoefficients object is provided, it should have the same attributes.
    Returns
    -------
    dict
        A dictionary with the aerodynamic derivatives for the section.
    
    Notes
    -----
    The aerodynamic derivatives are calculated based on the load coefficients provided. 
    The load coefficients should be normalized with respect to both D (Cd) and B (Cl and Cm).
    ADs are normalized using B only.
    """

    if type(load_coefficients)==dict: 
        Cd = load_coefficients['Cd']
        dCd = load_coefficients['dCd']
        Cl = load_coefficients['Cl']
        dCl = load_coefficients['dCl']
        Cm = load_coefficients['Cm']
        dCm = load_coefficients['dCm']
    else:
        Cd, dCd = load_coefficients.Cd, load_coefficients.dCd
        Cl, dCl = load_coefficients.Cl, load_coefficients.dCl
        Cm, dCm = load_coefficients.Cm, load_coefficients.dCm

    ad_dict = dict()
    ad_dict['P1'], ad_dict['P2'], ad_dict['P3'] = lambda v: -2*Cd*D/B*v, lambda v: 0*v, lambda v: dCd*D/B*v**2
    ad_dict['P4'], ad_dict['P5'], ad_dict['P6'] = lambda v: 0*v, lambda v: (Cl-dCd*D/B)*v, lambda v: 0*v

    ad_dict['H1'], ad_dict['H2'], ad_dict['H3'] = lambda v: -(dCl+Cd*D/B)*v, lambda v: 0*v, lambda v: dCl*v**2
    ad_dict['H4'], ad_dict['H5'], ad_dict['H6'] = lambda v: 0*v, lambda v: -2*Cl*v, lambda v: 0*v

    ad_dict['A1'], ad_dict['A2'], ad_dict['A3'] = lambda v: -dCm*v, lambda v: 0*v, lambda v: dCm*v**2
    ad_dict['A4'], ad_dict['A5'], ad_dict['A6'] = lambda v: 0*v, lambda v: -2*Cm*v, lambda v: 0*v

    return ad_dict


def compute_aero_matrices(U, AD, B, elements, T_wind, phi, 
                          omega_reduced=None, print_progress=False, rho=1.225):
    """Compute the aerodynamic matrices for a given set of elements.
    
    Parameters
    ----------
    U : float or function
        Wind speed or function to calculate wind speed at a given point.
    AD : ADs
        Aerodynamic derivatives for the section.
    B : float
        Width of the section.
    elements : list
        List of elements for which to compute the aerodynamic matrices.
    T_wind : np.ndarray
        Transformation matrix from global to wind coordinates.
    phi : np.ndarray
        Mode shapes for the elements.
    omega_reduced : np.ndarray, optional
        Reduced frequencies for which to compute the aerodynamic matrices.
    print_progress : bool, optional
        If True, print progress of the computation.
    rho : float, optional
        Air density, default is 1.225 kg/m^3.
        
    Returns
    -------
    Kae : np.ndarray
        Aerodynamic stiffness matrix.
    Cae : np.ndarray    
        Aerodynamic damping matrix.
    """
    
    if omega_reduced is None:
        omega_reduced = np.linspace(0.015, 2.0, 75)
    
    n_modes = phi.shape[1]

    Kae = np.zeros([n_modes, n_modes, len(omega_reduced)])
    Cae = np.zeros([n_modes, n_modes, len(omega_reduced)])
    
    for element_ix, element in enumerate(elements):
        
        if callable(U):
            U_el_glob = U(element.get_cog())
        else:
            U_el_glob = U*1
            
        U_el = normal_wind(T_wind, element.T0, U=U_el_glob)

        v = U_el/(B*omega_reduced)
            
        for k, v_k in enumerate(v):
            k_aero, c_aero = element_aero_mats(B, omega_reduced[k], 
                                               AD.evaluate_all(v_k), 
                                               element.L, T=element.T0, 
                                               phi=phi[element.global_dofs, :], rho=rho)

            Kae[:, :, k] = Kae[:, :, k] + k_aero
            Cae[:, :, k] = Cae[:, :, k] + c_aero
        
        if print_progress:
            pp(element_ix+1, len(elements), sym='=', postfix=' ESTABLISHING WIND EXCITATION')
            print('')
    
    Cae = interp1d(omega_reduced, Cae, kind='quadratic', fill_value='extrapolate', bounds_error=False) 
    Kae = interp1d(omega_reduced, Kae, kind='quadratic', fill_value='extrapolate', bounds_error=False)


    return Kae, Cae


def compute_aero_matrices_sets(U, AD, B, elements, T_wind, phi_dict, 
                          omega_reduced=None, omega=None, print_progress=False, sets=None):
    
    """Compute the aerodynamic matrices for a given set of elements.
    
    Parameters
    ----------
    U : float or function
        Wind speed or function to calculate wind speed at a given point.
    AD : dict
        Dictionary of aerodynamic derivatives for the section.
    B : float
        Width of the section.
    elements : dict
        Dictionary of elements (BEEF elements) for which to compute the aerodynamic matrices.
    T_wind : np.ndarray
        Transformation matrix from global to wind coordinates.
    phi : np.ndarray
        Mode shapes for the elements.
    omega_reduced : np.ndarray, optional
        Reduced frequencies for which to compute the aerodynamic matrices.
    print_progress : bool, optional
        If True, print progress of the computation.
    sets : list, optional
        List of sets of elements for which to compute the aerodynamic matrices.

    Returns
    -------
    Kae : np.ndarray 
        Aerodynamic stiffness matrix.
    Cae : np.ndarray
        Aerodynamic damping matrix.
    """

    if sets is None:
         sets = elements.keys()

    if omega is None:
        return_as_function = True
    else:
        first_is_zero = omega[0]==0.0
        if first_is_zero:
            omega = omega[1:]
    
    if omega_reduced is None:
        omega_reduced = np.logspace(np.log10(0.01), np.log10(2), 100)     #standard values should be reasonable in most typical cases - change later!
    
    first_key = [str(key) for key in sets][0]
    n_modes = np.shape(phi_dict[first_key])[1]

    Kae = np.zeros([n_modes, n_modes, len(omega_reduced)])
    Cae = np.zeros([n_modes, n_modes, len(omega_reduced)])
    
    for set_name in sets:
        B_set = B[set_name]
        AD_set = AD[set_name]
        phi = phi_dict[set_name]
        elements_set = elements[set_name]

        for element_ix, element in enumerate(elements_set):
            T_el = element.T0
            U_el = normal_wind(T_wind, T_el, U=U)
            v = U_el/(B_set*omega_reduced)
            
            dof_range = np.hstack([element.nodes[0].global_dofs, element.nodes[1].global_dofs])

            for k, v_k in enumerate(v):
                k_aero, c_aero = element_aero_mats(B_set, omega_reduced[k], AD_set.evaluate_all(v_k), element.L, T=T_el, phi=phi[dof_range, :])
                Kae[:, :, k] += k_aero
                Cae[:, :, k] += c_aero
            
            if print_progress:
                pp(element_ix+1, len(elements_set), sym='>', postfix=f' finished with set "{set_name}".')

        if print_progress:
            print('')

        Cae = interp1d(omega_reduced, Cae, kind='quadratic',fill_value='extrapolate') 
        Kae = interp1d(omega_reduced, Kae, kind='quadratic', fill_value='extrapolate')

        if return_as_function:
            return Kae, Cae
        else:
            Cae = Cae(omega)
            Kae = Kae(omega)
            
            if first_is_zero:
                Cae = np.insert(Cae, 0, Cae[:,:,0]*0, axis=2)
                Kae = np.insert(Kae, 0, Kae[:,:,0]*0, axis=2)

            return Kae, Cae


def f_rf_fun(a, d, v):
    N = len(a) 
    f = np.array(a[0])*0j
    
    for l in range(0, 3):
        f = f + a[l] * (1j/v)**l

    for l in range(0, N-3):
        f = f + a[l+2]*(1j/v) / ((1j/v + d[l]))
    
    f = f*v**2    

    return f


def rf_ads(a, d):
    """ 
    Function to compute the aerodynamic derivatives for a given set of reduced frequencies using rational functions.

    Parameters
    ----------
    a : list
        List of coefficients for the rational function, assumed in the order of 
        P1, P2, P3, P4, P5, P6, H1, H2, H3, H4, H5, H6, A1, A2, A3, A4, A5, A6.
    d : list
        List of poles for the rational function, assumed in the order of 
        P1, P2, P3, P4, P5, P6, H1, H2, H3, H4, H5, H6, A1, A2, A3, A4, A5, A6.

    Returns
    -------
    dict
        A dictionary with the aerodynamic derivatives for the section.

    Notes
    -----
    The aerodynamic derivatives are calculated based on the coefficients and poles provided.
    The coefficients should be normalized with respect to both D (Cd) and B (Cl and Cm).
    
    """
    # B assumed to be implicitly included in RF factors
    ad_dict = dict()
    ad_keys = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 
               'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
               'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    
    imag_component_ad = ['P1', 'P2', 'P5', 'H1', 'H2', 'H5', 'A1', 'A2', 'A5']
    
    position_dict = {'P1': [0,0], 'P2': [0,2], 'P3': [0,2], 'P4': [0,0], 'P5': [0,1], 'P6': [0,1],
                 'H1': [1,1], 'H2': [1,2], 'H3': [1,2], 'H4': [1,1], 'H5': [1,0], 'H6': [1,0],
                 'A1': [2,1], 'A2': [2,2], 'A3': [2,2], 'A4': [2,1], 'A5': [2,0], 'A6': [2,0]}
    
    for key in ad_keys:
        row = position_dict[key][0]
        col = position_dict[key][1]
        a_key = [ai[row, col] for ai in a]

        if key in imag_component_ad:
            ad_dict[key] = lambda v, a=a_key: np.imag(f_rf_fun_legacy(a, d, v))
        else:
            ad_dict[key] = lambda v, a=a_key: np.real(f_rf_fun_legacy(a, d, v))

    return ad_dict


def distribute_to_dict(prefix, array, count_start=1):
    """ 
    Distributes a list of arrays to a dictionary with keys based on the prefix and index.

    Parameters
    ----------
    prefix : str
        Prefix for the keys in the dictionary.
    array : list
        List of arrays to be distributed.
    count_start : int, optional
        Starting index for the keys in the dictionary. Default is 1.

    Returns
    -------
    dict
        A dictionary with keys based on the prefix and index, and values from the input array.

    Notes
    -----
    The keys in the dictionary are formed by concatenating the prefix with the index of the array.

    Docstring is generated by GitHub Copilot.
    
    """
    array_dict = dict()
    for ix,array_i in enumerate(array):
        key = prefix + str(ix+count_start)
        array_dict[key] = array_i
        
    return array_dict
        

def distribute_multi_to_dict(prefixes, arrays):
    """
    Distributes a list of arrays to a dictionary with keys based on the prefix and index.

    Parameters
    ----------
    prefixes : list
        List of prefixes for the keys in the dictionary.
    arrays : list
        List of arrays to be distributed.

    Returns
    -------
    dict
        A dictionary with keys based on the prefix and index, and values from the input array.

    Notes
    -----
    The keys in the dictionary are formed by concatenating the prefix with the index of the array.
    Docstring is generated by GitHub Copilot."""
    array_dict = dict()
    
    for prefix_ix, prefix in enumerate(prefixes):
        for ix, array_i in enumerate(arrays[prefix_ix]):
            key = prefix + str(ix+1)
            array_dict[key] = array_i
            
    return array_dict


def unwrap_rf_parameters(parameters):
    """
    Unwraps the parameters from a dictionary into two lists: one for the coefficients and one for the poles.

    Parameters
    ----------
    parameters : dict
        Dictionary containing the parameters with keys starting with 'a' for coefficients and 'd' for poles.
    
    Returns
    -------
    tuple
        tuple containing two lists: one for the coefficients and one for the poles.
    
    Notes
    -----
    Docstring is generated by GitHub Copilot.
    """

    keys = list(parameters.keys())
    a_ixs = np.where([word.startswith('a') for word in keys])[0]
    d_ixs  = np.where([word.startswith('d') for word in keys])[0]
    a_nums = np.array([int(string.split('a')[1]) for string in np.array(keys)[a_ixs]])
    d_nums = np.array([int(string.split('d')[1]) for string in np.array(keys)[d_ixs]])
    
    a = [np.zeros([3,3])]*(max(a_nums))
    d = [0]*(max(d_nums))
    
    for a_num in a_nums:
        a[a_num-1] = np.array(parameters['a%i' %a_num])
    
    for d_num in d_nums:
        d[d_num-1] = parameters['d%i' %d_num]
    
    d = np.array(d)
    return a,d


def normal_wind(T_g2wi, T_g2el, U=1.0):
    """
    Computes the wind speed in the local coordinates of the element.

    Parameters
    ----------
    T_g2wi : np.ndarray
        Transformation matrix from global to wind coordinates.
    T_g2el : np.ndarray 
        Transformation matrix from global to element coordinates.
    U : float, optional
        Wind speed in the global coordinates. Default is 1.0.
        
    Returns
    -------
    np.ndarray
        Wind speed in the local coordinates of the element.
    """

    T_wi2el = T_g2el @ T_g2wi.T
    e_wind_local = (T_wi2el @ np.array([1, 0, 0])[np.newaxis,:].T).flatten()
    
    Un = U * np.sqrt(e_wind_local[1]**2+e_wind_local[2]**2)
    return Un


def el_mat_generic(Ayy,Ayz,Ayt,Azy,Azz,Azt,Aty,Atz,Att,L):
    mat = np.zeros([12,12])

    mat[0:6, 0:6] = np.array([
        [0,         0,          0,          0,          0,              0           ],
        [0,         156*Ayy,    156*Ayz,    147*Ayt,    -22*L*Ayz,      22*L*Ayy    ],
        [0,         156*Azy,    156*Azz,    147*Azt,    -22*L*Azz,      22*L*Azy    ],
        [0,         147*Aty,    147*Atz,    140*Att,     -21*L*Atz,      21*L*Aty   ],
        [0,         -22*L*Azy,  -22*L*Azz,  -21*L*Azt,  4*L**2*Azz,     -4*L**2*Azy ],
        [0,         22*L*Ayy,   22*L*Ayz,   21*L*Ayt,   -4*L**2*Ayz,    4*L**2*Ayy  ],
    ])

    mat[0:6, 6:12] = np.array([
        [0,         0,          0,          0,          0,              0            ],
        [0,         54*Ayy,    54*Ayz,      63*Ayt,     13*L*Ayz,       -13*L*Ayy    ],
        [0,         54*Azy,    54*Azz,      63*Azt,     13*L*Azz,       -13*L*Azy    ],
        [0,         63*Aty,    63*Atz,      70*Att,     14*L*Atz,       -14*L*Aty    ],
        [0,         -13*L*Azy,  -13*L*Azz,  -14*L*Azt,  -3*L**2*Azz,     3*L**2*Azy  ],
        [0,         13*L*Ayy,   13*L*Ayz,   14*L*Ayt,   3*L**2*Ayz,     -3*L**2*Ayy  ],
    ])

    mat[6:12, 0:6] = np.array([
        [0,         0,          0,          0,          0,              0            ],
        [0,         54*Ayy,    54*Ayz,      63*Ayt,     -13*L*Ayz,       13*L*Ayy    ],
        [0,         54*Azy,    54*Azz,      63*Azt,     -13*L*Azz,       13*L*Azy    ],
        [0,         63*Aty,    63*Atz,      70*Att,     -14*L*Atz,       14*L*Aty    ],
        [0,         13*L*Azy,  13*L*Azz,    14*L*Azt,   -3*L**2*Azz,     3*L**2*Azy  ],
        [0,         -13*L*Ayy, -13*L*Ayz,   -14*L*Ayt,   3*L**2*Ayz,     -3*L**2*Ayy ],
    ])

    mat[6:12,6:12] = np.array([
        [0,         0,          0,          0,          0,              0               ],
        [0,         156*Ayy,    156*Ayz,    147*Ayt,    22*L*Ayz,      -22*L*Ayy        ],
        [0,         156*Azy,    156*Azz,    147*Azt,    22*L*Azz,      -22*L*Azy        ],
        [0,         147*Aty,    147*Atz,    140*Att,    21*L*Atz,      -21*L*Aty        ],
        [0,         22*L*Azy,   22*L*Azz,   21*L*Azt,    4*L**2*Azz,   -4*L**2*Azy      ],
        [0,         -22*L*Ayy,   -22*L*Ayz,   -21*L*Ayt,   -4*L**2*Ayz,    4*L**2*Ayy   ],
    ])

    return mat

def element_aero_mats(B, omega, ad_dict, L, T=None, phi=None, rho=1.225):
    # Called for selected reduced velocity, specified by omega value (implicitly mean wind).
    # Corresponding values of P,H and A are used for given mean wind velocity.

    # Stiffness
    Ayy = 1/2*rho*B**2*omega**2*ad_dict['P4']
    Ayz = 1/2*rho*B**2*omega**2*ad_dict['P6']
    Ayt = -1/2*rho*B**2*omega**2*B*ad_dict['P3']

    Azy = 1/2*rho*B**2*omega**2*ad_dict['H6']
    Azz = 1/2*rho*B**2*omega**2*ad_dict['H4']
    Azt = -1/2*rho*B**2*omega**2*B*ad_dict['H3']

    Aty = -1/2*rho*B**2*omega**2*B*ad_dict['A6']
    Atz = -1/2*rho*B**2*omega**2*B*ad_dict['A4']
    Att = 1/2*rho*B**2*omega**2*B**2*ad_dict['A3']

    k_aero = L/420 * el_mat_generic(Ayy,Ayz,Ayt,Azy,Azz,Azt,Aty,Atz,Att,L)

    
    # Damping
    Ayy = 1/2*rho*B**2*omega*ad_dict['P1']
    Ayz = 1/2*rho*B**2*omega*ad_dict['P5']
    Ayt = -1/2*rho*B**2*omega*B*ad_dict['P2']

    Azy = 1/2*rho*B**2*omega*ad_dict['H5']
    Azz = 1/2*rho*B**2*omega*ad_dict['H1']    
    Azt = -1/2*rho*B**2*omega*B*ad_dict['H2']
    
    Aty = -1/2*rho*B**2*omega*B*ad_dict['A5']
    Atz = -1/2*rho*B**2*omega*B*ad_dict['A1']
    Att = 1/2*rho*B**2*omega*B**2*ad_dict['A2']

    c_aero = L/420 * el_mat_generic(Ayy,Ayz,Ayt,Azy,Azz,Azt,Aty,Atz,Att,L)

    if (T is None and phi is None)!=True:
        if T is not None:     #if no transformation matrix is given, a local matrix is output
            if np.shape(T)[0]==6:
                T = np.kron(np.eye(2), T)   #two times 6dof matrix, block diagonal
            if np.shape(T)[0]==3:
                T = np.kron(np.eye(4), T)   #four times 3dof matrix, block diagonal
            elif np.shape(T)[0]!=12:
                raise ValueError('Wrong size of T (should be 3x3, 6x6 or 12x12')
        else:
            T = np.eye(12)
        
        if phi is not None:
            T = T @ phi
            
        k_aero = T.T @ k_aero @ T
        c_aero = T.T @ c_aero @ T
    
    return k_aero, c_aero


# Spectra
def kaimal_auto(omega, Lx, A, sigma, V):
    """
    Kaimal auto-spectral density function.

    Parameters
    ----------
    omega : float

        Angular frequency.
    Lx : float
        Length scale.
    A : float
        Amplitude.
    sigma : float
        Standard deviation of the turbulence.
    V : float   
        Mean wind speed.

    Returns
    -------
    S : float
        Auto-spectral density function value.
    """

    f = omega/(2*np.pi)
    fhat = f*Lx/V
    S = (sigma**2*(A*fhat)/(1+(1.5*A*fhat))**(5/3))/f

    return S/(2*np.pi)

def von_karman_auto(omega, Lx, sigma, V):
    """
    Von Karman auto-spectral density function.
    
    Parameters
    ----------
    omega : float
        Angular frequency.
    Lx : float
        Length scale.
    sigma : float
        Standard deviation of the turbulence.
    V : float
        Mean wind speed.
    Returns
    -------
    S : float
        Auto-spectral density function value.
    """
   
    A1 = [
        0.0,
        0.0,
        755.2,
        ]
    
    A2 = [
        70.8,
        0.0,
        283.2,
        ]
    
    rr = [
        5/6,
        11/6,
        11/6,
        ]
    
    f = omega/(2*np.pi)
    fhat = f*Lx/V
    S = (sigma**2*( (4*fhat)*(1+A1*fhat**2)  )/ (1+A2*fhat**2)**(rr))/f
    
    return S/(2*np.pi)

def generic_kaimal_matrix(omega, nodes, T_wind, A, sigma, C, Lx, U, spectrum_type='kaimal'):
    """
    Computes the cross-spectral density matrix for a given set of nodes.
    
    Parameters
    ----------
    omega : float

        Angular frequency.
    nodes : list
        List of BEEF nodes for which to compute the cross-spectral density matrix.
    T_wind : np.ndarray
        Transformation matrix from global to wind coordinates.
    A : float or np.ndarray
        Parameter(s) A in Kaimal auto-spectral density.
    sigma : float or np.ndarray
        Standard deviation of the turbulence (either single or all components).
    C : float or np.ndarray  
        Coefficient(s) for the cross-spectral density matrix.
    LX : float or np.ndarray
        Length scale(s) for the cross-spectral density matrix.
    U : function
        Function (of x, y and z) to calculate mean wind speed at a given point.
    spectrum_type : str, optional
        Type of spectrum to use. Default is 'kaimal'. Other options are 'vonKarman' or 'Karman'.

    """

    # Adopted from MATLAB version. `nodes` is list with beef-nodes.        
    V = np.zeros(len(nodes))      # Initialize vector with mean wind in all nodes
    Su = np.zeros([len(nodes), len(nodes)])     # One-point spectra for u component in all nodes
    Sv = np.zeros([len(nodes), len(nodes)])     # One-point spectra for v component in all nodes
    Sw = np.zeros([len(nodes), len(nodes)])     # One-point spectra for w component in all nodes
    xyz = np.zeros([len(nodes), 3])  # Nodes in wind coordinate system

    for node_ix, node in enumerate(nodes):
        xyz[node_ix,:] = (T_wind @ node.coordinates).T #Transform node coordinates to the wind coordinate system
        V[node_ix] = U(node.coordinates) # Mean wind velocity in the nodes

        if 'karman' in spectrum_type.lower():
            Su[node_ix,:], Sv[node_ix,:], Sw[node_ix,:] = von_karman_auto(omega, Lx, sigma, V[node_ix]) 
        elif spectrum_type.lower() == 'kaimal':
            Su[node_ix,:], Sv[node_ix,:], Sw[node_ix,:] = kaimal_auto(omega, Lx, A, sigma, V[node_ix]) # One point spectra for u component in all nodes
        else:
            raise ValueError('spectrum_type must either be defined as "vonKarman"/"Karman" or "Kaimal"')

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    dxdx = x[np.newaxis,:] - x[np.newaxis,:].T # Matrix with all distances between nodes in x direction
    dydy = y[np.newaxis,:] - y[np.newaxis,:].T # Matrix with all distances between nodes in y direction
    dzdz = z[np.newaxis,:] - z[np.newaxis,:].T # Matrix with all distances between nodes in z direction
    
    invV = 2/(V[np.newaxis,:]+V[np.newaxis,:].T) # Inverse mean wind velocity for all combination of nodes

    Suu = np.sqrt(Su)*np.sqrt(Su).T*np.exp(
            -invV*omega/(2*np.pi)*np.sqrt(
                (C[0,0]*dxdx)**2 + (C[1,0]*dydy)**2 + (C[2,0]*dzdz)**2)
        )
    
    Svv = np.sqrt(Sv)*np.sqrt(Sv).T*np.exp(
            -invV*omega/(2*np.pi)*np.sqrt(         
                (C[0,1]*dxdx)**2 + (C[1,1]*dydy)**2 + (C[2,1]*dzdz)**2)
        )
    
    Sww = np.sqrt(Sw)*np.sqrt(Sw).T*np.exp(
            -invV*omega/(2*np.pi)*np.sqrt( 
                (C[0,2]*dxdx)**2 + (C[1,2]*dydy)**2 + (C[2,2]*dzdz)**2)
        )
    
    SvSv = np.zeros([3*len(nodes), 3*len(nodes)]) # Cross sectral density matrix containing all the turbulence components
    SvSv[0::3, 0::3] = Suu
    SvSv[1::3, 1::3] = Svv
    SvSv[2::3, 2::3] = Sww
    
    return SvSv


def loadmatrix_fe(V, load_coefficients, rho, B, D, admittance=None):
    """
    Computes the aerodynamic load matrix for a given set of parameters.
    
    Parameters
    ----------
    V : float
        Wind speed.
    load_coefficients : dict
        Dictionary containing the aerodynamic coefficients.
    rho : float
        Air density.
    B : float
        Width of the section.
    D : float
        Height of the section.
    admittance : function, optional
        Function to compute the admittance function. Default is None.
        
    Returns
    -------
    BqBq : function
        Function to compute the aerodynamic load matrix for a given frequency.
    """

    if admittance is None :
        admittance = lambda omega_k: np.ones( (4,3) )

    Cd = load_coefficients['Cd']
    dCd = load_coefficients['dCd']
    Cl = load_coefficients['Cl']
    dCl = load_coefficients['dCl']
    Cm = load_coefficients['Cm']
    dCm = load_coefficients['dCm']
    
    # Equation 7 from Oiseth, 2010
    BqBq = lambda omega_k: 1/2*rho*V*B*admittance(omega_k*B/V/2/np.pi)*np.array([[0,            0,          0],
                                                                                [0,     2*D/B*Cd,          (D/B*dCd-Cl)],
                                                                                [0,         2*Cl,          (dCl+D/B*Cd)],
                                                                                [0,      -2*B*Cm,          -B*dCm]])
                                          
    return BqBq

def loadmatrix_fe_static(V, load_coefficients, rho, B, D ):
    """
    Computes the aerodynamic load matrix for a given set of parameters in static conditions.
    
    Parameters
    ----------
    V : float
        Wind speed.
    load_coefficients : dict
        Dictionary containing the aerodynamic coefficients.
    rho : float
        Air density.
    B : float
        Width of the section.
    D : float
        Height of the section.
    
    Returns
    -------
    BqBq : np.ndarray
        Aerodynamic load matrix.
    """
    
    Cd = load_coefficients['Cd']
    Cl = load_coefficients['Cl']
    Cm = load_coefficients['Cm']
        
    BqBq = 1/2*rho*V**2*B*np.array([[ 0 ],
                                    [ D/B*Cd ],
                                    [ Cl  ],
                                    [ -B*Cm ]])
    return BqBq

def loadvector(T_el, Bq, T_wind, L, static = False):
    """
    Computes load vectors for a given set of parameters.
    
    Parameters
    ----------
    T_el : np.ndarray
        Transformation matrix from global to element coordinates.
    Bq : np.ndarray
        Aerodynamic load matrix.
    T_wind : np.ndarray
        Transformation matrix from global to wind coordinates.
    L : float
        Length of the element.
    static : bool, optional
        If True, compute the load vector for static conditions. Default is False.

    Returns
    -------
    R1 : np.ndarray
        Load vector for element node 1.
    R2 : np.ndarray
        Load vector for element node 2.

    """

    G = np.zeros([12,4])
    G[0,0] = L/2
    G[1,1] = L/2
    G[2,2] = L/2
    G[3,3] = L/2
    G[6,0] = L/2
    G[7,1] = L/2
    G[8,2] = L/2
    G[9,3] = L/2
    G[4,2] = -L**2/12
    G[5,1] = L**2/12
    G[10,2] = L**2/12
    G[11,1] = -L**2/12
 
    # Transform from wind coordinates to local element coordinates 
    
    T = T_el @ T_wind.T   
    
    T_full = blkdiag(T_el, 4)     # Block diagonal - repeated 4 times to transform both trans and rot DOFs at each node (2+2)
    
    # T_full.T transforms L-->G
    if static is False:
        R =  T_full.T @ G @ Bq @ T 
    else:
        R =  T_full.T @ G @ Bq 
        
    R1 =  R[0:6]         # Element node 1
    R2 =  R[6:12]        # Element node 2


    return R1, R2


def windaction(omega, S, load_coefficients, elements, T_wind, 
               phi, B, D, U, omega_reduced=None, rho=1.225, print_progress=True,
               section_lookup=None, nodes=None, admittance=None):
    """
    Computes the wind cross-spectral density matrix on a set of elements using the specified parameters.
    
    Parameters
    ----------
    omega : np.ndarray
        Angular frequency.
    S : function
        Function to compute the cross-spectral density matrix.
    load_coefficients : dict
        Dictionary containing the aerodynamic coefficients.
    elements : list
        List of elements for which to compute the wind cross-spectral density matrix.
    T_wind : np.ndarray
        Transformation matrix from global to wind coordinates.
    phi : np.ndarray
        Transformation matrix from global to element coordinates.
    B : float  
        Width of the section.
    D : float
        Height of the section.
    U : function
        Function to calculate mean wind speed at a given point.
    omega_reduced : np.ndarray, optional    
        Reduced angular frequency. Default is None.
    rho : float, optional
        Air density. Default is 1.225.
    print_progress : bool, optional
        If True, print progress. Default is True.
    section_lookup : dict, optional
        Dictionary to look up sections for each element. Default is None.
    nodes : list, optional
        List of BEEF nodes for which to compute the wind cross-spectral density matrix. Default is None.
    admittance : function, optional
        Function to compute the admittance function. Default is None.
    
    Returns
    ------- 
    genSqSq_reduced : np.ndarray
        Cross-spectral density matrix for the wind action.

    """
    
    if nodes is None:
        nodes = list(set([a for b in [el.nodes for el in elements] for a in b]))

    n_dofs = 6
    
    # Ensure that first omega value is not 0 when using logspace omega axis
    if omega_reduced is None:
        if np.min(omega) == 0:
            omega_sorted = np.sort(omega)
            omega_start = omega_sorted[1]
        else:
            omega_start = np.min(omega)    
        
        omega_reduced = np.logspace(np.log10(omega_start), np.log10(np.max(omega)), num=50) # A log frequency axis that is used to obtain the cross-spectral density matrix
        
    genSqSq_reduced = np.zeros([phi.shape[1], phi.shape[1], len(omega_reduced)]) # Initialize the cross-spectral density matrix
    
    # Establish RG matrix (common for all freqs)
    

    if section_lookup is None:
        lc_fun = lambda el: load_coefficients
        B_fun = lambda el: B
        D_fun = lambda el: D
        admittance_fun = lambda el: admittance
    else:
        def get_sec(el):
            for key in section_lookup:
                if el in section_lookup[key]:    
                    return key 

        lc_fun = lambda el: load_coefficients[get_sec(el)]
        B_fun = lambda el: B[get_sec(el)]
        D_fun = lambda el: D[get_sec(el)]
    
    if admittance is None: # omit the frequency loop if ADmittance is not included - faster !
        RG = np.zeros([len(nodes)*n_dofs, 3])
        for el in elements:            
            node1_dofs = el.nodes[0].global_dofs
            node2_dofs = el.nodes[1].global_dofs

            mean_wind = U(el.get_cog())         
            Vn = normal_wind(T_wind, el.T0)*mean_wind        # Find the normal wind
            BqBq = loadmatrix_fe(Vn, lc_fun(el), rho, B_fun(el), D_fun(el))
            R1, R2 = loadvector(el.T0, BqBq(1), T_wind, el.L) # Obtain the load vector for each element

            RG[node1_dofs, :] = RG[node1_dofs, :] + R1   # Add the contribution from the element (end 1) to the system
            RG[node2_dofs, :] = RG[node2_dofs, :] + R2   # Add the contribution from the element (end 2) to the system

            # Make block matrix
            RG_block = np.zeros([6*len(nodes), 3*len(nodes)])
            
            for node in nodes:
                ix = node.index
                n = np.r_[6*ix:6*ix+6]
                m = np.r_[3*ix:3*ix+3]
                RG_block[np.ix_(n,m)] = RG[n,:]     #verified with MATLAB version for beam example

            for k, omega_k in enumerate(omega_reduced): 
                if print_progress:
                    pp(k+1, len(omega_reduced), sym='=', postfix=' ESTABLISHING WIND EXCITATION')
                    print('')
                    
                phiT_RG_block = phi.T @ RG_block
                genSqSq_reduced[:, :, k] = phiT_RG_block @ S(omega_k) @ phiT_RG_block.T  # to modal coordinates

    else: # admittance is given - triple loop (the old way, slower)
        admittance_fun = lambda el: admittance[get_sec(el)]

        for k, omega_k in enumerate(omega_reduced): 
            if print_progress:
                pp(k+1, len(omega_reduced), sym='=', postfix=' ESTABLISHING WIND EXCITATION')
                print('')
        
            # Establish RG matrix 
            RG = np.zeros([len(nodes)*n_dofs, 3])
            
            for el in elements:            
                node1_dofs = el.nodes[0].global_dofs
                node2_dofs = el.nodes[1].global_dofs

                mean_wind = U(el.get_cog())         
                Vn = normal_wind(T_wind, el.T0)*mean_wind        # Find the normal wind
                BqBq = loadmatrix_fe(Vn, lc_fun(el), rho, B_fun(el), D_fun(el), admittance=admittance_fun(el))
                R1, R2 = loadvector(el.T0, BqBq(omega_k), T_wind, el.L) # Obtain the load vector for each element

                RG[node1_dofs, :] = RG[node1_dofs, :] + R1   # Add the contribution from the element (end 1) to the system
                RG[node2_dofs, :] = RG[node2_dofs, :] + R2   # Add the contribution from the element (end 2) to the system
                
                
            # Make block matrix
            RG_block = np.zeros([6*len(nodes), 3*len(nodes)])
            
            for node in nodes:
                ix = node.index
                n = np.r_[6*ix:6*ix+6]
                m = np.r_[3*ix:3*ix+3]
                RG_block[np.ix_(n,m)] = RG[n,:]     #verified with MATLAB version for beam example
                
            phiT_RG_block = phi.T @ RG_block
            genSqSq_reduced[:, :, k] = phiT_RG_block @ S(omega_k) @ phiT_RG_block.T  # to modal coordinates


    # Interpolate results to full frequency axis
    genSqSq = interp1d(omega_reduced, genSqSq_reduced, kind='quadratic', axis=2, fill_value=0, bounds_error=False)
    
    return genSqSq

def windaction_static(load_coefficients, elements, T_wind, 
               phi, B, D, U, rho=1.225, print_progress=True,
               section_lookup=None, nodes=None):
    
    """
    Computes the wind load vector for a set of elements using the specified parameters.
    
    Parameters
    ----------
    load_coefficients : dict
        Dictionary containing the aerodynamic coefficients.
    elements : list
        List of BEEF elements for which to compute the wind load vector.
    T_wind : np.ndarray
        Transformation matrix from global to wind coordinates.
    phi : np.ndarray
        Transformation matrix from global to element coordinates.
    B : float
        Width of the section.
    D : float
        Height of the section.
    U : function
        Function to calculate mean wind speed at a given point.
    rho : float, optional
        Air density. Default is 1.225.
    print_progress : bool, optional
        If True, print progress. Default is True.
    section_lookup : dict, optional
        Dictionary to look up sections for each element. Default is None.
    nodes : list, optional
        List of BEEF nodes for which to compute the wind load vector. Default is None.
    
    Returns
    -------
    genF : np.ndarray
        Wind load vector for the specified elements.
        """
    
    if nodes is None:
        nodes = list(set([a for b in [el.nodes for el in elements] for a in b]))

    n_dofs = 6
    
    if section_lookup is None:
        lc_fun = lambda el: load_coefficients
        B_fun = lambda el: B
        D_fun = lambda el: D
    else:
        def get_sec(el):
            for key in section_lookup:
                if el in section_lookup[key]:    
                    return key 

        lc_fun = lambda el: load_coefficients[get_sec(el)]
        B_fun = lambda el: B[get_sec(el)]
        D_fun = lambda el: D[get_sec(el)]
            
        # Establish RG matrix 
        RG = np.zeros([len(nodes)*n_dofs])
        
        for el in elements:            
            node1_dofs = el.nodes[0].global_dofs
            node2_dofs = el.nodes[1].global_dofs

            mean_wind = U(el.get_cog())         
            Vn = normal_wind(T_wind, el.T0)*mean_wind        # Find the normal wind
            BqBq = loadmatrix_fe_static(Vn, lc_fun(el), rho, B_fun(el), D_fun(el))
            R1, R2 = loadvector(el.T0, BqBq, T_wind, el.L, static=True) # Obtain the load vector for each element

            RG[node1_dofs] = RG[node1_dofs] + R1[:,0]   # Add the contribution from the element (end 1) to the system
            RG[node2_dofs] = RG[node2_dofs] + R2[:,0]    # Add the contribution from the element (end 2) to the system
            
        # Make block matrix
        RG_block = np.zeros([6*len(nodes)])
        
        for node in nodes:
            ix = node.index
            n = np.r_[6*ix:6*ix+6]
            RG_block[np.ix_(n)] = RG[n]
            
        genF = phi.T @ RG_block 
    
    return genF

def K_from_ad(ad, V, w, B, rho):
    """
    Compute the aerodynamic stiffness matrix K from aerodynamic derivatives.

    Parameters
    ----------
    ad : dict
        Dictionary containing aerodynamic derivative functions. Expected keys are:
        'P4', 'P6', 'P3', 'H6', 'H4', 'H3', 'A6', 'A4', 'A3'. Each value should be a callable
        that takes a single argument (the reduced velocity `v`).
    V : float
        Wind velocity.
    w : float
        Circular frequency (rad/s). If zero, returns a zero matrix.
    B : float
        Reference length (e.g., bridge deck width).
    rho : float
        Air density.

    Returns
    -------
    k : ndarray of shape (3, 3)
        Aerodynamic stiffness matrix.

    Notes
    -----
    If `w` is zero, the function returns a 3x3 zero matrix.
    Otherwise, the matrix is computed using the provided aerodynamic derivatives evaluated at the reduced velocity.
    """
    
    if w==0:
        k = np.zeros([3,3])
    else:
        v = V / (B*w) # reduced velocity  
        
        k = (0.5*rho*B**2*w**2 * 
                np.vstack([[ad['P4'](v), ad['P6'](v), -B*ad['P3'](v)],
                           [ad['H6'](v), ad['H4'](v), -B*ad['H3'](v)],
                           [-B*ad['A6'](v), -B*ad['A4'](v), B**2*ad['A3'](v)]]))
        
        
    return k


def C_from_ad(ad, V, w, B, rho):
    """
    Compute the aerodynamic damping matrix K from aerodynamic derivatives.

    Parameters
    ----------
    ad : dict
        Dictionary containing aerodynamic derivative functions. Expected keys are: 
        'P1', 'P5', 'P2', 'H5', 'H1', 'H2', 'A5', 'A1', 'A2'. Each value should be a callable
        that takes a single argument (the reduced velocity `v`).
    V : float
        Wind velocity.
    w : float
        Circular frequency (rad/s). If zero, returns a zero matrix.
    B : float
        Reference length (e.g., bridge deck width).
    rho : float
        Air density.

    Returns
    -------
    k : ndarray of shape (3, 3)
        Aerodynamic stiffness matrix.
        
    Notes
    -----
    If `w` is zero, the function returns a 3x3 zero matrix.
    Otherwise, the matrix is computed using the provided aerodynamic derivatives evaluated at the reduced velocity.
    """
    if w==0:
        c = np.zeros([3,3])
    else:
        v = V / (B*w) # reduced velocity    
    
        c = (0.5*rho*B**2*w * 
            np.vstack([[ad['P1'](v), ad['P5'](v), -B*ad['P2'](v)],
                      [ad['H5'](v), ad['H1'](v), -B*ad['H2'](v)],
                      [-B*ad['A5'](v), -B*ad['A1'](v), B**2*ad['A2'](v)]]))  
    
    return c


def phi_aero_sum(mat, phi, x):
    """
    Computes the aerodynamic matrix (damping or stiffness) in modal coordinates.

    Parameters
    ----------
    mat : ndarray
        Aerodynamic matrix in global coordinates.
    phi : ndarray
        Transformation matrix from global to modal coordinates.
    x : ndarray
        Vector of x-coordinates for integration.

    Returns
    -------
    mat : ndarray
        Aerodynamic matrix in modal coordinates.
    """

    n_modes = phi.shape[1]
    n_points = len(x)
    
    mat_int = np.zeros([n_modes, n_modes, n_points])
    
    for p in range(n_points):
        phi_point = phi[p*6+1:p*6+4, :]
        mat_int[:, :, p] = phi_point.T @ mat @ phi_point

    mat = np.trapz(mat_int, x=x, axis=2)

    return mat


def function_sum(fun, const, fun_factor=1):
    """
    Function to sum a function and a constant. The function is multiplied by a factor.
    
    Parameters
    ----------
    fun : callable
        Function to be summed with the constant.
    const : float
        Constant to be summed with the function.
    fun_factor : float, optional
        Factor to multiply the function. Default is 1.
    
    Returns
    -------
    fsum : callable
        Function that computes the sum of the function and the constant.
        
    """
    def fsum(x):
        if fun is None:
            return const
        else:
            return fun(x)*fun_factor + const
    
    return fsum


def get_aero_cont_adfun(ad_dict_fun, V, B, rho, phi, x):
    """
    Computes the aerodynamic matrices in modal coordinates.

    Parameters
    ----------
    ad_dict_fun : callable
        Function that returns the aerodynamic derivatives for a given reduced velocity.
    V : float
        Wind velocity.
    B : float
        Reference length (e.g., bridge deck width).
    rho : float
        Air density.
    phi : ndarray
        Transformation matrix from global to modal coordinates.
    x : ndarray
        Vector of x-coordinates for integration.

    Returns
    ------- 
    K : callable
        Function that computes the aerodynamic stiffness matrix in modal coordinates.
    C : callable
        Function that computes the aerodynamic damping matrix in modal coordinates.
    
    
    """
    def K(w):
        n_modes = phi.shape[1]
        n_points = len(x)
        
        mat_int = np.zeros([n_modes, n_modes, n_points])
        
        for p in range(n_points):
            phi_point = phi[p*6+1:p*6+4, :]
            kae = K_from_ad(ad_dict_fun(x[p]), V, w, B, rho)
            mat_int[:, :, p] = phi_point.T @ kae @ phi_point

        return np.trapz(mat_int, x=x, axis=2)
    
    
    def C(w):
        n_modes = phi.shape[1]
        n_points = len(x)
        
        mat_int = np.zeros([n_modes, n_modes, n_points])
        
        for p in range(n_points):
            phi_point = phi[p*6+1:p*6+4, :]
            kae = C_from_ad(ad_dict_fun(x[p]), V, w, B, rho)
            mat_int[:, :, p] = phi_point.T @ kae @ phi_point

        return np.trapz(mat_int, x=x, axis=2)
    
        
    return K, C


def get_aero_cont_addict(ad_dict, V, B, rho, phi, x):
    """ 
    Computes the aerodynamic matrices in modal coordinates.

    Parameters
    ----------
    ad_dict : dict
        Dictionary containing aerodynamic derivative functions. Expected keys are:
        'P4', 'P6', 'P3', 'H6', 'H4', 'H3', 'A6', 'A4', 'A3'. Each value should be a callable
        that takes a single argument (the reduced velocity `v`).
    V : float
        Wind velocity.
    B : float
        Reference length (e.g., bridge deck width).
    rho : float
        Air density.
    phi : ndarray
        Transformation matrix from global to modal coordinates.
    x : ndarray
        Vector of x-coordinates for integration.

    Returns
    -------
    K : callable
        Function that computes the aerodynamic stiffness matrix in modal coordinates.
    C : callable
        Function that computes the aerodynamic damping matrix in modal coordinates.
    
    
    """
    def K(w):
        kae = K_from_ad(ad_dict, V, w, B, rho)
        return phi_aero_sum(kae, phi, x)
    
    def C(w):
        cae = C_from_ad(ad_dict, V, w, B, rho)
        return phi_aero_sum(cae, phi, x)
        
    return K, C


def itflutter_cont(Ms, Cs, Ks, phi, x, ad_dict, B, V=0.0, rho=1.225, dV=1, 
              overshoot_factor=0.5, itmax={}, omega_ref=None,
              tol={}, print_progress=True, keep_all=False, track_by_psi=True):
    
    """
    Iterative flutter analysis for continuous systems using the iterative eigensolver.
    
    Parameters
    ----------
    Ms : ndarray
        Mass matrix for in-vacuum and dry sub-structure.
    Cs : ndarray
        Damping matrix for in-vacuum and dry sub-structure.
    Ks : ndarray
        Stiffness matrix for in-vacuum and dry sub-structure.
    phi : ndarray
        Transformation matrix from global to modal coordinates.
    x : ndarray
        Vector of x-coordinates for integration.
    ad_dict : dict or callable
        Dictionary containing aerodynamic derivative functions or a function that returns the aerodynamic derivatives.
        If a dictionary, it should contain the following keys: 'P4', 'P6', 'P3', 'H6', 'H4', 'H3', 'A6', 'A4', 'A3'.
        Each value should be a callable that takes a single argument (the reduced velocity `v`).
    B : float
        Reference length (e.g., bridge deck width).
    V : float, optional
        Initial wind velocity. Default is 0.0.
    rho : float, optional
        Air density. Default is 1.225.
    dV : float, optional
        Increment for wind velocity. Default is 1.
    overshoot_factor : float, optional
        Factor to reduce the increment for wind velocity in case of overshooting. Default is 0.5.
    itmax : dict, optional
        Dictionary containing maximum iterations for velocity and frequency. Default is {'V': 50, 'f': 15}.
    omega_ref : float, optional
        Reference angular frequency. If None, it is computed from the eigenvalues of the system.
    tol : dict, optional
        Dictionary containing tolerance for velocity and frequency. Default is {'V': 1e-3, 'f': 1e-4}.
    print_progress : bool, optional
        If True, print progress. Default is True.
    keep_all : bool, optional
        If True, keep all results. Default is False.
    track_by_psi : bool, optional
        If True, track the eigenvectors by their similarity. Default is True.

    Returns
    -------
    res : dict
        Dictionary containing the results of the flutter analysis. The keys are:
    'V' : list of wind velocities at which flutter was detected.
    'lambd' : list of eigenvalues at which flutter was detected.
    'critical_mode' : list of indices of the critical mode at each velocity.
    'critical_psi' : list of eigenvectors corresponding to the critical mode at each velocity.
    """
    
    if callable(ad_dict):
        get_aero = get_aero_cont_adfun
    else:
        get_aero = get_aero_cont_addict

    itmax_ = {'V':50, 'f': 15}
    itmax_.update(**itmax)
    itmax = dict(itmax_)

    tol_ = {'V': 1e-3, 'f': 1e-4}
    tol_.update(**tol)
    tol = tol_
    
    res = dict()
    res['V'] = []
    res['lambd'] = []
    res['critical_mode'] = []
    res['critical_psi'] = []
    
    converged = False
    psi_prev = None
    
    if omega_ref is None:
        A = statespace(Ks, Cs, Ms)
        lambd_ref, psi = np.linalg.eig(A)  
        omega_initial = np.sort(np.abs(np.imag(lambd_ref)))[::2]
        omega_ref = omega_initial[0]
        
    for it_vel in range(itmax['V']):
        Kae, Cae = get_aero(ad_dict, V, B, rho, phi, x)
        getK = function_sum(Kae, Ks, fun_factor=-1)
        getC = function_sum(Cae, Cs, fun_factor=-1)
        getM = function_sum(None, Ms, fun_factor=-1)

        lambd, psi, not_converged = iteig(getK, getC, getM, tol=tol['f'], 
                                          keep_full=True, mac_min=0.0, itmax=itmax['f'])

        if len(not_converged)>0:
            lambd[not_converged] = -np.inf + 0j
            if print_progress:
                if len(not_converged)<10:
                    nc_modes = 'index '+ ', '.join([str(i) for i in not_converged])
                else:
                    nc_modes = '>10'
                print(f'** Non-converged modes ({nc_modes}) from iterative eigensolution disregarded! **')
                
        if it_vel!=0 and track_by_psi:
            ixs, __, __, __ = restructure_as_ref(psi_prev, psi)
            
            psi = psi[:, ixs]
            lambd = lambd[ixs] 
        
        psi_prev = psi*1
            
        critical_mode = np.argmax(np.real(lambd))
        real_lambd = np.max(np.real(lambd)) 
        critical_omega = np.abs(np.imag(lambd[critical_mode]))
        
        if keep_all or real_lambd<=0:
            res['critical_mode'].append(critical_mode)
            res['lambd'].append(lambd)
            res['V'].append(V)
            res['critical_psi'].append(psi[:,critical_mode])
        
        if dV < tol['V'] and real_lambd<=0:
            converged = True
            if print_progress:
                print(conv_text)
                print(f'Flutter estimated to occur at V = {V:.2f} m/s ({critical_omega:.2f} rad/s) ==>  v = {V/(B*critical_omega):.2f})\n')
            
            break
        elif real_lambd<0:
            if print_progress:
                print(f'Increasing velocity V = {V:.2f} --> {V+dV:.2f}.')
            V = V + dV
        else:
            if print_progress:
                print(f'Overshot. Reducing velocity V = {V:.2f} --> {V-dV/2:.2f}. Reducing step size dV = {dV:.2f} --> {dV/2:.2f}')
                
            dV = overshoot_factor*dV  # adjusting the velocity increment, and step backwards   
            V = V - dV

    if not converged and print_progress:
        print('Not able to converge within specified maximum iterations for specified tolerance criteria.')
    
    res = {key: np.array(res[key]) for key in ['critical_mode', 'critical_psi', 'V', 'lambd']}
    
    return res



def itflutter_cont_naive(Ms, Cs, Ks, phi, x, ad_dict, B, V=0.0, rho=1.225, dV=1, 
              overshoot_factor=0.5, itmax={}, tol={}, print_progress=True):
    """
    Iterative flutter analysis for continuous systems using a naive iteration.
    
    Parameters
    ----------
    Ms : ndarray
        Mass matrix for in-vacuum and dry sub-structure.
    Cs : ndarray
        Damping matrix for in-vacuum and dry sub-structure.
    Ks : ndarray
        Stiffness matrix for in-vacuum and dry sub-structure.
    phi : ndarray
        Transformation matrix from global to modal coordinates.
    x : ndarray
        Vector of x-coordinates for integration.
    ad_dict : dict or callable
        Dictionary containing aerodynamic derivative functions or a function that returns the aerodynamic derivatives.
        If a dictionary, it should contain the following keys: 'P4', 'P6', 'P3', 'H6', 'H4', 'H3', 'A6', 'A4', 'A3'.
        Each value should be a callable that takes a single argument (the reduced velocity `v`).
    B : float
        Reference length (e.g., bridge deck width).
    V : float, optional 
        Initial wind velocity. Default is 0.0.
    rho : float, optional   
        Air density. Default is 1.225.
    dV : float, optional
        Increment for wind velocity. Default is 1.
    overshoot_factor : float, optional
        Factor to reduce the increment for wind velocity in case of overshooting. Default is 0.5.
    itmax : dict, optional
        Dictionary containing maximum iterations for velocity and frequency. Default is {'V': 50, 'f': 15}.
    tol : dict, optional
        Dictionary containing tolerance for velocity and frequency. Default is {'V': 1e-3, 'f': 1e-4}.
    print_progress : bool, optional
        If True, print progress. Default is True.

    Returns
    ------- 
    res : dict
        Dictionary containing the results of the flutter analysis. The keys are:
    'V' : list of wind velocities at which flutter was detected.
    'lambd' : list of eigenvalues at which flutter was detected.
    'critical_mode' : list of indices of the critical mode at each velocity.
    'critical_psi' : list of eigenvectors corresponding to the critical mode at each velocity.

    """
        
    if callable(ad_dict):
        get_aero = get_aero_cont_adfun
    else:
        get_aero = get_aero_cont_addict
    
    itmax_ = {'V':50, 'f': 15}
    itmax_.update(**itmax)
    itmax = itmax_
    
    tol_ = {'V': 1e-3, 'f': 1e-4}
    tol_.update(**tol)
    tol = tol_
    
    res = dict()
    res['V'] = []
    res['lambd'] = []
    res['critical_mode'] = []
    res['critical_psi'] = []
    
    converged = False
        
    for it_vel in range(itmax['V']):
        Kae, Cae = get_aero(ad_dict, V, B, rho, phi, x)
        getK = function_sum(Kae, Ks, fun_factor=-1)
        getC = function_sum(Cae, Cs, fun_factor=-1)
        getM = function_sum(None, Ms, fun_factor=-1)

        lambd, psi = iteig_naive(getK, getC, getM, tol=tol['f'], itmax=itmax['f'])
        
        complex_ix = np.imag(lambd) != 0
        
        critical_mode = np.argmax(np.real(lambd[complex_ix]))
        critical_mode = np.where(complex_ix)[0][critical_mode]
        
        real_lambd = np.max(np.real(lambd)) 
        critical_omega = np.abs(np.imag(lambd[critical_mode]))
        
        if real_lambd<=0:
            res['critical_mode'].append(critical_mode)
            res['lambd'].append(lambd)
            res['V'].append(V)
            res['critical_psi'].append(psi[:,critical_mode])
        
        if dV < tol['V'] and real_lambd<=0:
            
            converged = True
            if print_progress:

                print(conv_text)
                print(f'Flutter estimated to occur at V = {V:.2f} m/s ({critical_omega:.2f} rad/s) ==>  v = {V/(B*critical_omega):.2f})\n')
            
            break
        elif real_lambd<=0:
            if print_progress:
                print(f'Increasing velocity V = {V:.2f} --> {V+dV:.2f}.')
            V = V + dV
        else:
            if print_progress:
                print(f'Overshot. Reducing velocity V = {V:.2f} --> {V-dV/2:.2f}. Reducing step size dV = {dV:.2f} --> {dV/2:.2f}')
                
            dV = overshoot_factor*dV  # adjusting the velocity increment, and step backwards   
            V = V - dV

    if not converged and print_progress:
        print('Not able to converge within specified maximum iterations for specified tolerance criteria.')
    
    res = {key: np.array(res[key]) for key in ['critical_mode', 'critical_psi', 'V', 'lambd']}
    
    return res