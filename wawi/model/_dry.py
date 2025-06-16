import numpy as np

'''
DRY MODES SUBMODULE
'''

class ModalDry:
    """
    Represents a modal system with dry (proportional) damping.

    Parameters
    ----------
    phi : dict
        Dictionary of mode shapes. Keys are identifiers, values are numpy arrays (each column is a mode shape).
    phi_x : dict, optional
        Dictionary of derivatives of mode shapes. Default is empty dict.
    local_phi : bool, optional
        Whether the mode shapes are local. Default is False.
    k : array_like, optional
        Modal stiffnesses. Required if m or omega_n is None.
    omega_n : array_like, optional
        Natural frequencies. Required if m or k is None.
    m : array_like, optional
        Modal masses. Required if k or omega_n is None. Default is 1.0.
    xi0 : float or array_like, optional
        Modal damping ratios. Default is 0.0.
    n_modes : int, optional
        Number of modes to consider. If None, inferred from phi. Default is None.
    m_min : float, optional
        Minimum modal mass to consider. Modes with mass below this are ignored. Default is 0.0.

    Notes
    -----
    This documentation was automatically generated using GitHub Copilot.
    """
    def __init__(self, phi, phi_x=dict(), local_phi=False, k=None, omega_n=None, m=1.0, xi0=0.0, n_modes=None, m_min=0.0):
        """
        Initialize ModalDry object.

        Parameters
        ----------
        phi : dict
            Dictionary of mode shapes.
        phi_x : dict, optional
            Dictionary of derivatives of mode shapes.
        local_phi : bool, optional
            Whether the mode shapes are local.
        k : array_like, optional
            Modal stiffnesses.
        omega_n : array_like, optional
            Natural frequencies.
        m : array_like, optional
            Modal masses.
        xi0 : float or array_like, optional
            Modal damping ratios.
        n_modes : int, optional
            Number of modes to consider.
        m_min : float, optional
            Minimum modal mass to consider.

        Raises
        ------
        ValueError
            If exactly two of m, k, omega_n are not provided.
        """

        self._n_modes = n_modes
        self.phi_full = {key: np.array(phi[key]) for key in phi}
        self.local_phi = local_phi  # is assumed for all relevant phi
        self.phi_x = phi_x
       
        n_none = np.sum([var_i is None for var_i in [m, k, omega_n]])
        if n_none != 1:
            raise ValueError('Exactly two of the variables m, k and omega_n has to be input. This is to ensure consistency and sufficient info. Force m=None if onlyk and omega_n are specified')

        # Variations of input of m,k and omega_n
        if m is not None and omega_n is not None:
            self._k = np.array(m)*np.array(omega_n)**2
            self._m = np.array(m)
        elif k is not None and omega_n is not None:
            self._k = np.array(k)
            self._m = np.array(k)/np.array(omega_n)**2
        else:
            self._k = np.array(k)
            self._m = np.array(m)
        
        self.m_min = m_min
        self._xi0 = xi0

        if self._m is not None and np.ndim(self._m) == 0:
            self._m = np.array([self._m]*self.n_modes)

    @property
    def mode_ix(self):
        """
        Indices of modes to consider, filtered by m_min.

        Returns
        -------
        numpy.ndarray
            Indices of valid modes.
        """
        if hasattr(self, 'm_min'):
            return np.where((self._m>self.m_min)[:self.n_modes])[0]
        else:
            return np.arange(0,self.n_modes)

    @property
    def omega_n(self):
        """
        Natural frequencies of the modes.

        Returns
        -------
        numpy.ndarray
            Natural frequencies.
        """
        return (self.k/self.m)**0.5
        
    @property
    def wn(self):
        """
        Alias for omega_n.

        Returns
        -------
        numpy.ndarray
            Natural frequencies.
        """
        return self.omega_n
    
    @property
    def omega_d(self):
        """
        Damped natural frequencies of the modes.

        Returns
        -------
        numpy.ndarray
            Damped natural frequencies.
        """
        return np.sqrt(1 - self.xi0) * self.omega_n
    
    @property
    def wd(self):
        """
        Alias for omega_d.

        Returns
        -------
        numpy.ndarray
            Damped natural frequencies.
        """
        return self.omega_d
        
    @property
    def fn(self):
        """
        Natural frequencies in Hz.

        Returns
        -------
        numpy.ndarray
            Natural frequencies in Hz.
        """
        return self.omega_n/2/np.pi
    
    @property
    def Tn(self):
        """
        Natural periods of the modes.

        Returns
        -------
        numpy.ndarray
            Natural periods.
        """
        return 2*np.pi/self.omega_n
    
    @property
    def fd(self):
        """
        Damped natural frequencies in Hz.

        Returns
        -------
        numpy.ndarray
            Damped natural frequencies in Hz.
        """
        return self.omega_d/2/np.pi
    
    @property
    def Td(self):
        """
        Damped natural periods of the modes.

        Returns
        -------
        numpy.ndarray
            Damped natural periods.
        """
        return 2*np.pi/self.omega_d
    

    @property
    def n_modes(self):
        """
        Number of modes considered.

        Returns
        -------
        int
            Number of modes.
        """
        if self._n_modes is None:
            return list(self.phi_full.values())[0].shape[1]
        else:
            return self._n_modes

    @n_modes.setter
    def n_modes(self, n):
        """
        Set the number of modes.

        Parameters
        ----------
        n : int
            Number of modes.
        """
        self._n_modes = n
        
    @property
    def xi0(self):
        """
        Modal damping ratios.

        Returns
        -------
        numpy.ndarray
            Modal damping ratios.

        Raises
        ------
        ValueError
            If xi0 does not have a valid length.
        """
        if self._xi0 is None:
            return 0.0
        elif np.ndim(self._xi0) == 0:
            return np.array([self._xi0]*self.n_modes)[self.mode_ix]
        else:
            if len(self._xi0) == len(self._m):   #full model
                return self._xi0[:self.n_modes][self.mode_ix]
            elif len(self._xi0) == self.n_modes:            #truncated
                return self._xi0[self.mode_ix]
            elif len(self._xi0) == len(self.mode_ix):       #truncated & filtered
                return self._xi0
            else:
                raise ValueError('''Specified xi0 must be scalar or with same length as total number of modes, 
                                    number of truncated modes (by n_modes), or filtered number of modes (by n_modes and m_min)
                                    ''')

    @xi0.setter
    def xi0(self, xi0):
        """
        Set modal damping ratios.

        Parameters
        ----------
        xi0 : float or array_like
            Modal damping ratios.
        """
        self._xi0 = xi0
        
    @property
    def m(self):
        """
        Modal masses of the considered modes.

        Returns
        -------
        numpy.ndarray
            Modal masses.
        """
        return self._m[self.mode_ix]
    
    @m.setter
    def m(self, m):
        """
        Set modal masses.

        Parameters
        ----------
        m : array_like
            Modal masses.
        """
        self._m = m
    
    @property
    def k(self):
        """
        Modal stiffnesses of the considered modes.

        Returns
        -------
        numpy.ndarray
            Modal stiffnesses.
        """
        return self._k[self.mode_ix]

    @k.setter
    def k(self, k):
        """
        Set modal stiffnesses.

        Parameters
        ----------
        k : array_like
            Modal stiffnesses.
        """
        self._k = k
        
    def get_phi(self, key='full', use_n_modes=True):
        """
        Get mode shapes for a given key.

        Parameters
        ----------
        key : str, optional
            Key for mode shapes. Default is 'full'.
        use_n_modes : bool, optional
            Whether to truncate to n_modes. Default is True.

        Returns
        -------
        numpy.ndarray
            Mode shapes for the given key.
        """
        if use_n_modes:
            return self.phi_full[key][:, self.mode_ix]
        else:
            return self.phi_full[key]
    
    @property
    def K(self):
        """
        Modal stiffness matrix.

        Returns
        -------
        numpy.ndarray
            Diagonal modal stiffness matrix.
        """
        return np.diag(self.k)

    @property
    def C(self):
        """
        Modal damping matrix.

        Returns
        -------
        numpy.ndarray
            Modal damping matrix.
        """
        return (2*np.sqrt(self.K*self.M)*np.diag(self.xi0))

    @property
    def M(self):
        """
        Modal mass matrix.

        Returns
        -------
        numpy.ndarray
            Diagonal modal mass matrix.
        """
        return np.diag(self.m)