import numpy as np

'''
DRY MODES SUBMODULE
'''

class ModalDry:
    def __init__(self, phi, phi_x=dict(), local_phi=False, k=None, omega_n=None, m=1.0, xi0=0.0, n_modes=None, m_min=0.0):
        
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
        if hasattr(self, 'm_min'):
            return np.where((self._m>self.m_min)[:self.n_modes])[0]
        else:
            return np.arange(0,self.n_modes)

    @property
    def omega_n(self):
        return (self.k/self.m)**0.5
        
    @property
    def wn(self):
        return self.omega_n
    
    @property
    def omega_d(self):
        return np.sqrt(1 - self.xi0) * self.omega_n
    
    @property
    def wd(self):
        return self.omega_d
        
    @property
    def fn(self):
        return self.omega_n/2/np.pi
    
    @property
    def Tn(self):
        return 2*np.pi/self.omega_n
    
    @property
    def fd(self):
        return self.omega_d/2/np.pi
    
    @property
    def Td(self):
        return 2*np.pi/self.omega_d
    

    @property
    def n_modes(self):
        if self._n_modes is None:
            return list(self.phi_full.values())[0].shape[1]
        else:
            return self._n_modes

    @n_modes.setter
    def n_modes(self, n):
        self._n_modes = n
        
    @property
    def xi0(self):
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
        self._xi0 = xi0
        
    @property
    def m(self):
        return self._m[self.mode_ix]
    
    @m.setter
    def m(self, m):
        self._m = m
    
    @property
    def k(self):
        return self._k[self.mode_ix]

    @k.setter
    def k(self, k):
        self._k = k
        
    def get_phi(self, key='full', use_n_modes=True):
        if use_n_modes:
            return self.phi_full[key][:, self.mode_ix]
        else:
            return self.phi_full[key]
    
    @property
    def K(self):
        return np.diag(self.k)

    @property
    def C(self):
        return (2*np.sqrt(self.K*self.M)*np.diag(self.xi0))

    @property
    def M(self):
        return np.diag(self.m)