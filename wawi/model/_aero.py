import numpy as np
import json
from wawi.wind import quasisteady_ads, ADs, compute_aero_matrices, generic_kaimal_matrix
from wawi.general import fun_scale, fun_sum
from beef.rotation import rodrot

'''
AERO SUBMODULE
'''
class Aero:
    '''A class for handling aerodynamic calculations.
    
    Attributes
    ----------
    sections : dict, optional
        A dictionary of AeroSection objects. The default is None.
    phi_key : str, optional
        Key for accessing the mode shapes. The default is 'full'.
    element_assignments : dict, optional
        A dictionary assigning elements to sections. The default is None.
    windstate : Windstate, optional
        A Windstate object defining the wind conditions. The default is None.
    phi : numpy.ndarray, optional
        Mode shapes. The default is None.
    phi_ixs : dict, optional
        Indices for accessing mode shapes. The default is an empty dictionary.
    eldef : dict, optional
        Element definitions. The default is an empty dictionary.
    Kfun : callable, optional
        Function for calculating the aerodynamic stiffness matrix. The default is None.
    Cfun : callable, optional
        Function for calculating the aerodynamic damping matrix. The default is None.
    Sqq_aero : numpy.ndarray, optional
        Aerodynamic power spectral density matrix. The default is None.
    '''
    def __init__(self, sections=None, phi_key='full', element_assignments=None, windstate=None):
        '''
        Initializes the Aero class.

        Parameters
        ----------
        sections : dict, optional
            A dictionary of AeroSection objects. The default is None.
        phi_key : str, optional
            Key for accessing the mode shapes. The default is 'full'.
        element_assignments : dict, optional
            A dictionary assigning elements to sections. The default is None.
        windstate : Windstate, optional
            A Windstate object defining the wind conditions. The default is None.
        '''
        self.sections = sections
        self.elements = ensure_list_of_ints(element_assignments)
        self.phi_key = phi_key
        self.phi = None
        self.phi_ixs = dict()
        self.eldef = dict()

        self.windstate = windstate
        
        self.Kfun = None
        self.Cfun = None

    def get_phi(self, group):
        '''
        Returns the mode shapes for a given group.

        Parameters
        ----------
        group : str
            The group for which to retrieve the mode shapes.

        Returns
        -------
        numpy.ndarray
            The mode shapes for the specified group.
        '''
        return self.phi[self.phi_ixs[group], :]

    @property
    def windstate(self):
        '''
        Returns the current windstate.

        Returns
        -------
        Windstate
            The current windstate.
        '''
        return self._windstate
    
    @windstate.setter
    def windstate(self, val):
        '''
        Sets the windstate and resets the aerodynamic stiffness and damping functions.

        Parameters
        ----------
        val : Windstate
            The new windstate to set.
        '''
        self._windstate = val
        self.Cfun = None
        self.Kfun = None
        self.Sqq_aero = None

    @property
    def K(self):
        '''
        Returns the aerodynamic stiffness matrix.

        Returns
        -------
        numpy.ndarray or float
            The aerodynamic stiffness matrix, or 0.0 if not defined.
        '''
        if self.Kfun is None:
            return 0.0
        else:
            return self.Kfun
        
    @property
    def C(self):
        '''
        Returns the aerodynamic damping matrix.

        Returns
        -------
        numpy.ndarray or float
            The aerodynamic damping matrix, or 0.0 if not defined.
        '''
        if self.Cfun is None:
            return 0.0
        else:
            return self.Cfun
               
    def get_generic_kaimal(self, nodes=None, group=None):
        '''
        Returns a function for calculating the generic Kaimal matrix.

        Parameters
        ----------
        nodes : list, optional
            List of node indices. The default is None.
        group : str, optional
            The group for which to calculate the Kaimal matrix. The default is None.

        Returns
        -------
        callable
            A function that takes frequency as input and returns the generic Kaimal matrix.

        Raises
        ------
        ValueError
            If neither nodes nor group is provided.
        '''
        if (nodes is None) and (group is None):
            raise ValueError('Input either nodes or group!')
        elif group is not None:
            nodes = self.eldef[group].nodes           

        return lambda om: generic_kaimal_matrix(om, nodes, self.windstate.T, self.windstate.A, 
                                                self.windstate.sigma, self.windstate.C, self.windstate.Lx, self.windstate.U, spectrum_type=self.windstate.spectrum_type)

    

    def get_aero_matrices(self, omega_reduced=None, aero_sections=None, print_progress=False):
        '''
        Computes the aerodynamic stiffness and damping matrices.

        Parameters
        ----------
        omega_reduced : numpy.ndarray, optional
            Reduced frequencies. The default is None.
        aero_sections : list, optional
            List of aero section names to include in the calculation. The default is None, which includes all sections.
        print_progress : bool, optional
            Whether to print progress during the calculation. The default is False.

        Returns
        -------
        tuple of callables
            A tuple containing the aerodynamic stiffness and damping functions.
        '''
        if aero_sections is None:
            aero_sections = self.elements.keys()
        
        Cae_m = [None]*len(aero_sections)
        Kae_m = [None]*len(aero_sections)
        
        for ix, sec in enumerate(aero_sections):
            phi = self.get_phi(sec)
            U = self.windstate.U
            AD = self.sections[sec].ADs
            B = self.sections[sec].B
            els = self.elements[sec]
            T_wind = self.windstate.T

            Kae_m[ix], Cae_m[ix] = compute_aero_matrices(U, AD, B, els, T_wind, phi, 
                                     omega_reduced=omega_reduced, print_progress=print_progress, rho=self.windstate.rho)  
            
        return fun_sum(*Kae_m), fun_sum(*Cae_m)
    
    
    def prepare_aero_matrices(self, omega=None, print_progress=False, aero_sections=None):
        '''
        Prepares the aerodynamic stiffness and damping matrices by creating functions.

        Parameters
        ----------
        omega : numpy.ndarray, optional
            Frequencies for the aerodynamic matrices. The default is None.
        print_progress : bool, optional
            Whether to print progress during the calculation. The default is False.
        aero_sections : list, optional
            List of aero section names to include in the calculation. The default is None, which includes all sections.
        '''
        self.Kfun, self.Cfun = self.get_aero_matrices(omega_reduced=omega, 
                                                      print_progress=print_progress, 
                                                      aero_sections=aero_sections)
        

'''
AERO SECTION CLASS
'''
def ensure_list_of_ints(d):
    '''
    Ensures that all values in a dictionary are lists of integers.

    Parameters
    ----------
    d : dict
        The dictionary to process.

    Returns
    -------
    dict
        The dictionary with all values converted to lists of integers.
    '''
    for key in d:
        d[key] = [int(di) for di in d[key]]
    
    return d


class AeroSection:
    '''
    A class defining an aerodynamic section.

    Attributes
    ----------
    D : float, optional
        Diameter of the section. The default is None.
    B : float, optional
        Width of the section. The default is None.
    ADs : ADs, optional
        Object containing aerodynamic coefficients. The default is None.
    Cd : float, optional
        Drag coefficient. The default is 0.0.
    dCd : float, optional
        Derivative of the drag coefficient. The default is 0.0.
    Cm : float, optional
        Moment coefficient. The default is 0.0.
    dCm : float, optional
        Derivative of the moment coefficient. The default is 0.0.
    Cl : float, optional
        Lift coefficient. The default is 0.0.
    dCl : float, optional
        Derivative of the lift coefficient. The default is 0.0.
    admittance : callable, optional
        Admittance function. The default is None.
    quasisteady : bool
        Indicates whether the aerodynamic coefficients are quasisteady.
    '''
    def __init__(self, D=None, B=None, ADs=None, Cd=0.0, dCd=0.0, Cm=0.0, dCm=0.0, Cl=0.0, dCl=0.0, admittance=None):
        '''
        Initializes the AeroSection class.

        Parameters
        ----------
        D : float, optional
            Diameter of the section. The default is None.
        B : float, optional
            Width of the section. The default is None.
        ADs : ADs, optional
            Object containing aerodynamic coefficients. The default is None.
        Cd : float, optional
            Drag coefficient. The default is 0.0.
        dCd : float, optional
            Derivative of the drag coefficient. The default is 0.0.
        Cm : float, optional
            Moment coefficient. The default is 0.0.
        dCm : float, optional
            Derivative of the moment coefficient. The default is 0.0.
        Cl : float, optional
            Lift coefficient. The default is 0.0.
        dCl : float, optional
            Derivative of the lift coefficient. The default is 0.0.
        admittance : callable, optional
            Admittance function. The default is None.
        '''
        self.D = D
        self.B = B
        self.Cd = Cd
        self.dCd = dCd
        self.Cm = Cm
        self.dCm = dCm
        self.Cl = Cl
        self.dCl = dCl
        self.admittance = admittance
        
        if ADs is None:
            self.add_quasisteady_ads()
            self.quasisteady = True
        else:
            self.ADs = ADs
            self.quasisteady = False        
    
    def assign(self, **kwargs):
        '''
        Assigns multiple attributes to the AeroSection object.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments where the key is the attribute name and the value is the attribute value.
        '''
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    @property
    def all_lc(self):
        '''
        Returns all load coefficients as a dictionary.

        Returns
        -------
        dict
            A dictionary containing all load coefficients.
        '''
        keys = ['Cd', 'Cm', 'Cl', 'dCd', 'dCm', 'dCl']
        return {key: getattr(self, key) for key in keys}
        
    def add_quasisteady_ads(self):
        '''
        Adds quasisteady aerodynamic coefficients to the AeroSection object.
        '''
        self.ADs = ADs(**quasisteady_ads(self.D, self.B, self.all_lc), ad_type='quasisteady')

    def __str__(self):
        '''
        Returns a string representation of the AeroSection object.

        Returns
        -------
        str
            A string representation of the AeroSection object.
        '''
        return f"<AeroSection> (D = {self.D}, B = {self.B}, Cd = {self.Cd}, Cd' = {self.dCd}, Cl = {self.Cl}, Cl' = {self.dCl}, Cm = {self.Cm}, Cm' = {self.dCm}, Admittance = {self.admittance})"
        

'''
WIND STATE CLASS
'''

class Windstate:
    '''
    A class defining the wind state.

    Attributes
    ----------
    U0 : float
        Mean wind speed at reference height.
    direction : float
        Wind direction in degrees (clockwise positive).
    Au : float, optional
        Scaling factor for the u-component of turbulence. The default is 0.0.
    Av : float, optional
        Scaling factor for the v-component of turbulence. The default is 0.0.
    Aw : float, optional
        Scaling factor for the w-component of turbulence. The default is 0.0.
    Iu : float, optional
        Turbulence intensity for the u-component. The default is 0.0.
    Iv : float, optional
        Turbulence intensity for the v-component. The default is 0.0.
    Iw : float, optional
        Turbulence intensity for the w-component. The default is 0.0.
    Cuy : float, optional
        Correlation coefficient between u and v. The default is 0.0.
    Cuz : float, optional
        Correlation coefficient between u and w. The default is 0.0.
    Cvy : float, optional
        Correlation coefficient between v and y. The default is 0.0.
    Cvz : float, optional
        Correlation coefficient between v and z. The default is 0.0.
    Cwy : float, optional
        Correlation coefficient between w and y. The default is 0.0.
    Cwz : float, optional
        Correlation coefficient between w and z. The default is 0.0.
    Lux : float, optional
        Turbulence length scale for the u-component. The default is 0.0.
    Lvx : float, optional
        Turbulence length scale for the v-component. The default is 0.0.
    Lwx : float, optional
        Turbulence length scale for the w-component. The default is 0.0.
    x_ref : numpy.ndarray, optional
        Reference coordinates. The default is np.array([0,0,0]).
    scaling : callable, optional
        Scaling function for the wind speed. The default is None.
    name : str, optional
        Name of the wind state. The default is None.
    spectrum_type : str, optional
        Type of turbulence spectrum. The default is 'kaimal'.
    rho : float, optional
        Air density. The default is 1.225.
    options : dict, optional
        Additional options. The default is None.
    '''
    def __init__(self, U0, direction, Au=0.0, Av=0.0, Aw=0.0, 
                 Iu=0.0, Iv=0.0, Iw=0.0, Cuy=0.0, Cuz=0.0, Cvy=0.0, Cvz=0.0, Cwy=0.0, Cwz=0.0, Lux=0.0, Lvx=0.0, Lwx=0.0,                
                 x_ref=np.array([0,0,0]), scaling=None, name=None, spectrum_type='kaimal', rho=1.225, options=None):
        '''
        Initializes the Windstate class.

        Parameters
        ----------
        U0 : float
            Mean wind speed at reference height.
        direction : float
            Wind direction in degrees (clockwise positive).
        Au : float, optional
            Scaling factor for the u-component of turbulence. The default is 0.0.
        Av : float, optional
            Scaling factor for the v-component of turbulence. The default is 0.0.
        Aw : float, optional
            Scaling factor for the w-component of turbulence. The default is 0.0.
        Iu : float, optional
            Turbulence intensity for the u-component. The default is 0.0.
        Iv : float, optional
            Turbulence intensity for the v-component. The default is 0.0.
        Iw : float, optional
            Turbulence intensity for the w-component. The default is 0.0.
        Cuy : float, optional
            Correlation coefficient between u and v. The default is 0.0.
        Cuz : float, optional
            Correlation coefficient between u and w. The default is 0.0.
        Cvy : float, optional
            Correlation coefficient between v and y. The default is 0.0.
        Cvz : float, optional
            Correlation coefficient between v and z. The default is 0.0.
        Cwy : float, optional
            Correlation coefficient between w and y. The default is 0.0.
        Cwz : float, optional
            Correlation coefficient between w and z. The default is 0.0.
        Lux : float, optional
            Turbulence length scale for the u-component. The default is 0.0.
        Lvx : float, optional
            Turbulence length scale for the v-component. The default is 0.0.
        Lwx : float, optional
            Turbulence length scale for the w-component. The default is 0.0.
        x_ref : numpy.ndarray, optional
            Reference coordinates. The default is np.array([0,0,0]).
        scaling : callable, optional
            Scaling function for the wind speed. The default is None.
        name : str, optional
            Name of the wind state. The default is None.
        spectrum_type : str, optional
            Type of turbulence spectrum. The default is 'kaimal'.
        rho : float, optional
            Air density. The default is 1.225.
        options : dict, optional
            Additional options. The default is None.
        '''
        
        self.U0 = U0
        self.direction = direction  # interpreted as positive in clock-wise direction and defines origin and not heading!

        self.A = np.array([Au, Av, Aw])
        self.I = np.array([Iu, Iv, Iw])
        
        self.C = np.array([[0,   0,   0],
                           [Cuy, Cvy, Cwy],
                           [Cuz, Cvz, Cwz]])
        
        self.Lx = np.array([Lux, Lvx, Lwx])
        
        self.spectrum_type = spectrum_type
        self.options = options

        if scaling is None:
            self.scaling = lambda x: 1.0    #{x} = [x,y,z]
        else:
            self.scaling = scaling

        self.x_ref = x_ref
        self.name = name
        self.rho = rho

    def __str__(self):
        '''
        Returns a string representation of the Windstate object.

        Returns
        -------
        str
            A string representation of the Windstate object.
        '''
        string = f'''\
WAWI WindState 
--------------
U={self.U0:.1f}m/s, direction={self.direction:.1f}deg
A=[{self.Au:.2f}, {self.Av:.2f}, {self.Aw:.2f}]
I=[{self.Iu:.2f}, {self.Iv:.2f}, {self.Iw:.2f}]
Cux, Cvx, Cwx = [{self.C[0,0]}, {self.C[0,1]}, {self.C[0,2]}]
Cuy, Cvy, Cwy = [{self.C[1,0]}, {self.C[1,1]}, {self.C[1,2]}]
Cuz, Cvz, Cwz = [{self.C[2,0]}, {self.C[2,1]}, {self.C[2,2]}]
Lx = [{self.Lx[0]:.2f}, {self.Lx[1]:.2f}, {self.Lx[2]:.2f}]
'''
        
        return string

    
    @property
    def Iu(self):
        '''
        Returns the turbulence intensity for the u-component.

        Returns
        -------
        float
            The turbulence intensity for the u-component.
        '''
        return self.I[0]
    @Iu.setter
    def Iu(self, val):
        '''
        Sets the turbulence intensity for the u-component.

        Parameters
        ----------
        val : float
            The new turbulence intensity for the u-component.
        '''
        self.I[0] = val
    
    @property
    def Iv(self):
        '''
        Returns the turbulence intensity for the v-component.

        Returns
        -------
        float
            The turbulence intensity for the v-component.
        '''
        return self.I[1]
    @Iv.setter
    def Iv(self, val):
        '''
        Sets the turbulence intensity for the v-component.

        Parameters
        ----------
        val : float
            The new turbulence intensity for the v-component.
        '''
        self.I[1] = val 

    @property
    def Iw(self):
        '''
        Returns the turbulence intensity for the w-component.

        Returns
        -------
        float
            The turbulence intensity for the w-component.
        '''
        return self.I[2]    
    @Iw.setter
    def Iw(self, val):
        '''
        Sets the turbulence intensity for the w-component.

        Parameters
        ----------
        val : float
            The new turbulence intensity for the w-component.
        '''
        self.I[2] = val   

    @property
    def sigma(self):
        '''
        Returns the standard deviations of the wind speed components.

        Returns
        -------
        numpy.ndarray
            The standard deviations of the wind speed components.
        '''
        return self.I*self.U0

    @property 
    def T(self):
        '''
        Returns the rotation matrix for the wind direction.

        Returns
        -------
        numpy.ndarray
            The rotation matrix for the wind direction.
        '''
        return rodrot((-self.direction+180)*np.pi/180)
    
    @property
    def U(self):
        '''
        Returns the scaled mean wind speed.

        Returns
        -------
        callable
            A function that takes coordinates as input and returns the scaled mean wind speed.
        '''
        return fun_scale(self.scaling, self.U0)

    @property
    def V(self):
        '''
        Alias for U.

        Returns
        -------
        callable
            A function that takes coordinates as input and returns the scaled mean wind speed.
        '''
        return self.U
    
    @property
    def V0(self):
        '''
        Returns the mean wind speed at reference height.

        Returns
        -------
        float
            The mean wind speed at reference height.
        '''
        return self.U0

    @property
    def Au(self):
        '''
        Returns the scaling factor for the u-component of turbulence.

        Returns
        -------
        float
            The scaling factor for the u-component of turbulence.
        '''
        return self.A[0]
    @Au.setter
    def Au(self, val):
        '''
        Sets the scaling factor for the u-component of turbulence.

        Parameters
        ----------
        val : float
            The new scaling factor for the u-component of turbulence.
        '''
        self.A[0] = val
    
    @property
    def Av(self):
        '''
        Returns the scaling factor for the v-component of turbulence.

        Returns
        -------
        float
            The scaling factor for the v-component of turbulence.
        '''
        return self.A[1]
    @Av.setter
    def Av(self, val):
        '''
        Sets the scaling factor for the v-component of turbulence.

        Parameters
        ----------
        val : float
            The new scaling factor for the v-component of turbulence.
        '''
        self.A[1] = val

    @property
    def Aw(self):
        '''
        Returns the scaling factor for the w-component of turbulence.

        Returns
        -------
        float
            The scaling factor for the w-component of turbulence.
        '''
        return self.A[2]
    @Aw.setter
    def Aw(self, val):
        '''
        Sets the scaling factor for the w-component of turbulence.

        Parameters
        ----------
        val : float
            The new scaling factor for the w-component of turbulence.
        '''
        self.A[2] = val
    
    @property
    def sigma_u(self):
        '''
        Returns the standard deviation of the u-component of wind speed.

        Returns
        -------
        float
            The standard deviation of the u-component of wind speed.
        '''
        return self.sigma[0]

    @property
    def sigma_v(self):
        '''
        Returns the standard deviation of the v-component of wind speed.

        Returns
        -------
        float
            The standard deviation of the v-component of wind speed.
        '''
        return self.sigma[1]

    @property
    def sigma_w(self):
        '''
        Returns the standard deviation of the w-component of wind speed.

        Returns
        -------
        float
            The standard deviation of the w-component of wind speed.
        '''
        return self.sigma[2]

    @property
    def Cux(self):
        '''
        Returns the correlation coefficient between u and x.

        Returns
        -------
        float
            The correlation coefficient between u and x.
        '''
        return self.C[0,0]
    @Cux.setter
    def Cux(self, val):
        '''
        Sets the correlation coefficient between u and x.

        Parameters
        ----------
        val : float
            The new correlation coefficient between u and x.
        '''
        self.C[0,0] = val

    @property
    def Cuy(self):
        '''
        Returns the correlation coefficient between u and y.

        Returns
        -------
        float
            The correlation coefficient between u and y.
        '''
        return self.C[1,0]
    @Cuy.setter
    def Cuy(self, val):
        '''
        Sets the correlation coefficient between u and y.

        Parameters
        ----------
        val : float
            The new correlation coefficient between u and y.
        '''
        self.C[1,0] = val
    
    @property
    def Cuz(self):
        '''
        Returns the correlation coefficient between u and z.

        Returns
        -------
        float
            The correlation coefficient between u and z.
        '''
        return self.C[2,0]
    @Cuz.setter
    def Cuz(self, val):
        '''
        Sets the correlation coefficient between u and z.

        Parameters
        ----------
        val : float
            The new correlation coefficient between u and z.
        '''
        self.C[2,0] = val

    @property
    def Cvx(self):
        '''
        Returns the correlation coefficient between v and x.

        Returns
        -------
        float
            The correlation coefficient between v and x.
        '''
        return self.C[0,1]
    @Cvx.setter
    def Cvx(self, val):
        '''
        Sets the correlation coefficient between v and x.

        Parameters
        ----------
        val : float
            The new correlation coefficient between v and x.
        '''
        self.C[0,1] = val

    @property
    def Cvy(self):
        '''
        Returns the correlation coefficient between v and y.

        Returns
        -------
        float
            The correlation coefficient between v and y.
        '''
        return self.C[1,1]
    @Cvy.setter
    def Cvy(self, val):
        '''
        Sets the correlation coefficient between v and y.

        Parameters
        ----------
        val : float
            The new correlation coefficient between v and y.
        '''
        self.C[1,1] = val
        
    @property
    def Cvz(self):
        '''
        Returns the correlation coefficient between v and z.

        Returns
        -------
        float
            The correlation coefficient between v and z.
        '''
        return self.C[2,1]
    @Cvz.setter
    def Cvz(self, val):
        '''
        Sets the correlation coefficient between v and z.

        Parameters
        ----------
        val : float
            The new correlation coefficient between v and z.
        '''
        self.C[2,1] = val

    @property
    def Cwx(self):
        '''
        Returns the correlation coefficient between w and x.

        Returns
        -------
        float
            The correlation coefficient between w and x.
        '''
        return self.C[0,2]
    @Cwx.setter
    def Cwx(self, val):
        '''
        Sets the correlation coefficient between w and x.

        Parameters
        ----------
        val : float
            The new correlation coefficient between w and x.
        '''
        self.C[0,2] = val

    @property
    def Cwy(self):
        '''
        Returns the correlation coefficient between w and y.

        Returns
        -------
        float
            The correlation coefficient between w and y.
        '''
        return self.C[1,2]
    @Cwy.setter
    def Cwy(self, val):
        '''
        Sets the correlation coefficient between w and y.

        Parameters
        ----------
        val : float
            The new correlation coefficient between w and y.
        '''
        self.C[1,2] = val
       
    @property
    def Cwz(self):
        '''
        Returns the correlation coefficient between w and z.

        Returns
        -------
        float
            The correlation coefficient between w and z.
        '''
        return self.C[2,2]
    @Cwz.setter
    def Cwz(self, val):
        '''
        Sets the correlation coefficient between w and z.

        Parameters
        ----------
        val : float
            The new correlation coefficient between w and z.
        '''
        self.C[2,2] = val
    
    @property
    def Lux(self):
        '''
        Returns the turbulence length scale for the u-component.

        Returns
        -------
        float
            The turbulence length scale for the u-component.
        '''
        return self.Lx[0]
    @Lux.setter
    def Lux(self, val):
        '''
        Sets the turbulence length scale for the u-component.

        Parameters
        ----------
        val : float
            The new turbulence length scale for the u-component.
        '''
        self.Lx[0] = val
    
    @property
    def Lvx(self):
        '''
        Returns the turbulence length scale for the v-component.

        Returns
        -------
        float
            The turbulence length scale for the v-component.
        '''
        return self.Lx[1]
    @Lvx.setter
    def Lvx(self, val):
        '''
        Sets the turbulence length scale for the v-component.

        Parameters
        ----------
        val : float
            The new turbulence length scale for the v-component.
        '''
        self.Lx[1] = val
    
    @property
    def Lwx(self):
        '''
        Returns the turbulence length scale for the w-component.

        Returns
        -------
        float
            The turbulence length scale for the w-component.
        '''
        return self.Lx[2]
    @Lwx.setter
    def Lwx(self, val):
        '''
        Sets the turbulence length scale for the w-component.

        Parameters
        ----------
        val : float
            The new turbulence length scale for the w-component.
        '''
        self.Lx[2] = val
    
   # Alternative constructor
    @classmethod
    def from_json(cls, json_file, **kwargs):
        '''
        Creates a Windstate object from a JSON file.

        Parameters
        ----------
        json_file : str
            Path to the JSON file.
        **kwargs : dict
            Additional keyword arguments to pass to the Windstate constructor.

        Returns
        -------
        Windstate
            A Windstate object created from the JSON file.
        '''

        with open(json_file, 'r') as fileobj:
            data = json.load(fileobj)

        direction = data.pop('direction')
        U0 = data.pop('U0')
 
        if 'scaling' in data:
            scaling = eval(data.pop('scaling'))
        else:
            scaling = None
    
        # Update options if provided (to enable overriding options from screening setup)
        if 'options' in data:
            options = data['options']
        else:
            options = {}

        return cls(U0, direction, scaling=scaling, options=options, **data)