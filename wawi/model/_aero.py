import numpy as np
import json
from wawi.wind import quasisteady_ads, ADs, compute_aero_matrices, generic_kaimal_matrix
from wawi.general import fun_scale, fun_sum
from beef.rotation import rodrot

'''
AERO SUBMODULE
'''
class Aero:
    def __init__(self, sections=None, phi_key='full', element_assignments=None, windstate=None):
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
        return self.phi[self.phi_ixs[group], :]

    @property
    def windstate(self):
        return self._windstate
    
    @windstate.setter
    def windstate(self, val):
        self._windstate = val
        self.Cfun = None
        self.Kfun = None
        self.Sqq_aero = None

    @property
    def K(self):
        if self.Kfun is None:
            return 0.0
        else:
            return self.Kfun
        
    @property
    def C(self):
        if self.Cfun is None:
            return 0.0
        else:
            return self.Cfun
               
    def get_generic_kaimal(self, nodes=None, group=None):
        if (nodes is None) and (group is None):
            raise ValueError('Input either nodes or group!')
        elif group is not None:
            nodes = self.eldef[group].nodes           

        return lambda om: generic_kaimal_matrix(om, nodes, self.windstate.T, self.windstate.A, 
                                                self.windstate.sigma, self.windstate.C, self.windstate.Lx, self.windstate.U, spectrum_type=self.windstate.spectrum_type)

    

    def get_aero_matrices(self, omega_reduced=None, aero_sections=None, print_progress=False):
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
        self.Kfun, self.Cfun = self.get_aero_matrices(omega_reduced=omega, 
                                                      print_progress=print_progress, 
                                                      aero_sections=aero_sections)
        

'''
AERO SECTION CLASS
'''
def ensure_list_of_ints(d):
    for key in d:
        d[key] = [int(di) for di in d[key]]
    
    return d


class AeroSection:
    def __init__(self, D=None, B=None, ADs=None, Cd=0.0, dCd=0.0, Cm=0.0, dCm=0.0, Cl=0.0, dCl=0.0, admittance=None):
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
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    @property
    def all_lc(self):
        keys = ['Cd', 'Cm', 'Cl', 'dCd', 'dCm', 'dCl']
        return {key: getattr(self, key) for key in keys}
        
    def add_quasisteady_ads(self):
        self.ADs = ADs(**quasisteady_ads(self.D, self.B, self.all_lc), ad_type='quasisteady')

    def __str__(self):
        return f"<AeroSection> (D = {self.D}, B = {self.B}, Cd = {self.Cd}, Cd' = {self.dCd}, Cl = {self.Cl}, Cl' = {self.dCl}, Cm = {self.Cm}, Cm' = {self.dCm}, Admittance = {self.admittance})"
        

'''
WIND STATE CLASS
'''

class Windstate:
    def __init__(self, U0, direction, Au=0.0, Av=0.0, Aw=0.0, 
                 Iu=0.0, Iv=0.0, Iw=0.0, Cuy=0.0, Cuz=0.0, Cvy=0.0, Cvz=0.0, Cwy=0.0, Cwz=0.0, Lux=0.0, Lvx=0.0, Lwx=0.0,                
                 x_ref=np.array([0,0,0]), scaling=None, name=None, spectrum_type='kaimal', rho=1.225, options=None):
        
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
        return self.I[0]
    @Iu.setter
    def Iu(self, val):
        self.I[0] = val
    
    @property
    def Iv(self):
        return self.I[1]
    @Iv.setter
    def Iv(self, val):
        self.I[1] = val 

    @property
    def Iw(self):
        return self.I[2]    
    @Iw.setter
    def Iw(self, val):
        self.I[2] = val   

    @property
    def sigma(self):
        return self.I*self.U0

    @property 
    def T(self):
        return rodrot((-self.direction+180)*np.pi/180)
    
    @property
    def U(self):
        return fun_scale(self.scaling, self.U0)

    @property
    def V(self):
        return self.U
    
    @property
    def V0(self):
        return self.U0

    @property
    def Au(self):
        return self.A[0]
    @Au.setter
    def Au(self, val):
        self.A[0] = val
    
    @property
    def Av(self):
        return self.A[1]
    @Av.setter
    def Av(self, val):
        self.A[1] = val

    @property
    def Aw(self):
        return self.A[2]
    @Aw.setter
    def Aw(self, val):
        self.A[2] = val
    
    @property
    def sigma_u(self):
        return self.sigma[0]

    @property
    def sigma_v(self):
        return self.sigma[1]

    @property
    def sigma_w(self):
        return self.sigma[2]

    @property
    def Cux(self):
        return self.C[0,0]
    @Cux.setter
    def Cux(self, val):
        self.C[0,0] = val

    @property
    def Cuy(self):
        return self.C[1,0]
    @Cuy.setter
    def Cuy(self, val):
        self.C[1,0] = val
    
    @property
    def Cuz(self):
        return self.C[2,0]
    @Cuz.setter
    def Cuz(self, val):
        self.C[2,0] = val

    @property
    def Cvx(self):
        return self.C[0,1]
    @Cvx.setter
    def Cvx(self, val):
        self.C[0,1] = val

    @property
    def Cvy(self):
        return self.C[1,1]
    @Cvy.setter
    def Cvy(self, val):
        self.C[1,1] = val
        
    @property
    def Cvz(self):
        return self.C[2,1]
    @Cvz.setter
    def Cvz(self, val):
        self.C[2,1] = val

    @property
    def Cwx(self):
        return self.C[0,2]
    @Cwx.setter
    def Cwx(self, val):
        self.C[0,2] = val

    @property
    def Cwy(self):
        return self.C[1,2]
    @Cwy.setter
    def Cwy(self, val):
        self.C[1,2] = val
       
    @property
    def Cwz(self):
        return self.C[2,2]
    @Cwz.setter
    def Cwz(self, val):
        self.C[2,2] = val
    
    @property
    def Lux(self):
        return self.Lx[0]
    @Lux.setter
    def Lux(self, val):
        self.Lx[0] = val
    
    @property
    def Lvx(self):
        return self.Lx[1]
    @Lvx.setter
    def Lvx(self, val):
        self.Lx[1] = val
    
    @property
    def Lwx(self):
        return self.Lx[2]
    @Lwx.setter
    def Lwx(self, val):
        self.Lx[2] = val
    
   # Alternative constructor
    @classmethod
    def from_json(cls, json_file, **kwargs):

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