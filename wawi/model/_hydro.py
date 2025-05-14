import numpy as np
import json
from inspect import isfunction
from scipy.interpolate import interp1d

from wawi.general import blkdiag, interp1d_angular
from wawi.wave import maxincrement, dispersion_relation_scalar, xsim
import wawi.wave
import pyvista as pv
from scipy.linalg import block_diag

'''
HYDRO SUBMODULE
'''     
class Hydro:    
    def __init__(self, pontoons=None, phi_key='hydro', environment=None, phases_lead=False, seastate=None):
            self.pontoons = pontoons
            self.phi_key = phi_key
            self._phases_lead = False
            self._phi = None
            self.seastate = seastate

            if environment is None:
                self.environment = Environment()
            
            self.assign_environment()
 
    @property
    def phases_lead(self):
        return self._phases_lead
    
    @phases_lead.setter
    def phases_lead(self, val):
        self._phases_lead = val
        self.assign_to_pontoon_types(phases_lead=val)
     
    @property
    def phi(self):
        if self._phi is not None:
            return self._phi
        else:
            return np.eye(self.ndofs)
        
    @phi.setter
    def phi(self, phi):
        self._phi = phi

    
    @property
    def pontoon_types(self):
        ptypes = [p.pontoon_type for p in self.pontoons]
        unique_types = []
        for pt in ptypes:
            if pt not in unique_types:
                unique_types.append(pt)
        return unique_types
    
        
    def reset_linearized_drag(self):
        for p in self.pontoons:
            p.Cquad_lin = np.diag(np.zeros(6))
    
    def assign_direct_cquad(self, cquad_dict=None):
        for p in self.pontoons:
            p.Cquad = np.zeros(6)
        
        if cquad_dict is None:
            cquad_dict = {}
            
        for pontoon_label in cquad_dict:
            self.get_pontoon(pontoon_label).Cquad = cquad_dict[pontoon_label]
        
    def assign_environment(self):
        for pontoon in self.pontoons:
            pontoon.environment = self.environment
            
    def assign_to_pontoons(self, **kwargs):
        for p in self.pontoons:
            for k, v in kwargs.items():
                setattr(p, k, v)
    
    def assign_to_pontoon_types(self, **kwargs):
        for pt in self.pontoon_types:
            for k, v in kwargs.items():
                setattr(pt, k, v)
                
    def get_all_cquad(self, local=False):
        
        if local:
            cquad = [p.cquad_local for p in self.pontoons]
        else:
            cquad = [p.cquad for p in self.pontoons]
            
        return block_diag(*cquad)
    
    
        
    def get_all_Cd(self):
        Cd = np.zeros(6*len(self.pontoons))
        for ix, p in enumerate(self.pontoons):
            Cd[ix*6:ix*6+6] = p.Cd
            
        return Cd
    
    def get_all_area(self):
        area = np.zeros(6*len(self.pontoons))
        for ix, p in enumerate(self.pontoons):
            area[ix*6:ix*6+6] = p.area
            
        return area

        
    def get_from_all(self, par):
        vals = []
        for p in self.pontoons:
            vals.append(getattr(p, par))
            
        return vals
                    
    def get_pontoon(self, label):
        matching_pontoons = [p for p in self.pontoons if p.label == label]
        if len(matching_pontoons)>1:
            raise ValueError('Multiple matching pontoons')
        elif len(matching_pontoons)==0:
            return None
        else:
            return matching_pontoons[0]
        
            
    @property
    def nodelabels(self):
        return np.array([pont.node.label for pont in self.pontoons])
    
    @property
    def ndofs(self):
        return len(self.pontoons)*6

'''
ENVIRONMENT CLASS
'''
class Environment:
    def __init__(self, g=9.80655, rho=1.025e3, depth=np.inf, waterlevel=0):
        self.g = g
        self.rho = rho
        self.depth = depth
        self.waterlevel = waterlevel


'''
PONTOON CLASS
'''

class Pontoon:
    def __init__(self, node, pontoon_type, rotation=0, label='pontoon', 
        current_affects_Q=None, seastate=None, environment=None, current_affects_k=None):

        self.node = node
        self.pontoon_type = pontoon_type
        self.rotation = rotation

        self.label = label

        self.seastate = seastate
        self.environment = environment     
        self._current_affects_Q = current_affects_Q
        self._current_affects_k = current_affects_k
        
        self.Cquad = np.zeros(6)   # local, direct quadratic damping
        self._Cd = None


    def __repr__(self):
        return f'Pontoon object: {self.label}'
    def __str__(self):
        return f'Pontoon object: {self.label} at {self.node}'

    def sea_get(self, parameter):
        return self.seastate.get_at_pos(parameter, self.node.x[0], self.node.x[1])


    @property
    def current_affects_Q(self):
        if self._current_affects_Q is not None:
            return self._current_affects_Q
        elif (self.seastate is not None) and ('current_affects_Q' in self.seastate.pontoon_options):
            return self.seastate.pontoon_options['current_affects_Q']
        
    @current_affects_Q.setter
    def current_affects_Q(self, val):
        self._current_affects_Q = val
    
    @property
    def current_affects_k(self):
        if self._current_affects_k is not None:
            return self._current_affects_k
        elif (self.seastate is not None) and ('current_affects_k' in self.seastate.pontoon_options):
            return self.seastate.pontoon_options['current_affects_k']
        
    @current_affects_k.setter
    def current_affects_k(self, val):
        self._current_affects_k = val

    @property
    def Cquad_lin(self):
        if hasattr(self, '_Cquad_lin'):
            return self._Cquad_lin
        else:
            return np.zeros([6, 6])
        
    @Cquad_lin.setter
    def Cquad_lin(self, Cquad_lin):
        self._Cquad_lin = Cquad_lin
        
    @property
    def tmat(self):
        return self.get_tmat_from_rotation()
    
    @property
    def cquad(self):
        T = self.tmat
        return T.T @ self.cquad_local @ T
    
    @property
    def cquad_local(self):
        rho = self.environment.rho
        
        # Directly specified C_quad
        if hasattr(self, 'Cquad'):     #to ensure backwards compatibility
            Cquad = self.Cquad
        else:
            Cquad = np.zeros(6)

        # C_quad from drag definitions
        Cquad_Cd = 0.5 * rho * self.Cd * self.area
        
        return np.diag(Cquad_Cd + Cquad)
        
        
    @property
    def Cd(self):
        if hasattr(self, '_Cd') and self._Cd is not None:
            val = self._Cd
        else:
            val = self.pontoon_type.Cd
            
        if len(val)==3:
            val = np.hstack([val, [0,0,0]])
        else:
            val = np.array(val)
            
        return val
    
    @Cd.setter
    def Cd(self, Cd):
        self._Cd = Cd
    
    @property
    def area(self):
        val = self.pontoon_type.area
            
        if len(val)==3:
            val = np.hstack([val, [0, 0, 0]])
        else:
            val = np.array(val)
            
        return val
    
    # To simplify expressions inside class:
    @property
    def S(self):
        return self.sea_get('S')

    @property
    def D(self):
        return self.sea_get('D')

    @property
    def U(self):
        return self.sea_get('U')

    @property
    def thetaU(self):
        return self.sea_get('thetaU')

    @property
    def depth(self):
        return self.sea_get('depth')

    @property
    def theta0(self):
        return self.sea_get('theta0')

    @property
    def theta_shift(self):
        return self.sea_get('theta_shift')
    
    def doppler_shift_omega(self, omega_k, theta, kappa=None):
        if kappa is None:
            kappa = self.get_kappa(omega_k, theta=theta, U=0.0)

        omega_shifted = theta*0.0
        
        for ix, kappai in enumerate(kappa):
            thetai_abs = theta[ix] + self.theta_shift
            Uproj = self.U * np.cos(self.thetaU - thetai_abs)
            omega_shifted[ix] = omega_k - Uproj * kappai

        return omega_shifted
    
 
    def get_kappa(self, omega_k, theta=0.0, U=None):
        if not self.current_affects_k:
            U = 0.0
        elif U is None:
            U = self.U*1.0

        if U == 0.0:
            kappa = dispersion_relation_scalar(omega_k, self.depth, 
                                               g=self.environment.g)
            
            if np.ndim(theta)!=0:
                kappa = kappa*np.ones(len(theta))
        else:
            kappa = theta*0.0
            for ix, thetai in enumerate(theta):
                thetai_abs = thetai + self.theta_shift
                Uproj = U * np.cos(self.thetaU - thetai_abs)
                kappa[ix] = dispersion_relation_scalar(omega_k, h=self.depth, 
                                                       g=self.environment.g, 
                                                       U=Uproj)

        return kappa
    
    def get_n(self):
        return np.array([np.cos(self.rotation), np.sin(self.rotation), 0])
               
    def get_tmat_from_rotation(self, e2=np.array([0, 0, 1])):
        n = self.get_n()
        e1 = np.cross(e2, n)
        return blkdiag(np.vstack([n, e1, e2]), 2)


    def get_K(self, omega, local=False):
        if local:
            T = np.eye(6)
        else:
            T = self.tmat
            
        if self.pontoon_type.K is not None:
            return T.T @ self.pontoon_type.K(omega) @ T
        else:
            return T*0

        
    def get_C(self, omega,  local=False):
        if local:
            T = np.eye(6)
        else:
            T = self.tmat
            

        return T.T @ (self.pontoon_type.C(omega)) @ T    
    
    
    def get_M(self, omega, local=False):
        if local:
            T = np.eye(6)
        else:
            T = self.tmat
            
        return T.T @ self.pontoon_type.M(omega) @ T   
    
    
    def get_Q0_legacy(self, omega_k, local=False):
        if local:
            T = np.eye(6)
        else:
            T = self.tmat
            
        return T.T @ self.pontoon_type.Q(omega_k)
    
    
    def get_Q0(self, omega, local=False):
        if local:
            T = np.eye(6)
        else:
            T = self.tmat
            
        if np.ndim(omega)==0:    #omega is scalar
            return T.T @ self.pontoon_type.Q(omega)
        else:
            Qeval = self.pontoon_type.Q(omega)
            for k in range(len(omega)):
                Qeval[:,:,k] = T.T @ Qeval[:,:,k]
                
            return Qeval
    
    
    def get_Q(self, omega_k, theta, local=False, theta_interpolation='quadratic'):
        if (not self.current_affects_Q) or (self.U == 0.0):    #omega is scalar
            Qk = interp1d_angular(self.pontoon_type.theta + self.rotation - self.theta_shift, 
                                    self.get_Q0(omega_k, local=local), axis=1, fill_value=0, 
                                    kind=theta_interpolation, bounds_error=False, 
                                    assume_sorted=False)(theta)
            return Qk
        else:
            omega = self.doppler_shift_omega(omega_k, theta)
            Q0 = self.get_Q0(omega, local=local)
            Qk = np.zeros([6, len(theta)]).astype(complex)
            for ix in range(len(theta)):   # TODO: first implementation for loop - will make quicker by avoiding loop later
                Qk[:, ix] = interp1d_angular(self.pontoon_type.theta + self.rotation - self.theta_shift, 
                                        Q0[:, :, ix], axis=1, fill_value=0, 
                                        kind=theta_interpolation, bounds_error=False, 
                                        assume_sorted=False)(theta[ix])
        return Qk     
       
   
    def evaluate_Q(self, omega=None, theta=None, 
                   theta_interpolation='quadratic', local=False):
        
        if omega is None:
            omega = self.pontoon_type.original_omegaQ
        
        if theta is None:
            theta = self.pontoon_type.theta
        
        if np.ndim(theta)==0:
            theta = np.array([theta])

        Q = np.zeros([6, len(theta), len(omega)]).astype('complex')
        
        for k, omega_k in enumerate(omega):
            Q[:, :, k] = self.get_Q(omega_k, theta,
                                    local=local, theta_interpolation=theta_interpolation)
            
        return Q, theta, omega
    
    
    def get_coh_2d(self, omega_k, theta, x0=[0,0], sign=-1):
        kappa = self.get_kappa(omega_k, theta=theta)
        
        # Minus sign in coh confirmed matching with phases lags in OrcaFlex
        
        coh = np.exp(self.pontoon_type.phase_sign*1j*kappa* ((self.node.x[0]-x0[0])*np.cos(theta+self.theta_shift) + 
                                 (self.node.x[1]-x0[1])*np.sin(theta+self.theta_shift)))
        return coh
    
    
    def get_Z(self, omega_k, theta_int, theta_interpolation='quadratic', 
              local=False, x0=[0,0]):   
        
        if theta_int is None:   # => long-crested   
            theta_int = np.array([self.theta0 - self.theta_shift])
                        
        if self.D.__code__.co_argcount==2:    # count number of inputs
            D = self.D(theta_int, omega_k)
        else:
            D = self.D(theta_int)

        # Interpolate hydrodynamic transfer function, get 2d coh and est. Z
        Q_int = self.get_Q(omega_k, theta_int, local=local, 
                           theta_interpolation=theta_interpolation)
        
        coh_2d = self.get_coh_2d(omega_k, theta_int, x0=x0)
        Z = np.sqrt(self.S(omega_k)) * Q_int * np.tile(np.sqrt(D), [6, 1]) * np.tile(coh_2d, [6, 1])
        
        return Z
    
    def get_theta_int(self, omega_k, ds=None, max_rel_error=0.01):
        if ds is None:
            ds = self.max_distance
            
        dtheta = np.min([maxincrement(ds, omega_k, 0, 2*np.pi, max_rel_error), 2*np.pi/20])   #minimum 20 divisions

        return np.arange(-np.pi, np.pi, dtheta)     
    

'''
SEASTATE CLASS
'''

class Seastate:
    def __init__(self, Tp, Hs, gamma, theta0, s=np.inf, depth=None, origin=None, ranges=None, 
                 options={}, pontoon_options={}, name=None, plot_label=None, centered_dirdist=True,
                 U=None, thetaU=None, use_robust_D=True, angle_unit='deg'):

        if angle_unit == 'deg':
            ang_conv = np.pi/180

        if origin is None:
            self.origin = np.array([0,0])
        else:
            self.origin = origin

        if depth is None:
            self.depth = lambda x,y: np.inf
            
        if ranges is None:
            self.ranges = dict(x=[-np.inf, np.inf], y=[-np.inf, np.inf])
        else:
            self.ranges = ranges
            
        self.name = name
        if plot_label is None:
            self.plot_label = name
        else:
            self.plot_label = plot_label
            
        self.pars = ['Tp', 'Hs', 'gamma', 'theta0', 's']
        self.pars_pretty = dict(Tp='$T_p$ [s]', Hs='$H_s$ [m]', gamma=r'$\gamma$', theta0=r'$\theta_0$ $[^\circ]$', s='s')

        if s is None:
            s = np.inf
        if U is None:
            U = 0.0
        if thetaU is None:
            thetaU = 0.0

        self.Tp = Tp
        self.Hs = Hs
        self.gamma = gamma
        self.s = s
        self.theta0 = self.scale(theta0, ang_conv)
        self.centered_dirdist = centered_dirdist
        self.use_robust_D = use_robust_D

        self.U = U
        self.thetaU = self.scale(thetaU, ang_conv)

        self.x0 = self.origin[0]
        self.y0 = self.origin[1]    
        self.options = {'keep_coherence': True}     #default options
        self.options.update(**options)       
        
        self.pontoon_options = {'current_affects_Q': True, 'current_affects_k': True}
        self.pontoon_options.update(**pontoon_options)

        self.fun_pars = [par for par in self.pars if self.isfun(par)]


    def __str__(self):
        string = f'''\
WAWI SeaState 
--------------
Hs={self.Hs:.2f}m
Tp={self.Tp:.2f}s
theta0={self.theta0*180/np.pi:.1f}deg
gamma={self.gamma:.2f}
s={self.s:.1f}
U (current)={self.U:.1f}m/s
thetaU (current)={self.thetaU*180/np.pi:.1f}deg
'''
        
        return string


    @property
    def homogeneous(self):
        if len(self.fun_pars)==0:
            return True         
        else:
            return False
        
    @property
    def theta_shift(self):
        if self.centered_dirdist:
            return self.theta0 
        else:
            return 0.0       
        
    @property
    def short_crested(self):
        return self.s not in [np.inf, None]
    
    @property
    def D(self):
        if not self.short_crested:
            return lambda x, y: lambda theta: 1*(theta==self.evpar('theta0', x, y)-self.evpar('theta_shift',x,y))
        else:
            if self.use_robust_D:
                Dfun = wawi.wave.dirdist_robust
            else:
                Dfun = wawi.wave.dirdist_decimal  
                
            return lambda x,y: Dfun(self.evpar('s', x, y), theta0=self.evpar('theta0', x, y)-self.evpar('theta_shift',x,y))
        
        
    @property
    def S(self):
        return lambda x,y: wawi.wave.jonswap(self.evpar('Hs', x, y), 
                self.evpar('Tp', x, y), self.evpar('gamma', x, y))

    @property
    def theta_int(self):
        if self.short_crested == False:
            th = np.array([self.theta0 - self.theta_shift])
        elif 'truncate_theta' in self.options:
            theta_lims = np.array(self.options['truncate_theta'])
            dtheta = self.options['dtheta']
            th = np.arange(theta_lims[0], theta_lims[1]+dtheta, dtheta)
        else:
            th = None
        
        return th

    def simulate(self, x, y, omega, fs=None, theta=None, phase=None, return_phases=False, 
                 print_progress=True, time_history=True, grid_mode=True):
        
        if theta is None:
            theta = self.theta_int
            
        output = xsim(x, y, self.S, self.D, omega, theta=theta, fs=fs, 
             grid_mode=grid_mode, print_progress=print_progress, time_history=time_history, 
             phase=phase, return_phases=return_phases, theta_shift=self.theta_shift)
        
        if return_phases:
            eta, t, phase = output
            return eta, t, phase
        elif time_history:
            eta, t = output
            return eta, t
        else:
            return output

            

    @staticmethod
    def scale(par, factor):
        if isfunction(par):
            if 'x' in par.__code__.co_varnames and 'y' in par.__code__.co_varnames:
                return lambda x,y: par(x, y)*factor
            elif 'x' in par.__code__.co_varnames:
                return lambda x: par(x)*factor
            elif 'y' in par.__code__.co_varnames:
                return lambda y: par(y)*factor              
        else:
            return par*factor    

    @classmethod
    def from_json(cls, json_file, **kwargs):
        with open(json_file, 'r') as fileobj:
            data = json.load(fileobj)
        
        Tp = eval(data['S']['Tp'])
        Hs = eval(data['S']['Hs'])
        gamma = eval(data['S']['gamma'])
        
        theta0 = eval(data['D']['theta0'])  #assumed in deg
        s = eval(data['D']['s'])

        if 'current' in data:
            U = eval(data['current']['U'])
            thetaU = eval(data['current']['thetaU'])
        else:
            U = None
            thetaU = None
        
        if 'options' in data:
            options = data['options']
        else:
            options = {}
            
        if 'pontoon_options' in data:
            pontoon_options = data['pontoon_options']
        else:
            pontoon_options = {}
        
        if 'origin' in data:
            origin = np.array(data['origin'])
        else:
            origin = np.array([0,0])
            
        if 'ranges' in data:
            ranges = data['ranges']
            if 'x' not in ranges:
                ranges['x'] = [-np.inf, np.inf]
            if 'y' not in ranges:
                ranges['y'] = [-np.inf, np.inf]
        else:
            ranges = None

        if 'name' in data:
            name = data['name']
        else:
            name = 'Unnamed seastate'
        
        if 'plot_label' in data:
            plot_label = data['plot_label']
        else:
            plot_label = None
            
        if 'centered_dirdist' in options:
            centered_dirdist = options.pop('centered_dirdist')
        else:
            centered_dirdist = True

        return cls(Tp, Hs, gamma, theta0, s, origin=origin, ranges=ranges, 
                   name=name, plot_label=plot_label, options=options, 
                   U=U, thetaU=thetaU, pontoon_options=pontoon_options, 
                   centered_dirdist=centered_dirdist,
                   **kwargs)
        
    
    def evpar(self, parameter, x, y):
        
        # Constant extrapolate if outside borders
        if x>np.max(self.ranges['x']): x = np.max(self.ranges['x']) 
        if x<np.min(self.ranges['x']): x = np.min(self.ranges['x'])        
        if y>np.max(self.ranges['y']): y = np.max(self.ranges['y']) 
        if y<np.min(self.ranges['y']): y = np.min(self.ranges['y'])        
        
        par = getattr(self, parameter)
        if isfunction(par):
            if 'x' in par.__code__.co_varnames and 'y' in par.__code__.co_varnames:
                return par(x - self.x0, y - self.y0)
            elif 'x' in par.__code__.co_varnames:
                return par(x - self.x0)
            elif 'y' in par.__code__.co_varnames:
                return par(y - self.y0)                
        else:
            return par
    
    def get_at_pos(self, parameter, x, y):
        par = getattr(self, parameter)
        if isfunction(par):
            if 'x' in par.__code__.co_varnames and 'y' in par.__code__.co_varnames:
                return par(x - self.x0, y - self.y0)
            elif 'x' in par.__code__.co_varnames:
                return par(x - self.x0)
            elif 'y' in par.__code__.co_varnames:
                return par(y - self.y0)                
        else:
            return par
            

    def isfun(self, parameter):
        return isfunction(getattr(self, parameter))

    def with_unit(self, parameter):
        return self.pars_pretty[parameter]


'''
NODE CLASS (MINIMUM CLASS FOR WHEN BEEF MODEL IS NOT APPENDED TO MODEL OBJECT)
'''

class Node:
    def __init__(self, label, x=0.0, y=0.0, z=0.0):
        self.x0 = np.array([x,y,z,0.0,0.0,0.0]).astype(float)
        self.x = np.array([x,y,z,0.0,0.0,0.0]).astype(float)
        self.label = label
        
    def __repr__(self):
        return f'Node {self.label}'
    
    def __str__(self):
        return f'Node {self.label}'
    

'''
PONTOONTYPE CLASS
'''

class PontoonType:
    def __init__(self, K=None, C=None, M=None, Q=None, original_omega=None, original_omegaQ=None,
                 theta=None, label='unnamed', Cd=None, area=None, stl_path=None):
        self.K = K
        self.C = C
        self.M = M
        self.Q = Q
        self.original_omega = original_omega
        self.original_omegaQ = original_omegaQ
        self.theta = theta
        self.label = label
        self.stl_path = stl_path
        
        self.phases_lead = False

        self.generate_mesh()
        self._Cd = Cd
        self._area = area
        
    def __repr__(self):
        return f'PontoonType object: {self.label}'
    def __str__(self):
        return f'PontoonType object: {self.label}'

    def generate_mesh(self):
        if self.stl_path is not None:           
            stl = pv.STLReader(self.stl_path)
            self.mesh = stl.read().translate([0,0,0])
        else:
            self.mesh = None
            
    @property
    def phase_sign(self):
        if hasattr(self, 'phases_lead'):
            return -int(1 - 2*self.phases_lead*1)
        else:
            return -1   #standard is phases lag
    
    @property
    def Cd(self):
        if hasattr(self, '_Cd') and self._Cd is not None:
            return self._Cd
        else:
            return np.array([0,0,0,0,0,0])
        

    @Cd.setter
    def Cd(self, Cd):
        self._Cd = Cd

    @property
    def area(self):
        if hasattr(self, '_area') and self._area is not None:
            return self._area
        else:
            return np.array([0,0,0,0,0,0])
        
    @area.setter
    def area(self, area):
        self._area = area    
    
            
    @classmethod
    def from_numeric(cls, interpolation_kind='linear', fill_value_added='extrapolate', A=None, M0=None, B=None, Kh=None, 
                     omega=None, Q=None, theta=None, omegaQ=None, **kwargs):
        
        if omegaQ is None:
            omegaQ = omega*1
        
        if A is None and M0 is None:
            M = None
        else:
            if A is None:
                A = np.zeros([M0.shape + [len(omega)]])
            elif M0 is None:
                M0 = A[:,:,0]*0
            M = interp1d(omega, (A.T+M0.T).T, axis=2, fill_value=fill_value_added, kind=interpolation_kind)
        
        if B is None:
            C = None
        else:
            C = interp1d(omega, B, axis=2, fill_value=fill_value_added, kind=interpolation_kind)
        
        if Kh is not None:
            K = lambda omega: Kh
        else:
            K = None
        
        if Q is not None:
            Q = interp1d(omegaQ, Q, fill_value=0.0, kind=interpolation_kind, axis=2, bounds_error=False)
    
        return cls(M=M, C=C, K=K, Q=Q, original_omega=omega, original_omegaQ=omegaQ, theta=theta, **kwargs)


    @classmethod
    def from_wadam(cls, path, interpolation_kind='linear', fill_value_added='extrapolate',
                   include=['added mass', 'added damping', 'restoring stiffness', 'inertia'], **kwargs):
        
        from wawi.io import import_wadam_mat, import_wadam_hydro_transfer
        
        A, B, Kh, M0, omega = import_wadam_mat(path)
        omega_Q, theta, Q = import_wadam_hydro_transfer(path)
        
        if 'added mass' not in include:
            A = A*0
        
        if 'added damping' not in include:
            B = B*0
            
        if 'restoring stiffness' not in include:
            Kh = Kh*0
            
        if 'inertia' not in include:
            M0 = M0*0
            
        M = interp1d(omega, (A.T+M0.T).T, axis=2, fill_value=fill_value_added, kind=interpolation_kind)
        C = interp1d(omega, B, axis=2, fill_value=fill_value_added, kind=interpolation_kind)
        K = lambda omega: Kh
        
        Q = interp1d(omega_Q, Q, fill_value=0.0, kind=interpolation_kind, axis=2)
        
        return cls(M=M, C=C, K=K, Q=Q, original_omega=omega, theta=theta, **kwargs)