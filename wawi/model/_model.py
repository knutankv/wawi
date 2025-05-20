import numpy as np
import beef
from beef import fe
from copy import deepcopy as copy
import pyvista as pv
from beef.rotation import rodrot, R_from_drot

from pathlib import Path

from ._hydro import Hydro, Pontoon, Environment

from wawi.structural import var_from_modal, expmax_from_modal, peakfactor_from_modal
from wawi.modal import maxreal, normalize_phi, iteig_freq, iteig
from wawi.general import transform_3dmat, eval_3d_fun, fun_const_sum, eval_3d_fun
from wawi.structural import freqsim
from wawi.tools import print_progress as pp
from wawi.wave import stochastic_linearize, harmonic_linearize
from wawi.wind import windaction, windaction_static
# from wawi.io import import_folder

import dill


import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from scipy.signal import savgol_filter

'''
RESULTS SUBMODULE
'''

class Results:
    def __init__(self, psi=None, lambd=None, S=None, omega=None, model=None):
        self.psi = psi
        self.lambd = lambd
        self.S = S
        self.omega = omega
        self.include_eig = None
        self.model = model
        
    @property
    def mdiag(self):
        Kfun, Cfun, Mfun = self.model.get_system_matrices(self.include_eig)
        m = np.zeros(self.lambd.shape[0])
        for ix, wi in enumerate(self.wd):
            m[ix] = np.real(np.diag(np.real(self.psi.T) @ Mfun(wi) @ np.real(self.psi))[ix])
        
        m[m<0] = np.nan
        return m
    
    @property
    def kdiag(self):
        Kfun, Cfun, Mfun = self.model.get_system_matrices(self.include_eig)
        k = np.zeros(self.lambd.shape[0])
        for ix, wi in enumerate(self.wd):
            k[ix] = np.real(np.diag(np.real(self.psi.T) @ Kfun(wi) @ np.real(self.psi))[ix])
            
        k[k<0] = np.nan
        
        return k
    
    @property
    def cdiag(self):
        Kfun, Cfun, Mfun = self.model.get_system_matrices(self.include_eig)
        c = np.zeros(self.lambd.shape[0])
        for ix, wi in enumerate(self.wd):
            c[ix] = np.real(np.diag(np.real(self.psi.T) @ Cfun(wi) @ np.real(self.psi))[ix])       
            
        c[c<0] = np.nan
        
        return c
    
    @property
    def xi(self):
        if self.lambd is not None:
            return -np.real(self.lambd)/np.abs(self.lambd)

    @property
    def wn(self):
        if self.lambd is not None:
            return np.abs(self.lambd)

    @property
    def wd(self):
        if self.lambd is not None:
            return np.abs(np.imag(self.lambd))
   
    @property
    def fn(self):
        if self.lambd is not None:
            return np.abs(self.lambd)/2/np.pi      
        
    @property
    def fd(self):
        if self.lambd is not None:
            return np.abs(np.imag(self.lambd))/2/np.pi                              

    @property
    def Tn(self):
        if self.lambd is not None:
            return 1/(np.abs(self.lambd)/2/np.pi)    
        
    @property
    def Td(self):
        if self.lambd is not None:
            return 1/(np.abs(np.imag(self.lambd))/2/np.pi)   

'''
DRAG ELEMENT CLASS
'''
class DragElement:
    def __init__(self, element, Cd, D=1.0, rho=1025.0, group=None, eltype=None):

        self.group = group
        self._D = D
        self._Cd = Cd
        self.element = element
        self.rho = rho
        self._eltype = eltype
        
    

    @property
    def eltype(self):
        if hasattr(self, '_eltype') and self._eltype is not None:
            return self._eltype
        elif isinstance(self.element, fe.BeamElement3d):
            return 'beam'
        else:
            return 'bar'
        
    @eltype.setter
    def eltype(self, eltype):
        self._eltype = eltype
        

    @property
    def cquad_local(self):
        c = (0.5 * self.Cd * self.rho * self.D)[:3]        #Linearized as 0.25 Cd D L sqrt(8/pi) * stdudot according to A Wang- seems to work well for linearized,check nonlin also
        from beef.general import bar_foundation_stiffness, generic_beam_mat
    
        if self.eltype == 'bar':
            return bar_foundation_stiffness(self.element.L, *(c*0.5))
        elif self.eltype == 'beam':  
            val_dict = {'yy': c[1], 'zz': c[2]}
            return generic_beam_mat(self.element.L, **val_dict)
        elif self.eltype in ['node', 'lumped']:
            C = np.zeros([12,12])
            c = 0.5*c*self.element.L # half on each end
            C[:3,:3] = np.diag(c)
            C[6:6+3,6:6+3] = np.diag(c)
            return C
        else:
            return np.zeros([12,12])
   
    @property
    def cquad(self):                   
        return self.element.tmat.T @ self.cquad_local @ self.element.tmat  
   
    
    def get_cquad_lin(self, var_udot, stochastic=True, 
                      input_is_local=False, ensure_local_output=False):
        # var_udot is covariance matrix of velocity of nodes
        # local_input: whether or not the var_udot is referring to a local csys - if not, it will be transformed to local
        # prior to C computation
        # local_output: whether or not the output should be local - global output is standard
        if not input_is_local:     
            Tin = self.element.tmat
        else:
            Tin = np.eye(12)
            
        if not ensure_local_output:
            Tout = self.element.tmat
        else:
            Tout = np.eye(12)
        
        if stochastic:
            std_udot = np.sqrt(np.abs(Tin @ var_udot @ Tin.T))
            C = stochastic_linearize(self.cquad_local, std_udot)
        else:
            udot = np.diag(np.sqrt(np.abs(Tin @ var_udot @ Tin.T)))    
            C = harmonic_linearize(self.cquad_local, udot)
            
        return Tout.T @ C @ Tout
    
    
    # Get and set methods D and Cd
    @property  
    def D(self):
        if np.ndim(self._D) == 0:
            return np.hstack([self._D*np.ones(3), [0, 0, 0]])
        elif len(self._D) == 3:
            return np.hstack([self._D, [0,0,0]])
        else:
            return self._D
    
    @D.setter
    def D(self, D):
        self._D = D
        
    @property        
    def Cd(self):
        if np.ndim(self._Cd) == 0:
            return np.hstack([self._Cd*np.ones(3), [0,0,0]])
        elif len(self._Cd) == 3:
            return np.hstack([self._Cd, [0,0,0]])
        else:
            return self._Cd
    
    @Cd.setter
    def Cd(self, Cd):
        self._Cd = Cd

    def __repr__(self):
        return f'DragElement <{self.group}> [{self.element.label}]'

    def __str__(self):
        return f'DragElement <{self.group}> [{self.element.label}]'

'''
MODEL CLASS
'''

class Model:
    def __init__(self, hydro=None, aero=None, eldef=None, modal_dry=None, seastate=None, windstate=None,
                 n_dry_modes=None, x0_wave=None, phases_lead=False,
                 use_multibody=True, avoid_eldef=False, drag_elements={}):
        
        self.results = Results(model=self)
        self.hydro = hydro
        self.aero = aero
        self.modal_dry = modal_dry  # ModalDry object
        self.eldef = eldef  # BEEF ElDef object
        self.f_static = None
        
        self.n_dry_modes = n_dry_modes
        
        if modal_dry is not None:
            self.modal_dry.n_modes = n_dry_modes

        if not avoid_eldef and self.eldef is None:
            self.construct_simple_eldef(node_labels=hydro.nodelabels)
        
        if self.aero is not None:
            self.assign_windstate(windstate)

        if self.hydro is not None:
            self.prepare_waveaction()
            self.assign_seastate(seastate)

            if x0_wave is None:
                x,y = self.get_all_pos()
                x0_wave = np.array([np.mean(x), np.mean(y)])
                self.x0 = x0_wave

        if modal_dry is not None:        
            self.assign_dry_modes() 

        self.use_multibody = use_multibody # use multibody if available
        self.phases_lead = phases_lead  # define if phases lead or lag. When lags (lead=False): eta = exp(iwt-kx)
        
        if type(drag_elements) == dict:
            self.assign_drag_elements(drag_elements)
        else:
            self.drag_elements = drag_elements
            
        self.Cquad_lin = 0.0
    
    # Not used due to circular referencing.
    # @staticmethod
    # def from_folder(*args, **kwargs):
    #     '''
    #     Alternative constructor. Passes all inputs to `wawi.io.import_folder`.
    #     '''

    #     model = import_folder(*args, **kwargs)
        
    #     return model
    

    @staticmethod    #alternative constructor
    def from_wwi(path):
        with open(path, 'rb') as f:
            model = dill.load(f)
    
        return model


    def assign_drag_elements(self, drag_elements):
        drag_els = [None]*len(drag_elements)
        for ix, group_key in enumerate(drag_elements):
            els = self.eldef.get_els_with_sec(drag_elements[group_key]['sections'])
            
            # Filter if elements are specified as well
            if hasattr(drag_elements[group_key], 'elements'):
                els = [el for el in els if el in drag_elements[group_key]['elements']]
                
            Cd = drag_elements[group_key]['Cd']
            D = drag_elements[group_key]['D']
            
            if 'rho' in drag_elements[group_key] and drag_elements[group_key]['rho'] is not None:
                rho = drag_elements[group_key]['rho']
            else:
                rho = self.hydro.environment.rho
                
            if 'eltype' in drag_elements[group_key]:
                eltype = drag_elements[group_key]['eltype']
            else:
                eltype = None
            
            drag_els[ix] = [DragElement(el, Cd, D=D, rho=rho, group=group_key, eltype=eltype) for el in els]
        
        self.drag_elements = [a for b in drag_els for a in b]
    
        
    def get_modal_cquad(self, dry=False, psi=None):
        if psi is None:
            if dry:
                psi = np.eye(self.n_modes)
            else:
                psi = self.results.psi
                
        c_quad = np.zeros(self.n_modes)

        if 'full' in self.modal_dry.phi_full and hasattr(self, 'Cquad'):
            phi0 = np.real(self.get_dry_phi(key='full'))
            phi = np.real(phi0 @ psi)
            Cquad_model = self.Cquad
            
            for mode in range(self.n_modes):
                phiabs = np.diag(np.abs(phi[:, mode]))
                c_quad[mode] = c_quad[mode] + phi[:,mode].T @ (Cquad_model * phiabs) @ phi[:,mode]
            
        if 'hydro' in self.modal_dry.phi_full:
            phi0 = self.get_dry_phi(key='hydro')
            phi = np.real(phi0 @ psi)
            Cquad_pontoons = self.hydro.get_all_cquad()
                        
            for mode in range(len(c_quad)):
                phiabs = np.diag(np.abs(phi[:, mode]))
                c_quad[mode] = c_quad[mode] + phi[:,mode].T @ (Cquad_pontoons * phiabs) @ phi[:,mode]
  
        return c_quad
    
    @property
    def tmat_full(self):
        all_tmats = []
        for node in self.eldef.nodes:
            all_tmats.append(self.eldef.get_node_csys(node.label))
        return block_diag(*all_tmats)

    @property
    def Cquad_lin_pontoons(self):    # Only pontoons
        mat = self.initialize_const_matrix()
        for ix, p in enumerate(self.hydro.pontoons):        
            mat[ix*6:ix*6+6, ix*6:ix*6+6] = p.Cquad_lin
        return self.hydro.phi.T @ mat @ self.hydro.phi
    
    
    @property
    def n_modes(self):
        return self.modal_dry.n_modes

    @n_modes.setter
    def n_modes(self, n):
        self.modal_dry.n_modes = n
        self.assign_dry_modes()
        

    @property
    def Cquad(self):    # Only drag elements
        # Requires global representation (phi_full) currently. Consider supporting modal form referring to specific keys.
        # self.local only refers to phi of pontoons
        C = np.zeros([self.eldef.ndofs, self.eldef.ndofs])
        
        if (hasattr(self, 'drag_elements')) and (self.drag_elements is not None) and (len(self.drag_elements) > 0):    
            for e in self.drag_elements:
                 ix = e.element.global_dofs
                 C[np.ix_(ix,ix)] = C[np.ix_(ix,ix)] + e.cquad
             
        return C
        
    def get_Cquad_lin(self, var_udot, local=None, stochastic=True):    # Only drag elements
        
        if local is None:
            local = self.local*1

        if not hasattr(self, 'drag_elements') or len(self.drag_elements) == 0:
            return np.zeros([self.eldef.ndofs, self.eldef.ndofs])
        else:
            C = np.zeros([self.eldef.ndofs, self.eldef.ndofs])
    
        if np.ndim(var_udot)==1:
            var_udot = np.diag(var_udot)
            
        for e in self.drag_elements:
             ix = e.element.global_dofs
             C[np.ix_(ix,ix)] = C[np.ix_(ix,ix)] + e.get_cquad_lin(var_udot[np.ix_(ix,ix)], 
                                                        input_is_local=local, 
                                                        ensure_local_output=local, 
                                                        stochastic=stochastic)    
   
        return C

        
            
    def construct_simple_eldef(self, node_labels=None):
        if node_labels is None:
            node_labels = np.arange(1,len(self.hydro.pontoons)+1,1)
            
        element_matrix = beef.nodes_to_beam_element_matrix(node_labels)
        node_matrix = np.vstack([np.hstack([node_labels[ix], self.hydro.pontoons[ix].node.x0[:3]]) for ix in range(len(node_labels))])
        part = fe.Part(node_matrix, element_matrix)

        for element in part.elements:
            element.assign_e2(np.array([0,1,0]))
            
        part.assign_global_dofs()
        part.update_all_geometry()

        self.eldef = part
        self.connect_eldef()
        
        if 'hydro' in self.modal_dry.phi_full:
            self.modal_dry.phi_full['full'] = self.modal_dry.phi_full['hydro']
    
        
    def initialize_const_matrix(self, astype=float):
        return np.zeros([len(self.hydro.pontoons)*6, len(self.hydro.pontoons)*6]).astype(astype)

    def initialize_freq_matrix(self, n_freq, astype=float):
        return np.zeros([len(self.hydro.pontoons)*6, len(self.hydro.pontoons)*6, n_freq]).astype(astype)

    def __repr__(self):
        return f'WAWI model <{self.__hash__()}>'
    
    def __str__(self):
        return f'WAWI model <{self.__hash__()}>'
    
    def to_wwi(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f, -1)
        
    
    def get_all_pos(self):
        x = np.array([pont.node.x[0] for pont in self.hydro.pontoons])
        y = np.array([pont.node.x[1] for pont in self.hydro.pontoons])
        return x,y
    
    
    def assign_dry_modes(self):
        if self.hydro is not None:
            self.hydro.phi = self.modal_dry.get_phi(key=self.hydro.phi_key)
        if self.aero is not None:
            self.aero.phi = self.modal_dry.get_phi(key=self.aero.phi_key)


    def connect_eldef(self):
        if self.eldef is not None:
            if self.hydro is not None:
                # Connect pontoons to nodes in beef eldef
                for pontoon in self.hydro.pontoons:
                    pontoon.node = self.eldef.get_node(pontoon.node.label)
            
            # Establish connectivity to aerodynamic sections
            if self.aero is not None:
                self.aero.eldef = dict()
                node_labels_org = self.eldef.get_node_labels()                                   
                
                for group in self.aero.elements:
                    self.aero.eldef[group] = self.eldef.get_element_subset(self.aero.elements[group], renumber=False)  
                    node_labels_sorted = np.array([nl for nl in node_labels_org if nl in self.aero.eldef[group].get_node_labels()])
                    self.aero.eldef[group].arrange_nodes(node_labels_sorted)
                    
                    # These refer to the full dof set, no renumbering yet
                    self.aero.phi_ixs[group] = self.aero.eldef[group].global_dofs*1   

                    # Rearrange dofs based on nodes in subselection, so global dof ixs refer to subset eldef
                    self.aero.eldef[group].assign_node_dofcounts()
                    self.aero.eldef[group].assign_global_dofs() 
                    self.aero.elements[group] = self.aero.eldef[group].elements
                    
                
    def plot_mode(self, mode_ix, use_dry=False, scale=300, title=None, plot_wind_axes=False, 
                  plot_states=['deformed', 'undeformed'], plot_wave_direction=False, **kwargs):
        
        if use_dry: 
            phi_plot = self.get_dry_phi(key='full')
        else:
            phi_plot = np.real(maxreal(self.get_phi(key='full')))
        
        self.eldef.deform_linear(phi_plot[:, mode_ix]*scale, only_deform=True)
        
        if title is None:
            title = f'Mode {mode_ix+1}'
            
        pl = self.plot(plot_states=plot_states, title=title, plot_wind_axes=plot_wind_axes, plot_wave_direction=plot_wave_direction, **kwargs)
        return pl
    
    def export_modeshapes(self, folder, n_modes=None, format='pdf', title=None, zoom=1.0, **kwargs):
        if title is None:
            title = lambda mode: f'Mode {mode+1}:\nTd = {self.results.Td[mode]:.2f}s\nxi = {self.results.xi[mode]*100:.2f}%'
            
        elif type(title) is str:
            title = lambda mode: title + ''

        folder = Path(folder)
        if n_modes is None:
            n_modes = self.n_modes

        for mode in range(n_modes):
            pl = self.plot_mode(mode, show=False, title=title(mode), **kwargs) 
            save_path = folder / f'mode{mode+1:004}.{format}'
            pl.camera.zoom(zoom)
            pl.save_graphic(save_path)


    def copy(self):
        return copy(self)
    
    def plot(self, use_R=False, plot_water=True, 
             waterplane_padding=[1800, 1800], plot_wind_axes=True, wind_ax=[0],
             title=None, show=True, plot_wind_at_centre=True, axis_scaling=100, plot_states=['undeformed'],
             plot_wave_direction=True, wave_origin='center', pontoons_on=['deformed', 'undeformed'],
             thickness_scaling=None, annotate_pontoon_type=False, **kwargs):
        
        if thickness_scaling == 'area':
            lambda sec: np.sqrt(sec.A)
    
        
        tmat_settings = dict(show_edges=False)
        tmat_colors = ['#ff0000', '#00ff00', '#0000ff']
        
        if self.eldef is None:
            pl = pv.Plotter()
        else:
            pl = self.eldef.plot(show=False, plot_states=plot_states,thickness_scaling=thickness_scaling, **kwargs)

        bounds = np.array(pl.bounds)
        origin = (bounds[0::2] + bounds[1::2])/2
        
        if plot_wind_at_centre:
            wind_origin = origin*1
            wind_origin[2] = 10
        else:
            wind_origin = np.array(self.aero.windstate.x_ref)
        
        if self.hydro is not None:
            for pontoon in self.hydro.pontoons:
                if pontoon.pontoon_type.mesh is not None:
                    for state in pontoons_on:
                        x_field = 'x' if state=='deformed' else 'x0'
                        
                        if state in plot_states:
                            # Establish rotation tensor (from global undeformed to global deformed)
                            if use_R and hasattr(pontoon.node, 'R') and state == 'deformed':
                                Tn = pontoon.node.R
                            else:
                                node_rots = getattr(pontoon.node, x_field)[3:]
                                Tn = R_from_drot(node_rots)
                                
                            # Establish rotation tensor (from local to global undeformed)
                            Tzrot = rodrot(pontoon.rotation, style='column')
                            
                            # Stack 4x4 rotation tensor
                            T = np.eye(4)
                            T[:3, :3] = Tn @ Tzrot 
                            
                            mesh_current = copy(pontoon.pontoon_type.mesh)
                            mesh_current = mesh_current.transform(T).translate(getattr(pontoon.node, x_field)[:3])
                            pl.add_mesh(mesh_current)

        # Plot wind arrow
        if plot_wind_axes and self.aero is not None and self.aero.windstate is not None:
            for ax in wind_ax:
                vec = self.aero.windstate.T[ax, :]*axis_scaling
                pl.add_arrows(wind_origin, vec, color=tmat_colors[ax], **tmat_settings)
            
            pl.add_point_labels(np.vstack([wind_origin]), [f'U={self.aero.windstate.U0:.2f} m/s from heading {self.aero.windstate.direction:.1f} deg (CW)'])
            
        # Plot wave arrow        
        if plot_wave_direction and self.hydro is not None and self.hydro.seastate is not None:
            if self.hydro.seastate.homogeneous:
                if wave_origin=='center':
                    wave_origin = origin*1
                    
                vec = rodrot(self.hydro.seastate.theta0)[0,:]*axis_scaling
                pl.add_arrows(np.array(wave_origin-vec*3), vec, color='black', **tmat_settings)
                pl.add_point_labels(np.vstack([wave_origin-vec*3]), [f'dir={self.hydro.seastate.theta0*180/np.pi:.1f} deg, Hs={self.hydro.seastate.Hs:.2f} m, Tp={self.hydro.seastate.Tp:.1f} s'])
            else:
                vec = []
                pos = []
                fun_pars = self.hydro.seastate.fun_pars
                for pontoon in self.hydro.pontoons:
                    pars = dict(theta0=f'theta0 = {pontoon.sea_get("theta0")*180/np.pi:.1f}deg',
                                Hs=f'Hs = {pontoon.sea_get("Hs"):.2f}m',
                                Tp=f'Tp = {pontoon.sea_get("Tp"):.1f}s')
                    
                    vec = rodrot(pontoon.sea_get('theta0'))[0,:]*axis_scaling
                    
                    if wave_origin=='center':
                        wave_origin = origin*1
                        
                    string = ','.join([pars[key] for key in fun_pars])
                    
                    pl.add_arrows(np.array(pontoon.node.x0[:3]+pontoon.tmat[0,:][:3]*200), vec, color='black', **tmat_settings)
                    pl.add_point_labels(np.vstack([pontoon.node.x0[:3]]), [string])
           
        if annotate_pontoon_type:
            for pontoon in self.hydro.pontoons:
                pl.add_point_labels(np.vstack([pontoon.node.x0[:3]]), [pontoon.pontoon_type.label])
       
                    
        # Water plane    
        if plot_water:
            sizes = np.abs(bounds[0::2]-bounds[1::2])
            xsize = sizes[0]+2*waterplane_padding[0]
            ysize = sizes[1]+2*waterplane_padding[1]

            if self.hydro is not None and self.hydro.environment is not None:
                origin[2] = self.hydro.environment.waterlevel
            else:
                origin[2] = 0.0
            water_level = pv.Plane(i_size=xsize, j_size=ysize, center=origin)
            pl.add_mesh(water_level, color='#00aaff', opacity=0.3)
        
        if title is not None:
            pl.add_title(title, color='black', font_size=12)
        
        if show:
            pl.show()
        
        return pl


    @property
    def theta_int(self):
        return self.hydro.seastate.theta_int

    
    @property
    def local(self):
        if self.modal_dry is not None:
            return self.modal_dry.local_phi
        else:
            return False
    
    @property
    def dry_K(self):
        if self.modal_dry is None:
            return 0
        else:
            return self.modal_dry.K

    @property
    def dry_C(self):
        if self.modal_dry is None:
            return 0
        else:
            return self.modal_dry.C

    @property
    def dry_M(self):
        if self.modal_dry is None:
            return 0
        else:
            return self.modal_dry.M

    def get_dry_phi(self, key='hydro'):
        if self.modal_dry is None:
            return np.eye(self.hydro.ndofs)
        else:
            return self.modal_dry.get_phi(key=key)
        
    def get_phi(self, key='hydro', normalize=True, ensure_maxreal=True):
        phi_tot = self.get_dry_phi(key=key) @ self.results.psi
        
        if ensure_maxreal:
            phi_tot = maxreal(phi_tot)
            
        if normalize:
            phi_tot, __ = normalize_phi(phi_tot)
        
        return phi_tot

    def get_result_psd(self, key='hydro', index=None, convert_to=None, modes=None):
        # index: applied after transformation to requested csys (convert_to)

        ix, ix_3d = self.get_mode_ix(modes)            

        if key is not None:    
            sel_phi = self.get_dry_phi(key=key)[:, ix]
        else:
            return self.results.S

        if key in ['hydro', 'full']:   #only supported for hydro and full currently
            if key == 'full':
                tmat = self.tmat_full*1
                if convert_to is not None:
                    print('Local nodal csys is strictly not possible - averaging introduced (use with care).')
            else:
                tmat = self.tmat*1

            if (convert_to == 'global') and (self.local):
                sel_phi = tmat.T @ sel_phi
    
            elif (convert_to == 'local') and (not self.local):
                sel_phi = tmat @ sel_phi

        elif convert_to is not None:
            raise ValueError('convert_to only supported for key="hydro" or "full"; use convert_to=None (output will be given in csys of phi matrix with specified key.')
    
        if index is not None and np.ndim(index)==0:
            index = np.array([index])
            sel_phi = sel_phi[index, :]
        elif index is not None:
            sel_phi = sel_phi[index, :]

        psd = transform_3dmat(self.results.S[ix_3d], sel_phi.T)
        
        if psd.shape[0]==1:
            psd = psd.flatten()
            psd = np.real(psd)
            
        return psd 

    def get_result_std(self, key=None, h=lambda om: 1.0, modes=None):
        ix, ix_3d = self.get_mode_ix(modes)  
 
        if key is None:
            return np.sqrt(np.diag(np.trapz(np.real(self.results.S[ix_3d]*h(self.results.omega)), self.results.omega, axis=2)))
        else:
            return np.sqrt(var_from_modal(self.results.omega, self.results.S[ix_3d]*h(self.results.omega), self.get_dry_phi(key=key)[:,ix]))
    
    def get_result_expmax(self, T, key=None, h=lambda om: 1.0, modes=None):
        ix, ix_3d = self.get_mode_ix(modes)   
        if key is None:
            return expmax_from_modal(self.results.omega, self.results.S[ix_3d]*h(self.results.omega), np.eye(self.results.S.shape[0])[:, ix], T)
        else:
            return expmax_from_modal(self.results.omega, self.results.S[ix_3d]*h(self.results.omega), self.get_dry_phi(key=key)[:,ix], T)  

  
    def get_result_peakfactor(self, T, key='hydro', h=lambda om: 1.0, modes=None):
        ix, ix_3d = self.get_mode_ix(modes)  
        return peakfactor_from_modal(self.results.omega, self.results.S[ix_3d]*h(self.results.omega), self.get_dry_phi(key=key)[:,ix], T)

    def get_gen_S(self, psi=None):
        if psi is None:
            psi = np.real(self.results.psi) # self.results.psi has already been run through maxreal
    
        n_modes = psi.shape[0]

        # CPSD matrices: dry modal response
        Sy = self.results.S*1

        # Estimate Sg
        Sg = Sy*0
        psi_inv = np.linalg.pinv(psi)
        Sg = transform_3dmat(Sy[:n_modes,:n_modes,:], psi_inv.T)

        return Sg
    
    
    def get_mode_ix(self, modes):
        if modes is None:
            modes = np.arange(0, self.results.S.shape[0])
            
        if np.ndim(modes)==0:
            modes = np.array([modes])
        
        return np.array(modes), np.ix_(modes,modes, range(self.results.S.shape[2]))
        
    @property
    def n_pontoons(self):
        return len(self.hydro.pontoons)
    
    @property
    def tmat(self):
        return block_diag(*[pont.tmat for pont in self.hydro.pontoons])


    
    @property
    def pontoon_x(self):
        return np.array([p.node.x[0] for p in self.hydro.pontoons])

    @property
    def pontoon_y(self):
        return np.array([p.node.x[1] for p in self.hydro.pontoons])
    
    @property
    def pontoon_z(self):
        return np.array([p.node.x[2] for p in self.hydro.pontoons])
    

    def plot_2d_tmats(self, scale=50, show_labels=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        for pontoon in self.hydro.pontoons:
            x,y,__ = pontoon.node.coordinates
            tmat = pontoon.tmat[:2,:2]*scale
            ex, ey = tmat[0,:], tmat[1,:]
            
            ax.plot([x,x+ex[0]], [y,y+ex[1]], color='blue')
            ax.plot([x,x+ey[0]], [y,y+ey[1]], color='red')

            if show_labels:
                ax.text(x,y,pontoon.label, fontsize=7)
            
        ax.axis('equal')
        return ax

    @classmethod
    def from_nodes_and_types(cls, nodes, pontoon_types, rotation=0, prefix_label='pontoon-', 
                             labels=None, pontoon_props=dict(), **kwargs):     
        if np.ndim(rotation)==0:
            rotation = [rotation]*len(nodes)
            
        pontoons = [None]*len(nodes)
        for ix, node in enumerate(nodes):
            if labels is None:
                label = prefix_label+str(ix+1)
            else:
                label = labels[ix]
            
            pontoons[ix] = Pontoon(node, pontoon_types[ix], rotation=rotation[ix], label=label)            
            
        for pontoon in pontoons:
            for key in pontoon_props:
                setattr(pontoon, key, pontoon_props[key])
        if len(pontoons)>0:
            return cls(hydro=Hydro(pontoons), **kwargs)
        else:
            return cls(**kwargs)
    
            
    def run_eig(self, normalize=False, print_progress=True, freq_kind=True, 
                include=['aero', 'hydro', 'drag_elements', 'drag_pontoons'], 
                smooth=None, aero_sections=None, **kwargs):
        
        include_dict = self.establish_include_dict(include)

        if (('aero' in include_dict['k'] or 'aero' in include_dict['c']) and hasattr(self, 'aero')
            and self.aero is not None and self.aero.Kfun is None):
                self.aero.prepare_aero_matrices(aero_sections=aero_sections)
       
        Kfun, Cfun, Mfun = self.get_system_matrices(include=include_dict)

        if smooth is not None:
            omega = smooth.pop('omega')
            if Kfun is not None:
                K = np.stack([Kfun(omi) for omi in omega], axis=2)
                K = savgol_filter(K, **smooth)
                Kfun = interp1d(omega, K, kind='quadratic')
                
            if Cfun is not None:
                C = np.stack([Cfun(omi) for omi in omega], axis=2)
                C = savgol_filter(C, **smooth)
                Cfun = interp1d(omega, C, kind='quadratic')
                
            if Mfun is not None:
                M = np.stack([Mfun(omi) for omi in omega], axis=2)
                M = savgol_filter(M, **smooth)
                Mfun = interp1d(omega, M, kind='quadratic')
        
        if freq_kind:
            fun = iteig_freq
        else:
            fun = iteig
            
        lambd, psi, __ = fun(Kfun, Cfun, Mfun, print_progress=print_progress, 
                               normalize=normalize, **kwargs)
        
        self.results.psi = maxreal(psi)
        self.results.lambd = lambd
        self.results.include_eig = include


    def run_static(self, aero_sections=None, include_selfexctied=['aero']):
        if not hasattr(self.aero, 'F0_m') or (self.aero.F0_m is None):     # check if already computed static forces
            self.precompute_windaction(static=True, aero_sections=aero_sections)     # compute if not present

        if ('aero' in include_selfexctied) and (self.aero is not None) and (self.aero.Kfun is None):
            K_ae, __ = self.get_aero_matrices(omega_reduced=0, aero_sections=aero_sections)
        else:
            K_ae = 0.0

        Ktot = -K_ae + self.dry_K
        self.results.y_static = np.linalg.inv(Ktot) @ self.aero.F0_m
      

    def run_freqsim(self, omega, omega_aero=None, omega_hydro=None, 
                    print_progress=True, interpolation_kind='linear',
                    max_rel_error=0.01, drag_iterations=0, tol=1e-2, include_action=['hydro', 'aero'],
                    include_selfexcited=['hydro', 'aero', 'drag_elements', 'drag_pontoons'], ensure_at_peaks=True, 
                    theta_interpolation='linear', reset_Cquad_lin=True, aero_sections=None):

        include_dict = self.establish_include_dict(include_selfexcited)

        if (('aero' in include_dict['k'] or 'aero' in include_dict['c']) and hasattr(self, 'aero')
             and (self.aero is not None) and (self.aero.Kfun is None)):
                self.aero.prepare_aero_matrices(omega=omega_aero, aero_sections=aero_sections)

        if reset_Cquad_lin and hasattr(self, 'hydro') and self.hydro is not None:
            self.hydro.reset_linearized_drag()
            self.Cquad_lin = 0.0
        
        if ensure_at_peaks and hasattr(self.results, 'wd') and self.results.wd is not None:
            wd = self.results.wd
            omega_peaks = wd[(wd>=np.min(omega)) & (wd<=np.max(omega))]
            omega = np.unique(np.hstack([omega, omega_peaks]))
        
        if omega_hydro is None:
            omega_hydro = omega*1
        
        if omega_aero is None:
            omega_aero = omega*1
        
        # Excitation
        Sqq_m = 0.0
        
        if (self.hydro is not None) and (self.hydro.seastate is not None) and ('hydro' in include_action):
            if hasattr(self.hydro, 'Sqq_hydro') and (self.hydro.Sqq_hydro is not None):
                Sqq_m = Sqq_m + self.hydro.Sqq_hydro(omega)
            else:
                Sqq_hydro = self.evaluate_waveaction(omega_hydro, print_progress=print_progress, 
                                     max_rel_error=max_rel_error, theta_interpolation=theta_interpolation)
                
                Sqq_m = Sqq_m + interp1d(omega_hydro, Sqq_hydro, kind=interpolation_kind, axis=2, 
                                         fill_value=0.0, bounds_error=False)(omega)
            
        if (self.aero is not None) and (self.aero.windstate is not None) and ('aero' in include_action):
            if hasattr(self.aero, 'Sqq_aero') and (self.aero.Sqq_aero is not None):
                Sqq_m = Sqq_m + self.aero.Sqq_aero(omega)
            else:
                Sqq_aero = self.evaluate_windaction(omega_aero, print_progress=print_progress, aero_sections=aero_sections)
                Sqq_m = Sqq_m + interp1d(omega_aero, Sqq_aero, kind=interpolation_kind, 
                                         axis=2, fill_value=0.0, bounds_error=False)(omega)

        def runsim():
            Hnum = eval_3d_fun(self.get_frf_fun(include=include_dict), omega)  # System
            Srr_m = freqsim(Sqq_m, Hnum) # Power-spectral density method
            
            return Srr_m
        
        Srr_m = runsim()

        # Linearized drag damping           
        if drag_iterations>0:
            Clin_pontoons = np.diag(np.zeros(len(self.hydro.pontoons)*6))   # initialize for convergence check
            
            if ('drag_elements' in include_dict['c']) and hasattr(self, 'Cquad'):
                Cquad_model = self.Cquad * 1
                model_converged = False
                phi_model = self.get_dry_phi(key='full')
                vnodes = np.zeros(phi_model.shape[0])
            else:
                Cquad_model = None
                model_converged = True
                
            if 'drag_pontoons' in include_dict['c']:
                pontoons_converged = False
                vp = np.zeros(Clin_pontoons.shape[0])
                Cquad_pontoons = self.hydro.get_all_cquad()
            else:
                Cquad_pontoons = None
                pontoons_converged = True
            
        for n in range(drag_iterations):
            if print_progress:
                pp(n+1, drag_iterations, postfix=f' DRAG LINEARIZATION - RUNNING ITERATION {n+1}')

            #------ PONTOONS ----------------------------
            if 'drag_pontoons' in include_dict['c']:
                vp_prev = vp * 1
                vp = np.sqrt(var_from_modal(omega, Srr_m*omega**2, self.get_dry_phi(key='hydro')))
                Cquad_lin_pontoons = stochastic_linearize(Cquad_pontoons, vp)
                for ix, p in enumerate(self.hydro.pontoons):
                    p.Cquad_lin = Cquad_lin_pontoons[ix*6:ix*6+6, ix*6:ix*6+6]
                            
                if np.linalg.norm(vp - vp_prev) < tol*(np.max([np.linalg.norm(vp), np.linalg.norm(vp_prev)])):
                    pontoons_converged = True
            
            #------ DRAG ELEMENTS -----------------------
            if ('drag_elements' in include_dict['c']) and (Cquad_model is not None): 
                vnodes_prev = vnodes*1
                var_udot = var_from_modal(omega, Srr_m*omega**2, phi_model, only_diagonal=False)
                self.Cquad_lin = phi_model.T @ self.get_Cquad_lin(var_udot, local=False) @ phi_model
                vnodes = np.sqrt(np.diag(var_udot))
                if np.linalg.norm(vnodes - vnodes_prev) < tol*(np.max([np.linalg.norm(vnodes), np.linalg.norm(vnodes_prev)])):
                    model_converged = True
                
            # Convergence check
            if pontoons_converged and model_converged and print_progress:
                print('\n STOPPING ITERATION. Linearized damping converged by assertion with specified tolerance criterion.')
                break
            
            Srr_m = runsim()
        
        # STORE RESULTS
        self.results.omega = omega*1
        self.results.S = Srr_m*1

    def assign_windstate(self, windstate):
        self.aero.windstate = windstate

    def assign_seastate(self, seastate=None):
        # Reset pontoon settings
        self.hydro.assign_to_pontoons(**dict(current_affects_k=None, current_affects_Q=None))
        self.hydro.Sqq_hydro = None
        
        if seastate is None:
            seastate = self.hydro.seastate
        else:
            self.hydro.seastate = seastate

        for pontoon in self.hydro.pontoons:
            pontoon.seastate = seastate
        
        if seastate is not None:
            self.hydro.assign_to_pontoons(**self.hydro.seastate.pontoon_options)

    def prepare_waveaction(self):
        x, y = self.get_all_pos()
    
        xmesh, ymesh = np.meshgrid(x,x), np.meshgrid(y,y)
        dx = xmesh[0]-xmesh[1]
        dy = ymesh[0]-ymesh[1] 
        ds = np.sqrt(dx**2+dy**2)
        
        min_ix = np.argmin(np.max(ds, axis=0))
        max_distances = ds[min_ix,:] 

        for ix, p in enumerate(self.hydro.pontoons):
            p.max_distance = max_distances[ix]
        
        pont_max_distance = self.hydro.pontoons[np.argmax(max_distances)]
        self.get_theta_int = pont_max_distance.get_theta_int
  
    
       
    def get_waveaction(self, omega_k, max_rel_error=0.01, 
                       theta_interpolation='linear', theta_int=None, transform_by_phi=True):
        
        if theta_int is None and self.theta_int is None:
            theta_int = self.get_theta_int(omega_k, max_rel_error=max_rel_error)
        elif theta_int is None:
            theta_int = self.theta_int
        
        Z = np.zeros([self.hydro.ndofs, len(theta_int)]).astype('complex')
     
        for pontoon_index, pontoon in enumerate(self.hydro.pontoons):                
            Z[pontoon_index*6:pontoon_index*6+6, :] = pontoon.get_Z(omega_k, theta_int, 
                                                                    theta_interpolation=theta_interpolation, 
                                                                    local=self.local, x0=self.x0)
        
        if self.hydro.seastate.short_crested:
            # first and last point in trapezoidal integration has 1/2 as factor, others have 1
            # verified to match for loop over angles and trapz integration.
            dtheta = theta_int[1] - theta_int[0]
            Z[:, 0] = np.sqrt(0.5)*Z[:, 0]
            Z[:, -1] = np.sqrt(0.5)*Z[:, -1]       
            Sqq0 = dtheta * Z @ Z.conj().T
        else:
            Sqq0 = Z @ Z.conj().T
        
        if not self.hydro.seastate.options['keep_coherence']:
            Sqq0 = block_diag(*[Sqq0[i*6:(i+1)*6, i*6:(i+1)*6] for i in range(int(Sqq0.shape[0]/6))])
        
        if transform_by_phi:
            return self.hydro.phi.T @ Sqq0 @ self.hydro.phi
        else:
            return Sqq0

    
    def evaluate_windaction(self, omega=None, aero_sections=None, print_progress=True, static=False, **kwargs):
        if aero_sections is None:
            aero_sections = self.aero.elements.keys()

        Sae_m = 0.0
        T = self.aero.windstate.T
        U = self.aero.windstate.U
        rho = self.aero.windstate.rho

        # Sections needs to be merged - all elements are therefore unwrapped
        els = [a for b in [self.aero.elements[sec] for sec in aero_sections] for a in b]   # all requested sections, flattened
        eldef = self.eldef.get_element_subset(self.eldef.get_elements([el.label for el in els]), renumber=False)    # create new eldef for requested elements   
        phi = self.get_dry_phi(key='full')[eldef.global_dofs, :]    # grab relevant phi components
        eldef.assign_global_dofs()
        nodes = eldef.nodes*1
        els = eldef.elements*1
        
        lc = {sec: self.aero.sections[sec].all_lc for sec in aero_sections}                 # dict with all load coefficients
        B = {sec: self.aero.sections[sec].B for sec in aero_sections}                       # dict with all load coefficients
        D = {sec: self.aero.sections[sec].D for sec in aero_sections}                       # dict with all load coefficients
        S = self.aero.get_generic_kaimal(nodes=nodes)
        section_lookup = {sec: self.aero.elements[sec] for sec in aero_sections}

        if static:
            F0_m = windaction_static(lc, els, T, phi, 
                                B, D, U, print_progress=print_progress, rho=rho,
                                section_lookup=section_lookup, nodes=nodes)  
            return F0_m
        else:
            admittance = {sec: self.aero.sections[sec].admittance for sec in aero_sections}  
            Sae_m_fun = windaction(omega, S, lc, els, T, phi, 
                                B, D, U, print_progress=print_progress, rho=rho,
                                section_lookup=section_lookup, nodes=nodes, admittance=admittance, **kwargs)   
        
            Sae_m = np.stack([Sae_m_fun(om_k) for om_k in omega], axis=2)

            return Sae_m
    
    def evaluate_windaction_static(self, aero_sections=None, print_progress=True, **kwargs):
        return self.evaluate_windaction(aero_sections=None, print_progress=True, static=True, **kwargs)

    def precompute_windaction(self, omega, include=['dynamic'], interpolation_kind='linear', **kwargs):

        if 'dynamic' in include:
            self.aero.Sqq_aero = interp1d(omega, self.evaluate_windaction(omega=omega, static=False, **kwargs), 
                                      kind=interpolation_kind, axis=2, fill_value=0.0, bounds_error=False)
        if 'static' in include:
            self.aero.F0_m = self.evaluate_windaction(static=True, **kwargs)


    def precompute_waveaction(self, omega, interpolation_kind='linear', method='standard', **kwargs):
        if method=='standard':
            Sqq0 = self.evaluate_waveaction(omega, **kwargs)
            
        elif method=='fft':
            if not self.hydro.seastate.homogeneous:
                raise ValueError('Only method standard" is supported for inhomogeneous conditions')
            # Sqq0 = transform_3dmat(waveaction_fft(self.hydro.pontoons, omega, **kwargs), self.hydro.phi)
            raise NotImplementedError('FFT not implemented yet. Will be at some point.')
        
        elif method in ['fourier', 'cos2s', 'cos2s-fourer'] :
            if not self.hydro.homogeneous:
                raise ValueError('Only method standard" is supported for inhomogeneous conditions')
            
            raise NotImplementedError('Fourier (cos 2s) not implemented yet. Will be at some point.')
        
        self.hydro.Sqq_hydro = interp1d(omega, Sqq0, kind=interpolation_kind, axis=2, fill_value=0.0, bounds_error=False)
            


    def evaluate_waveaction(self, omega, max_rel_error=0.01, print_progress=True, theta_int=None,
                            theta_interpolation='quadratic', transform_by_phi=True, **kwargs):
        
        if theta_int is None:
            theta_int = self.theta_int

        if transform_by_phi:
            ndofs = self.hydro.phi.shape[1]
        else:
            ndofs = self.hydro.phi.shape[0]
        
        Sqq = np.zeros([ndofs, ndofs, len(omega)]).astype('complex')

        for k, omega_k in enumerate(omega):
            Sqq[:,:,k] = self.get_waveaction(omega_k, max_rel_error=max_rel_error, theta_int=theta_int, transform_by_phi=transform_by_phi,
                                             theta_interpolation=theta_interpolation, **kwargs)
            
            if print_progress:
                pp(k+1, len(omega), postfix=' ESTABLISHING WAVE EXCITATION     ')
                
        return Sqq
   
    # FRF
    def get_added_frf(self, omega_k, inverse=False):
        if inverse:
            return -omega_k**2*self.get_added_M(omega_k) + 1j*omega_k*self.get_added_C(omega_k) + self.get_added_K(omega_k)
        else:   
            return np.linalg.inv(-omega_k**2*self.get_added_M(omega_k) + 
                                 1j*omega_k*self.get_added_C(omega_k) + self.get_added_K(omega_k))

    def get_dry_frf(self, omega_k, inverse=False):
        if inverse:
            return -omega_k**2*self.dry_M + 1j*omega_k*self.dry_C + self.dry_K
        else:   
            return np.linalg.inv(-omega_k**2*self.dry_M + 1j*omega_k*self.dry_C + self.dry_K)

    def get_aero_K(self, omega_k):
        if self.aero is not None:
            K_aero = self.aero.Kfun(omega_k)
        else:
            K_aero = 0.0
        
        return -K_aero
          
    def get_aero_C(self, omega_k):        
     
        if self.aero is not None:
            C_aero = self.aero.Cfun(omega_k)
        else:
            C_aero = 0.0
            
        return -C_aero
              
            
    def get_aero_M(self, omega_k):
        return 0.0

    def get_hydro_K(self, omega_k):
        if self.hydro is not None:
            mat = self.initialize_const_matrix()
            
            for ix, p in enumerate(self.hydro.pontoons):        
                mat[ix*6:ix*6+6, ix*6:ix*6+6] = p.get_K(omega_k, local=self.local)
            return self.hydro.phi.T @ mat @ self.hydro.phi
        else:
            return 0.0
         
    def get_hydro_C(self, omega_k):        
        if self.hydro is not None:
            mat = self.initialize_const_matrix()
            for ix, p in enumerate(self.hydro.pontoons):        
                mat[ix*6:ix*6+6, ix*6:ix*6+6] = p.get_C(omega_k, local=self.local)
            return self.hydro.phi.T @ mat @ self.hydro.phi
        else:
            return 0.0
              
    
    def get_hydro_M(self, omega_k):
        if self.hydro is not None:
            mat = self.initialize_const_matrix()
            for ix, p in enumerate(self.hydro.pontoons):        
                mat[ix*6:ix*6+6, ix*6:ix*6+6] = p.get_M(omega_k, local=self.local)
    
            return self.hydro.phi.T @ mat @ self.hydro.phi
        else:
            return 0.0

    
    def get_added_K(self, omega_k):
        if self.hydro is not None:
            mat = self.initialize_const_matrix()
            
            for ix, p in enumerate(self.hydro.pontoons):        
                mat[ix*6:ix*6+6, ix*6:ix*6+6] = p.get_K(omega_k, local=self.local)
            K_hydro = self.hydro.phi.T @ mat @ self.hydro.phi
        else:
            K_hydro = 0.0
            
        if self.aero is not None:
            K_aero = self.aero.Kfun(omega_k)
        else:
            K_aero = 0.0
        
        return K_hydro - K_aero
            
    def get_added_C(self, omega_k):        
        if self.hydro is not None:
            mat = self.initialize_const_matrix()
            for ix, p in enumerate(self.hydro.pontoons):        
                mat[ix*6:ix*6+6, ix*6:ix*6+6] = p.get_C(omega_k, local=self.local)
            C_hydro = self.hydro.phi.T @ mat @ self.hydro.phi
        else:
            C_hydro = 0.0
            
        if self.aero is not None:
            C_aero = self.aero.Cfun(omega_k)
        else:
            C_aero = 0.0
            
        return C_hydro - C_aero
              
            
    def get_added_M(self, omega_k):
        if self.hydro is not None:
            mat = self.initialize_const_matrix()
            for ix, p in enumerate(self.hydro.pontoons):        
                mat[ix*6:ix*6+6, ix*6:ix*6+6] = p.get_M(omega_k, local=self.local)
    
            return self.hydro.phi.T @ mat @ self.hydro.phi
        else:
            return 0.0

    @staticmethod
    def establish_include_dict(include):
        if type(include) is not dict:
            include_dict = dict()
            keys = ['k', 'c', 'm']
            for key in keys:
                include_dict[key] = include*1
            return include_dict
        else:
            replacement_keys = dict(stiffness='k', mass='m', damping='c')
            include_dict = {replacement_keys.get(k, k): v for k, v in include.items()}
            return include_dict
            
    def get_system_matrices(self, include=['hydro', 'aero', 'drag_elements', 'drag_pontoons']):
        include_dict = self.establish_include_dict(include)

        # Stiffness
        if ('hydro' in include_dict['k'] and self.hydro is not None) and ('aero' in include_dict['k'] and self.aero is not None):
            Kfun = fun_const_sum(self.get_added_K, self.dry_K)
        elif ('hydro' in include_dict['k'] and self.hydro is not None):
            Kfun = fun_const_sum(self.get_hydro_K, self.dry_K)  
        elif ('aero' in include_dict['k'] and self.aero is not None):
            Kfun = fun_const_sum(self.get_aero_K, self.dry_K)
        else:
            Kfun = lambda omega_k: self.dry_K
            
        # Damping
        dry_C = self.dry_C*1
        if 'drag_elements' in include_dict['c']:
            dry_C += self.Cquad_lin 
        if ('drag_pontoons' in include_dict['c'] and self.hydro is not None):
            dry_C += self.Cquad_lin_pontoons
            
        if ('hydro' in include_dict['c'] and self.hydro is not None) and ('aero' in include_dict['c']  and self.aero is not None):
            Cfun = fun_const_sum(self.get_added_C, dry_C)
        elif ('hydro' in include_dict['c'] and self.hydro is not None):
            Cfun = fun_const_sum(self.get_hydro_C, dry_C)
        elif 'aero' in include_dict['c'] and self.aero is not None:
            Cfun = fun_const_sum(self.get_aero_C, dry_C)
        else:
            Cfun = lambda omega_k: dry_C

        # Mass
        if ('hydro' in include_dict['m'] and self.hydro is not None) and ('aero' in include_dict['m']  and self.aero is not None):
            Mfun = fun_const_sum(self.get_added_M, self.dry_M)
        elif ('hydro' in include_dict['m'] and self.hydro is not None):
            Mfun = fun_const_sum(self.get_hydro_M, self.dry_M)
        elif 'aero' in include_dict['m'] and self.aero is not None: 
            Mfun = fun_const_sum(self.get_aero_M, self.dry_M)
        else:
            Mfun = lambda omega_k: self.dry_M

        return Kfun, Cfun, Mfun
    
    
    def get_frf_fun(self, include=['hydro', 'aero', 'drag_elements'], opt = 0):
        Kfun, Cfun, Mfun = self.get_system_matrices(include)

        def frf(omega_k):
            return np.linalg.inv(-omega_k**2*Mfun(omega_k) + 1j*omega_k*Cfun(omega_k) + Kfun(omega_k))
        
        def imp(omega_k):
            return (-omega_k**2*Mfun(omega_k) + 1j*omega_k*Cfun(omega_k) + Kfun(omega_k))
        if opt == 0:
            return frf
        else:
            return imp
        
    def get_node_ix(self, nodelabel):
        ix = np.where(self.eldef.get_node_labels()==nodelabel)[0]
        if len(ix)>0:
            ix = int(ix[0])
        else:
            ix = None

        return ix
    
    def get_node_ixs(self, nodelabels):
        return [self.get_node_ix(nodelabel) for nodelabel in nodelabels]
    
    def get_aerosection_phi_and_x(self, key):
        ixs = self.aero.phi_ixs[key]
        x = np.vstack([node.coordinates for node in self.aero.eldef[key].nodes])

        return ixs, x
