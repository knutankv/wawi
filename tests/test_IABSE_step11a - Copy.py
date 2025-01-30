
import pytest
import numpy as np
from math import isclose
import dill

# use the local wawi (Github folder) instead of the installed version (remove this when using the installed version)
import sys
import os
sys.path.insert(0, os.path.abspath('C:\\Users\\aksef\\Documents\\GitHub\\wawi'))

# import functions
from wawi.io import import_folder
from wawi.model import Windstate
from wawi.wind import ADs, flatplate_ads
from wawi.wind import itflutter_cont_naive
from wawi.general import eval_3d_fun

model_folder = './tests/models/model_2a'

def AD_dict(AD_funs):
    AD_s = dict(
        A1 = lambda v: -AD_funs['a_fun'][0](v), # sign convention
        A2 = AD_funs['a_fun'][1],
        A3 = AD_funs['a_fun'][2],
        A4 = lambda v: -AD_funs['a_fun'][3](v),
        A5 = lambda v: -AD_funs['a_fun'][4](v),
        A6 = lambda v: -AD_funs['a_fun'][5](v),
        
        H1 = AD_funs['h_fun'][0],
        H2 = lambda v: -AD_funs['h_fun'][1](v),
        H3 = lambda v: -AD_funs['h_fun'][2](v),
        H4 = AD_funs['h_fun'][3],
        H5 = AD_funs['h_fun'][4],
        H6 = AD_funs['h_fun'][5],
        
        P1 = AD_funs['p_fun'][0],
        P2 = lambda v: -AD_funs['p_fun'][1](v),
        P3 = lambda v: -AD_funs['p_fun'][2](v),
        P4 = AD_funs['p_fun'][3],
        P5 = AD_funs['p_fun'][4],
        P6 = AD_funs['p_fun'][5],
        )
    return AD_s

def iabse_2a_windstate(mean_v):
    windstate = Windstate(mean_v, 
                                  90, 
                                  Iu=0.1, 
                                  Iw=0.05, 
                                  Au=6.8, Aw=9.4,  # not used in von Karman
                                  Cuy=10.0, Cwy=6.5,
                                  Cuz=10.0, Cwz=3.0,
                                  Lux=200.0, Lwx=20.0,
                                  x_ref=[0,0,0], rho=1.22,
                                    options = {
                                        'spectra_type': 'vonKarman'
                                        }
                                  )

    return windstate

davenport = lambda fred: 2*(7*fred-1+np.exp(-7*fred))/(7*fred)**2

omega = np.linspace(0.001, 6, 1000)

# import the model and assign properties
model = import_folder(model_folder)
model.modal_dry.xi0 = .3e-2

# assign ADs (BB3 ADs)
with open( model_folder + '/AD_funs_BB3_scanlan.pkl', 'rb') as file:
    AD_funs = dill.load(file)
AD_s = AD_dict(AD_funs)
for key in model.aero.sections:
    model.aero.sections[key].ADs = ADs(**AD_s)

V = 30
# assign windstate
model.aero.windstate = iabse_2a_windstate(30)
# admittance
for key in model.aero.sections:
    model.aero.sections[key].Admittance = lambda fred: np.full((4, 3), davenport(fred))

model.run_eig(w_initial=model.modal_dry.omega_n.tolist(), freq_kind=True, itmax=100)

# run analysis
model.run_freqsim(omega,
                    include_selfexcited=['aero'], 
                    include_action=['aero'],
                    print_progress=False, merge_aero_sections=True) 
# RMS responses
stds = model.get_result_std(key = 'full')
# global dofs
global_dof_ix1 = model.eldef.node_dof_lookup(36)[1:4]
global_dof_ix2 = model.eldef.node_dof_lookup(26)[1:4]

assert isclose(stds[global_dof_ix1][0], RMS_horz_exp1, rel_tol = 10e-2) # midspan horizontal 
assert isclose(stds[global_dof_ix1][1], RMS_vert_exp1, rel_tol = 10e-2) # midspan vertical 
assert isclose(stds[global_dof_ix1][2]*31/2, RMS_tors_exp1, rel_tol = 10e-2) # midspan torsional 

assert isclose(stds[global_dof_ix2][0], RMS_horz_exp2, rel_tol = 15e-2) # q-span horizontal 
assert isclose(stds[global_dof_ix2][1], RMS_vert_exp2, rel_tol = 15e-2) # q-span vertical 
assert isclose(stds[global_dof_ix2][2]*31/2, RMS_tors_exp2, rel_tol = 20e-2) # q-span torsional 
