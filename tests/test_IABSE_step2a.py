import pytest
import numpy as np
from math import isclose
import dill

# use the local wawi (Github folder) instead of the installed version (remove this when using the installed version)
import sys
import os
sys.path.insert(0, os.path.abspath('C:\\Users\\aksef\\Documents\\GitHub\\wawi'))

from wawi.io import import_folder
from wawi.model import Windstate
from wawi.wind import ADs
from wawi.wind import itflutter_cont_naive

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

@pytest.mark.parametrize(
    'V, expected_f, expected_zeta, tol',
    [(15, 0.274, 0.008 , 0.1), 
     (30, 0.267, 0.014, 0.1), 
     (45, 0.254 , 0.019, 0.1),
     (60, 0.234, 0.015, 0.1),
     (68, 0.220 , 0.004, 0.15)],
)

def test_in_wind_frequencies_and_damping( V, expected_f, expected_zeta, tol):
    # importing the relevant model
    model = import_folder(model_folder)
    model.modal_dry.xi0 = .3e-2

    # assign ADs (BB3 ADs)
    with open( model_folder + '/AD_funs_BB3_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    AD_s = AD_dict(AD_funs)
    for key in model.aero.sections:
        model.aero.sections[key].ADs = ADs(**AD_s)

    model.aero.windstate = iabse_2a_windstate(V)
    model.run_eig(w_initial=model.modal_dry.omega_n.tolist(), freq_kind=True, itmax=100)
    lambd = model.results.lambd

    f9 =  np.abs(lambd[8].imag) / (2*np.pi) 
    zeta9 =  -lambd[8].real/ np.abs(lambd[8])

    assert isclose(f9, expected_f, rel_tol = tol)
    assert isclose(zeta9, expected_zeta, rel_tol = tol)

# flutter speed
def test_flutter():
    expected_flutter_speed = 69.8 # m/s (reference value from paper)
    tol = 0.01
    with open( model_folder + '/AD_funs_BB3_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    ad_dict = AD_dict(AD_funs)

    # importing the relevant model
    model = import_folder(model_folder)
    model.modal_dry.xi0 = .3e-2
    B = model.aero.sections['girder'].B

    Ms = model.dry_M
    Cs = model.dry_C
    Ks = model.dry_K
    phi = model.get_dry_phi('full')

    # calculate the flutter speed
    x_coor = [node.coordinates[0] for node in model.eldef.nodes ]

    res = itflutter_cont_naive(Ms, Cs, Ks, phi, x_coor, ad_dict, B, V=0.1, rho=1.22, dV=5, 
                overshoot_factor=0.5, itmax={}, tol={}, print_progress=True)

    assert isclose(res['V'][-1], expected_flutter_speed, rel_tol = tol)


# spectra of vertical and torsional response at 45 m/s
def test_RS_spectra():
   
    V = 45

    # omega_axis 
    omega = (2*np.pi)*np.array([0.086,	0.1,	0.123,	0.142,	0.151,	0.179,	0.2,	0.236,	0.247,	0.262,	0.278,	0.283,	0.3, 0.379,])

    Sz_midspan_expected = np.array([2.5816E+00	,
                                    3.9291E+00	,
                                    1.2792E+00	,
                                    8.3683E-01	,
                                    3.6548E-01	,
                                    6.1153E-02	,
                                    2.5616E-02	,
    ])
    Szeq_midspan_expected = np.array([9.7440E-03	,
                                    4.1529E-03	,
                                    1.1582E-02	,
                                    1.2131E-02	,
                                    1.2402E-02	,
                                    1.4801E-02	,
                                    2.0026E-02	,
                                    9.3995E-02	,
                                    3.5115E-01	,
                                    2.9514E-01	,
                                    3.1238E-02	,
                                    2.0048E-02	,
                                    6.3728E-03	,
    ])
    Sz_qspan_expected = np.array([5.7483E+00	,
                                2.3605E+00	,
                                7.0894E-01	,
                                4.9097E-01	,
                                2.1767E-01	,
                                3.9699E-02	,
                                1.7792E-02	,
    ])
    Szeq_qspan_expected = np.array([4.0150E-03	,
                                    3.5677E-03	,
                                    4.8657E-03	,
                                    5.4711E-03	,
                                    5.5522E-03	,
                                    6.1715E-03	,
                                    7.8843E-03	,
                                    3.2631E-02	,
                                    1.1894E-01	,
                                    1.0044E-01	,
                                    1.1859E-02	,
                                    8.1749E-03	,
                                    3.9972E-03	,
                                    1.5413E-02	,
    ])

    # import the model and assign properties
    model = import_folder(model_folder)
    model.modal_dry.xi0 = .3e-2

    # assign ADs (BB3 ADs)
    with open( model_folder + '/AD_funs_BB3_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    AD_s = AD_dict(AD_funs)
    for key in model.aero.sections:
        model.aero.sections[key].ADs = ADs(**AD_s)

    # assign windstate
    model.aero.windstate = iabse_2a_windstate(V)
    # admittance
    for key in model.aero.sections:
        model.aero.sections[key].Admittance = lambda fred: np.full((4, 3), davenport(fred))

    # run analysis
    model.run_freqsim(omega,
                        include_selfexcited=['aero'], 
                        include_action=['aero'],
                        print_progress=False, merge_aero_sections=True,
                                            ensure_at_peaks=False) # keep the given frequency axis


    # extract PSD - at the midpsan
    global_dof_ix_ms = model.eldef.node_dof_lookup(36) # node 36 is the midspan node
    S = model.get_result_psd(key='full', 
                            index=global_dof_ix_ms[1:4])
    # extract PSD - at the quarter-span
    global_dof_ix_qs = model.eldef.node_dof_lookup(26) # node 26 is the quarter-span node
    Sq = model.get_result_psd(key='full', 
                            index=global_dof_ix_qs[1:4])

    assert np.allclose((2*np.pi)*S[1,1,range(len(Sz_midspan_expected))], Sz_midspan_expected, rtol=1e-1)
    assert np.allclose((31/2)**2*(2*np.pi)*S[2,2,range(len(Szeq_midspan_expected))], Szeq_midspan_expected, rtol=2e-1)

    assert np.allclose((2*np.pi)*Sq[1,1,range(len(Sz_qspan_expected))], Sz_qspan_expected, rtol=1e-1)
    assert np.allclose((31/2)**2*(2*np.pi)*Sq[2,2,range(len(Szeq_qspan_expected))], Szeq_qspan_expected, rtol=35e-2)

# buffeting responses (RMS values)
@pytest.mark.parametrize(
    'Vbuff, RMS_horz_exp1, RMS_vert_exp1, RMS_tors_exp1, RMS_horz_exp2, RMS_vert_exp2, RMS_tors_exp2',
    [(15.,   0.019,	0.058,	0.009, 0.015,	0.078,	0.006  ), 
     (30.,	0.128,	0.238,	0.051, 0.091,	0.296,	0.035  ), 
     (45.,	0.316,	0.508,	0.150, 0.225,	0.563,	0.103  ),
     (60.,	0.557,	0.879,	0.417, 0.413,	0.880,	0.273  ),
    ],
    )

def test_RMS_response(Vbuff, RMS_horz_exp1, RMS_vert_exp1, RMS_tors_exp1, RMS_horz_exp2, RMS_vert_exp2, RMS_tors_exp2):
    omega = np.linspace(0.0001, 4, 3000)

    # import the model and assign properties
    model = import_folder(model_folder)
    model.modal_dry.xi0 = .3e-2

    # assign ADs (BB3 ADs)
    with open( model_folder + '/AD_funs_BB3_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    AD_s = AD_dict(AD_funs)
    for key in model.aero.sections:
        model.aero.sections[key].ADs = ADs(**AD_s)

    # assign windstate
    model.aero.windstate = iabse_2a_windstate(Vbuff)
    # admittance
    for key in model.aero.sections:
        model.aero.sections[key].Admittance = lambda fred: np.full((4, 3), davenport(fred))

    #model.run_eig(w_initial=model.modal_dry.omega_n.tolist(), freq_kind=True, itmax=100) # alters integration 

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

    assert isclose(stds[global_dof_ix1][0], RMS_horz_exp1, rel_tol = 15e-2) # midspan horizontal 
    assert isclose(stds[global_dof_ix1][1], RMS_vert_exp1, rel_tol = 15e-2) # midspan vertical 
    assert isclose(stds[global_dof_ix1][2]*31/2, RMS_tors_exp1, rel_tol = 15e-2) # midspan torsional 

    assert isclose(stds[global_dof_ix2][0], RMS_horz_exp2, rel_tol = 15e-2) # q-span horizontal 
    assert isclose(stds[global_dof_ix2][1], RMS_vert_exp2, rel_tol = 15e-2) # q-span vertical 
    assert isclose(stds[global_dof_ix2][2]*31/2, RMS_tors_exp2, rel_tol = 20e-2) # q-span torsional 


