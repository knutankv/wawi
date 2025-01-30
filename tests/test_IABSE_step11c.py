
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
from wawi.wind import ADs
from wawi.wind import itflutter_cont_naive
from wawi.general import eval_3d_fun

model_folder = './tests/models/model_11c'

   
def AD_dict(AD_funs):
    AD_s = dict(
        A1 = AD_funs['a_fun'][0],
        A2 = AD_funs['a_fun'][1],
        A3 = AD_funs['a_fun'][2],
        A4 = AD_funs['a_fun'][3],
        A5 = AD_funs['a_fun'][4],
        A6 = AD_funs['a_fun'][5],
        
        H1 = AD_funs['h_fun'][0],
        H2 = AD_funs['h_fun'][1],
        H3 = AD_funs['h_fun'][2],
        H4 = AD_funs['h_fun'][3],
        H5 = AD_funs['h_fun'][4],
        H6 = AD_funs['h_fun'][5],
        
        P1 = AD_funs['p_fun'][0],
        P2 = AD_funs['p_fun'][1],
        P3 = AD_funs['p_fun'][2],
        P4 = AD_funs['p_fun'][3],
        P5 = AD_funs['p_fun'][4],
        P6 = AD_funs['p_fun'][5],
        )
    return AD_s


def iabse_11c_windstate(mean_v):
    windstate = Windstate(mean_v, 
                                 90, 
                                 Iu=0.1, 
                                 Iw=0.05, 
                                 Au=6.8, Aw=9.4,  # not used in von Karman
                                 Cuy=0.0, Cwy=0.0,# full coherence assumed
                                 Cuz=0.0, Cwz=0.0,
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
    [(45, np.array([0.0520,	0.1029,	0.2488,]), np.array([0.0142,	0.1660,	0.0429,]), 0.1), 
     (60, np.array([0.0520,	0.1025,	0.2275,]), np.array([0.0169,	0.2879,	0.0366,]), 0.15), 
     (72, np.array([0.0520,	0.0933,	0.2049]), np.array([0.0209,	0.5099,	0.0015,]), 0.25)],
)

def test_in_wind_frequencies_and_damping( V, expected_f, expected_zeta, tol):

    # importing the relevant model
    model = import_folder(model_folder)
    model.modal_dry.xi0 = .3e-2

    # assign ADs (Storebelt ADs)
    with open( model_folder + '/AD_funs_SB_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    AD_s = AD_dict(AD_funs)
    model.aero.sections['girder0'].ADs = ADs(**AD_s)

    model.aero.windstate = iabse_11c_windstate(V)
    model.run_eig(w_initial=[0.3, 0.6, 1.5], freq_kind=True, itmax=50)
    lambd = model.results.lambd

    f3 =  np.abs(lambd[2].imag) / (2*np.pi) 
    zeta3 =  -lambd[2].real/ np.abs(lambd[2])

    assert isclose(f3, expected_f[2], rel_tol = tol)
    assert isclose(zeta3, expected_zeta[2], rel_tol = tol)


# flutter speed
def test_flutter():

    expected_flutter_speed = 72.3
    tol = 0.01
    with open( model_folder + '/AD_funs_SB_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    ad_dict = AD_dict(AD_funs)

    # importing the relevant model
    model = import_folder(model_folder)
    model.modal_dry.xi0 = .3e-2
    B = model.aero.sections['girder0'].B

    Ms = model.dry_M
    Cs = model.dry_C
    Ks = model.dry_K
    phi = model.get_dry_phi('full')

    # calculate the flutter speed
    res = itflutter_cont_naive(Ms, Cs, Ks, phi, np.array([0,1]), ad_dict, B, V=0.1, rho=1.22, dV=5, 
                overshoot_factor=0.5, itmax={}, tol={}, print_progress=True)

    assert isclose(res['V'][-1], expected_flutter_speed, rel_tol = tol)

# buffeting at 45 m/s (compare with reference values)
def test_mean_speed_45():

    # at 45 m/s & f = 0.278 Hz
    V = 45
    f = 0.278

    # expected values (IABSE report)
    swind_exp = np.array( [ [7.2139, 0], [0, 5.2599] ] )
    # admittance matrix (BqBq matrix in wawi)
    Bq_exp = np.array( [ 
        [	0.0088	,	0.0116	],
        [	0.0076	,	0.2536	],
        [	0.099	,	2.0687	],
    ] )*1e4
    # impedance matrix
    h_exp = np.array( [
        [	-6.74+0.11j	,	-0.01+0.01j	,	-0.29-0.26j	],
        [	0.07+0.04j	,	-6.25+0.80j	,	-19.40-0.37j	],
        [	0.06+0.23j	,	0.50+5.39j	,	-158.52+79.71j	],
    ] )*1e4
    # sign convention
    h_exp[2, :] *= -1
    h_exp[:, 2] *= -1

    # PSD response
    S_exp = np.array( [ [0.26-0j, 1+1j, 1-1j ], 
                        [1-1j, 20-0j, 0-13j ], 
                        [1+1j, 0+13j, 8-0j ], 
                        ])*1e-4
    # sign convention 
    S_exp[2, :] *= -1
    S_exp[:, 2] *= -1

    model = import_folder(model_folder)
    model.aero.windstate = iabse_11c_windstate(V)

    wind_spectrum_func = model.aero.get_generic_kaimal(nodes = model.eldef.nodes)
    wind_spectrum_uu = 2*np.pi*wind_spectrum_func(f*2*np.pi)[0,0]
    wind_spectrum_ww = 2*np.pi*wind_spectrum_func(f*2*np.pi)[2,2]

    model.modal_dry.xi0 = .3e-2
    # assign ADs (Storebelt ADs)
    with open( model_folder + '/AD_funs_SB_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    AD_s = AD_dict(AD_funs)
    model.aero.sections['girder0'].ADs = ADs(**AD_s)
    model.aero.prepare_aero_matrices()
    omega = np.array([0.001, f])*np.pi*2
    HH = eval_3d_fun(model.get_frf_fun(opt = 1), omega)

    model.aero.sections['girder0'].Admittance = lambda fred: np.full((4, 3), davenport(fred))
    model.run_freqsim(omega,
                    include_selfexcited=['aero'], 
                    include_action=['aero'],
                    print_progress=False, merge_aero_sections=True)

    global_dof_ix = np.array([1,2,3])
    S = model.get_result_psd(key='full', 
                            index=global_dof_ix)
    
    assert isclose(wind_spectrum_ww, swind_exp[1,1], rel_tol = 0.01)
    assert isclose(wind_spectrum_uu, swind_exp[0,0], rel_tol = 0.01)
    assert np.allclose(np.diag(HH[:,:,1]), np.diag(h_exp), rtol=1e-2)
    assert np.allclose((2*np.pi)*np.diag(S[:,:,-1]), np.diag(S_exp), rtol=5e-2)


@pytest.mark.parametrize(
    'Vbuff, RS_horz_exp, RS_vert_exp, RS_tors_exp',
    [
    (15, 
    np.array([4.00E-02,	2.12E-02,	4.12E+00,	7.00E-05,	6.03E-07,	2.03E-07,	9.11E-08,	3.36E-08,	1.09E-08,]), 
    np.array([6.36E-02,	6.05E-02,	6.64E-02,	9.94E-01,	2.93E-04,	9.17E-05,	1.73E-05,	9.17E-06,	3.41E-06,]),
    np.array([1.52E-03,	1.38E-03,	8.59E-04,	6.95E-05,	2.86E-04,	5.35E-04,	1.16E-02,	5.03E-04,	3.11E-05,]),    
    ), 

    (30, 
    np.array([3.30E-01,	2.76E-01,	8.88E+01,	2.70E-03,	2.84E-05,	1.15E-05,	3.72E-06,	1.85E-06,	6.55E-07,]), 
    np.array([5.89E-01,	5.75E-01,	8.34E-01,	8.70E+00,	1.65E-02,	7.54E-03,	4.73E-04,	2.98E-04,	1.64E-04,]),
    np.array([1.40E-02,	1.33E-02,	1.07E-02,	8.31E-05,	1.60E-02,	4.26E-02,	9.19E-02,	1.40E-02,	1.46E-03,]),    
    ), 

    (45, 
     np.array([1.16E+00,	1.10E+00,	4.02E+02,	1.74E-02,	3.07E-04,	2.02E-04,	2.56E-05,	1.55E-05,	5.92E-06,]), 
     np.array([2.58E+00,	2.55E+00,	3.56E+00,	1.74E+01,	2.18E-01,	2.11E-01,	2.03E-03,	1.45E-03,	1.08E-03,]),
     np.array([6.12E-02,	5.91E-02,	4.50E-02,	2.94E-04,	1.95E-01,	1.11E+00,	1.89E-01,	5.54E-02,	9.36E-03,]),
    ), 

    (60, 
     np.array([3.00E+00,	2.99E+00,	1.01E+03,	6.09E-02,	3.95E-03,	2.53E-03	,8.00E-05,	5.47E-05,	2.47E-05]), 
     np.array([9.43,	9.35,	10.9,	24.7,	2.46,	1.92,	0.00423,	0.00351,	0.00328,]),
     np.array([0.221,	0.215,	0.143,	0.000666,	2,	9.1	, 0.279,	0.115,	0.0272,]),
    ), 
    ]
)

# buffeting responses (spectra)
def test_response_spectra(Vbuff, RS_horz_exp, RS_vert_exp, RS_tors_exp):

    omega_axis = np.array( [0.001, 0.01, 0.052, 0.1, 0.2, 0.234, 0.278, 0.3, 0.35] )*np.pi*2

    # import the model and assign properties
    model = import_folder(model_folder)
    model.aero.windstate = iabse_11c_windstate(Vbuff)
    model.modal_dry.xi0 = .3e-2
    # assign ADs (Storebelt ADs)
    with open( model_folder + '/AD_funs_SB_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    AD_s = AD_dict(AD_funs)
    model.aero.sections['girder0'].ADs = ADs(**AD_s)
    model.aero.sections['girder0'].Admittance = lambda fred: np.full((4, 3), davenport(fred))

    # run analysis
    model.run_freqsim(omega_axis,
                      include_selfexcited=['aero'], 
                      include_action=['aero'],
                      print_progress=False, merge_aero_sections=True)
    
    # extract PSD
    global_dof_ix = np.array([1,2,3])
    S = model.get_result_psd(key='full', 
                            index=global_dof_ix)
    
    assert np.allclose((2*np.pi)*S[0,0,:], RS_horz_exp, rtol=5e-2)
    assert np.allclose((2*np.pi)*S[1,1,:], RS_vert_exp, rtol=5e-2)
    assert np.allclose((31/2)**2*(2*np.pi)*S[2,2,:], RS_tors_exp, rtol=5e-2)


# buffeting responses (RMS values)  - for integration we use a finer frequency axis

@pytest.mark.parametrize(
    'Vbuff, RMS_horz_exp, RMS_vert_exp, RMS_tors_exp',
    [(15, 0.0952,	0.173,	0.0211, ), 
     (30, 0.4319,	0.5596,	0.1134,), 
     (45, 0.99,	0.9888,	0.2984, ),  # note that the lateral response is given wrong in the table in the paper. It is taken from the spectral plot.
     (60, 1.7349,	1.5557,	0.7181 ), 
    ])

def test_RMS_response(Vbuff, RMS_horz_exp, RMS_vert_exp, RMS_tors_exp):

    omega_axis = np.linspace(0.001, 6, 1000)

    model = import_folder(model_folder)
    model.aero.windstate = iabse_11c_windstate(Vbuff)
    model.modal_dry.xi0 = .3e-2
    # assign ADs (Storebelt ADs)
    with open( model_folder + '/AD_funs_SB_scanlan.pkl', 'rb') as file:
        AD_funs = dill.load(file)
    AD_s = AD_dict(AD_funs)
    model.aero.sections['girder0'].ADs = ADs(**AD_s)
    model.aero.sections['girder0'].Admittance = lambda fred: np.full((4, 3), davenport(fred))
    
    # run analysis
    model.run_freqsim(omega_axis,
                    include_selfexcited=['aero'], 
                    include_action=['aero'],
                    print_progress=False, merge_aero_sections=True)
    
    # RMS responses
    stds = model.get_result_std(key = 'full')

    assert isclose(stds[1], RMS_horz_exp, rel_tol = 15e-2) # horizontal  # 5-15% std reported in the paper
    assert isclose(stds[2], RMS_vert_exp, rel_tol = 15e-2) # vertical  # 5-15% std reported in the paper
    assert isclose(stds[3]*31/2, RMS_tors_exp, rel_tol = 15e-2) # torsional # 10-20% std reported in the paper
