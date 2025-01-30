
import pytest
import numpy as np
from math import isclose

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

model_folder = './tests/models/model_11a'

def iabse_11a_windstate(mean_v):
    windstate = Windstate(mean_v, 
                                    90, 
                                    Iu=0.0, 
                                    Iw=0.05, 
                                    Au=6.8, Aw=9.4, 
                                    Cuy=0.0, Cwy=0.0,
                                    Cuz=0.0, Cwz=0.0,
                                    Lux=0.0, Lwx=20.0,
                                    x_ref=[0,0,0], rho=1.22,
                                    options = {
                                        'spectra_type': 'vonKarman'
                                        }
                                    )
    return windstate

davenport = lambda fred: 2*(7*fred-1+np.exp(-7*fred))/(7*fred)**2


# in-wind frequencies and damping ratio

@pytest.mark.parametrize(
    'V, expected_f, expected_zeta, tol',
    [(15, np.array([0.0987, 0.279]), np.array([0.0399, 0.0096]), 0.05), 
     (30, np.array([0.0999, 0.2691]), np.array([0.0921, 0.0189]), 0.1), 
     (45, np.array([0.1014, 0.2561]), np.array([0.1689, 0.0309]), 0.1), 
     (60, np.array([0.1027, 0.2340]), np.array([0.3034, 0.0418]), 0.15), 
     (75, np.array([0.0829, 0.1994]), np.array([0.5349, 0.0148]), 0.25)],
)


# aerodynamic stability

# input - wind velocities / output - frequencies, damping ratios ( test only the unstable mode)

def test_in_wind_frequencies_and_damping( V, expected_f, expected_zeta, tol):

    # importing the relevant model
    model = import_folder('./tests/models/model_11a')
    model.modal_dry.xi0 = .3e-2

    # assign flat plate ADs (Theodorsen)
    model.aero.sections['girder0'].ADs = ADs(**flatplate_ads())

    model.aero.windstate = iabse_11a_windstate(V)
    model.run_eig(w_initial=[0.5, 1.25], freq_kind=True, itmax=50)
    lambd = model.results.lambd

    f2 =  np.abs(lambd[1].imag) / (2*np.pi) 
    zeta2 =  -lambd[1].real/ np.abs(lambd[1])

    assert isclose(f2, expected_f[1], rel_tol = tol)
    assert isclose(zeta2, expected_zeta[1], rel_tol = tol)


# flutter speed
def test_flutter():

    expected_flutter_speed = 77.45
    tol = 0.01
    ad_dict = flatplate_ads()

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
    

# buffeting at 45 m/s (compare with analytical)
def test_mean_speed_45():

    # at 45 m/s & f = 0.278 Hz
    V = 45
    f = 0.278

    # expected values (IABSE report)
    sww = 5.26 # wind spectrum
    # impedance matrix
    h_exp = np.array( [[-0.0619+0.0056j, -(-0.1492-0.0811j)], [-(0.01+0.0419j), -1.1559+0.5382j]] )*1e6
    # PSD response
    S_exp = np.transpose(np.array( [ [0.0167, -0.007j], [0.007j, 0.00293] ] ))


    model = import_folder(model_folder)
    model.aero.windstate = iabse_11a_windstate(V)

    wind_spectrum_func = model.aero.get_generic_kaimal(nodes = model.eldef.nodes)
    wind_spectrum_ww = 2*np.pi*wind_spectrum_func(f*2*np.pi)[2,2]

    model.modal_dry.xi0 = .3e-2
    model.aero.sections['girder0'].ADs = ADs(**flatplate_ads())
    model.aero.prepare_aero_matrices()
    omega = np.array([0.001, f])*np.pi*2
    HH = eval_3d_fun(model.get_frf_fun(opt = 1), omega)

    model.aero.sections['girder0'].Admittance = lambda fred: np.full((4, 3), davenport(fred))
    model.run_freqsim(omega,
                    include_selfexcited=['aero'], 
                    include_action=['aero'],
                    print_progress=False, merge_aero_sections=True)

    global_dof_ix = np.array([2,3])
    S = model.get_result_psd(key='full', 
                            index=global_dof_ix)
    
    assert isclose(wind_spectrum_ww, sww, rel_tol = 0.01)
    assert np.allclose(HH[:,:,1], h_exp, rtol=1e-2)
    assert np.allclose((2*np.pi)*S[:,:,-1], S_exp, rtol=5e-2)


# buffeting responses (spectra)

@pytest.mark.parametrize(
    'Vbuff, RS_vert_exp, RS_tors_exp',
    [
    (15, np.array([1.24E-01,	1.19E-01,	1.37E-01,	4.09E+00,	5.66E-04,	1.75E-04,	2.15E-04,	2.18E-05,	7.02E-06,]), 
      np.array([2.53E-03,	2.39E-03,	1.56E-03,	1.85E-04,	4.94E-04,	9.05E-04,	7.07E-02,	1.02E-03,	5.71E-05])   ), 

    (30, np.array([1.19E+00,	1.15E+00,	1.75E+00,	2.06E+01,	3.02E-02,	1.27E-02,	4.61E-03,	9.67E-04,	3.75E-04,]), 
     np.array([2.43E-02,	2.31E-02,	1.98E-02,	9.30E-04,	2.59E-02,	6.16E-02,	4.23E-01,	3.46E-02,	2.94E-03,])  ), 

    (45, np.array([5.60E+00	,	5.40E+00	,	7.45E+00	,	3.43E+01	,	3.41E-01	,	2.50E-01	,	1.67E-02	,	5.85E-03	,	2.78E-03]), 
     np.array([1.15E-01	,	1.08E-01	,	8.46E-02	,	1.55E-03	,	2.85E-01	,	1.11E+00	,	7.06E-01	,	1.52E-01	,	2.07E-02])  ), 

    (60, np.array([2.39E+01	,	2.25E+01	,	2.36E+01	,	4.45E+01	,	3.44E+00	,	6.03E+00	,	3.41E-02	,	1.62E-02	,	9.21E-03,]), 
     np.array([4.89E-01	,	4.51E-01	,	2.67E-01	,	2.00E-03	,	2.77E+00	,	2.38E+01	,	8.28E-01	,	3.06E-01	,	6.37E-02,])  ), 

    (75, np.array([1.47E+02	,	1.24E+02	,	6.17E+01	,	5.32E+01	,	9.48E+02	,	1.79E+00	,	5.34E-02	,	3.04E-02	,	1.98E-02,]), 
     np.array([3.02E+00	,	2.50E+00	,	6.99E-01	,	2.38E-03	,	7.29E+02	,	6.17E+00	,	8.39E-01	,	4.26E-01	,	1.26E-01,])  ), 

     ]
)



# buffeting responses (spectra)
def test_response_spectra(Vbuff, RS_vert_exp, RS_tors_exp):

    omega_axis = np.array( [0.001, 0.01, 0.052, 0.1, 0.2, 0.234, 0.278, 0.3, 0.35] )*np.pi*2

    model = import_folder(model_folder)
    model.aero.windstate = iabse_11a_windstate(Vbuff)
    model.modal_dry.xi0 = .3e-2
    model.aero.sections['girder0'].ADs = ADs(**flatplate_ads())
    model.aero.sections['girder0'].Admittance = lambda fred: np.full((4, 3), davenport(fred))
    
    model.run_freqsim(omega_axis,
                    include_selfexcited=['aero'], 
                    include_action=['aero'],
                    print_progress=False, merge_aero_sections=True)

    global_dof_ix = np.array([2,3])
    S = model.get_result_psd(key='full', 
                            index=global_dof_ix)

    assert np.allclose((2*np.pi)*S[0,0,:], RS_vert_exp, rtol=5e-2)
    assert np.allclose((31/2)**2*(2*np.pi)*S[1,1,:], RS_tors_exp, rtol=5e-2)



@pytest.mark.parametrize(
    'Vbuff, RMS_vert_exp, RMS_tors_exp',
    [(15, 0.2603, 0.0419  ), 
     (30, 0.778 , 0.2027 ), 
     (45, 1.3404, 0.4792 ), 
     (60, 2.1601, 0.9306 ), 
     (75, 4.4848, 2.8414 )],
)

# buffeting responses (RMS values)  - for integration we use a finer frequency axis
def test_RMS_response(Vbuff, RMS_vert_exp, RMS_tors_exp):

    omega_axis = np.linspace(0.001, 6, 1000)

    model = import_folder(model_folder)
    model.aero.windstate = iabse_11a_windstate(Vbuff)
    model.modal_dry.xi0 = .3e-2
    model.aero.sections['girder0'].ADs = ADs(**flatplate_ads())
    model.aero.sections['girder0'].Admittance = lambda fred: np.full((4, 3), davenport(fred))

    model.run_freqsim(omega_axis,
                    include_selfexcited=['aero'], 
                    include_action=['aero'],
                    print_progress=False, merge_aero_sections=True)
    
    stds = model.get_result_std(key = 'full')

    assert isclose(stds[2], RMS_vert_exp, rel_tol = 10e-2) # vertical  # 5-10% std reported in the paper
    assert isclose(stds[3]*31/2, RMS_tors_exp, rel_tol = 20e-2) # torsional # 10-25% std reported in the paper
