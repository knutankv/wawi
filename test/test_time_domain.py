import unittest
import warnings
import numpy as np
from numpy.testing import assert_allclose

from wawi.time_domain import simulate_mdof, simulate_mdof_direct


def cpsd_function(n_dofs, complex_valued=False, indefinite=False, seed=0):
    '''Kaimal-like auto spectra with exponential coherence between random points (optionally with phase).'''
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-50, 50, [n_dofs, 3])
    d = np.linalg.norm(xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :], axis=2)
    sigma = rng.uniform(1, 3, n_dofs)
    V = 30.0

    def S_fun(omega):
        omega = np.atleast_1d(omega)
        f = omega[:, np.newaxis]/(2*np.pi)
        fhat = f*100/V
        S_auto = sigma**2*6.8*fhat/(1 + 1.5*6.8*fhat)**(5/3)/f/(2*np.pi)
        coh = np.exp(-omega[:, np.newaxis, np.newaxis]/(2*np.pi)*8*d/V)
        if indefinite:      # coherence above 1 between two points at low frequencies: those matrices need the PSD fallback
            low = omega < 1.0
            coh[low, 0, 1] = coh[low, 1, 0] = 1.05
        S = np.sqrt(S_auto)[:, :, np.newaxis]*np.sqrt(S_auto)[:, np.newaxis, :]*coh
        if complex_valued:
            S = S*np.exp(1j*omega[:, np.newaxis, np.newaxis]*(xyz[:, 0][np.newaxis, :, np.newaxis] - xyz[:, 0][np.newaxis, np.newaxis, :])/V)
        return S

    return S_fun


class TestSimulateDirect(unittest.TestCase):
    T, fs = 128.0, 4.0

    def compare(self, S_fun, n_dofs, **kwargs):
        domega = 2*np.pi/self.T
        omega = np.arange(3, 250)*domega        # band on fft frequencies
        S = np.moveaxis(S_fun(omega), 0, 2).copy()

        np.random.seed(11)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            p_ref, t_ref, alpha = simulate_mdof(S, omega, fs=self.fs, tmax=self.T, return_phase_angles=True, **kwargs)
            p, t = simulate_mdof_direct(S_fun, n_dofs, self.fs, self.T, omega_min=omega[0], omega_max=omega[-1],
                                        phase_angles=alpha, batch_size=50, **kwargs)
        assert_allclose(t, t_ref)
        assert_allclose(p, p_ref, rtol=0, atol=1e-10*np.abs(p_ref).max())

    def test_equals_simulate_mdof(self):
        self.compare(cpsd_function(8), 8)

    def test_complex_cpsd(self):
        self.compare(cpsd_function(6, complex_valued=True), 6)

    def test_component_scaling(self):
        self.compare(cpsd_function(6), 6, component_scaling=[1.0, 10.0, 0.1])

    def test_psd_fallback(self):
        S_fun = cpsd_function(10, indefinite=True, seed=2)
        with self.assertWarnsRegex(RuntimeWarning, 'not positive definite'):
            simulate_mdof_direct(S_fun, 10, self.fs, self.T, omega_min=3*2*np.pi/self.T, omega_max=249*2*np.pi/self.T)
        self.compare(S_fun, 10)

    def test_same_random_stream(self):
        '''Without given phase angles, both functions draw the same phase angles from np.random.'''
        S_fun = cpsd_function(5)
        omega = np.arange(3, 250)*2*np.pi/self.T
        np.random.seed(4)
        p_ref, _ = simulate_mdof(np.moveaxis(S_fun(omega), 0, 2).copy(), omega, fs=self.fs, tmax=self.T)
        np.random.seed(4)
        p, _ = simulate_mdof_direct(S_fun, 5, self.fs, self.T, omega_min=omega[0], omega_max=omega[-1])
        assert_allclose(p, p_ref, rtol=0, atol=1e-10*np.abs(p_ref).max())


if __name__ == '__main__':
    unittest.main()
