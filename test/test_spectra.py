import unittest
from unittest import TestCase as TC

from wawi.model import Model, Seastate, Windstate
from wawi.wave import jonswap, jonswap_numerical, jonswap_dnv
import numpy as np
from numpy.testing import assert_array_equal
        
class TestJonswap(TC):
    def test_numerical(self):
        omega = np.array([0.1,1,2,3,4,9.2])
        Tp = 1.234
        Hs = 2.321
        gamma = 3.2

        S1 = jonswap(Hs, Tp, gamma)(omega)
        S2 = jonswap_numerical(Hs, Tp, gamma, omega)

        assert_array_equal(S1, S2)

    def test_swh(self):
        omega = np.arange(0, 2.0, 0.001)
        Hs = 2.0
        Tp = 4.5
        gamma = 3.0

        Snum = jonswap_numerical(Hs, Tp, gamma, omega)
        Hs_S = 4*np.sqrt(np.trapz(Snum, omega))
        self.assertAlmostEqual(Hs_S, Hs)