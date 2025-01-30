
# test wind functions 

# wind state 

# from wawi.model import Windstate

# use the local wawi (Github folder) instead of the installed version
import sys
import os
sys.path.insert(0, os.path.abspath('C:\\Users\\aksef\\Documents\\GitHub\\wawi'))
from wawi.model import Windstate


from math import isclose


def test_windstate():

    V0 = 40.0 # these could be also parametrized

    dir = 45 
    Iu=2.0/V0
    Iw=1.0/V0 
    Au=30 
    Aw=3 
    Cuy=10.0
    Cwy=8.0
    Cuz=10.0
    Cwz=8.0
    Lux=60
    Lwx=60
    x_ref=[0,0,0]
    rho=1.25

    tol = 1e-5

    # Act
    windstate = Windstate(V0, 
                        dir, 
                        Iu=Iu, 
                        Iw=Iw, 
                        Au=Au, Aw=Aw, 
                        Cuy=Cuy, Cwy=Cwy,
                        Cuz=Cuz, Cwz=Cwz,
                        Lux=Lux, Lwx=Lwx,
                        x_ref=x_ref, rho=rho)

    # Assert
    assert isclose(windstate.V0, V0, rel_tol = tol)
    assert isclose(windstate.direction, dir, rel_tol = tol)
    assert isclose(windstate.Iu, Iu, rel_tol = tol)
    assert isclose(windstate.Iw, Iw, rel_tol = tol)
    assert isclose(windstate.Au, Au, rel_tol = tol)
    assert isclose(windstate.Aw, Aw, rel_tol = tol)

    assert isclose(windstate.Cuy, Cuy, rel_tol = tol)
    assert isclose(windstate.Cwy, Cwy, rel_tol = tol)
    assert isclose(windstate.Cuz, Cuz, rel_tol = tol)
    assert isclose(windstate.Cwz, Cwz, rel_tol = tol)

    assert isclose(windstate.Lux, Lux, rel_tol = tol)
    assert isclose(windstate.Lwx, Lwx, rel_tol = tol)
    assert isclose(windstate.rho, rho, rel_tol = tol)

    assert all(abs(a - b) <= tol for a, b in zip(windstate.x_ref, x_ref))





