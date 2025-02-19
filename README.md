![WAWI logo](https://raw.githubusercontent.com/knutankv/wawi/main/wawi-logo-animated.svg)
=======================

What is wawi?
=======================
WAWI is a Python toolbox for prediction of response of structures exposed to wind and wave excitation. The package is still under development in its alpha stage, and documentation and testing will be completed along the way.


Installation 
========================
Either install via PyPI as follows:

```
pip install wawi
```

or install directly from github:

```
pip install git+https://www.github.com/knutankv/wawi.git@main
```


Quick start
=======================
Assuming a premade WAWI-model is created and saved as `MyModel.wwi´, it can be imported as follows:

```python
from wawi.model import Model, Windstate, Seastate

model = Model.load('MyModel.wwi')
model.n_modes = 50                  # number of dry modes to use for computation
omega = np.arange(0.001, 2, 0.01)   # frequency axis to use for FRF
```

A windstate (U=20 m/s with origin 90 degrees and other required properties) and a seastate (Hs=2.1m, Tp=8.3s, gamma=8, s=12, heading 90 deg) is created and assigned to the model:

```python
# Wind state
U0 = 20.0
direction = 90.0
windstate = Windstate(U0, direction, Iu=0.136, Iw=0.072, Au=6.8, Aw=9.4, Cuy=10.0, Cwy=6.5, Lux=115, Lwx=9.58)
model.assign_windstate(windstate)

# Sea state
Hs = 2.1
Tp = 8.3
gamma = 8
s = 12
theta0 = 90.0
seastate = Seastate(Tp, Hs, gamma, theta0, s)
model.assign_seastate(seastate)
```

Then, response predictions can be run by the `run_freqsim` method or iterative modal analysis (combined system) conducted by `run_eig`:

```python
model.run_eig(include=['hydro', 'aero'])
model.run_freqsim(omega)
```

The results are stored in `model.results`, and consists of modal representation of the response (easily converted to relevant physical counterparts using built-in methods) or modal parameters of the combined system (natural frequencies, damping ratio, mode shapes). 

For more details and recommendations regarding the analysis setup, it is referred to the examples provided and the code reference.

Examples
=======================
Examples are provided as Jupyter Notebooks in the [examples folder](https://github.com/knutankv/wawi/tree/main/examples).

References
=======================

Citation
=======================
Zenodo research entry:

Support
=======================
Please [open an issue](https://github.com/knutankv/wawi/issues/new) for support.

