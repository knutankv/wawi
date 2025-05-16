![WAWI logo](https://raw.githubusercontent.com/knutankv/wawi/main/wawi-logo-animated.svg)
=======================

What is WAWI?
=======================
WAWI is a Python toolbox for prediction of response of structures exposed to wind and wave excitation, using a multimodal frequency-domain approach. It supports special features such as:

* Hydrodynamic added mass, radiation damping and hydrodynamic force transfer function from e.g. WAMIT analysis
* Quasisteady wind forcing for buffeting analysis
* Combined effects of wind and waves
* Iterative multimodal flutter
* Iterative modal analysis
* Modelling of motion-induced aerodynamic forces using aerodynamic derivatives
* Inhomogeneous sea states
* Inhomogeneous mean wind (other parameters planned for)
* Current effects on wave excitation
* Stochastic linearization methodology to support linearized effect of quadratic drag damping (both line elements and pontoon objects)
* Object-oriented model setup, including FE description (using Python package BEEF) of beams exposed to aerodynamic forcing

Planned implemented in the near future:

* Fully inhomogeneous wind state definition (all wind field parameters)
* Hydrodynamic interaction effects from multibody analyses
* Second-order wave excitation effects
* Definition of ADs (aerodynamic derivatives) using rational functions

The package is still under development in its alpha stage, and documentation and testing will be completed along the way.


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


How does WAWI work?
======================
By representing both aerodynamic and hydrodynamic motion-induced forces and excitation using a coordinate basis defined by the dry in-vacuum mode shapes of the structure, WAWI is able to versatily predict response based on input from any commercial FE software. The main structure used for response prediction is given in this figure:
![Model](https://raw.githubusercontent.com/knutankv/wawi/main/docs/flowchart.svg)

The object structure of a WAWI model is given here:
![Modelattributes](https://raw.githubusercontent.com/knutankv/wawi/main/docs/structure.svg)

Further details regarding hydrodynamic definitions initiated by the `Hydro` class is given below:
![Hydro](https://raw.githubusercontent.com/knutankv/wawi/main/docs/hydro_part.svg)

Further details regarding aerodynamic definitions initiated by the `Aero` class is given below:
![Aero](https://raw.githubusercontent.com/knutankv/wawi/main/docs/aero_part.svg)


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
windstate = Windstate(U0, direction, Iu=0.136, Iw=0.072,
                      Au=6.8, Aw=9.4, Cuy=10.0, Cwy=6.5,  
                      Lux=115, Lwx=9.58, spectrum_type='kaimal')
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

The model is plotted by envoking this command:

```python
model.plot()
```

which gives this plot of the model and the wind and wave states:
![Model](https://raw.githubusercontent.com/knutankv/wawi/main/docs/model.png)

Then, response predictions can be run by the `run_freqsim` method or iterative modal analysis (combined system) conducted by `run_eig`:

```python
model.run_eig(include=['hydro', 'aero'])
model.run_freqsim(omega)
```

The results are stored in `model.results`, and consists of modal representation of the response (easily converted to relevant physical counterparts using built-in methods) or modal parameters of the combined system (natural frequencies, damping ratio, mode shapes). 

The resulting first mode shape is plotted as follows:

```python
model.plot_mode(0)
```

This results in this plot:
![Mode 1](https://raw.githubusercontent.com/knutankv/wawi/main/docs/mode1.png)

For more details and recommendations regarding the analysis setup, it is referred to the examples provided and the code reference.

Examples
=======================
Examples are provided as Jupyter Notebooks in the [examples folder](https://github.com/knutankv/wawi/tree/main/examples).

The examples are structured in the following folders based on their topic:

* **0 Model generation and setup** - *Describing how a model is defined, either using input files together with the function `wawi.io.import_folder` or using the classes available in the `wawi.model` module directly (in the latter case, the open source Python library BEEF is used to create the required parameters directly in the notebook).*
* **1 Modal analysis** - *Describing how to set up iterative (and incremental) modal analyses to represent the contributions from aerodynamics and hydrodynamices. Also, an example showing how to set up a multi-modal flutter analysis is given.*
* **2 Response prediction** - *Describing how to conduct response analyses using WAWI. This includes assigning wind states, sea states, and the necessary commands to run a frequency domain analysis. Furthermore, more advanced topics such as wave-current interaction, inhomogeneous waves and stochastic linearization to represent quadratic drag damping are given in separate examples. Three models are considered: (i) a simple curved floating bridge, (ii) a single beam, (iii) a suspension bridge.*
* **3 Software interfacing** - *Describing how to export necessary data from other software (limited to Abaqus for now) to construct a WAWI model.*

References
=======================
The following papers provide background for the implementation:

* Beam (FE) description of aerodynamic forces: [Øiseth et al. (2012)](https://www.sciencedirect.com/science/article/abs/pii/S0168874X11001880)
* Wave modelling and response prediction: [Kvåle et al. (2016)](https://www.sciencedirect.com/science/article/abs/pii/S004579491500334X)
* Inhomogeneous wave modelling: [Kvåle et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0141118723003437)
* Hydrodynamic interaction effects: [Fenerci et al. (2022)](https://www.sciencedirect.com/science/article/pii/S095183392200017X)
* Wave-current interaction: [Fredriksen et al. (2024)](https://www.researchgate.net/profile/Arnt-Fredriksen/publication/386453916_On_the_wave-current_interaction_effect_on_linear_motion_for_floating_bridges/links/6751a40fabddbb448c65cbef/On-the-wave-current-interaction-effect-on-linear-motion-for-floating-bridges.pdf)


Citation
=======================
Zenodo research entry: [![DOI](https://zenodo.org/badge/921621297.svg)](https://doi.org/10.5281/zenodo.14895014)

Support
=======================
Please [open an issue](https://github.com/knutankv/wawi/issues/new) for support.

