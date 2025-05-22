![WAWI logo](https://raw.githubusercontent.com/knutankv/wawi/main/wawi-logo-animated.svg)

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

![Model attributes](https://raw.githubusercontent.com/knutankv/wawi/main/docs/structure.svg)

Further details regarding hydrodynamic definitions initiated by the `Hydro` class is given below:

![Hydro](https://raw.githubusercontent.com/knutankv/wawi/main/docs/hydro_part.svg)

Further details regarding aerodynamic definitions initiated by the `Aero` class is given below:

![Aero](https://raw.githubusercontent.com/knutankv/wawi/main/docs/aero_part.svg)

Wave conditions
----------------------
The wave field definition is based on the assumption that the two-dimensional wave spectral density can be decomposed into a directional distribution and a one-dimensional wave spectral density, i.e.,  $S_\eta(\omega,\theta) = D(\theta) S(\omega)$. The two factors are defined using these well-known formulations:

* $S(\omega)$: JONSWAP spectrum  (see [Hasselmann et al., 1973](https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_3262854))
* $D(\theta)$: cos-2s directional distribution (see Longuet-Higgins et al., 1963)

Currents are defined by a homogeneous current speed `U` and corresponding direction `thetaU`.

An example of a two-dimensional wave spectral density based on (arbitrarily chosen) parameters $H_s = 2.1$ m, $T_p = 2.1$ s, $\gamma = 4.0$, $s = 10$ and $\theta_0 = 75^\circ$ is shown in this plot:

![2D wave PSD](https://raw.githubusercontent.com/knutankv/wawi/main/docs/wave_S2d.png)

It is noted that you can easily assign custom functions of the `S` and `D` of the seastate (or customly on all pontoons for full control) instead of relying on the built in JONSWAP and cos-2s definitions.

Furthermore, as described in relevant examples, all sea state parameters can be defined as functions of x and y, to accomodate inhomogeneous sea states. In [Kvåle et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0141118723003437), the effects of inhomogeneous sea states were analysed and were shown to be large for swell sea states with spherical wave fronts as illustrated here:

![Animated inhomogeneous swell](https://raw.githubusercontent.com/knutankv/wawi/main/docs/inhomogeneous_swell.gif)

Wind conditions
----------------------
The wind field is defined by single-point turbulence wind spectra (for all turbulence components $u$, $v$ and $w$) and coherence definitions.

By default, wind spectra can be defined using these two definitions:

* Kaimal spectrum defined by length scale parameters ($L^x_u$, $L^x_v$, $L^x_w$), spectral shape parameters ($A_u$, $A_v$ and $A_w$) and turbulence intensities ($I_u$, $I_v$ and $I_w$); see [Kaimal et al., 1972](https://www.climatexchange.nl/projects/alteddy/papers/Kaimal-1972.pdf)
* von Karmán spectrum defined by only length scale parameters ($L^x_u$, $L^x_v$, $L^x_w$) and turbulence intensities ($I_u=\sigma_u/U$, $I_v=\sigma_v/U$ and $I_w=\sigma_w/U$); see [von Kármán, 1948](http://dx.doi.org/10.1073/pnas.34.11.530)

Furthermore, the coherence of the wind field is defined by the nine decay parameters $C_{ux}$, $C_{vx}$, $C_{wx}$, $C_{uy}$, $C_{vy}$, $C_{wy}$, $C_{uz}$ $C_{vz}$, $C_{wz}$; see e.g. [Simiu and Scanlan, 1996](https://library.wur.nl/WebQuery/titel/1606468) for details.

An example of turbulence spectral densities based on (arbitrarily chosen) parameters $I_u=0.136$, $I_v=0.0$, $I_w=0.072$, $L^x_u=115$, $L^x_w=9.58$, $A_u=6.8$ (only relevant for Kaimal-type) and $A_w=9.4$ (only relevant for Kaimal-type) is given below:

![1D turbulence PSD](https://raw.githubusercontent.com/knutankv/wawi/main/docs/wind_psd.png)


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

* Wave modelling and response prediction: [Kvåle et al. (2016)](https://www.sciencedirect.com/science/article/abs/pii/S004579491500334X)
* Inhomogeneous wave modelling: [Kvåle et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0141118723003437)
* Hydrodynamic interaction effects: [Fenerci et al. (2022)](https://www.sciencedirect.com/science/article/pii/S095183392200017X)
* Beam (FE) description of aerodynamic forces: [Øiseth et al. (2012)](https://www.sciencedirect.com/science/article/abs/pii/S0168874X11001880)
* Wave-current interaction: [Fredriksen et al. (2024)](https://www.researchgate.net/profile/Arnt-Fredriksen/publication/386453916_On_the_wave-current_interaction_effect_on_linear_motion_for_floating_bridges/links/6751a40fabddbb448c65cbef/On-the-wave-current-interaction-effect-on-linear-motion-for-floating-bridges.pdf)


Citation
=======================
Please cite the use of this software as follows:

Kvåle, K. A., Fenerci, A., Petersen, Ø. W., & Øiseth, O. A. (2025). WAWI. Zenodo. https://doi.org/10.5281/zenodo.14895014
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14895014.svg)](https://doi.org/10.5281/zenodo.14895014)

Support
=======================
Please [open an issue](https://github.com/knutankv/wawi/issues/new) for support.

