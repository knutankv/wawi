# WAWI GUI

A browser-based graphical user interface for [WAWI](https://github.com/knutankv/wawi), the toolbox for
wave- and wind-induced response prediction of structures (floating bridges, suspension bridges, …).

The GUI runs a small local Python server that holds a live WAWI session. The page in your browser drives that
session: every analysis is the real `wawi` code, run in a background job queue with live progress.

## Install and start

The GUI is part of the `wawi` package (`wawi.gui`), so no separate installation is needed:

```bash
pip install wawi            # or: pip install -e .  from the repository root

wawi-gui                    # or: python -m wawi.gui
```

From Python: `import wawi.gui; wawi.gui.launch()`.

A browser tab opens at `http://localhost:8050/?token=…`. Stop the server with Ctrl+C in the terminal.

Options: `--port 8060`, `--no-browser`, `--host 0.0.0.0` (expose on the network – see *Security*),
`--token <fixed token>`, `--verbose`.

No extra packages are needed: the server uses only the Python standard library, and the charts use the
`plotly.min.js` that ships with the `plotly` Python package (a wawi dependency). If that is missing, the page
falls back to the Plotly CDN.

## What you can do

| Area | Functionality (wawi API used) |
|---|---|
| **Model → Open & save** | Load `.wwi` (`Model.from_wwi`, `wawi.io.load_model`), import model folders with all file options (`wawi.io.import_folder`), convert folders to `.wwi` (`convert_to_wwi`), save / download (`Model.to_wwi`), upload files/folders from the browser, keep several models as variables (`Model.copy`) and switch between them |
| **Model → Build from objects** | Pontoon types from NPZ (`PontoonType.from_numeric`), WADAM (`from_wadam`), WAMIT (`io.import_wamit_mat/force`) or expressions; pontoon table (edit, paste, load `pontoon.json`); dry modes from `modal.json`/NPZ/expressions (`ModalDry`); environment; `Model.from_nodes_and_types` or hydro-only `Model(hydro=Hydro(…))` |
| **Model → Overview / Edit** | Dry modal table, pontoons, pontoon types, aero sections, drag elements, FE summary. Edit `n_modes`, `xi0`, `m_min`, `x0`, phases lead, environment, pontoon-type `Cd`/area, per-pontoon `Cd`, `Cquad` and current flags, aero coefficients/admittance/ADs (quasi-steady, flat plate, custom), `assign_drag_elements`, remove parts, reset results/precomputed actions |
| **Model → Pontoon types / Aerodynamics** | M(ω), C(ω), K(ω) and |Q(ω,θ)| of each pontoon type; aerodynamic derivatives of each section |
| **Model → 3D view** | Interactive geometry with pontoons, water plane, wave and wind directions; dry/wet mode shapes and static deformation with scale and animation; optional native pyvista rendering (`Model.plot`, `plot_mode`), `export_modeshapes`, `plot_2d_tmats` |
| **Environment** | `Seastate` with every parameter (homogeneous or as functions of x, y; current; options, pontoon options, ranges, origin), JSON open/save (`Seastate.from_json` format), S(ω), D(θ), 2D spectrum, parameters per pontoon, `plot_S2d`, `plot_dir_and_crests`; `Windstate` (Kaimal/von Kármán, intensities, length scales, all 9 coherence coefficients, inhomogeneous mean-wind scaling), JSON open/save, spectra, coherence, mean wind along the deck; sea-surface simulation (`Seastate.simulate`, `animate_surface` GIF) |
| **Analysis** | Iterative modal analysis (`run_eig`, `iteig_freq`/`iteig` with all solver options and smoothing); in-wind modal sweep over wind speeds; frequency-domain response (`run_freqsim` with all arguments, incl. stochastic drag linearisation); static wind response; multimodal flutter (`wind.itflutter_cont` / `_naive`); sea-state screening (`ScreeningCase`, grid or paired, or from JSON); load time histories (`evaluate_waveaction`/`evaluate_windaction` + `time_domain.simulate_mdof`); `prepare_aero_matrices`, `precompute_waveaction`, `precompute_windaction` |
| **Results** | Mode tables and shapes, modal mass/stiffness/damping (`Results.mdiag/kdiag/cdiag`); std, expected maxima, peak factors along the structure for any mode-shape key (`get_result_std/expmax/peakfactor`, displacement/velocity/acceleration or custom h(ω), local/global csys, mode subsets, aero-section subsets, pinned curves for comparison); auto and cross spectra by node, pontoon or index (`get_result_psd`, `get_gen_S`); static displacements; system matrices and FRF (`get_system_matrices`, `get_frf_fun`); CSV and NPZ export |
| **Calculators** | Wave spectra (`jonswap`, `jonswap_dnv`, `pm2`, `peak_enhancement`, `swh_from_gamma_alpha_Tp`), spreading functions, dispersion relation with depth and current, wave coherence (`get_coh_cos2s`), turbulence spectra and Beaufort scale, terrain roughness (`wind_code`), flat-plate / quasi-steady ADs, spectral statistics (`wawi.random`), Gumbel paper (`prob`), Rayleigh damping, JONSWAP fitting (`signal.fit_jonswap`), time-series simulation, linearised drag |
| **API explorer** | Every class, function, method and property of every wawi module, with signatures and docstrings, callable with your own arguments (results stored as session variables, matplotlib figures shown) |
| **Variables & console** | Inspect any object (attributes, properties, methods, dict/list items), plot arrays, download variables (NPZ/CSV/JSON/pickle), and a Python console that runs in the same session |

Long computations run in a background queue; the bar at the bottom shows progress (wawi's own progress
output) and the full log, and running jobs can be cancelled.

Expressions: almost every input field accepts a Python expression evaluated in the session namespace, e.g.
`np.arange(0.001, 2, 0.005)`, `lambda x: 2.1 - 0.5*x/1500`, `model.results.omega` or a variable you created
in the console. Available names: `np`, `plt`, `wawi`, `model`, `seastate`, `windstate`, the wawi modules
(`wave`, `wind`, `structural`, `modal`, `general`, `plot`, `time_domain`, `fe`, … plus `wio`, `wrandom`,
`wsignal` for `wawi.io/random/signal`) and the classes (`Model`, `Seastate`, `Windstate`, `PontoonType`, …).

## Notes

* **Abaqus export** (`wawi.ext.abq`) only runs inside Abaqus/CAE's Python and is therefore listed but not
  callable from the GUI.
* **NumPy 2:** wawi/BEEF still use a few aliases removed in NumPy 2 (`np.trapz`, `np.Inf`, `np.in1d`). The GUI
  restores them at start-up so those code paths work; this is reported in the API explorer.
* **Workarounds for wawi issues** (applied in the GUI only, wawi itself is not modified):
  `Model.run_static` calls methods that don't exist, so the static analysis is implemented with the same
  formulation in the GUI; the end of `itflutter_cont` can fail on ragged result lists, which the GUI tolerates;
  `Seastate(depth=…)` ignores the depth and `angle_unit='rad'` fails, both handled when building sea states;
  `Seastate.__str__` fails for inhomogeneous sea states, so the GUI formats those itself.
* **pyvista rendering** is optional. It needs pyvista with working off-screen rendering; the interactive 3D view
  does not depend on it.

## Security

The server executes Python code you type (it is a front end to a Python session). By default it listens on
127.0.0.1 only, every request must carry the random session token from the start-up URL, and requests with a
foreign `Host` header are refused. Only use `--host 0.0.0.0` on a trusted network.

## Layout

```
wawi/gui/
  server.py        HTTP server, token check, routing, uploads/downloads
  session.py       namespace, background job queue, stdout capture, cancellation
  bootstrap.py     imports wawi, builds the session namespace, NumPy-2 aliases
  api_model.py     load/import/build/edit/save, geometry, mode shapes, pontoon types, ADs, pyvista
  api_env.py       sea state, wind state, sea-surface simulation
  api_analysis.py  eig, sweep, freqsim, static, flutter, screening, load simulation, preparation
  api_results.py   statistics, spectra, static results, system matrices, export
  api_tools.py     calculators
  api_explorer.py  API explorer, variables, inspector, console
  static/          index.html, app.css and the JavaScript front end
```
