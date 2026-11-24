"""Import wawi and build the namespace shared by the console, explorer and forms."""
import importlib
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # figures are rendered to PNG for the browser

import numpy as np  # noqa: E402

WAWI_MODULES = ["model", "io", "wave", "wind", "structural", "modal", "general", "plot",
                "time_domain", "random", "signal", "prob", "tools", "fe", "identification",
                "wind_code"]

IMPORT_STATUS = {}


def _try(name):
    try:
        mod = importlib.import_module(name)
        IMPORT_STATUS[name] = {"ok": True, "version": getattr(mod, "__version__", None)}
        return mod
    except Exception as exc:  # pragma: no cover - depends on the user's environment
        IMPORT_STATUS[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return None


COMPAT_SHIMS = []


def install_numpy_compat():
    """wawi/BEEF still call a few NumPy aliases removed in NumPy 2.x; restore them."""
    for name, target in (("trapz", "trapezoid"), ("Inf", "inf"), ("NaN", "nan"), ("in1d", "isin")):
        try:
            getattr(np, name)
        except AttributeError:
            if hasattr(np, target):
                setattr(np, name, getattr(np, target))
                COMPAT_SHIMS.append(f"np.{name} -> np.{target}")


def build_namespace():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    install_numpy_compat()
    ns = {"np": np, "numpy": np}
    import matplotlib.pyplot as plt
    ns["plt"] = plt

    wawi = _try("wawi")
    if wawi is None:
        raise SystemExit("Could not import wawi: " + IMPORT_STATUS["wawi"]["error"] +
                         "\nInstall it with:  pip install wawi")
    ns["wawi"] = wawi
    for m in WAWI_MODULES:
        mod = _try(f"wawi.{m}")
        if mod is not None:
            ns[m if m not in ("io", "random", "signal") else f"w{m}"] = mod
    IMPORT_STATUS["wawi.ext.abq"] = {"ok": False, "error": "Runs only inside the Abaqus/CAE Python interpreter"}

    # Convenience: the main classes directly
    wm = ns.get("model")
    if wm is not None:
        for cls in ["Model", "Seastate", "Windstate", "Hydro", "Aero", "AeroSection", "Pontoon",
                    "PontoonType", "Node", "ModalDry", "Environment", "ScreeningCase", "DragElement", "Results"]:
            if hasattr(wm, cls):
                ns[cls] = getattr(wm, cls)
        ns["wmodel"] = wm
        del ns["model"]  # 'model' is reserved for the active Model object
    ns["model"] = None
    ns["seastate"] = None
    ns["windstate"] = None

    for opt in ["pyvista", "beef", "dill", "plotly", "scipy"]:
        _try(opt)
    ns["__file_dir__"] = os.getcwd()
    return ns
