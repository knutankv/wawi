"""System, file and model endpoints (load / import / build / edit / save / inspect / geometry)."""
import json
import os
import string
import sys
import time
from pathlib import Path

import numpy as np

from .bootstrap import IMPORT_STATUS
from .common import (state_str, DOF_NAMES, Params, arclength, collect_figures, deg, ensure_list, fig_to_png,
                     model_phi_keys, node_xyz, register_download, temp_path)
from .serialize import arr, clean_float, jsonable, short_repr
from .server import api
from .session import UserError


# =====================================================================  system / jobs
@api("system/state")
def system_state(sess, p):
    m = sess.model
    info = {
        "python": sys.version.split()[0],
        "imports": IMPORT_STATUS,
        "cwd": os.getcwd(),
        "home": str(Path.home()),
        "busy": sess.jobs.busy,
        "jobs": [j.to_dict() for j in list(sess.jobs.jobs.values())[-40:]][::-1],
        "model": model_summary(m) if m is not None else None,
        "seastate": state_brief(sess.ns.get("seastate")),
        "windstate": state_brief(sess.ns.get("windstate")),
    }
    return info


def state_brief(s):
    if s is None:
        return None
    try:
        return {"type": type(s).__name__, "str": state_str(s), "name": getattr(s, "name", None)}
    except Exception as exc:
        return {"type": type(s).__name__, "str": f"<error: {exc}>"}


def model_summary(m):
    d = {"repr": short_repr(m)}
    try:
        d["n_modes"] = int(m.n_modes) if m.modal_dry is not None else None
        d["n_modes_total"] = int(list(m.modal_dry.phi_full.values())[0].shape[1]) if m.modal_dry is not None else None
        d["has_hydro"] = m.hydro is not None
        d["has_aero"] = m.aero is not None
        ws = m.aero.windstate if m.aero is not None else None
        d["wind_U0"] = clean_float(getattr(ws, "U0", None)) if ws is not None else None
        d["has_eldef"] = m.eldef is not None
        d["n_pontoons"] = len(m.hydro.pontoons) if m.hydro is not None else 0
        d["aero_sections"] = list(m.aero.sections.keys()) if m.aero is not None and m.aero.sections else []
        d["phi_keys"] = model_phi_keys(m)
        d["n_drag_elements"] = len(getattr(m, "drag_elements", []) or [])
        d["seastate"] = state_str(m.hydro.seastate) if m.hydro is not None and m.hydro.seastate is not None else None
        d["windstate"] = state_str(m.aero.windstate) if m.aero is not None and m.aero.windstate is not None else None
        r = m.results
        d["has_eig"] = r.lambd is not None
        d["has_freqsim"] = r.S is not None
        d["has_static"] = getattr(r, "y_static", None) is not None
        d["include_eig"] = jsonable(r.include_eig)
        d["aero_prepared"] = m.aero is not None and getattr(m.aero, "Kfun", None) is not None
        d["wave_precomputed"] = m.hydro is not None and getattr(m.hydro, "Sqq_hydro", None) is not None
        d["wind_precomputed"] = m.aero is not None and getattr(m.aero, "Sqq_aero", None) is not None
        d["path"] = getattr(m, "_gui_path", None)
    except Exception as exc:
        d["error"] = str(exc)
    return d


@api("job/get")
def job_get(sess, p):
    job = sess.jobs.jobs.get(int(p["id"]))
    if job is None:
        raise UserError(f"Unknown job {p['id']}")
    return job.to_dict(with_log=True, with_result=job.status == "done")


@api("job/cancel")
def job_cancel(sess, p):
    return sess.jobs.cancel(int(p["id"])).to_dict()


@api("job/clear")
def job_clear(sess, p):
    for jid in [j for j, job in sess.jobs.jobs.items() if job.status not in ("queued", "running")]:
        sess.jobs.jobs.pop(jid)
    return True


# =====================================================================  file system
@api("fs/list")
def fs_list(sess, p):
    raw = (p.get("path") or "").strip()
    path = Path(raw).expanduser() if raw else Path(os.getcwd())
    if path.is_file():
        path = path.parent
    if not path.exists():
        raise UserError(f"Folder does not exist: {path}")
    entries = []
    try:
        for e in os.scandir(path):
            if e.name.startswith(".") and not p.get("hidden"):
                continue
            try:
                is_dir = e.is_dir()
                st = e.stat()
                entries.append({"name": e.name, "dir": is_dir, "size": None if is_dir else st.st_size,
                                "mtime": st.st_mtime})
            except OSError:
                continue
    except PermissionError:
        raise UserError(f"Permission denied: {path}")
    entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
    drives = []
    if os.name == "nt":
        drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    # Detect wawi model folders for convenience
    names = {e["name"] for e in entries}
    return {"path": str(path.resolve()), "parent": str(path.resolve().parent), "entries": entries,
            "home": str(Path.home()), "cwd": os.getcwd(), "drives": drives, "sep": os.sep,
            "is_model_folder": "modal.json" in names}


# =====================================================================  load / import / save
def _set_model(sess, model, path=None):
    sess.ns["model"] = model
    try:
        model._gui_path = str(path) if path else None
    except Exception:
        pass
    sess.meta.pop("flutter", None)


@api("model/load")
def model_load(sess, p):
    P = Params(sess, p)
    path = P.path("path")
    method = P.str("method", "auto")

    def work():
        import wawi.io
        from wawi.model import Model
        t = time.time()
        print(f"Loading {path} ...")
        if method == "wwi" or (method == "auto" and path.is_file()):
            model = Model.from_wwi(str(path))
        else:
            model = wawi.io.load_model(str(path), save_if_nonexistent=P.bool("save_if_nonexistent"),
                                       sort_nodes_by_x=P.bool("sort_nodes_by_x", True))
        _set_model(sess, model, path)
        print(f"Loaded in {time.time() - t:.1f} s")
        return model_summary(model)
    return sess.run("load", f"Load model {path.name}", work)


@api("model/import_folder")
def model_import_folder(sess, p):
    P = Params(sess, p)
    folder = P.path("folder", kind="dir")
    names = ["pontoon_stl", "aero_sections", "modal", "pontoon", "eldef", "orientations",
             "drag_elements", "pontoon_types"]
    kwargs = {}
    for n in names:
        if n in p:
            v = p.get(n)
            kwargs[n] = None if (v is None or str(v).strip() == "") else str(v).strip()
    kwargs["sort_nodes_by_x"] = P.bool("sort_nodes_by_x", True)
    kwargs["interpolation_kind"] = P.str("interpolation_kind", "linear")

    def work():
        from wawi.io import import_folder
        t = time.time()
        print(f"Importing model folder {folder} ...")
        model = import_folder(str(folder), **kwargs)
        if P.has("n_modes"):
            model.n_modes = P.int("n_modes")
        if P.has("xi0"):
            model.modal_dry.xi0 = P.ev("xi0")
        _set_model(sess, model, folder)
        print(f"Imported in {time.time() - t:.1f} s")
        return model_summary(model)
    return sess.run("import", f"Import folder {folder.name}", work)


@api("model/save")
def model_save(sess, p):
    P = Params(sess, p)
    model = sess.require_model()
    path = P.path("path", must_exist=False)
    if path.suffix == "":
        path = path.with_suffix(".wwi")

    def work():
        path.parent.mkdir(parents=True, exist_ok=True)
        model.to_wwi(str(path))
        print(f"Saved model to {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return {"path": str(path)}
    return sess.run("save", f"Save model {path.name}", work)


@api("model/download")
def model_download(sess, p):
    model = sess.require_model()
    name = (p.get("name") or "model.wwi").strip()

    def work():
        tmp = temp_path(".wwi")
        model.to_wwi(str(tmp))
        return {"download": register_download(sess, tmp, name)}
    return sess.run("save", "Prepare model download", work)


@api("model/convert_to_wwi")
def model_convert(sess, p):
    P = Params(sess, p)
    folder = P.path("model_path", kind="dir")
    output = P.str("output")

    def work():
        import wawi.io
        wawi.io.convert_to_wwi(str(folder), output=output)
        out = output or str(folder / "model.wwi")
        print(f"Wrote {out}")
        return {"path": out}
    return sess.run("convert", f"Convert {folder.name} to .wwi", work)


@api("model/close")
def model_close(sess, p):
    def work():
        sess.ns["model"] = None
        return True
    return sess.run("edit", "Close model", work)


@api("model/copy_to_var")
def model_copy(sess, p):
    model = sess.require_model()
    name = (p.get("name") or "").strip()
    if not name.isidentifier():
        raise UserError("Variable name must be a valid Python identifier")

    def work():
        sess.ns[name] = model.copy()
        return {"name": name}
    return sess.run("edit", f"Copy model to '{name}'", work)


@api("model/activate_var")
def model_activate(sess, p):
    name = (p.get("name") or "").strip()
    obj = sess.ns.get(name)
    from wawi.model import Model
    if not isinstance(obj, Model):
        raise UserError(f"'{name}' is not a wawi Model")

    def work():
        _set_model(sess, obj, getattr(obj, "_gui_path", None))
        return model_summary(obj)
    return sess.run("edit", f"Activate model '{name}'", work)


# =====================================================================  object-based builder
@api("model/build")
def model_build(sess, p):
    """Build a model from pontoon types + pontoon table (+ optional dry modes)."""
    P = Params(sess, p)
    ptypes_spec = p.get("ptypes") or []
    pontoons_spec = p.get("pontoons") or []
    modal_spec = p.get("modal") or {"source": "none"}
    if not ptypes_spec:
        raise UserError("Define at least one pontoon type.")
    if not pontoons_spec:
        raise UserError("Define at least one pontoon.")

    def work():
        from wawi.model import Model, Node, PontoonType, Hydro, Pontoon, ModalDry
        import wawi.io as wio
        types = {}
        for spec in ptypes_spec:
            label = spec.get("label") or f"type{len(types) + 1}"
            src = spec.get("source", "npz")
            extra = {}
            if str(spec.get("Cd") or "").strip():
                extra["Cd"] = np.atleast_1d(sess.ev(spec["Cd"])).astype(float)
            if str(spec.get("area") or "").strip():
                extra["area"] = np.atleast_1d(sess.ev(spec["area"])).astype(float)
            stl = (spec.get("stl_path") or "").strip() or None
            path = Path((spec.get("path") or "").strip()).expanduser()
            interp = spec.get("interpolation_kind") or "linear"
            print(f"Pontoon type '{label}' from {src}: {path}")
            if src == "npz":
                D = np.load(path)
                k = {kk: (spec.get(f"key_{kk}") or default) for kk, default in
                     [("A", "M"), ("B", "C"), ("omega", "omega"), ("Q", "Q"), ("theta", "theta_Q"),
                      ("omegaQ", "omega_Q"), ("Kh", "K"), ("M0", "M0"), ("Cd", "Cd"), ("area", "A")]}
                kw = dict(A=D[k["A"]], B=D[k["B"]], omega=D[k["omega"]])
                if k["Q"] in D:
                    kw.update(Q=D[k["Q"]], theta=D[k["theta"]], omegaQ=D[k["omegaQ"]] if k["omegaQ"] in D else None)
                if k["Kh"] in D:
                    kw["Kh"] = D[k["Kh"]]
                if k["M0"] in D:
                    kw["M0"] = D[k["M0"]]
                if "Cd" not in extra and k["Cd"] in D:
                    extra["Cd"] = D[k["Cd"]]
                if "area" not in extra and k["area"] in D and k["area"] != k["A"]:
                    extra["area"] = D[k["area"]]
                types[label] = PontoonType.from_numeric(interpolation_kind=interp, label=label,
                                                        stl_path=stl, **kw, **extra)
            elif src == "wadam":
                include = spec.get("include") or ["added mass", "added damping", "restoring stiffness", "inertia"]
                types[label] = PontoonType.from_wadam(str(path), interpolation_kind=interp, include=include,
                                                      label=label, stl_path=stl, **extra)
            elif src == "wamit":
                rho = float(sess.ev(spec.get("rho") or "1020"))
                L = float(sess.ev(spec.get("L") or "1.0"))
                g = float(sess.ev(spec.get("g") or "9.80665"))
                A, B, K0, M0, omega = wio.import_wamit_mat(str(path), rho=rho, L=L, g=g)
                omegaQ, theta, Q = wio.import_wamit_force(str(path), rho=rho, L=L, g=g)
                if spec.get("theta_deg", True):
                    theta = np.radians(theta)
                types[label] = PontoonType.from_numeric(interpolation_kind=interp, A=A, M0=M0, B=B, Kh=K0,
                                                        omega=omega, Q=Q, theta=theta, omegaQ=omegaQ,
                                                        label=label, stl_path=stl, **extra)
            elif src == "expr":
                kw = {kk: sess.ev(spec.get(kk)) for kk in ["A", "M0", "B", "Kh", "omega", "Q", "theta", "omegaQ"]
                      if str(spec.get(kk) or "").strip()}
                types[label] = PontoonType.from_numeric(interpolation_kind=interp, label=label,
                                                        stl_path=stl, **kw, **extra)
            else:
                raise UserError(f"Unknown pontoon type source {src}")

        nodes, ptypes, rots, labels = [], [], [], []
        for ix, row in enumerate(pontoons_spec):
            t = row.get("type") or next(iter(types))
            if t not in types:
                raise UserError(f"Pontoon {row.get('label')}: unknown type '{t}'")
            nlabel = row.get("node")
            nlabel = int(sess.ev(nlabel)) if str(nlabel or "").strip() else ix + 1
            nodes.append(Node(nlabel, float(sess.ev(row.get("x") or 0)), float(sess.ev(row.get("y") or 0)),
                              float(sess.ev(row.get("z") or 0))))
            ptypes.append(types[t])
            rots.append(np.radians(float(sess.ev(row.get("rotation") or 0))))
            labels.append(row.get("label") or f"pontoon-{ix + 1}")

        modal_dry = None
        src = modal_spec.get("source", "none")
        if src == "json":
            with open(Path(modal_spec["path"]).expanduser()) as f:
                modal = json.load(f)
            xi0 = np.array(modal.pop("xi0")) if "xi0" in modal else 0.0
            m_min = modal.pop("m_min", 0.0)
            phi_x = modal.pop("phi_x", None)
            local_phi = modal.pop("local", False)
            phi = modal.pop("phi")
            modal_dry = ModalDry(phi, xi0=xi0, local_phi=local_phi, phi_x=phi_x, m_min=m_min, **modal)
        elif src == "npz":
            D = np.load(Path(modal_spec["path"]).expanduser())
            phi = {k[4:]: D[k] for k in D.files if k.startswith("phi_")}
            if not phi:
                raise UserError("NPZ must contain arrays named phi_<key> (e.g. phi_hydro, phi_full)")
            kw = {k: D[k] for k in ("m", "k", "omega_n") if k in D.files}
            if "m" not in kw:
                kw["m"] = None
            modal_dry = ModalDry(phi, xi0=float(D["xi0"]) if "xi0" in D.files else 0.0, **kw)
        elif src == "expr":
            phi = sess.ev(modal_spec.get("phi"))
            kw = {}
            for k in ("m", "k", "omega_n"):
                kw[k] = sess.ev(modal_spec.get(k)) if str(modal_spec.get(k) or "").strip() else None
            modal_dry = ModalDry(phi, xi0=sess.ev(modal_spec.get("xi0") or "0.0"), **kw)

        if modal_dry is not None:
            if "hydro" not in modal_dry.phi_full:
                raise UserError("The dry modes must contain a 'hydro' mode shape matrix (6 DOFs per pontoon).")
            model = Model.from_nodes_and_types(nodes, ptypes, rotation=rots, labels=labels, modal_dry=modal_dry)
        else:
            pontoons = [Pontoon(nodes[i], ptypes[i], rotation=rots[i], label=labels[i]) for i in range(len(nodes))]
            model = Model(hydro=Hydro(pontoons), avoid_eldef=True)
            print("No dry modes defined: model supports wave-excitation evaluation and load simulation "
                  "(hydro.phi = identity), but not eigenvalue/response analyses.")
        env = p.get("environment") or {}
        for k in ("g", "rho", "depth", "waterlevel"):
            if str(env.get(k) or "").strip():
                setattr(model.hydro.environment, k, float(sess.ev(env[k])))
        if P.has("n_modes") and model.modal_dry is not None:
            model.n_modes = P.int("n_modes")
        _set_model(sess, model, None)
        print(f"Built model with {len(nodes)} pontoons and {len(types)} pontoon type(s).")
        return model_summary(model)
    return sess.run("build", "Build model from objects", work)


# =====================================================================  information
def _expr_src(v):
    """Python source that re-creates a scalar/array value (for pre-filling edit forms)."""
    if v is None:
        return ""
    a = np.asarray(v)
    if a.ndim == 0:
        return repr(float(a))
    if a.size <= 400:
        return f"np.array({np.round(a.astype(float), 10).tolist()!r})"
    return ""



@api("model/info")
def model_info(sess, p):
    m = sess.require_model()
    d = {"summary": model_summary(m)}
    md = m.modal_dry
    if md is not None:
        try:
            mi = np.asarray(md.mode_ix)
            d["modal"] = {
                "n_modes": int(md.n_modes),
                "n_total": int(list(md.phi_full.values())[0].shape[1]),
                "m_min": clean_float(getattr(md, "m_min", 0.0)),
                "local_phi": bool(md.local_phi),
                "mode_ix": arr(mi),
                "fn": arr(md.fn), "Tn": arr(md.Tn), "wn": arr(md.wn),
                "xi0": arr(np.atleast_1d(md.xi0)), "m": arr(md.m), "k": arr(md.k),
                "xi0_raw": _expr_src(md._xi0),
                "phi": {k: list(np.shape(v)) for k, v in md.phi_full.items()},
                "phi_x": {k: list(np.shape(v)) for k, v in (md.phi_x or {}).items()},
            }
        except Exception as exc:
            d["modal"] = {"error": str(exc)}
    if m.hydro is not None:
        h = m.hydro
        rows = []
        for pt in h.pontoons:
            xyz = node_xyz(pt.node)
            rows.append({"label": pt.label, "node": jsonable(getattr(pt.node, "label", None)),
                         "x": clean_float(xyz[0]), "y": clean_float(xyz[1]), "z": clean_float(xyz[2]),
                         "rotation": deg(pt.rotation), "type": pt.pontoon_type.label,
                         "Cd": arr(pt.Cd), "area": arr(pt.area),
                         "current_affects_Q": jsonable(getattr(pt, "_current_affects_Q", None)),
                         "current_affects_k": jsonable(getattr(pt, "_current_affects_k", None)),
                         "Cquad": arr(getattr(pt, "Cquad", np.zeros(6)))})
        types = []
        for pt in h.pontoon_types:
            t = {"label": pt.label, "has_M": pt.M is not None, "has_C": pt.C is not None,
                 "has_K": pt.K is not None, "has_Q": pt.Q is not None,
                 "Cd": arr(pt.Cd), "area": arr(pt.area), "stl_path": pt.stl_path,
                 "phases_lead": bool(getattr(pt, "phases_lead", False))}
            if pt.original_omega is not None:
                om = np.asarray(pt.original_omega)
                t["omega"] = [clean_float(om.min()), clean_float(om.max()), int(om.size)]
            if pt.theta is not None:
                th = np.asarray(pt.theta)
                t["theta_deg"] = arr(np.degrees(th), 2)
            types.append(t)
        env = h.environment
        d["hydro"] = {"pontoons": rows, "types": types, "phases_lead": bool(h.phases_lead),
                      "phi_key": h.phi_key,
                      "environment": {"g": clean_float(env.g), "rho": clean_float(env.rho),
                                      "depth": clean_float(env.depth), "waterlevel": clean_float(env.waterlevel)},
                      "seastate": state_str(h.seastate) if h.seastate is not None else None,
                      "x0": arr(getattr(m, "x0", None))}
    if m.aero is not None:
        a = m.aero
        secs = []
        for name, s in (a.sections or {}).items():
            els = a.elements.get(name, []) if a.elements else []
            secs.append({"name": name, "B": jsonable(s.B), "D": jsonable(s.D),
                         **{k: jsonable(getattr(s, k)) for k in ["Cd", "dCd", "Cl", "dCl", "Cm", "dCm"]},
                         "admittance": short_repr(s.admittance, 100) if s.admittance is not None else None,
                         "ad_type": getattr(s.ADs, "type", None), "quasisteady": bool(getattr(s, "quasisteady", False)),
                         "n_elements": len(els)})
        d["aero"] = {"sections": secs, "phi_key": a.phi_key,
                     "windstate": state_str(a.windstate) if a.windstate is not None else None,
                     "prepared": a.Kfun is not None}
    dels = getattr(m, "drag_elements", None) or []
    groups = {}
    for e in dels:
        g = groups.setdefault(e.group, {"group": e.group, "n": 0, "Cd": arr(e.Cd), "D": arr(e.D),
                                        "rho": clean_float(e.rho), "eltype": e.eltype})
        g["n"] += 1
    d["drag_elements"] = list(groups.values())
    if m.eldef is not None:
        try:
            secs = {}
            for el in m.eldef.elements:
                sec = getattr(el, "section", None)
                nm = getattr(sec, "name", None) if sec is not None else None
                secs[str(nm)] = secs.get(str(nm), 0) + 1
            d["eldef"] = {"n_nodes": len(m.eldef.nodes), "n_elements": len(m.eldef.elements),
                          "ndofs": int(m.eldef.ndofs), "sections": secs}
        except Exception as exc:
            d["eldef"] = {"error": str(exc)}
    d["flags"] = {"use_multibody": bool(getattr(m, "use_multibody", True)),
                  "phases_lead": bool(getattr(m, "phases_lead", False)), "local": bool(m.local)}
    return d


# =====================================================================  editing
@api("model/set")
def model_set(sess, p):
    """Apply modifications to the active model. Each key is optional."""
    m = sess.require_model()
    P = Params(sess, p)

    def reset_aero():
        if m.aero is not None:
            m.aero.Kfun = None
            m.aero.Cfun = None
            m.aero.Sqq_aero = None
            if hasattr(m.aero, "F0_m"):
                m.aero.F0_m = None

    def work():
        changed = []
        if P.has("n_modes"):
            m.n_modes = P.int("n_modes")
            reset_aero()
            if m.hydro is not None:
                m.hydro.Sqq_hydro = None
            changed.append(f"n_modes = {m.n_modes}")
        if P.has("xi0"):
            m.modal_dry.xi0 = P.ev("xi0")
            changed.append("xi0")
        if P.has("m_min"):
            m.modal_dry.m_min = P.num("m_min")
            changed.append("m_min")
        if "phases_lead" in p and m.hydro is not None:
            m.hydro.phases_lead = P.bool("phases_lead")
            m.phases_lead = P.bool("phases_lead")
            changed.append("phases_lead")
        if "use_multibody" in p:
            m.use_multibody = P.bool("use_multibody")
        if P.has("x0"):
            m.x0 = np.asarray(P.ev("x0"), dtype=float)
            changed.append("x0")
        env = p.get("environment")
        if env and m.hydro is not None:
            for k in ("g", "rho", "depth", "waterlevel"):
                if str(env.get(k, "")).strip() != "":
                    setattr(m.hydro.environment, k, float(sess.ev(env[k])))
            m.hydro.assign_environment()
            changed.append("environment")
        for spec in p.get("ptypes") or []:
            pt = next((t for t in m.hydro.pontoon_types if t.label == spec["label"]), None)
            if pt is None:
                raise UserError(f"No pontoon type {spec['label']}")
            for k in ("Cd", "area"):
                if str(spec.get(k, "")).strip():
                    setattr(pt, k, np.atleast_1d(sess.ev(spec[k])).astype(float))
            changed.append(f"pontoon type {pt.label}")
        for spec in p.get("pontoons") or []:
            pt = m.hydro.get_pontoon(spec["label"])
            if pt is None:
                raise UserError(f"No pontoon {spec['label']}")
            if str(spec.get("Cd", "")).strip():
                pt.Cd = np.atleast_1d(sess.ev(spec["Cd"])).astype(float)
            if str(spec.get("Cquad", "")).strip():
                pt.Cquad = np.atleast_1d(sess.ev(spec["Cquad"])).astype(float)
            for k in ("current_affects_Q", "current_affects_k"):
                if k in spec:
                    v = spec[k]
                    setattr(pt, k, None if v in (None, "", "default") else bool(v in (True, "true", "True", 1)))
            changed.append(f"pontoon {pt.label}")
        if "assign_to_pontoons" in p:
            kw = {k: sess.ev(v) for k, v in (p.get("assign_to_pontoons") or {}).items()}
            m.hydro.assign_to_pontoons(**kw)
            changed.append("all pontoons: " + ", ".join(kw))
        for spec in p.get("aero_sections") or []:
            s = m.aero.sections.get(spec["name"])
            if s is None:
                raise UserError(f"No aero section {spec['name']}")
            for k in ("B", "D", "Cd", "dCd", "Cl", "dCl", "Cm", "dCm"):
                if str(spec.get(k, "")).strip() != "":
                    setattr(s, k, sess.ev(spec[k]))
            if "admittance" in spec:
                s.admittance = sess.ev(spec["admittance"]) if str(spec["admittance"] or "").strip() else None
            ads = (spec.get("ads") or "keep").strip()
            if ads == "quasisteady":
                s.add_quasisteady_ads()
                s.quasisteady = True
            elif ads == "flatplate":
                from wawi.wind import ADs, flatplate_ads
                s.ADs = ADs(**flatplate_ads(), ad_type="flatplate")
                s.quasisteady = False
            elif ads == "expr":
                from wawi.wind import ADs
                val = sess.ev(spec.get("ads_expr"))
                s.ADs = val if isinstance(val, ADs) else ADs(**val, ad_type="custom")
                s.quasisteady = False
            elif ads == "keep" and getattr(s, "quasisteady", False):
                s.add_quasisteady_ads()  # regenerate with updated coefficients
            reset_aero()
            changed.append(f"aero section {spec['name']}")
        if p.get("drag_elements_json"):
            data = p["drag_elements_json"]
            data = json.loads(data) if isinstance(data, str) else data
            m.assign_drag_elements(data)
            changed.append(f"drag elements ({len(m.drag_elements)})")
        if p.get("direct_cquad"):
            m.hydro.assign_direct_cquad(sess.ev(p["direct_cquad"]))
            changed.append("direct Cquad")
        for what in ensure_list(p.get("remove")):
            if what == "aero":
                m.aero = None
            elif what == "drag_elements":
                m.drag_elements = []
            elif what == "seastate" and m.hydro is not None:
                m.hydro.seastate = None
            elif what == "windstate" and m.aero is not None:
                m.aero.windstate = None
            changed.append(f"removed {what}")
        if p.get("reset_results"):
            from wawi.model import Results
            m.results = Results(model=m)
            changed.append("results reset")
        if p.get("clear_precomputed"):
            reset_aero()
            if m.hydro is not None:
                m.hydro.Sqq_hydro = None
            changed.append("precomputed actions cleared")
        print("Updated: " + (", ".join(changed) if changed else "nothing"))
        return {"changed": changed, "summary": model_summary(m)}
    return sess.run("edit", "Edit model", work)


# =====================================================================  geometry / 3D
def _model_nodes_elements(m):
    """Return node labels, coordinates, element connectivity (index pairs) and labels."""
    if m.eldef is not None:
        nodes = m.eldef.nodes
        labels = [int(n.label) for n in nodes]
        xyz = np.vstack([node_xyz(n) for n in nodes]) if nodes else np.zeros((0, 3))
        index = {l: i for i, l in enumerate(labels)}
        conn, ellabels, elsec = [], [], []
        for el in m.eldef.elements:
            try:
                conn.append([index[int(el.nodes[0].label)], index[int(el.nodes[1].label)]])
                ellabels.append(int(el.label))
                sec = getattr(el, "section", None)
                elsec.append(getattr(sec, "name", None) if sec is not None else None)
            except Exception:
                continue
        return labels, xyz, conn, ellabels, elsec, True
    if m.hydro is not None:
        nodes = [pt.node for pt in m.hydro.pontoons]
        labels = [jsonable(n.label) for n in nodes]
        xyz = np.vstack([node_xyz(n) for n in nodes])
        conn = [[i, i + 1] for i in range(len(nodes) - 1)]
        return labels, xyz, conn, list(range(len(conn))), [None] * len(conn), False
    return [], np.zeros((0, 3)), [], [], [], False


MAX_MESH_TRIANGLES = 4000  # per pontoon type, to keep the payload and the 3D view light


def _mesh_triangles(mesh):
    """Triangulated surface of a pyvista mesh (local pontoon coordinates) as JSON-able arrays."""
    surf = mesh.extract_surface() if not hasattr(mesh, "faces") else mesh
    tri = surf.triangulate()
    if tri.n_cells > MAX_MESH_TRIANGLES:
        try:
            tri = tri.decimate(1.0 - MAX_MESH_TRIANGLES / tri.n_cells).triangulate()
        except Exception:
            pass
    faces = np.asarray(tri.faces).reshape(-1, 4)
    faces = faces[faces[:, 0] == 3, 1:]
    pts = np.asarray(tri.points, dtype=float)
    return {"xyz": arr(pts, 5), "ijk": faces.astype(int).tolist(), "n_tri": int(len(faces))}


def _pontoon_meshes(m):
    """Meshes of the pontoon types and the placement (rotation matrix) of each pontoon, as in Model.plot."""
    from beef.rotation import R_from_drot, rodrot
    meshes, place = {}, []
    for pt in m.hydro.pontoons:
        ptype = pt.pontoon_type
        mesh = getattr(ptype, "mesh", None)
        if mesh is None:
            place.append(None)
            continue
        key = str(ptype.label)
        if key not in meshes:
            try:
                meshes[key] = _mesh_triangles(mesh)
            except Exception as exc:
                meshes[key] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            x0 = np.asarray(getattr(pt.node, "x0", np.zeros(6)), dtype=float)
            Tn = R_from_drot(x0[3:6]) if len(x0) >= 6 else np.eye(3)
            R = Tn @ rodrot(pt.rotation, style="column")
        except Exception:
            R = np.eye(3)
        place.append({"type": key, "R": arr(np.asarray(R, dtype=float)), "xyz": arr(node_xyz(pt.node))})
    return meshes, place


@api("model/geometry")
def model_geometry(sess, p):
    m = sess.require_model()
    labels, xyz, conn, ellabels, elsec, has_eldef = _model_nodes_elements(m)
    out = {"nodes": {"label": labels, "xyz": arr(xyz)}, "elements": {"conn": conn, "label": ellabels, "section": elsec},
           "has_eldef": has_eldef}
    ext = (np.ptp(xyz, axis=0) if len(xyz) else np.ones(3))
    out["extent"] = arr(ext)
    if m.hydro is not None:
        out["pontoons"] = [{"label": pt.label, "xyz": arr(node_xyz(pt.node)), "rotation": float(pt.rotation),
                            "type": pt.pontoon_type.label, "tmat": arr(pt.tmat[:3, :3])} for pt in m.hydro.pontoons]
        out["waterlevel"] = clean_float(m.hydro.environment.waterlevel)
        meshes, place = _pontoon_meshes(m)
        if meshes:
            out["pontoon_meshes"] = meshes
            out["pontoon_place"] = place
        ss = m.hydro.seastate
        if ss is not None:
            try:
                if ss.homogeneous:
                    out["wave"] = {"homogeneous": True, "theta0": float(ss.theta0), "Hs": clean_float(ss.Hs),
                                   "Tp": clean_float(ss.Tp), "U": clean_float(ss.U), "thetaU": clean_float(ss.thetaU)}
                else:
                    out["wave"] = {"homogeneous": False, "per_pontoon": [
                        {"theta0": clean_float(pt.sea_get("theta0")), "Hs": clean_float(pt.sea_get("Hs")),
                         "Tp": clean_float(pt.sea_get("Tp"))} for pt in m.hydro.pontoons]}
            except Exception as exc:
                out["wave"] = {"error": str(exc)}
    if m.aero is not None:
        groups = {}
        for g, els in (m.aero.elements or {}).items():
            groups[g] = [int(getattr(e, "label", e)) for e in els]
        out["aero_groups"] = groups
        ws = m.aero.windstate
        if ws is not None:
            out["wind"] = {"U0": clean_float(ws.U0), "direction": clean_float(ws.direction),
                           "vec": arr(ws.T[0, :]), "x_ref": arr(np.asarray(ws.x_ref, dtype=float))}
    dels = getattr(m, "drag_elements", None) or []
    out["drag_elements"] = [int(e.element.label) for e in dels if hasattr(e.element, "label")]
    return out


@api("model/shape")
def model_shape(sess, p):
    """Mode shape (dry or wet) or static displacement mapped to node translations."""
    m = sess.require_model()
    P = Params(sess, p)
    source = P.str("source", "wet")
    mode = P.int("mode", 0)
    labels, xyz, conn, *_ = _model_nodes_elements(m)
    r = m.results
    info = {}
    from wawi.modal import maxreal
    if source == "static":
        y = getattr(r, "y_static", None)
        if y is None:
            raise UserError("No static results. Run the static analysis first.")
        vec = None
    else:
        if source == "wet" and r.psi is None:
            raise UserError("No wet/in-wind modes. Run the modal analysis first (or choose dry modes).")
    use_key = "full" if ("full" in model_phi_keys(m) and m.eldef is not None) else "hydro"
    n_shapes = int(m.n_modes) if m.modal_dry else 0
    if source == "flutter":
        # Critical mode shape at a speed step of the last flutter analysis ('mode' = step index, sorted by speed)
        fl = sess.meta.get("flutter")
        res = sess.ns.get("flutter_result")
        if not fl or res is None or not fl.get("steps"):
            raise UserError("No flutter results. Run the flutter analysis first (Analysis → Flutter).")
        steps = fl["steps"]
        if mode < 0 or mode >= len(steps):
            raise UserError(f"Speed step out of range (1 … {len(steps)})")
        if fl["n_modes"] != int(m.n_modes):
            raise UserError(f"The flutter analysis used {fl['n_modes']} modes, the model now has {m.n_modes}. Run it again.")
        s = steps[mode]
        psi = np.asarray(res["critical_psi"][s["i"]], dtype=complex)
        vec = maxreal(m.get_dry_phi(key=use_key) @ psi[:, None])[:, 0]
        n_shapes = len(steps)
        fd = s["omega"] / (2 * np.pi) if s["omega"] else None
        info.update(fd=clean_float(fd), Td=clean_float(1 / fd) if fd else None, xi=s["xi"], V=s["V"],
                    fn=clean_float(fd), Tn=clean_float(1 / fd) if fd else None)
        mode_txt = f"mode {s['mode']}" if s["mode"] is not None else "real root"
        info["label"] = (f"Flutter: critical {mode_txt} at V = {s['V']:.4g} m/s"
                         f" ({'stable' if s['stable'] else 'unstable'}, step {mode + 1}/{len(steps)})")
    elif source == "static":
        phi = m.get_dry_phi(key=use_key)
        vec = phi @ np.real(np.asarray(y)).ravel()
        info["label"] = "Static displacement"
    else:
        phi = m.get_dry_phi(key=use_key) if source == "dry" else maxreal(m.get_dry_phi(key=use_key) @ r.psi)
        if mode < 0 or mode >= phi.shape[1]:
            raise UserError(f"Mode index out of range (0 … {phi.shape[1] - 1})")
        vec = phi[:, mode]
        if source == "dry":
            md = m.modal_dry
            xi0 = np.atleast_1d(md.xi0)
            info.update(fn=clean_float(md.fn[mode]), Tn=clean_float(md.Tn[mode]),
                        xi=clean_float(xi0[mode] if xi0.size > mode else xi0[0]))
        else:
            info.update(fn=clean_float(r.fn[mode]), Tn=clean_float(r.Tn[mode]), xi=clean_float(r.xi[mode]),
                        fd=clean_float(r.fd[mode]), Td=clean_float(r.Td[mode]))
        info["label"] = f"{'Dry' if source == 'dry' else 'Wet/in-wind'} mode {mode + 1}"
    n_nodes = len(labels)
    disp = np.zeros((n_nodes, 3), dtype=complex)
    rot = np.zeros((n_nodes, 3), dtype=complex)
    if use_key == "full":
        for i, node in enumerate(m.eldef.nodes):
            gd = np.asarray(node.global_dofs)
            if len(gd) >= 3:
                disp[i] = vec[gd[:3]]
            if len(gd) >= 6:
                rot[i] = vec[gd[3:6]]
    else:
        v = np.asarray(vec)
        if m.local:  # hydro phi in local pontoon csys -> rotate to global
            T = m.tmat
            v = T.T @ v
        for i in range(n_nodes):
            disp[i] = v[i * 6:i * 6 + 3]
            rot[i] = v[i * 6 + 3:i * 6 + 6]
    if source != "static":
        # normalise to unit maximum translation
        mx = np.max(np.abs(disp)) if np.max(np.abs(disp)) > 0 else 1.0
        disp, rot = disp / mx, rot / mx
    ext = np.ptp(xyz, axis=0).max() if len(xyz) else 1.0
    info["auto_scale"] = clean_float(0.06 * ext / max(np.max(np.abs(disp)), 1e-30))
    info["key"] = use_key
    return {"disp": arr(disp), "rot": arr(rot), "info": info, "n_modes": n_shapes}


@api("model/pontoon_type_data")
def pontoon_type_data(sess, p):
    m = sess.require_model()
    label = p.get("label")
    pt = next((t for t in m.hydro.pontoon_types if t.label == label), None)
    if pt is None:
        raise UserError(f"No pontoon type '{label}'")
    om = np.asarray(pt.original_omega if pt.original_omega is not None else np.linspace(0.01, 3, 150), dtype=float)
    out = {"label": label, "omega": arr(om)}
    for q in ("M", "C", "K"):
        f = getattr(pt, q)
        if f is None:
            continue
        try:
            v = np.asarray(f(om))
            if v.shape == (6, 6):
                v = np.repeat(v[:, :, None], len(om), axis=2)
            out[q] = arr(np.real(v))
        except Exception as exc:
            out[q + "_error"] = str(exc)
    if pt.Q is not None:
        omq = np.asarray(pt.original_omegaQ if getattr(pt, "original_omegaQ", None) is not None else om, dtype=float)
        try:
            Q = np.asarray(pt.Q(omq))  # 6 x ntheta x nomega
            out["omegaQ"] = arr(omq)
            out["thetaQ_deg"] = arr(np.degrees(np.asarray(pt.theta, dtype=float)))
            out["Qabs"] = arr(np.abs(Q), 6)
            out["Qphase"] = arr(np.angle(Q), 4)
        except Exception as exc:
            out["Q_error"] = str(exc)
    return out


@api("model/aero_ads")
def aero_ads(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    name = p.get("section")
    if m.aero is None or name not in m.aero.sections:
        raise UserError("Unknown aero section")
    s = m.aero.sections[name]
    v = np.linspace(P.num("vmin", 0.01), P.num("vmax", 5.0), P.int("n", 200))
    out = {"v": arr(v), "type": getattr(s.ADs, "type", None), "ads": {}}
    for key in s.ADs.ad_keys:
        f = getattr(s.ADs, key)
        if f is None:
            continue
        try:
            out["ads"][key] = arr(np.real(np.asarray(f(v), dtype=complex)) * np.ones_like(v))
        except Exception as exc:
            out["ads"][key] = None
            out.setdefault("errors", {})[key] = str(exc)
    return out


# =====================================================================  pyvista rendering (optional)
def _render_pyvista(sess, fn, w, h):
    try:
        import pyvista as pv
    except Exception as exc:
        raise UserError(f"pyvista not available: {exc}")
    pv.OFF_SCREEN = True
    pl = fn()
    try:
        img = pl.screenshot(return_img=True, window_size=[w, h])
    finally:
        try:
            pl.close()
        except Exception:
            pass
    import matplotlib
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    return fig_to_png(fig, dpi=100)


@api("model/render")
def model_render(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    kind = P.str("kind", "model")
    w, h = P.int("width", 1400), P.int("height", 800)
    opts = P.ev("kwargs", {}) or {}

    def work():
        if kind == "mode":
            fn = lambda: m.plot_mode(P.int("mode", 0), use_dry=P.bool("use_dry"), scale=P.num("scale", 300),
                                     show=False, **opts)
        else:
            fn = lambda: m.plot(show=False, **opts)
        return {"image": _render_pyvista(sess, fn, w, h)}
    return sess.run("render", f"Render {kind} (pyvista)", work, wait=60)


@api("model/export_modeshapes")
def export_modeshapes(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    folder = P.path("folder", must_exist=False)
    opts = P.ev("kwargs", {}) or {}

    def work():
        import pyvista as pv
        pv.OFF_SCREEN = True
        folder.mkdir(parents=True, exist_ok=True)
        m.export_modeshapes(str(folder), n_modes=P.int("n_modes"), format=P.str("format", "png"),
                            zoom=P.num("zoom", 1.0), **opts)
        print(f"Exported mode shapes to {folder}")
        return {"folder": str(folder)}
    return sess.run("export", "Export mode shapes", work, wait=2)


@api("model/plot_2d_tmats")
def plot_2d_tmats(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    import matplotlib.pyplot as plt
    with sess.mpl_lock:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 6))
        m.plot_2d_tmats(scale=P.num("scale", 50), show_labels=P.bool("show_labels", True), ax=ax)
        return {"images": collect_figures(plt)}


@api("fs/read_json")
def fs_read_json(sess, p):
    P = Params(sess, p)
    path = P.path("path", kind="file")
    with open(path) as f:
        return json.load(f)
