"""Sea state and wind state endpoints."""
import json
from pathlib import Path

import numpy as np

from .common import state_str, Params, collect_figures, register_download, temp_path
from .serialize import arr, clean_float, jsonable
from .server import api
from .session import UserError

SEA_FIELDS = ["Tp", "Hs", "gamma", "theta0", "s", "U", "thetaU", "depth", "origin", "ranges",
              "options", "pontoon_options", "name", "plot_label", "centered_dirdist", "use_robust_D", "angle_unit"]
WIND_FIELDS = ["U0", "direction", "Au", "Av", "Aw", "Iu", "Iv", "Iw", "Cux", "Cvx", "Cwx", "Cuy", "Cvy", "Cwy",
               "Cuz", "Cvz", "Cwz", "Lux", "Lvx", "Lwx", "x_ref", "scaling", "name", "spectrum_type", "rho", "options"]


def _src(v):
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v)


# =====================================================================  Sea state
def build_seastate(sess, p):
    from wawi.model import Seastate
    P = Params(sess, p)
    for req in ("Tp", "Hs", "gamma", "theta0"):
        if not P.has(req):
            raise UserError(f"Sea state parameter '{req}' is required")
    unit = P.str("angle_unit", "deg")
    th0, thU = P.ev("theta0"), P.ev("thetaU", 0.0)
    if unit == "rad":
        th0 = Seastate.scale(th0, 180 / np.pi)
        thU = Seastate.scale(thU, 180 / np.pi)
    options = P.ev("options", {}) or {}
    pontoon_options = P.ev("pontoon_options", {}) or {}
    ss = Seastate(P.ev("Tp"), P.ev("Hs"), P.ev("gamma"), th0, s=P.ev("s", np.inf),
                  origin=P.ev("origin"), ranges=P.ev("ranges"), options=dict(options),
                  pontoon_options=dict(pontoon_options), name=P.str("name"), plot_label=P.str("plot_label"),
                  centered_dirdist=P.bool("centered_dirdist", True), U=P.ev("U", 0.0), thetaU=thU,
                  use_robust_D=P.bool("use_robust_D", True), angle_unit="deg")
    if P.has("depth"):
        d = P.ev("depth")
        ss.depth = d if callable(d) else (lambda x, y, _d=float(d): _d)
    ss._gui_src = {k: p.get(k) for k in SEA_FIELDS if k in p}
    return ss


def seastate_to_form(ss):
    """Best-effort reconstruction of form fields from a Seastate object."""
    src = dict(getattr(ss, "_gui_src", {}) or {})
    if src:
        return src

    def val(par, conv=1.0):
        v = getattr(ss, par)
        if callable(v):
            return "<function – redefine>"
        return repr(float(v) * conv) if np.ndim(v) == 0 else repr(v)
    return {"Tp": val("Tp"), "Hs": val("Hs"), "gamma": val("gamma"), "theta0": val("theta0", 180 / np.pi),
            "s": "np.inf" if not ss.short_crested else val("s"), "U": val("U"),
            "thetaU": val("thetaU", 180 / np.pi), "name": ss.name or "",
            "centered_dirdist": bool(ss.centered_dirdist), "use_robust_D": bool(ss.use_robust_D),
            "options": json.dumps(jsonable(ss.options)), "pontoon_options": json.dumps(jsonable(ss.pontoon_options)),
            "origin": repr(list(np.asarray(ss.origin, dtype=float))), "angle_unit": "deg"}


@api("seastate/build")
def seastate_build(sess, p):
    ss = build_seastate(sess, p)
    sess.ns["seastate"] = ss
    out = {"str": state_str(ss), "homogeneous": ss.homogeneous, "short_crested": ss.short_crested,
           "fun_pars": ss.fun_pars}
    if p.get("assign"):
        m = sess.require_model()
        if m.hydro is None:
            raise UserError("The model has no hydrodynamic part")
        sess.run("edit", "Assign sea state", lambda: m.assign_seastate(ss))
        out["assigned"] = True
    return out


@api("seastate/assign")
def seastate_assign(sess, p):
    m = sess.require_model()
    ss = sess.ns.get(p.get("var") or "seastate")
    if ss is None:
        raise UserError("No sea state defined")
    if m.hydro is None:
        raise UserError("The model has no hydrodynamic part")
    return sess.run("edit", "Assign sea state", lambda: (m.assign_seastate(ss), state_str(ss))[1])


@api("seastate/from_json")
def seastate_from_json(sess, p):
    from wawi.model import Seastate
    P = Params(sess, p)
    path = P.path("path", kind="file")
    ss = Seastate.from_json(str(path))
    with open(path) as f:
        data = json.load(f)
    opts = dict(data.get("options", {}))
    form = {"Tp": data["S"]["Tp"], "Hs": data["S"]["Hs"], "gamma": data["S"]["gamma"],
            "theta0": data["D"]["theta0"], "s": data["D"]["s"],
            "U": data.get("current", {}).get("U", ""), "thetaU": data.get("current", {}).get("thetaU", ""),
            "centered_dirdist": opts.pop("centered_dirdist", True),
            "options": json.dumps(opts) if opts else "", "pontoon_options": json.dumps(data.get("pontoon_options", {})) if data.get("pontoon_options") else "",
            "origin": json.dumps(data["origin"]) if "origin" in data else "",
            "ranges": json.dumps(data["ranges"]) if "ranges" in data else "",
            "name": data.get("name", ""), "plot_label": data.get("plot_label", "") or "", "angle_unit": "deg"}
    ss._gui_src = form
    sess.ns["seastate"] = ss
    return {"form": form, "str": state_str(ss)}


@api("seastate/from_model")
def seastate_from_model(sess, p):
    m = sess.require_model()
    if m.hydro is None or m.hydro.seastate is None:
        raise UserError("The model has no sea state assigned")
    sess.ns["seastate"] = m.hydro.seastate
    return {"form": seastate_to_form(m.hydro.seastate), "str": state_str(m.hydro.seastate)}


@api("seastate/to_json")
def seastate_to_json(sess, p):
    """Export the form as a wawi sea-state JSON file (readable by Seastate.from_json)."""
    f = p.get("form") or {}
    unit = f.get("angle_unit", "deg")

    def conv(v):
        v = _src(v).strip()
        if unit == "rad" and v:
            if "lambda" in v:
                raise UserError("JSON export with angle_unit='rad' is only supported for scalar angles")
            return f"np.degrees({v})"
        return v
    data = {"name": f.get("name") or "Unnamed seastate",
            "S": {"Tp": _src(f.get("Tp")), "Hs": _src(f.get("Hs")), "gamma": _src(f.get("gamma"))},
            "D": {"theta0": conv(f.get("theta0")), "s": _src(f.get("s")) or "np.inf"}}
    if _src(f.get("U")).strip():
        data["current"] = {"U": _src(f.get("U")), "thetaU": conv(f.get("thetaU")) or "0.0"}
    opts = {}
    if _src(f.get("options")).strip():
        opts = sessless_eval(f["options"])
    opts["centered_dirdist"] = bool(f.get("centered_dirdist", True))
    data["options"] = opts
    if _src(f.get("pontoon_options")).strip():
        data["pontoon_options"] = sessless_eval(f["pontoon_options"])
    if _src(f.get("origin")).strip():
        data["origin"] = list(map(float, sessless_eval(f["origin"])))
    if _src(f.get("ranges")).strip():
        data["ranges"] = sessless_eval(f["ranges"])
    if f.get("plot_label"):
        data["plot_label"] = f["plot_label"]
    return _write_json(sess, p, data, "seastate.json")


def sessless_eval(s):
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return eval(s, {"np": np, "inf": np.inf})


def _write_json(sess, p, data, default_name):
    path = (p.get("path") or "").strip()
    text = json.dumps(jsonable(data), indent=4)
    if path:
        Path(path).expanduser().write_text(text)
        return {"path": path, "json": text}
    tmp = temp_path(".json")
    tmp.write_text(text)
    return {"download": register_download(sess, tmp, default_name), "json": text}


def _get_state(sess, which, kind):
    if which == "model":
        m = sess.require_model()
        st = (m.hydro.seastate if m.hydro is not None else None) if kind == "sea" else \
             (m.aero.windstate if m.aero is not None else None)
    else:
        st = sess.ns.get("seastate" if kind == "sea" else "windstate")
    if st is None:
        raise UserError(f"No {'sea' if kind == 'sea' else 'wind'} state "
                        f"{'assigned to the model' if which == 'model' else 'defined'}. Click 'Build' first.")
    return st


@api("seastate/plot_data")
def seastate_plot_data(sess, p):
    P = Params(sess, p)
    ss = _get_state(sess, P.str("which", "editor"), "sea")
    x, y = P.num("x", 0.0), P.num("y", 0.0)
    omega = np.asarray(P.ev("omega", "np.linspace(0.01, 3.0, 300)"), dtype=float)
    theta = np.asarray(P.ev("theta", "np.linspace(-np.pi, np.pi, 361)"), dtype=float)
    S = np.asarray(ss.S(x, y)(omega), dtype=float)
    Dfun = ss.D(x, y)
    if ss.short_crested:
        D = np.asarray(Dfun(theta), dtype=float)
    else:
        D = np.zeros_like(theta)
    shift = float(ss.evpar("theta_shift", x, y))
    m0 = np.trapezoid(S, omega) if hasattr(np, "trapezoid") else np.trapz(S, omega)
    out = {"omega": arr(omega), "S": arr(S), "theta": arr(theta), "D": arr(D), "theta_shift": shift,
           "theta0_abs": clean_float(ss.evpar("theta0", x, y)),
           "Hs_m0": clean_float(4 * np.sqrt(max(m0, 0))),
           "Tp_peak": clean_float(2 * np.pi / omega[np.argmax(S)]) if S.size else None,
           "short_crested": ss.short_crested, "homogeneous": ss.homogeneous, "str": state_str(ss),
           "params_at_xy": {k: clean_float(ss.evpar(k, x, y)) for k in ["Hs", "Tp", "gamma", "s", "U"]}}
    if ss.short_crested:
        oi = np.linspace(omega.min(), omega.max(), min(len(omega), 150))
        ti = np.linspace(theta.min(), theta.max(), min(len(theta), 181))
        S2 = np.interp(oi, omega, S)[None, :] * np.asarray(Dfun(ti), dtype=float)[:, None]
        out["S2d"] = {"omega": arr(oi), "theta": arr(ti), "z": arr(S2, 8)}
    m = sess.model
    if m is not None and m.hydro is not None:
        rows = []
        for pt in m.hydro.pontoons:
            px, py = float(pt.node.x[0]) if getattr(pt.node, "x", None) is not None else 0.0, \
                float(pt.node.x[1]) if getattr(pt.node, "x", None) is not None else 0.0
            r = {"label": pt.label, "x": px, "y": py}
            for k in ["Hs", "Tp", "gamma", "s", "U"]:
                try:
                    r[k] = clean_float(ss.evpar(k, px, py))
                except Exception:
                    r[k] = None
            for k in ["theta0", "thetaU"]:
                try:
                    r[k] = clean_float(np.degrees(ss.evpar(k, px, py)))
                except Exception:
                    r[k] = None
            rows.append(r)
        out["pontoons"] = rows
    return out


@api("seastate/mpl")
def seastate_mpl(sess, p):
    """Matplotlib versions of wawi.plot.plot_S2d / plot_dir_and_crests."""
    P = Params(sess, p)
    ss = _get_state(sess, P.str("which", "editor"), "sea")
    kind = P.str("kind", "S2d")
    import matplotlib.pyplot as plt
    from wawi.plot import plot_S2d, plot_dir_and_crests
    with sess.mpl_lock:
        plt.close("all")
        x, y = P.num("x", 0.0), P.num("y", 0.0)
        if kind == "S2d":
            omega = np.asarray(P.ev("omega", "np.arange(0.001, 1.5, 0.01)"), dtype=float)
            theta = np.asarray(P.ev("theta", "np.arange(0, 2*np.pi, 0.01)"), dtype=float)
            D = ss.D(x, y)(theta) if ss.short_crested else None
            plot_S2d(ss.S(x, y)(omega), omega, theta, D=D)
        else:
            th0 = float(np.degrees(ss.evpar("theta0", x, y)))
            Tp = float(ss.evpar("Tp", x, y))
            U = float(ss.evpar("U", x, y) or 0.0)
            thU = float(np.degrees(ss.evpar("thetaU", x, y) or 0.0))
            fig, ax = plt.subplots(figsize=(8, 7))
            L = P.num("arrow_length", 500)
            plot_dir_and_crests(th0, Tp, arrow_length=L, n_repeats=P.int("n_repeats", 4), add_text=U == 0.0, ax=ax,
                                arrow_options=dict(width=L / 120, head_width=L / 25, edgecolor="#1f5fa8", facecolor="#1f5fa8"),
                                crest_options=dict(color="#1f5fa8"))
            if U != 0.0:
                plot_dir_and_crests(th0, Tp, U=U, thetaU=thU, arrow_length=L, n_repeats=P.int("n_repeats", 4), ax=ax,
                                    arrow_options=dict(width=L / 120, head_width=L / 25, edgecolor="#c0392b", facecolor="#c0392b"),
                                    crest_options=dict(color="#c0392b"))
                ax.set_title("Blue: without current, red: with current")
        return {"images": collect_figures(plt)}


@api("seastate/simulate")
def seastate_simulate(sess, p):
    P = Params(sess, p)
    ss = _get_state(sess, P.str("which", "editor"), "sea")
    x = np.asarray(P.ev("x", "np.arange(-500, 500, 10)"), dtype=float)
    y = np.asarray(P.ev("y", "np.arange(-200, 200, 10)"), dtype=float)
    omega = np.asarray(P.ev("omega", "np.arange(0.01, 3.0, 0.05)"), dtype=float)
    fs = P.num("fs")
    th = P.ev("theta")
    time_history = P.bool("time_history", True)
    max_frames = P.int("max_frames", 60)
    gif = P.bool("gif", False)

    def work():
        print(f"Simulating sea surface on {len(x)} x {len(y)} grid ...")
        out = ss.simulate(x, y, omega, fs=fs, theta=th, time_history=time_history, print_progress=True)
        eta, t = out if isinstance(out, tuple) else (out, None)
        eta = np.asarray(eta)
        sess.ns["eta_sim"] = eta
        sess.ns["t_sim"] = t
        sess.ns["x_sim"], sess.ns["y_sim"] = x, y
        res = {"x": arr(x), "y": arr(y), "vars": ["eta_sim", "t_sim", "x_sim", "y_sim"]}
        if eta.ndim == 3:
            nt = eta.shape[2]
            idx = np.unique(np.linspace(0, nt - 1, min(max_frames, nt)).astype(int))
            # limit payload
            while eta.shape[0] * eta.shape[1] * len(idx) > 3e6 and len(idx) > 2:
                idx = idx[::2]
            res["frames"] = arr(np.transpose(eta[:, :, idx], (2, 0, 1)), 4)
            res["t"] = arr(np.asarray(t)[idx])
            res["std"] = clean_float(np.std(eta))
            if gif:
                from wawi.plot import animate_surface
                import matplotlib.pyplot as plt
                with sess.mpl_lock:
                    plt.close("all")
                    tmp = temp_path(".gif")
                    sel = idx
                    animate_surface(eta[:, :, sel], x, y, np.asarray(t)[sel], filename=str(tmp), writer="pillow",
                                    fps=P.num("fps", 10))
                    plt.close("all")
                res["gif_download"] = register_download(sess, tmp, "sea_surface.gif")
        else:
            res["frames"] = [arr(eta, 4)]
            res["t"] = [0.0]
            res["std"] = clean_float(np.std(eta))
        return res
    return sess.run("simulate", "Simulate sea surface", work, wait=1)


# =====================================================================  Wind state
def build_windstate(sess, p):
    from wawi.model import Windstate
    P = Params(sess, p)
    if not P.has("U0") or not P.has("direction"):
        raise UserError("U0 and direction are required")
    kw = {}
    for k in ["Au", "Av", "Aw", "Iu", "Iv", "Iw", "Cuy", "Cuz", "Cvy", "Cvz", "Cwy", "Cwz", "Lux", "Lvx", "Lwx", "rho"]:
        if P.has(k):
            kw[k] = P.num(k)
    if P.has("x_ref"):
        kw["x_ref"] = np.asarray(P.ev("x_ref"), dtype=float)
    if P.has("scaling"):
        kw["scaling"] = P.ev("scaling")
    ws = Windstate(P.num("U0"), P.num("direction"), name=P.str("name"),
                   spectrum_type=P.str("spectrum_type", "kaimal"), options=P.ev("options"), **kw)
    for k in ["Cux", "Cvx", "Cwx"]:
        if P.has(k):
            setattr(ws, k, P.num(k))
    ws._gui_src = {k: p.get(k) for k in WIND_FIELDS if k in p}
    return ws


def windstate_to_form(ws):
    src = dict(getattr(ws, "_gui_src", {}) or {})
    if src:
        return src
    f = {"U0": ws.U0, "direction": ws.direction, "spectrum_type": ws.spectrum_type, "rho": ws.rho,
         "name": ws.name or "", "x_ref": repr(list(np.asarray(ws.x_ref, dtype=float)))}
    for k in ["Au", "Av", "Aw", "Iu", "Iv", "Iw", "Cux", "Cvx", "Cwx", "Cuy", "Cvy", "Cwy", "Cuz", "Cvz", "Cwz",
              "Lux", "Lvx", "Lwx"]:
        f[k] = clean_float(getattr(ws, k))
    return jsonable(f)


@api("windstate/build")
def windstate_build(sess, p):
    ws = build_windstate(sess, p)
    sess.ns["windstate"] = ws
    out = {"str": state_str(ws)}
    if p.get("assign"):
        m = sess.require_model()
        if m.aero is None:
            raise UserError("The model has no aerodynamic part")
        sess.run("edit", "Assign wind state", lambda: m.assign_windstate(ws))
        out["assigned"] = True
    return out


@api("windstate/assign")
def windstate_assign(sess, p):
    m = sess.require_model()
    ws = sess.ns.get(p.get("var") or "windstate")
    if ws is None:
        raise UserError("No wind state defined")
    if m.aero is None:
        raise UserError("The model has no aerodynamic part")
    return sess.run("edit", "Assign wind state", lambda: (m.assign_windstate(ws), state_str(ws))[1])


@api("windstate/from_json")
def windstate_from_json(sess, p):
    from wawi.model import Windstate
    P = Params(sess, p)
    path = P.path("path", kind="file")
    ws = Windstate.from_json(str(path))
    with open(path) as f:
        data = json.load(f)
    form = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in data.items()}
    ws._gui_src = form
    sess.ns["windstate"] = ws
    return {"form": form, "str": state_str(ws)}


@api("windstate/from_model")
def windstate_from_model(sess, p):
    m = sess.require_model()
    if m.aero is None or m.aero.windstate is None:
        raise UserError("The model has no wind state assigned")
    sess.ns["windstate"] = m.aero.windstate
    return {"form": windstate_to_form(m.aero.windstate), "str": state_str(m.aero.windstate)}


@api("windstate/to_json")
def windstate_to_json(sess, p):
    f = p.get("form") or {}
    data = {}
    for k in WIND_FIELDS:
        v = f.get(k)
        if v is None or (isinstance(v, str) and v.strip() == "") or k in ("Cux", "Cvx", "Cwx"):
            continue
        if k in ("name", "spectrum_type", "scaling"):
            data[k] = str(v)
        elif k in ("x_ref", "options"):
            data[k] = sessless_eval(v)
        else:
            data[k] = float(sessless_eval(v))
    return _write_json(sess, p, data, "windstate.json")


@api("windstate/plot_data")
def windstate_plot_data(sess, p):
    from wawi.wind import get_beaufort, kaimal_auto, von_karman_auto
    P = Params(sess, p)
    ws = _get_state(sess, P.str("which", "editor"), "wind")
    omega = np.asarray(P.ev("omega", "np.linspace(0.005, 3.0, 400)"), dtype=float)
    U = float(ws.U0)
    Sk = np.vstack([kaimal_auto(om, ws.Lx, ws.A, ws.sigma, U) for om in omega])
    Sv = np.vstack([von_karman_auto(om, ws.Lx, ws.sigma, U) for om in omega])
    f = P.num("f_coh", 0.1)
    dist = np.linspace(0, P.num("max_sep", 200.0), 200)
    coh = {}
    for ci, comp in enumerate("uvw"):
        for di, d in enumerate("xyz"):
            C = ws.C[di, ci]
            coh[f"C{comp}{d}"] = arr(np.exp(-f * C * dist / U)) if U > 0 else None
    try:
        bf = get_beaufort(U)
    except Exception:
        bf = None
    out = {"omega": arr(omega), "kaimal": arr(Sk.T), "vonkarman": arr(Sv.T), "spectrum_type": ws.spectrum_type,
           "sigma": arr(ws.sigma), "T": arr(ws.T), "beaufort": jsonable(bf), "coh_dist": arr(dist), "coh": coh,
           "f_coh": f, "str": state_str(ws)}
    m = sess.model
    if m is not None and m.aero is not None and m.eldef is not None:
        try:
            pts = []
            for g, els in m.aero.elements.items():
                for e in els:
                    for n in e.nodes:
                        pts.append(np.asarray(n.coordinates, dtype=float))
            pts = np.unique(np.vstack(pts), axis=0)
            Ux = np.array([ws.U(c) for c in pts], dtype=float)
            order = np.argsort(pts[:, 0])
            out["U_profile"] = {"x": arr(pts[order, 0]), "y": arr(pts[order, 1]), "z": arr(pts[order, 2]),
                                "U": arr(Ux[order])}
        except Exception as exc:
            out["U_profile_error"] = str(exc)
    return out
