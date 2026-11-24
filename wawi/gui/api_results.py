"""Results post-processing endpoints."""
import numpy as np

from .api_analysis import modal_table
from .common import DOF_NAMES, Params, arclength, model_phi_keys, node_xyz, register_download, temp_path
from .serialize import arr, clean_float, jsonable
from .server import api
from .session import UserError


def _h(sess, P):
    h = P.str("h", "disp")
    if h in ("disp", "", None):
        return (lambda om: 1.0), 0
    if h == "vel":
        return (lambda om: om ** 2), 2
    if h == "acc":
        return (lambda om: om ** 4), 4
    f = sess.ev(h)
    if not callable(f):
        raise UserError("Custom h must be a function of omega, e.g. lambda om: om**2")
    return f, None


def _key(P):
    k = P.str("key", "")
    return None if k in ("", "modal", "None") else k


@api("results/overview")
def results_overview(sess, p):
    m = sess.require_model()
    r = m.results
    md = m.modal_dry
    out = {"has_eig": r.lambd is not None, "has_freqsim": r.S is not None,
           "has_static": getattr(r, "y_static", None) is not None,
           "keys": {k: list(np.shape(v)) for k, v in (md.phi_full.items() if md is not None else [])},
           "phi_x": list((md.phi_x or {}).keys()) if md is not None else [],
           "pontoons": [pt.label for pt in m.hydro.pontoons] if m.hydro is not None else [],
           "aero_groups": list(m.aero.elements.keys()) if m.aero is not None and m.aero.elements else [],
           "modal": modal_table(r, m), "has_eldef": m.eldef is not None}
    if md is not None:
        out["dry"] = {"fn": arr(md.fn), "Tn": arr(md.Tn), "wn": arr(md.wn), "xi0": arr(np.atleast_1d(md.xi0)),
                      "m": arr(md.m), "k": arr(md.k)}
    if r.S is not None:
        out["freq"] = {"n": int(len(r.omega)), "min": clean_float(r.omega.min()), "max": clean_float(r.omega.max()),
                       "n_modes": int(r.S.shape[0])}
    return out


@api("results/modal_props")
def modal_props(sess, p):
    """Modal mass/stiffness/damping of the combined system (Results.mdiag/kdiag/cdiag)."""
    m = sess.require_model()
    if m.results.lambd is None:
        raise UserError("Run the modal analysis first")

    def work():
        r = m.results
        return {"mdiag": arr(r.mdiag), "kdiag": arr(r.kdiag), "cdiag": arr(r.cdiag)}
    return sess.run("results", "Modal mass/stiffness/damping", work, wait=30)


def _point_layout(m, key, n, subset=None):
    """Return list of (label, xyz, row indices) describing how result rows map to points."""
    md = m.modal_dry
    if key is None:
        return [(f"mode {i + 1}", None, [i]) for i in range(n)], ["value"]
    if subset:
        ixs, xyz = m.get_aerosection_phi_and_x(subset)
        ixs = np.asarray(ixs)
        pts = []
        for i in range(len(xyz)):
            pts.append((str(m.aero.eldef[subset].nodes[i].label), np.asarray(xyz[i], dtype=float),
                        list(ixs[i * 6:i * 6 + 6])))
        return pts, DOF_NAMES
    if key == "full" and m.eldef is not None and m.eldef.ndofs == n:
        return [(str(nd.label), node_xyz(nd), list(np.asarray(nd.global_dofs)[:6])) for nd in m.eldef.nodes], DOF_NAMES
    if key == "hydro" and m.hydro is not None and 6 * len(m.hydro.pontoons) == n:
        return [(pt.label, node_xyz(pt.node), list(range(i * 6, i * 6 + 6))) for i, pt in enumerate(m.hydro.pontoons)], DOF_NAMES
    px = (md.phi_x or {}).get(key) if md is not None else None
    if px is not None:
        px = np.asarray(px, dtype=float)
        npts = px.shape[0]
        if npts and n % npts == 0:
            c = n // npts
            names = DOF_NAMES if c == 6 else [f"c{j}" for j in range(c)]
            xyz = px if px.ndim == 2 else np.c_[px, np.zeros((npts, 2))]
            return [(str(i), np.r_[xyz[i], np.zeros(3)][:3], list(range(i * c, i * c + c))) for i in range(npts)], names
    if n % 6 == 0:
        return [(str(i), None, list(range(i * 6, i * 6 + 6))) for i in range(n // 6)], DOF_NAMES
    return [(str(i), None, [i]) for i in range(n)], ["value"]


@api("results/stats")
def results_stats(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    if m.results.S is None:
        raise UserError("No frequency-domain results. Run the response analysis first.")
    q = P.str("quantity", "std")
    key = _key(P)
    modes = P.ev("modes")
    if key is None and modes is None:
        modes = np.arange(m.results.S.shape[0])
    h, hpow = _h(sess, P)
    convert_to = P.str("convert_to")
    if q == "std":
        v = m.get_result_std(key=key, h=h, modes=modes, convert_to=convert_to)
    elif q == "expmax":
        v = m.get_result_expmax(P.num("T", 3600.0), key=key, h=h, modes=modes)
    elif q == "peakfactor":
        v = m.get_result_peakfactor(P.num("T", 3600.0), key=key or "hydro", h=h, modes=modes)
    else:
        raise UserError(f"Unknown quantity {q}")
    v = np.real(np.asarray(v)).ravel()
    pts, comps = _point_layout(m, key, len(v), subset=P.str("subset"))
    vals = np.full((len(pts), len(comps)), np.nan)
    xyz = np.full((len(pts), 3), np.nan)
    for i, (lab, x, rows) in enumerate(pts):
        rows = [r for r in rows if r < len(v)]
        vals[i, :len(rows)] = v[rows]
        if x is not None:
            xyz[i] = x
    pos = {"index": arr(np.arange(len(pts)))}
    if np.isfinite(xyz).all():
        pos.update(x=arr(xyz[:, 0]), y=arr(xyz[:, 1]), z=arr(xyz[:, 2]), arclength=arr(arclength(xyz)))
    sess.ns["last_stats"] = vals
    return {"values": arr(vals), "comps": comps, "labels": [pt[0] for pt in pts], "pos": pos,
            "quantity": q, "key": key, "raw_len": len(v)}


def _psd_index(m, P, sess):
    mode = P.str("sel", "index")
    key = _key(P)
    dofs = [int(d) for d in (P.list("dofs") or [0, 1, 2, 3, 4, 5])]
    if mode == "node":
        if m.eldef is None:
            raise UserError("Model has no FE element definition")
        lab = int(P.ev("node"))
        nd = m.eldef.get_node(lab)
        if nd is None:
            raise UserError(f"No node {lab}")
        gd = np.asarray(nd.global_dofs)
        return [int(gd[d]) for d in dofs], [f"node {lab} {DOF_NAMES[d]}" for d in dofs]
    if mode == "pontoon":
        labels = [pt.label for pt in m.hydro.pontoons]
        lab = P.str("pontoon")
        if lab not in labels:
            raise UserError(f"No pontoon {lab}")
        i = labels.index(lab)
        return [i * 6 + d for d in dofs], [f"{lab} {DOF_NAMES[d]}" for d in dofs]
    idx = np.atleast_1d(P.ev("index", "[0]")).astype(int).tolist()
    return idx, [f"{key or 'mode'}[{i}]" for i in idx]


@api("results/psd")
def results_psd(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    r = m.results
    if r.S is None:
        raise UserError("No frequency-domain results. Run the response analysis first.")
    key = _key(P)
    h, _ = _h(sess, P)
    om = np.asarray(r.omega)
    idx, names = _psd_index(m, P, sess)
    if P.str("key") == "generalized":
        Sg = m.get_gen_S()
        S = Sg[np.ix_(idx, idx, range(Sg.shape[2]))]
    elif key is None:
        S = r.S[np.ix_(idx, idx, range(r.S.shape[2]))]
    else:
        S = m.get_result_psd(key=key, index=np.array(idx), convert_to=P.str("convert_to"), modes=P.ev("modes"))
    S = np.asarray(S)
    if S.ndim == 1:
        S = S[None, None, :]
    S = S * np.asarray(h(om))
    n = S.shape[0]
    auto = np.real(np.stack([S[i, i, :] for i in range(n)]))
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    std = np.sqrt(np.abs(trap(auto, om, axis=1)))
    out = {"omega": arr(om), "names": names, "auto": arr(auto), "std": arr(std), "index": idx}
    if n <= 6:
        out["cross"] = arr(S)
    sess.ns["last_psd"] = S
    return out


@api("results/system")
def results_system(sess, p):
    """System matrices (modal) at a frequency and FRF magnitudes, from Model.get_system_matrices/get_frf_fun."""
    m = sess.require_model()
    P = Params(sess, p)
    include = p.get("include") or ["hydro", "aero", "drag_elements", "drag_pontoons"]
    if m.aero is not None and m.aero.windstate is None and "aero" in include:
        include = [i for i in include if i != "aero"]
    if m.aero is None:
        include = [i for i in include if i != "aero"]
    if "aero" in include and m.aero.Kfun is None:
        def prep():
            print("Preparing aerodynamic matrices (default reduced-frequency axis) ...")
            m.aero.prepare_aero_matrices(print_progress=True)
        sess.run("prep", "Prepare aerodynamic matrices", prep, wait=600)
    Kf, Cf, Mf = m.get_system_matrices(include)
    w = P.num("omega_k", 0.5)
    K, C, M = (np.real(np.atleast_2d(f(w))) for f in (Kf, Cf, Mf))
    out = {"K": arr(K), "C": arr(C), "M": arr(M), "omega_k": w}
    if P.bool("frf", True):
        omega = np.asarray(P.ev("omega", "np.linspace(0.01, 2, 300)"), dtype=float)
        modes = np.atleast_1d(P.ev("frf_modes", "[0, 1, 2]")).astype(int)
        frf = m.get_frf_fun(include=include)
        H = np.stack([np.abs(np.diag(frf(om))[modes]) for om in omega], axis=1)
        out["frf"] = {"omega": arr(omega), "modes": arr(modes), "absH": arr(H)}
    return out


@api("results/static")
def results_static(sess, p):
    m = sess.require_model()
    y = getattr(m.results, "y_static", None)
    if y is None:
        raise UserError("No static results")
    P = Params(sess, p)
    key = _key(P) or "full"
    phi = m.get_dry_phi(key=key)
    v = np.real(phi @ np.asarray(y).ravel())
    pts, comps = _point_layout(m, key, len(v), subset=P.str("subset"))
    vals = np.array([[v[r] if r < len(v) else np.nan for r in rows] + [np.nan] * (len(comps) - len(rows))
                     for _, _, rows in pts])
    xyz = np.array([x if x is not None else [np.nan] * 3 for _, x, _ in pts], dtype=float)
    pos = {"index": arr(np.arange(len(pts)))}
    if np.isfinite(xyz).all():
        pos.update(x=arr(xyz[:, 0]), y=arr(xyz[:, 1]), z=arr(xyz[:, 2]), arclength=arr(arclength(xyz)))
    return {"values": arr(vals), "comps": comps, "labels": [pt[0] for pt in pts], "pos": pos,
            "y_static": arr(np.real(np.asarray(y).ravel()))}


@api("results/export_npz")
def results_export(sess, p):
    m = sess.require_model()
    r = m.results
    data = {}
    for k in ("omega", "S", "psi", "lambd"):
        v = getattr(r, k, None)
        if v is not None:
            data[k] = np.asarray(v)
    if getattr(r, "y_static", None) is not None:
        data["y_static"] = np.asarray(r.y_static)
    if r.lambd is not None:
        for k in ("wn", "wd", "fn", "xi", "Tn"):
            data[k] = np.asarray(getattr(r, k))
    if not data:
        raise UserError("No results to export")
    path = (p.get("path") or "").strip()
    if path:
        np.savez(path, **data)
        return {"path": path, "keys": list(data)}
    tmp = temp_path(".npz")
    np.savez(tmp, **data)
    return {"download": register_download(sess, tmp, "wawi_results.npz"), "keys": list(data)}
