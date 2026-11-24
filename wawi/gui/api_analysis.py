"""Analysis endpoints: modal, frequency domain, static, aero preparation, actions, flutter, screening."""
import contextlib
import io
import sys
import time

import numpy as np

from .common import Params, arclength, ensure_list, node_xyz
from .serialize import arr, clean_float, jsonable
from .server import api
from .session import UserError


def _include(p, key, default):
    """Include specification: list of strings, or {'k': [...], 'c': [...], 'm': [...]}"""
    v = p.get(key)
    if v is None:
        return list(default)
    if isinstance(v, dict):
        return {k: list(vv) for k, vv in v.items()}
    return list(v)


def _aero_sections(P):
    v = P.list("aero_sections")
    return v or None


def mpc(phi):
    """Modal phase collinearity of each column (1 = real/standing mode, towards 0 = travelling/rotating).
    Closed form of wawi.modal.mpc that also handles exactly real shapes (no division by Sxy)."""
    phi = np.asarray(phi)
    if phi.ndim == 1:
        phi = phi[:, None]
    re_, im_ = np.real(phi), np.imag(phi)
    sxx, syy, sxy = (re_ * re_).sum(0), (im_ * im_).sum(0), (re_ * im_).sum(0)
    with np.errstate(all="ignore"):
        return ((sxx - syy) ** 2 + 4 * sxy ** 2) / (sxx + syy) ** 2


def modal_table(r, m=None):
    if r.lambd is None:
        return None
    out = {"wn": arr(r.wn), "wd": arr(r.wd), "fn": arr(r.fn), "fd": arr(r.fd), "Tn": arr(r.Tn),
           "Td": arr(r.Td), "xi": arr(r.xi), "lambd": arr(r.lambd), "include": jsonable(r.include_eig)}
    if m is not None and r.psi is not None:
        # MPC of the physical mode shapes (full model if available, else the hydro DOFs)
        try:
            from .common import model_phi_keys
            key = "full" if ("full" in model_phi_keys(m) and m.eldef is not None) else "hydro"
            out["mpc"] = arr(mpc(m.get_dry_phi(key=key) @ r.psi))
        except Exception:
            pass
    return out


# =====================================================================  modal analysis
@api("analysis/eig")
def run_eig(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    include = _include(p, "include", ["aero", "hydro", "drag_elements", "drag_pontoons"])
    kw = {}
    for k in ("itmax", "tol", "mac_min", "w_initial", "omega_ref", "reference_omega"):
        if P.has(k):
            kw[k] = P.ev(k)
    for k in ("keep_full", "print_warnings", "divergence_protection", "track_by_psi", "remove_velocity"):
        if k in p and p[k] is not None:
            kw[k] = P.bool(k)
    freq_kind = P.bool("freq_kind", True)
    if freq_kind:
        kw.pop("track_by_psi", None)
        kw.pop("remove_velocity", None)
        kw.pop("omega_ref", None)
    else:
        kw.pop("reference_omega", None)
    smooth = None
    if P.has("smooth_omega"):
        smooth = {"omega": np.asarray(P.ev("smooth_omega"), dtype=float),
                  "window_length": P.int("smooth_window", 11), "polyorder": P.int("smooth_polyorder", 3),
                  "axis": 2}
    extra = P.ev("kwargs", {}) or {}
    kw.update(extra)
    incl_flat = include if isinstance(include, list) else [v for vals in include.values() for v in vals]
    use_aero = "aero" in incl_flat and m.aero is not None
    ws = m.aero.windstate if m.aero is not None else None
    if use_aero and ws is None:
        raise UserError('"aero" is included but no wind state is assigned. Assign one (Environment → Wind state) or untick "aero".')
    U0_override = P.num("U0") if P.has("U0") else None

    def work():
        U0_orig = ws.U0 if ws is not None else None
        overridden = use_aero and U0_override is not None and U0_override != U0_orig
        try:
            if overridden:
                ws.U0 = float(U0_override)
                m.assign_windstate(ws)  # resets the aero matrices, so they are rebuilt for this U0
            if P.has("omega_aero") and use_aero:
                print("Preparing aerodynamic matrices ...")
                m.aero.prepare_aero_matrices(omega=np.asarray(P.ev("omega_aero"), dtype=float),
                                             aero_sections=_aero_sections(P))
            t = time.time()
            print(f"Running iterative modal analysis (include={include}) ...")
            if use_aero:
                src = "override" if overridden else "assigned wind state"
                print(f"Wind: U0 = {ws.U0:g} m/s ({src}), direction {ws.direction:g} deg; "
                      "self-excited aerodynamic stiffness/damping only (no static wind).")
            m.run_eig(normalize=P.bool("normalize"), print_progress=True, freq_kind=freq_kind, include=include,
                      smooth=smooth, aero_sections=_aero_sections(P), **kw)
            print(f"\nDone in {time.time() - t:.1f} s")
        finally:
            if overridden:
                ws.U0 = U0_orig
                m.assign_windstate(ws)  # back to the assigned wind state for later analyses
        out = modal_table(m.results, m)
        if out is not None and use_aero:
            out["U0"] = clean_float(float(U0_override) if overridden else U0_orig)
        return out
    return sess.run("eig", "Modal analysis (run_eig)", work, wait=1)


@api("analysis/eig_sweep")
def run_eig_sweep(sess, p):
    """Repeat run_eig for a list of mean wind speeds (in-wind modal properties vs. U)."""
    m = sess.require_model()
    P = Params(sess, p)
    if m.aero is None or m.aero.windstate is None:
        raise UserError("Assign a wind state to the model first")
    speeds = np.atleast_1d(np.asarray(P.ev("speeds"), dtype=float))
    include = _include(p, "include", ["aero", "hydro"])
    n_track = P.int("n_track", 10)
    freq_kind = P.bool("freq_kind", True)
    kw = P.ev("kwargs", {}) or {}
    omega_aero = P.ev("omega_aero")

    def work():
        ws = m.aero.windstate
        U_orig = ws.U0
        wn, xi = [], []
        try:
            for i, U in enumerate(speeds):
                ws.U0 = float(U)
                m.assign_windstate(ws)
                if omega_aero is not None:
                    m.aero.prepare_aero_matrices(omega=np.asarray(omega_aero, dtype=float),
                                                 aero_sections=_aero_sections(P))
                print(f"\n=== U = {U:.2f} m/s ({i + 1}/{len(speeds)}) ===")
                m.run_eig(include=include, print_progress=False, freq_kind=freq_kind,
                          aero_sections=_aero_sections(P), **kw)
                r = m.results
                k = min(n_track, len(r.lambd))
                wn.append(np.r_[r.wn[:k], [np.nan] * (n_track - k)])
                xi.append(np.r_[r.xi[:k], [np.nan] * (n_track - k)])
                print(f"[{'=' * int(20 * (i + 1) / len(speeds)):<20}] {100 * (i + 1) / len(speeds):3.0f}%")
        finally:
            ws.U0 = U_orig
            m.assign_windstate(ws)
        res = {"U": arr(speeds), "wn": arr(np.array(wn)), "xi": arr(np.array(xi))}
        sess.ns["eig_sweep"] = {"U": speeds, "wn": np.array(wn), "xi": np.array(xi)}
        return res
    return sess.run("eig", "In-wind modal sweep", work, wait=1)


# =====================================================================  frequency domain
@api("analysis/freqsim")
def run_freqsim(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    omega = np.asarray(P.ev("omega", "np.arange(0.001, 2, 0.005)"), dtype=float)
    kw = dict(
        omega_aero=None if not P.has("omega_aero") else np.asarray(P.ev("omega_aero"), dtype=float),
        omega_hydro=None if not P.has("omega_hydro") else np.asarray(P.ev("omega_hydro"), dtype=float),
        interpolation_kind=P.str("interpolation_kind", "linear"),
        max_rel_error=P.num("max_rel_error", 0.01),
        drag_iterations=P.int("drag_iterations", 0),
        tol=P.num("tol", 0.01),
        include_action=_include(p, "include_action", ["hydro", "aero"]),
        include_selfexcited=_include(p, "include_selfexcited", ["hydro", "aero", "drag_elements", "drag_pontoons"]),
        ensure_at_peaks=P.bool("ensure_at_peaks", True),
        theta_interpolation=P.str("theta_interpolation", "linear"),
        reset_Cquad_lin=P.bool("reset_Cquad_lin", True),
        aero_sections=_aero_sections(P),
    )
    extra = P.ev("kwargs", {}) or {}

    def work():
        if P.bool("prepare_aero") and kw["omega_aero"] is not None and m.aero is not None:
            print("Preparing aerodynamic matrices ...")
            m.aero.prepare_aero_matrices(omega=kw["omega_aero"], aero_sections=kw["aero_sections"])
        t = time.time()
        print(f"Running frequency-domain simulation with {len(omega)} frequencies ...")
        m.run_freqsim(omega, print_progress=True, **kw, **extra)
        print(f"\nDone in {time.time() - t:.1f} s")
        r = m.results
        return {"n_freq": int(len(r.omega)), "shape": list(r.S.shape),
                "omega": [clean_float(r.omega.min()), clean_float(r.omega.max())]}
    return sess.run("freqsim", "Frequency-domain response (run_freqsim)", work, wait=1)


@api("analysis/static")
def run_static(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    if m.aero is None or m.aero.windstate is None:
        raise UserError("Static wind analysis requires an aero model with assigned wind state")
    incl = _include(p, "include_selfexcited", ["aero"])

    def work():
        if not hasattr(m.aero, "F0_m") or m.aero.F0_m is None:
            print("Evaluating static wind action ...")
            m.aero.F0_m = m.evaluate_windaction(static=True, aero_sections=_aero_sections(P), print_progress=True)
        if "aero" in incl:
            if m.aero.Kfun is None:
                print("\nPreparing aerodynamic matrices (default reduced-frequency axis) ...")
                m.aero.prepare_aero_matrices(aero_sections=_aero_sections(P))
            w0 = P.num("omega_static", 0.015)
            K_ae = np.real(m.aero.Kfun(w0))
            print(f"Aerodynamic stiffness evaluated at omega = {w0} rad/s (quasi-static limit).")
        else:
            K_ae = 0.0
        Ktot = -K_ae + m.dry_K
        m.results.y_static = np.linalg.inv(Ktot) @ m.aero.F0_m
        print("Static modal response computed (results.y_static).")
        return {"y_static": arr(np.real(m.results.y_static))}
    return sess.run("static", "Static wind response", work, wait=2)


@api("analysis/prepare_aero")
def prepare_aero(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    if m.aero is None:
        raise UserError("Model has no aero part")
    if m.aero.windstate is None:
        raise UserError("Assign a wind state first")
    omega = None if not P.has("omega") else np.asarray(P.ev("omega"), dtype=float)

    def work():
        m.aero.prepare_aero_matrices(omega=omega, print_progress=True, aero_sections=_aero_sections(P))
        print("\nAerodynamic stiffness/damping interpolants prepared.")
        return True
    return sess.run("prep", "Prepare aerodynamic matrices", work, wait=2)


@api("analysis/precompute")
def precompute(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    what = P.str("what", "wave")
    omega = np.asarray(P.ev("omega", "np.arange(0.001, 2, 0.05)"), dtype=float)

    def work():
        if what == "wave":
            if m.hydro is None or m.hydro.seastate is None:
                raise UserError("Assign a sea state first")
            m.precompute_waveaction(omega, interpolation_kind=P.str("interpolation_kind", "linear"),
                                    max_rel_error=P.num("max_rel_error", 0.01),
                                    theta_interpolation=P.str("theta_interpolation", "quadratic"))
            print("\nWave excitation interpolant stored (model.hydro.Sqq_hydro).")
        else:
            if m.aero is None or m.aero.windstate is None:
                raise UserError("Assign a wind state first")
            m.precompute_windaction(omega, include=ensure_list(p.get("include") or ["dynamic"]),
                                    interpolation_kind=P.str("interpolation_kind", "linear"),
                                    aero_sections=_aero_sections(P))
            print("\nWind excitation stored (model.aero.Sqq_aero / F0_m).")
        return True
    return sess.run("precompute", f"Precompute {what} action", work, wait=1)


@api("analysis/load_simulation")
def load_simulation(sess, p):
    """Evaluate wave/wind excitation CPSD and simulate time histories (wawi.time_domain.simulate_mdof)."""
    m = sess.require_model()
    P = Params(sess, p)
    source = P.str("source", "wave")
    omega = np.asarray(P.ev("omega", "np.arange(0.05, 2.0, 0.05)"), dtype=float)
    fs, tmax = P.num("fs", 10.0), P.num("tmax", 300.0)
    transform_by_phi = P.bool("transform_by_phi", False)
    to_global = P.bool("to_global", False)

    def work():
        from wawi.time_domain import simulate_mdof
        if source == "wave":
            if m.hydro is None or m.hydro.seastate is None:
                raise UserError("Assign a sea state first")
            print("Evaluating wave excitation CPSD ...")
            Sqq = m.evaluate_waveaction(omega, transform_by_phi=transform_by_phi, print_progress=True,
                                        max_rel_error=P.num("max_rel_error", 0.01),
                                        theta_interpolation=P.str("theta_interpolation", "quadratic"))
        else:
            if m.aero is None or m.aero.windstate is None:
                raise UserError("Assign a wind state first")
            print("Evaluating (modal) wind excitation CPSD ...")
            Sqq = m.evaluate_windaction(omega, aero_sections=_aero_sections(P), print_progress=True)
        sess.ns["Sqq"] = Sqq
        sess.ns["omega_Sqq"] = omega
        print("\nSimulating time histories ...")
        seed = P.int("seed")
        if seed is not None:
            np.random.seed(seed)
        q, t = simulate_mdof(Sqq, omega, fs=fs, tmax=tmax, print_status=False)
        q = np.real(q)
        labels = None
        if source == "wave" and not transform_by_phi:
            from scipy.linalg import block_diag
            if to_global:
                Ttot = block_diag(*m.hydro.get_from_all("tmat"))
                q = Ttot.T @ q
            dofs = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            labels = [f"{pt.label} {d}" for pt in m.hydro.pontoons for d in dofs]
        else:
            labels = [f"mode {i + 1}" for i in range(q.shape[0])]
        sess.ns["q_sim"] = q
        sess.ns["t_q"] = t
        stride = max(1, int(np.ceil(q.shape[1] / 4000)))
        std = np.std(q, axis=1)
        print(f"Simulated {q.shape[0]} components, {q.shape[1]} samples. Stored as q_sim, t_q, Sqq, omega_Sqq.")
        stride = max(stride, int(np.ceil(q.shape[0] * q.shape[1] / 6e5)))
        return {"t": arr(np.asarray(t)[::stride]), "q": arr(q[:, ::stride], 6),
                "labels": labels, "std": arr(std), "vars": ["q_sim", "t_q", "Sqq", "omega_Sqq"],
                "Sqq_diag": arr(np.real(np.stack([np.diag(Sqq[:, :, k]) for k in range(Sqq.shape[2])], axis=1))),
                "omega": arr(omega)}
    return sess.run("loadsim", f"Simulate {source} load time histories", work, wait=1)


# =====================================================================  flutter
class _RaggedSafeNumpy:
    """Proxy for the numpy module: np.array falls back to object arrays for ragged input."""

    def __init__(self, np_mod):
        self._np = np_mod

    def __getattr__(self, k):
        return getattr(self._np, k)

    def array(self, obj, *a, **k):
        try:
            return self._np.array(obj, *a, **k)
        except ValueError:
            out = self._np.empty(len(obj), dtype=object)
            for i, o in enumerate(obj):
                out[i] = o
            return out


def _pad_rows(rows):
    rows = [np.atleast_1d(np.asarray(r)) for r in rows]
    if not rows:
        return np.zeros((0, 0), dtype=complex)
    n = max(len(r) for r in rows)
    out = np.full((len(rows), n), np.nan + 0j)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


class _Tee(io.TextIOBase):
    """Writes through to a stream and keeps a copy (to read wawi's convergence message)."""

    def __init__(self, stream):
        self.stream, self.buf = stream, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()

    def text(self):
        return self.buf.getvalue()


OSC_TOL = 1e-9  # |Im(lambda)| below this: non-oscillating (real) root


def _flutter_summary(res, sec, B, converged):
    """Interpret the itflutter result: flutter speed bracket, critical root, convergence."""
    V = np.asarray(res["V"], dtype=float)
    lam = _pad_rows(res["lambd"])
    out = {"V": arr(V), "section": sec, "converged": bool(converged)}
    if not len(V) or lam.ndim != 2 or not lam.size:
        out["message"] = "No results (no velocity step completed)."
        return out
    with np.errstate(all="ignore"):
        re_ = np.where(np.isfinite(lam.real), lam.real, -np.inf)
        out["wd"] = arr(np.abs(np.imag(lam)))
        out["xi"] = arr(-np.real(lam) / np.abs(lam))
    max_re = re_.max(axis=1)
    stable, unstable = max_re <= 0, max_re > 0
    V_st = V[stable].max() if stable.any() else None                       # highest stable speed
    above = unstable & (V > V_st) if V_st is not None else unstable
    V_un = V[above].min() if above.any() else None                          # lowest unstable speed above it
    out["V_stable"], out["V_unstable"] = clean_float(V_st), clean_float(V_un)

    # The roots come in complex-conjugate pairs (2 per mode), not necessarily adjacent when tracked by shape.
    # Keep one root per pair (positive frequency at the first step; tracking keeps column identity) and label
    # each by its mode number = frequency rank at the lowest speed. Real roots get no mode number.
    i0 = int(np.argmin(V))
    keep = np.where(lam[i0].imag > OSC_TOL)[0]
    root_mode = [None] * lam.shape[1]
    for rank, c in enumerate(keep[np.argsort(lam[i0].imag[keep])]):
        root_mode[int(c)] = rank + 1
    out["root_mode"] = root_mode

    def to_kept(i, k):
        """Column of the kept root representing root k at step i (its conjugate partner if k was dropped)."""
        if root_mode[k] is not None or not len(keep):
            return k
        return int(keep[np.argmin(np.abs(lam[i, keep] - np.conj(lam[i, k])))])

    # Critical root: at the lowest unstable step if any, otherwise at the highest stable step
    i = int(np.where(V == V_un)[0][0]) if V_un is not None else int(np.where(V == V_st)[0][0])
    k = int(np.argmax(re_[i]))
    l = lam[i, k]
    oscillating = bool(abs(l.imag) > OSC_TOL)
    kc = to_kept(i, k) if oscillating else k
    out.update(critical_col=kc, critical_mode=root_mode[kc], critical_index=i, oscillating=oscillating,
               omega_flutter=clean_float(abs(l.imag)), critical_re=clean_float(l.real))

    # Speed steps sorted by speed (for mode-shape plotting: critical mode shape at each step)
    crit_all = np.asarray(res.get("critical_mode", []), dtype=int)
    steps = []
    for j in np.argsort(V, kind="stable"):
        j = int(j)
        if j >= len(crit_all):
            continue
        kj = int(crit_all[j])
        lj = lam[j, kj]
        with np.errstate(all="ignore"):
            xi_j = -lj.real / abs(lj) if abs(lj) > 0 else None
        steps.append({"i": j, "V": clean_float(V[j]), "stable": bool(max_re[j] <= 0),
                      "mode": root_mode[to_kept(j, kj)] if abs(lj.imag) > OSC_TOL else None,
                      "omega": clean_float(abs(lj.imag)), "xi": clean_float(xi_j)})
    out["steps"] = steps
    flutter_step = [n for n, s in enumerate(steps) if V_st is not None and s["V"] == clean_float(V_st)]
    out["flutter_step"] = flutter_step[-1] if flutter_step else (len(steps) - 1 if steps else None)
    with np.errstate(all="ignore"):
        osc = np.abs(lam[i].imag) > OSC_TOL
        xi_i = -lam[i].real / np.abs(lam[i])
        out["min_xi"] = clean_float(np.nanmin(np.where(osc, xi_i, np.nan))) if osc.any() else None
    out["n_real_unstable"] = int(np.sum((np.abs(lam[i].imag) <= OSC_TOL) & (re_[i] > 0)))

    if V_st is not None and V_un is not None:
        out["V_flutter"] = clean_float(V_st)
        kind = ("flutter (oscillating root, omega = %.4g rad/s, reduced velocity V/(B*omega) = %.3g)" % (abs(l.imag), V_st / (B * abs(l.imag)))
                if oscillating else "a NON-OSCILLATING (real) root: static divergence or a spurious root, not flutter")
        mode_txt = f"critical mode {root_mode[kc]}" if root_mode[kc] is not None else "critical root (no mode number)"
        msg = f"Instability between V = {V_st:.6g} and {V_un:.6g} m/s, caused by {kind}; {mode_txt}."
        if out["min_xi"] is not None:
            msg += f" Minimum damping of oscillating modes at {V_un:.4g} m/s: {out['min_xi'] * 100:.4g} %."
        if not converged:
            msg += " NOT CONVERGED to the speed tolerance (increase 'Max speed iterations' or the tolerance)."
    elif V_un is None:
        out["V_flutter"] = None
        msg = (f"No instability found up to V = {V.max():.4g} m/s (all eigenvalues have negative real parts). "
               "Increase the start speed, step or number of speed iterations to search higher.")
    else:
        out["V_flutter"] = None
        msg = (f"Unstable already at the lowest speed computed (V = {V_un:.4g} m/s); "
               + ("oscillating root." if oscillating else "NON-OSCILLATING (real) root - check the model/ADs.")
               + " Lower the start speed.")
    out["message"] = msg
    return out


@api("analysis/flutter")
def run_flutter(sess, p):
    m = sess.require_model()
    P = Params(sess, p)
    if m.aero is None:
        raise UserError("Flutter analysis requires an aero model")
    sec = P.str("section") or next(iter(m.aero.sections))
    method = P.str("method", "cont")

    def work():
        from wawi import wind
        ixs, xyz = m.get_aerosection_phi_and_x(sec)
        phi = m.get_dry_phi("full")[ixs, :]
        coord = P.str("x_coord", "arclength")
        if coord == "arclength":
            x = arclength(xyz)
        else:
            x = np.asarray(xyz)[:, "xyz".index(coord)]
        src = P.str("ad_source", "section")
        if src == "flatplate":
            ad_dict = wind.flatplate_ads()
        elif src == "expr":
            v = sess.ev(p.get("ad_expr"))
            ad_dict = v.to_dict() if hasattr(v, "to_dict") else v
        else:
            ad_dict = m.aero.sections[sec].ADs.to_dict()
        B = P.num("B") or m.aero.sections[sec].B
        Ms, Cs, Ks = m.dry_M, m.dry_C, m.dry_K
        common = dict(V=P.num("V0", 0.1), rho=P.num("rho", 1.225), dV=P.num("dV", 5.0),
                      overshoot_factor=P.num("overshoot_factor", 0.5),
                      itmax={"V": P.int("itmax_V", 50), "f": P.int("itmax_f", 15)},
                      tol={"V": P.num("tol_V", 1e-3), "f": P.num("tol_f", 1e-4)}, print_progress=True)
        print(f"Flutter analysis on section '{sec}' with {phi.shape[1]} modes, B = {B} m ...")
        orig_np = wind.np
        wind.np = _RaggedSafeNumpy(orig_np)  # tolerate ragged result lists at the end of the wawi routine
        tee = _Tee(sys.stdout)
        try:
            with contextlib.redirect_stdout(tee):
                if method == "naive":
                    res = wind.itflutter_cont_naive(Ms, Cs, Ks, phi, x, ad_dict, B, **common)
                else:
                    res = wind.itflutter_cont(Ms, Cs, Ks, phi, x, ad_dict, B, keep_all=P.bool("keep_all", True),
                                              track_by_psi=P.bool("track_by_psi", True), **common)
        finally:
            wind.np = orig_np
        sess.ns["flutter_result"] = res
        out = _flutter_summary(res, sec, B, converged="Flutter estimated to occur" in tee.text())
        # for mode-shape plotting (critical mode shape per speed step, in dry-modal coordinates)
        sess.meta["flutter"] = {"steps": out.get("steps", []), "n_modes": int(phi.shape[1]), "section": sec}
        print("\n" + out["message"])
        print("Results stored as 'flutter_result'.")
        return out
    return sess.run("flutter", f"Flutter analysis ({sec})", work, wait=1)


# =====================================================================  screening
@api("analysis/screening")
def run_screening(sess, p):
    """Parameter screening over sea states (wawi.model.ScreeningCase) with run_freqsim per combination."""
    m = sess.require_model()
    P = Params(sess, p)
    from wawi.model import ScreeningCase
    if P.has("json_path"):
        case = ScreeningCase.from_json(str(P.path("json_path", kind="file")))
    else:
        ss = sess.ns.get("seastate") if P.str("seastate", "editor") == "editor" else m.hydro.seastate
        if ss is None:
            raise UserError("Build a sea state first (Environment → Sea state)")
        parvar = {}
        for k, v in (p.get("parvar") or {}).items():
            if str(v or "").strip():
                parvar[k] = np.atleast_1d(np.asarray(sess.ev(v), dtype=float))
        if not parvar:
            raise UserError("Specify at least one parameter variation")
        import copy
        case = ScreeningCase(copy.deepcopy(ss), parvar, independent=P.bool("independent", True), name=P.str("name"))
    omega = np.asarray(P.ev("omega", "np.arange(0.001, 2, 0.01)"), dtype=float)
    key = P.str("key", "hydro")
    quantity = P.str("quantity", "std")
    T = P.num("T", 3600.0)
    incl_sx = _include(p, "include_selfexcited", ["hydro", "aero", "drag_elements", "drag_pontoons"])
    incl_ac = _include(p, "include_action", ["hydro"])
    max_rel_error = P.num("max_rel_error", 0.01)

    def work():
        n = int(case.n)
        print(f"Screening {n} sea states ...")
        case.reset_ix()
        case.ix = -1
        rows, combos = [], []
        orig = m.hydro.seastate
        try:
            for i in range(n):
                ss = case.get_next_seastate()
                combo = case.get_combo()
                m.assign_seastate(ss)
                m.run_freqsim(omega, print_progress=False, include_selfexcited=incl_sx, include_action=incl_ac,
                              max_rel_error=max_rel_error)
                if quantity == "expmax":
                    v = m.get_result_expmax(T, key=key)
                else:
                    v = m.get_result_std(key=key)
                rows.append(np.real(v))
                combos.append({k: float(np.degrees(vv)) if k in ("theta0", "thetaU") else float(vv)
                               for k, vv in combo.items()})
                print(f"[{'=' * int(20 * (i + 1) / n):<20}] {100 * (i + 1) / n:3.0f}%  {combos[-1]}")
        finally:
            m.assign_seastate(orig)
        R = np.vstack(rows)
        sess.ns["screening_result"] = {"combos": combos, "values": R, "key": key, "quantity": quantity}
        print("\nStored as 'screening_result'.")
        return {"combos": combos, "values": arr(R), "key": key, "quantity": quantity,
                "parameters": list(case.parvar.keys())}
    return sess.run("screening", f"Screening ({case.n} cases)", work, wait=1)
