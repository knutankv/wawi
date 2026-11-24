"""Stand-alone calculators built on wawi's function modules (wave, wind, random, structural, signal ...)."""
import numpy as np

from .common import Params
from .serialize import arr, clean_float, jsonable
from .server import api
from .session import UserError

TOOLS = {}


def tool(name):
    def deco(fn):
        TOOLS[name] = fn
        return fn
    return deco


def trap(y, x):
    return (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y, x)


@api("tools/run")
def tools_run(sess, p):
    name = p.get("tool")
    fn = TOOLS.get(name)
    if fn is None:
        raise UserError(f"Unknown tool {name}")
    return fn(sess, Params(sess, p.get("params") or {}))


@tool("wave_spectrum")
def t_wave_spectrum(sess, P):
    from wawi import wave
    omega = np.asarray(P.ev("omega", "np.linspace(0.05, 3, 400)"), dtype=float)
    Hs, Tp, gamma = P.num("Hs", 2.0), P.num("Tp", 8.0), P.num("gamma", 3.3)
    sigma = list(np.atleast_1d(P.ev("sigma", "[0.07, 0.09]")).astype(float))
    series = {}
    kinds = P.list("kinds") or ["jonswap"]
    for k in kinds:
        if k == "jonswap":
            S = wave.jonswap(Hs, Tp, gamma, sigma=sigma)(omega)
        elif k == "jonswap_dnv":
            S = wave.jonswap_dnv(Hs, Tp, gamma, sigma=sigma)(omega)
        elif k == "pm2":
            S = wave.pm2(Hs, Tp, unit="rad/s")(omega)
        else:
            continue
        S = np.asarray(S, dtype=float)
        series[k] = {"S": arr(S), "Hs_m0": clean_float(4 * np.sqrt(trap(S, omega))),
                     "Tp_peak": clean_float(2 * np.pi / omega[np.nanargmax(S)])}
    pe = wave.peak_enhancement(gamma, Tp, sigma, normalize=False)(omega)
    out = {"omega": arr(omega), "series": series, "peak_enhancement": arr(pe)}
    if P.has("alpha"):
        out["Hs_from_alpha"] = clean_float(wave.swh_from_gamma_alpha_Tp(gamma, P.num("alpha"), Tp))
    return out


@tool("dirdist")
def t_dirdist(sess, P):
    from wawi import wave
    theta = np.asarray(P.ev("theta", "np.linspace(-np.pi, np.pi, 721)"), dtype=float)
    s = P.num("s", 10.0)
    th0 = np.radians(P.num("theta0", 0.0))
    out = {"theta": arr(theta), "series": {}}
    for k in P.list("kinds") or ["dirdist_robust", "dirdist_decimal"]:
        f = getattr(wave, k, None)
        if f is None:
            continue
        try:
            D = np.asarray(f(s, theta0=th0)(theta), dtype=float)
            out["series"][k] = {"D": arr(D), "integral": clean_float(trap(D, theta))}
        except Exception as exc:
            out["series"][k] = {"error": str(exc)}
    return out


@tool("dispersion")
def t_dispersion(sess, P):
    from wawi.wave import dispersion_relation_scalar
    omega = np.asarray(P.ev("omega", "np.linspace(0.05, 3, 200)"), dtype=float)
    depths = np.atleast_1d(P.ev("depth", "[np.inf, 50, 20]")).astype(float)
    g, U, thU = P.num("g", 9.81), P.num("U", 0.0), np.radians(P.num("theta_rel_U", 0.0))
    out = {"omega": arr(omega), "series": []}
    for h in depths:
        k = np.array([dispersion_relation_scalar(w, h=h, g=g, U=U, theta_rel_U=thU) for w in omega])
        out["series"].append({"depth": clean_float(h), "k": arr(k), "wavelength": arr(2 * np.pi / k),
                              "celerity": arr(omega / k)})
    return out


@tool("wave_coherence")
def t_wave_coherence(sess, P):
    from wawi.wave import dispersion_relation_scalar, get_coh_cos2s
    omega = np.asarray(P.ev("omega", "np.linspace(0.05, 2.0, 200)"), dtype=float)
    dx, dy = P.num("dx", 100.0), P.num("dy", 0.0)
    s, th0, depth = P.num("s", 10.0), np.radians(P.num("theta0", 0.0)), P.num("depth", np.inf)
    kappa = np.array([dispersion_relation_scalar(w, h=depth) for w in omega])[:, None]
    coh = get_coh_cos2s(kappa, dx, dy, s, th0, k_max=P.int("k_max", 10), input_is_kappa=True)
    return {"omega": arr(omega), "abs": arr(np.abs(coh)), "re": arr(np.real(coh)), "im": arr(np.imag(coh))}


@tool("wind_spectrum")
def t_wind_spectrum(sess, P):
    from wawi.wind import get_beaufort, kaimal_auto, von_karman_auto
    omega = np.asarray(P.ev("omega", "np.logspace(-3, 1, 300)"), dtype=float)
    U = P.num("U", 20.0)
    Lx = np.asarray(P.ev("Lx", "[115, 30, 9.58]"), dtype=float)
    A = np.asarray(P.ev("A", "[6.8, 9.4, 9.4]"), dtype=float)
    I = np.asarray(P.ev("I", "[0.136, 0.1, 0.072]"), dtype=float)
    sigma = I * U
    Sk = np.vstack([kaimal_auto(w, Lx, A, sigma, U) for w in omega]).T
    Sv = np.vstack([von_karman_auto(w, Lx, sigma, U) for w in omega]).T
    try:
        bf = get_beaufort(U)
    except Exception:
        bf = None
    return {"omega": arr(omega), "kaimal": arr(Sk), "vonkarman": arr(Sv), "sigma": arr(sigma), "beaufort": jsonable(bf),
            "var_kaimal": arr([trap(Sk[i], omega) for i in range(3)]),
            "var_vonkarman": arr([trap(Sv[i], omega) for i in range(3)])}


@tool("terrain")
def t_terrain(sess, P):
    from wawi.wind_code import terrain_roughness
    z = np.asarray(P.ev("z", "np.linspace(0.5, 200, 200)"), dtype=float)
    out = {"z": arr(z), "series": []}
    for z0 in np.atleast_1d(P.ev("z0", "[0.003, 0.01, 0.05, 0.3]")).astype(float):
        cr = np.array([terrain_roughness(zi, z0=z0, zmin=P.num("zmin", 1.0)) for zi in z])
        out["series"].append({"z0": clean_float(z0), "cr": arr(cr)})
    return out


@tool("ads")
def t_ads(sess, P):
    from wawi import wind
    v = np.linspace(P.num("vmin", 0.01), P.num("vmax", 10.0), P.int("n", 300))
    kind = P.str("kind", "flatplate")
    if kind == "flatplate":
        d = wind.flatplate_ads()
    else:
        lc = {k: P.num(k, 0.0) for k in ["Cd", "dCd", "Cl", "dCl", "Cm", "dCm"]}
        d = wind.quasisteady_ads(P.num("D", 3.0), P.num("B", 20.0), lc)
    return {"v": arr(v), "ads": {k: arr(np.real(np.asarray(f(v), dtype=complex)) * np.ones_like(v)) for k, f in d.items()}}


@tool("response_stats")
def t_response_stats(sess, P):
    from wawi import random as wr
    src = P.str("source", "jonswap")
    if src == "expr":
        S = np.real(np.asarray(P.ev("S"), dtype=complex)).ravel()
        omega = np.asarray(P.ev("omega"), dtype=float).ravel()
        if S.shape != omega.shape:
            raise UserError(f"S and omega must have equal length ({S.shape} vs {omega.shape})")
    else:
        from wawi.wave import jonswap
        omega = np.asarray(P.ev("omega", "np.linspace(0.05, 3, 600)"), dtype=float)
        S = jonswap(P.num("Hs", 2.0), P.num("Tp", 8.0), P.num("gamma", 3.3))(omega)
    T = P.num("T", 3600.0)
    m0, m2 = wr.m0(S, omega), wr.m2(S, omega)
    v0 = wr.v0(m0, m2)
    out = {"omega": arr(omega), "S": arr(S), "m0": clean_float(m0), "m2": clean_float(m2),
           "m4": clean_float(wr.stoch_mom(S, omega, n=4)),
           "std": clean_float(np.sqrt(m0)), "v0": clean_float(v0), "Tz": clean_float(wr.zero_crossing_period(S, omega)),
           "peakfactor": clean_float(wr.peakfactor(T, v0)), "expmax": clean_float(wr.expmax(T, v0, np.sqrt(m0))),
           "T": T}
    Ts = np.logspace(1, 5, 60)
    out["curve"] = {"T": arr(Ts), "kp": arr([wr.peakfactor(t, v0) for t in Ts])}
    return out


@tool("gumbel")
def t_gumbel(sess, P):
    from wawi.prob import gumbel_log
    u = np.asarray(P.ev("umax"), dtype=float).ravel()
    uo, ll = gumbel_log(u)
    return {"umax": arr(uo), "loglogF": arr(ll)}


@tool("rayleigh")
def t_rayleigh(sess, P):
    from wawi.structural import rayleigh, rayleigh_damping_fit
    xi, w1, w2 = P.num("xi", 0.005), P.num("omega_1", 0.2), P.num("omega_2", 3.0)
    c = rayleigh_damping_fit(xi, w1, w2)
    omega = np.linspace(P.num("omega_min", 0.02), P.num("omega_max", max(5.0, 1.5 * w2)), 300)
    return {"alpha": clean_float(c["mass"]), "beta": clean_float(c["stiffness"]), "omega": arr(omega),
            "xi": arr(rayleigh(c["mass"], c["stiffness"], omega))}


@tool("fit_jonswap")
def t_fit_jonswap(sess, P):
    from wawi.signal import fit_jonswap
    from wawi.wave import jonswap
    S = np.real(np.asarray(P.ev("S"), dtype=complex)).ravel()
    w = np.asarray(P.ev("omega"), dtype=float).ravel()
    sigma = list(np.atleast_1d(P.ev("sigma", "[0.07, 0.09]")).astype(float))
    res = fit_jonswap(S, w, sigma=sigma, initial_values=P.ev("initial_values"))
    fit = jonswap(res["Hs"], res["Tp"], res["gamma"], sigma=sigma)(w)
    return {"omega": arr(w), "S": arr(S), "fit": arr(fit), "Hs": clean_float(res["Hs"]), "Tp": clean_float(res["Tp"]),
            "gamma": clean_float(res["gamma"])}


@tool("time_sim")
def t_time_sim(sess, P):
    from wawi.time_domain import simulate_mdof
    S = np.asarray(P.ev("S"))
    omega = np.asarray(P.ev("omega"), dtype=float).ravel()
    if S.ndim == 1:
        S = S[None, None, :]
    if S.ndim != 3 or S.shape[2] != len(omega):
        raise UserError(f"S must be (n, n, n_freq) matching omega; got {S.shape} and {omega.shape}")
    if P.has("seed"):
        np.random.seed(P.int("seed"))
    q, t = simulate_mdof(S, omega, fs=P.num("fs", 10.0), tmax=P.num("tmax", 600.0))
    q = np.real(q)
    name = P.str("assign_to", "q_tool")
    sess.ns[name] = q
    sess.ns[name + "_t"] = t
    stride = max(1, int(np.ceil(q.size / 4e5)))
    return {"t": arr(np.asarray(t)[::stride]), "q": arr(q[:, ::stride], 6), "std": arr(np.std(q, axis=1)),
            "target_std": arr(np.sqrt(np.abs([trap(np.real(S[i, i]), omega) for i in range(S.shape[0])]))),
            "vars": [name, name + "_t"]}


@tool("linear_drag")
def t_linear_drag(sess, P):
    from wawi.wave import linear_drag_damping
    Cd = np.atleast_1d(P.ev("Cd", "[1.0, 1.0, 2.0]")).astype(float)
    std_udot = np.atleast_1d(P.ev("std_udot", "[0.1, 0.1, 0.1]")).astype(float)
    c = linear_drag_damping(Cd, std_udot, area=np.atleast_1d(P.ev("area", "1.0")).astype(float),
                            rho=P.num("rho", 1025.0), as_matrix=False)
    return {"c": arr(c)}
