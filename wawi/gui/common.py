"""Helpers shared by the API modules."""
import base64
import io
import secrets
import tempfile
from pathlib import Path

import numpy as np

from .session import UserError

DOF_NAMES = ["x", "y", "z", "θx", "θy", "θz"]


class Params:
    """Typed access to request parameters; string fields are evaluated as Python expressions."""

    def __init__(self, sess, raw):
        self.sess = sess
        self.raw = raw or {}

    def has(self, k):
        v = self.raw.get(k)
        return v is not None and not (isinstance(v, str) and v.strip() == "")

    def get(self, k, default=None):
        return self.raw.get(k, default)

    def str(self, k, default=None):
        v = self.raw.get(k)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return default
        return str(v).strip()

    def bool(self, k, default=False):
        v = self.raw.get(k, default)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def ev(self, k, default=None):
        """Evaluate field as Python expression (numbers/lists pass through)."""
        try:
            return self.sess.ev(self.raw.get(k), default)
        except UserError:
            raise
        except Exception as exc:
            raise UserError(f"Field '{k}': {type(exc).__name__}: {exc}")

    def num(self, k, default=None):
        v = self.ev(k, default)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            raise UserError(f"Field '{k}' must be a number (got {v!r})")

    def int(self, k, default=None):
        v = self.ev(k, default)
        return None if v is None else int(v)

    def list(self, k, default=None):
        v = self.raw.get(k, default)
        if v is None:
            return default
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return default
            if v[0] in "[(" or v.startswith("np."):
                return list(np.atleast_1d(self.sess.ev(v)))
            return [s.strip() for s in v.split(",") if s.strip()]
        return list(v)

    def path(self, k, must_exist=True, kind=None):
        v = self.str(k)
        if not v:
            raise UserError(f"Please specify a path for '{k}'.")
        p = Path(v).expanduser()
        if must_exist and not p.exists():
            raise UserError(f"Path does not exist: {p}")
        if kind == "file" and must_exist and not p.is_file():
            raise UserError(f"Not a file: {p}")
        if kind == "dir" and must_exist and not p.is_dir():
            raise UserError(f"Not a folder: {p}")
        return p


def fig_to_png(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def collect_figures(plt, close=True):
    imgs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        try:
            imgs.append(fig_to_png(fig))
        except Exception as exc:  # pragma: no cover
            imgs.append(None)
            print(f"Could not render figure {num}: {exc}")
    if close:
        plt.close("all")
    return [i for i in imgs if i]


def register_download(sess, path, name=None):
    did = secrets.token_hex(8)
    sess.downloads[did] = {"path": str(path), "name": name or Path(path).name}
    return did


def temp_path(suffix):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="wawi_gui_")
    f.close()
    return Path(f.name)


def node_xyz(node):
    c = getattr(node, "coordinates", None)
    if c is not None and len(np.atleast_1d(c)) >= 3:
        return np.asarray(c, dtype=float)[:3]
    x0 = getattr(node, "x0", None)
    if x0 is not None:
        return np.asarray(x0, dtype=float)[:3]
    return np.zeros(3)


def arclength(xyz):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or len(xyz) == 0:
        return np.arange(len(xyz))
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    return d


def deg(x):
    return float(np.degrees(x))


def model_phi_keys(model):
    md = getattr(model, "modal_dry", None)
    return list(md.phi_full.keys()) if md is not None else []


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def state_str(st):
    """str() of a Seastate/Windstate that also works for inhomogeneous (function-valued) parameters."""
    if st is None:
        return None
    try:
        return str(st)
    except Exception:
        pass
    lines = [f"WAWI {type(st).__name__} (inhomogeneous)", "--------------"]
    src = getattr(st, "_gui_src", None) or {}
    for k in ("Hs", "Tp", "gamma", "theta0", "s", "U", "thetaU", "U0", "direction"):
        if not hasattr(st, k):
            continue
        v = getattr(st, k)
        if callable(v):
            lines.append(f"{k} = {src.get(k) or 'function of (x, y)'}")
        else:
            try:
                vv = float(v)
                if k in ("theta0", "thetaU") and type(st).__name__ == "Seastate":
                    vv = np.degrees(vv)
                lines.append(f"{k} = {vv:.3g}")
            except Exception:
                lines.append(f"{k} = {v!r}")
    return "\n".join(lines)
