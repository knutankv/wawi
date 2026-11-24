"""API explorer (call any wawi function/method), variables, inspector and Python console."""
import ast
import importlib
import inspect
import json
import time

import numpy as np

from .bootstrap import COMPAT_SHIMS, IMPORT_STATUS, WAWI_MODULES
from .common import Params, collect_figures, register_download, temp_path
from .serialize import arr, first_doc, jsonable, safe_signature, short_repr, summarize, type_name
from .server import api
from .session import UserError

_TREE_CACHE = {}


def _params_of(obj):
    try:
        sig = inspect.signature(obj)
    except Exception:
        return None
    out = []
    for name, prm in sig.parameters.items():
        if name == "self":
            continue
        d = {"name": name, "kind": str(prm.kind).split(".")[-1]}
        if prm.default is not inspect.Parameter.empty:
            d["default"] = _default_src(prm.default)
        out.append(d)
    return out


def _default_src(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        if isinstance(v, float) and not np.isfinite(v):
            return "np.inf" if v > 0 else "-np.inf"
        return repr(v)
    if isinstance(v, np.ndarray):
        return f"np.array({v.tolist()!r})"
    if isinstance(v, (list, tuple, dict)):
        try:
            return repr(v)
        except Exception:
            pass
    if callable(v) and getattr(v, "__name__", "") == "<lambda>":
        try:
            return inspect.getsource(v).split("=", 1)[-1].strip().rstrip(",):")
        except Exception:
            return "lambda …"
    return short_repr(v, 80)


def _member_info(name, obj, qual):
    info = {"name": name, "qual": qual, "doc": first_doc(obj)}
    if inspect.isclass(obj):
        info["kind"] = "class"
        info["signature"] = safe_signature(obj)
        info["params"] = _params_of(obj)
        members = []
        for mname, raw in vars(obj).items():
            if mname.startswith("_") and mname != "__init__":
                continue
            if isinstance(raw, property):
                members.append({"name": mname, "kind": "property", "doc": first_doc(raw),
                                "settable": raw.fset is not None})
            elif isinstance(raw, staticmethod):
                f = raw.__func__
                members.append({"name": mname, "kind": "staticmethod", "signature": safe_signature(f),
                                "params": _params_of(f), "doc": first_doc(f)})
            elif isinstance(raw, classmethod):
                f = raw.__func__
                members.append({"name": mname, "kind": "classmethod", "signature": safe_signature(getattr(obj, mname)),
                                "params": _params_of(getattr(obj, mname)), "doc": first_doc(f)})
            elif inspect.isfunction(raw):
                if mname == "__init__":
                    continue
                members.append({"name": mname, "kind": "method", "signature": safe_signature(raw),
                                "params": _params_of(raw), "doc": first_doc(raw)})
        info["members"] = members
    elif callable(obj):
        info["kind"] = "function"
        info["signature"] = safe_signature(obj)
        info["params"] = _params_of(obj)
    return info


@api("explorer/tree")
def explorer_tree(sess, p):
    if "tree" in _TREE_CACHE and not p.get("refresh"):
        return _TREE_CACHE["tree"]
    mods = []
    names = ["wawi.model._model", "wawi.model._hydro", "wawi.model._aero", "wawi.model._dry", "wawi.model._screening"] + \
            [f"wawi.{m}" for m in WAWI_MODULES if m != "model"]
    for modname in names:
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:
            mods.append({"name": modname, "error": str(exc), "members": []})
            continue
        public = modname.replace("._model", "").replace("._hydro", "").replace("._aero", "").replace(
            "._dry", "").replace("._screening", "")
        members = []
        for name, obj in vars(mod).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != modname:
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            members.append(_member_info(name, obj, f"{public}.{name}"))
        members.sort(key=lambda m: (m["kind"] != "class", m["name"].lower()))
        mods.append({"name": modname, "public": public, "doc": first_doc(mod), "members": members})
    mods.append({"name": "wawi.ext.abq", "public": "wawi.ext.abq", "members": [],
                 "error": "Abaqus export helpers – only importable inside Abaqus/CAE (see examples/3 Software interfacing)."})
    tree = {"modules": mods, "imports": IMPORT_STATUS, "compat": COMPAT_SHIMS}
    _TREE_CACHE["tree"] = tree
    return tree


@api("explorer/doc")
def explorer_doc(sess, p):
    target = p.get("target", "")
    try:
        obj = eval(target, sess.ns)
    except Exception as exc:
        raise UserError(f"Cannot resolve {target!r}: {exc}")
    doc = inspect.getdoc(obj) or "(no docstring)"
    src = None
    if p.get("source"):
        try:
            src = inspect.getsource(obj)
        except Exception as exc:
            src = f"(source unavailable: {exc})"
    return {"target": target, "doc": doc, "signature": safe_signature(obj), "params": _params_of(obj),
            "source": src}


@api("explorer/call")
def explorer_call(sess, p):
    """Call any callable: target expression, positional args and keyword args as expressions."""
    target = (p.get("target") or "").strip()
    if not target:
        raise UserError("No target")
    args_src = [a for a in (p.get("args") or [])]
    kwargs_src = {k: v for k, v in (p.get("kwargs") or {}).items() if str(v).strip() != ""}
    assign = (p.get("assign_to") or "").strip()
    if assign and not assign.isidentifier():
        raise UserError("'Store result as' must be a valid Python identifier")
    call_src = f"{target}({', '.join([*map(str, args_src), *[f'{k}={v}' for k, v in kwargs_src.items()]])})"

    def work():
        fn = eval(target, sess.ns)
        args = [eval(a, sess.ns) if isinstance(a, str) else a for a in args_src]
        kwargs = {k: (eval(v, sess.ns) if isinstance(v, str) else v) for k, v in kwargs_src.items()}
        print(f">>> {(assign + ' = ') if assign else ''}{call_src}")
        import matplotlib.pyplot as plt
        with sess.mpl_lock:
            plt.close("all")
            t = time.time()
            res = fn(*args, **kwargs)
            dt = time.time() - t
            imgs = collect_figures(plt)
        name = assign or "_"
        sess.ns[name] = res
        print(f"\n[{dt:.2f} s] result stored as '{name}'")
        return {"stored_as": name, "summary": summarize(res), "images": imgs, "call": call_src}
    return sess.run("call", call_src[:120], work, wait=p.get("wait", 3))


@api("vars/list")
def vars_list(sess, p):
    out = []
    special = ["model", "seastate", "windstate"]
    items = {k: sess.ns.get(k) for k in special if sess.ns.get(k) is not None}
    items.update(sess.user_vars())
    for k, v in items.items():
        try:
            s = summarize(v, preview=False)
        except Exception as exc:
            s = {"type": type_name(v), "repr": f"<error {exc}>"}
        s["name"] = k
        out.append(s)
    return out


@api("vars/delete")
def vars_delete(sess, p):
    name = p.get("name")
    if name in ("model", "seastate", "windstate"):
        sess.ns[name] = None
    elif name in sess.ns and name not in sess.base_names:
        del sess.ns[name]
    return True


@api("vars/inspect")
def vars_inspect(sess, p):
    expr = (p.get("expr") or "").strip()
    try:
        obj = eval(expr, sess.ns)
    except Exception as exc:
        raise UserError(f"Cannot evaluate {expr!r}: {type(exc).__name__}: {exc}")
    info = summarize(obj)
    info["expr"] = expr
    attrs, methods = [], []
    if not isinstance(obj, (np.ndarray, str, bytes, int, float, complex, bool)) and not inspect.ismodule(obj):
        cls = type(obj)
        for name in sorted(set(dir(obj))):
            if name.startswith("__"):
                continue
            if name.startswith("_") and not p.get("private"):
                continue
            try:
                static = inspect.getattr_static(obj, name)
            except AttributeError:
                continue
            if isinstance(static, property) or (inspect.isdatadescriptor(static) and not isinstance(static, (staticmethod, classmethod))
                                                and name not in getattr(obj, "__dict__", {})):
                attrs.append({"name": name, "kind": "property", "doc": first_doc(static)})
                continue
            try:
                val = getattr(obj, name)
            except Exception as exc:
                attrs.append({"name": name, "kind": "error", "repr": str(exc)[:200]})
                continue
            if inspect.ismethod(val) or inspect.isfunction(val) or inspect.isbuiltin(val) or \
                    (callable(val) and isinstance(static, (staticmethod, classmethod))):
                methods.append({"name": name, "signature": safe_signature(val), "doc": first_doc(val),
                                "params": _params_of(val)})
            else:
                s = summarize(val, preview=False)
                attrs.append({"name": name, "kind": s.get("kind"), "type": s.get("type"), "repr": s.get("repr"),
                              "shape": s.get("shape")})
    if isinstance(obj, dict):
        info["entries"] = [{"key": short_repr(k, 60), "keyexpr": f"{expr}[{k!r}]", "repr": short_repr(v, 120),
                            "type": type_name(v)} for k, v in list(obj.items())[:200]]
    elif isinstance(obj, (list, tuple)):
        info["entries"] = [{"key": str(i), "keyexpr": f"{expr}[{i}]", "repr": short_repr(v, 120), "type": type_name(v)}
                           for i, v in enumerate(list(obj)[:200])]
    info["attrs"] = attrs
    info["methods"] = methods
    return info


@api("vars/set")
def vars_set(sess, p):
    """Evaluate expression and assign to a name or attribute (e.g. model.modal_dry.xi0)."""
    target = (p.get("target") or "").strip()
    expr = p.get("expr")
    if not target:
        raise UserError("No target")
    code = f"{target} = {expr}"

    def work():
        exec(compile(code, "<gui-set>", "exec"), sess.ns)
        print(f">>> {code}")
        return {"summary": summarize(eval(target, sess.ns), preview=False)}
    return sess.run("edit", code[:120], work)


@api("vars/data")
def vars_data(sess, p):
    """Numeric data of an expression for plotting (1D/2D, complex parts handled client-side)."""
    expr = (p.get("expr") or "").strip()
    try:
        v = eval(expr, sess.ns)
    except Exception as exc:
        raise UserError(f"Cannot evaluate {expr!r}: {exc}")
    a = np.asarray(v)
    if a.dtype == object:
        raise UserError(f"{expr} is not numeric")
    if a.size > 2_000_000:
        raise UserError(f"Array too large to plot ({a.size} elements); slice it first, e.g. {expr}[0, 0, :]")
    xexpr = (p.get("x") or "").strip()
    x = None
    if xexpr:
        x = np.asarray(eval(xexpr, sess.ns))
    return {"shape": list(a.shape), "data": arr(a), "x": arr(x) if x is not None else None, "complex": bool(np.iscomplexobj(a))}


@api("vars/download")
def vars_download(sess, p):
    expr = (p.get("expr") or "").strip()
    fmt = p.get("format", "npz")
    v = eval(expr, sess.ns)
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in expr)[:40] or "data"
    if fmt == "csv":
        a = np.asarray(v)
        if a.ndim > 2:
            raise UserError("CSV export supports 1D/2D arrays; slice first or use NPZ")
        tmp = temp_path(".csv")
        if np.iscomplexobj(a):
            a = np.column_stack([np.real(a).reshape(len(a), -1), np.imag(a).reshape(len(a), -1)])
        np.savetxt(tmp, np.atleast_1d(a), delimiter=",")
        return {"download": register_download(sess, tmp, safe + ".csv")}
    if fmt == "json":
        tmp = temp_path(".json")
        tmp.write_text(json.dumps(jsonable(v)))
        return {"download": register_download(sess, tmp, safe + ".json")}
    if fmt == "pickle":
        import dill
        tmp = temp_path(".pkl")
        with open(tmp, "wb") as f:
            dill.dump(v, f)
        return {"download": register_download(sess, tmp, safe + ".pkl")}
    tmp = temp_path(".npz")
    if isinstance(v, dict):
        np.savez(tmp, **{str(k): np.asarray(vv) for k, vv in v.items()
                         if np.asarray(vv).dtype != object})
    else:
        np.savez(tmp, data=np.asarray(v))
    return {"download": register_download(sess, tmp, safe + ".npz")}


@api("console/exec")
def console_exec(sess, p):
    code = p.get("code") or ""
    if not code.strip():
        raise UserError("Nothing to run")
    try:
        tree = ast.parse(code, "<console>", "exec")
    except SyntaxError as exc:
        raise UserError(f"SyntaxError: {exc.msg} (line {exc.lineno})")

    def work():
        import matplotlib.pyplot as plt
        last = None
        body = tree
        if body.body and isinstance(body.body[-1], ast.Expr):
            last = ast.Expression(body.body.pop().value)
        with sess.mpl_lock:
            plt.close("all")
            exec(compile(body, "<console>", "exec"), sess.ns)
            value = None
            if last is not None:
                value = eval(compile(last, "<console>", "eval"), sess.ns)
                if value is not None:
                    sess.ns["_"] = value
                    print(repr(value) if not isinstance(value, np.ndarray) or value.size < 2000
                          else f"ndarray{value.shape} {value.dtype}")
            imgs = collect_figures(plt)
        return {"images": imgs, "summary": summarize(value) if value is not None else None}
    first = code.strip().splitlines()[0][:80]
    return sess.run("console", f"Console: {first}", work, wait=p.get("wait", 5))


@api("mpl/plot")
def mpl_plot(sess, p):
    """Run plotting code / wawi.plot function and return PNGs (synchronous, short tasks)."""
    code = p.get("code") or ""

    def work():
        import matplotlib.pyplot as plt
        with sess.mpl_lock:
            plt.close("all")
            exec(compile(code, "<plot>", "exec"), sess.ns)
            return {"images": collect_figures(plt)}
    return sess.run("plot", "Matplotlib plot", work, wait=30)
