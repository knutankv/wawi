"""WAWI GUI – a browser-based graphical user interface for the WAWI toolbox.

Start with ``python -m wawi.gui``, the ``wawi-gui`` command, or ``wawi.gui.launch()``.
"""
__version__ = "0.1.0"


def launch(**kwargs):
    """Start the GUI server (blocking). See ``wawi.gui.server.serve`` for options."""
    from .server import serve
    return serve(**kwargs)
