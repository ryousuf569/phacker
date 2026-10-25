try:
    from . import _core
except ImportError:  # sdist built without a compiler
    _core = None


def backend():
    return "numpy" if _core is None else "cpp"
