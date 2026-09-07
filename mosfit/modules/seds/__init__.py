"""Lazy loading for `SED` modules.

SED classes are imported on first attribute access so ``import mosfit`` (and
fits that do not use the SESN SEDONA emulator) do not import ``torch``.
"""
import ast
import importlib
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))


def _discover_exports():
    """Map public names to submodule names without importing those modules."""
    name_to_mod = {}
    for filename in os.listdir(_DIR):
        if not filename.endswith('.py') or filename == '__init__.py':
            continue
        mod_name = filename[:-3]
        path = os.path.join(_DIR, filename)
        with open(path, encoding='utf-8') as fh:
            tree = ast.parse(fh.read(), filename=filename)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                name_to_mod[node.name] = mod_name
    return name_to_mod


_NAME_TO_MODULE = _discover_exports()
__all__ = list(_NAME_TO_MODULE)


def __getattr__(name):
    mod_name = _NAME_TO_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}')
    module = importlib.import_module('.' + mod_name, __name__)
    obj = getattr(module, name)
    setattr(sys.modules[__name__], name, obj)
    return obj


def __dir__():
    return sorted(set(globals()) | set(_NAME_TO_MODULE))
