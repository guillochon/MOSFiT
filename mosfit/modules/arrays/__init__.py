"""Initilization procedure for `Array` modules."""
import inspect
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
__all__ = []

for py in [
        f[:-3] for f in os.listdir(path)
        if f.endswith('.py') and f != '__init__.py'
]:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [
        x[1] for x in inspect.getmembers(mod)
        if (inspect.isroutine(x[1]) or inspect.isclass(x[1])
            ) and inspect.getmodule(x[1]) == mod
    ]
    for cls in classes:
        __all__.append(cls.__name__)
        setattr(sys.modules[__name__], cls.__name__, cls)
