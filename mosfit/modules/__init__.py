"""Initilization procedure for `Module`."""
from .arrays import *  # noqa: F403
from .constraints import *  # noqa: F403
from .datas import *  # noqa: F403
from .energetics import *  # noqa: F403
from .engines import *  # noqa: F403
from .module import Module
from .objectives import *  # noqa: F403
from .observables import *  # noqa: F403
from .outputs import *  # noqa: F403
from .parameters import *  # noqa: F403
from .photospheres import *  # noqa: F403
from . import seds
from .transforms import *  # noqa: F403

__all__ = ['Module']
__all__.extend(arrays.__all__)  # noqa: F405
__all__.extend(constraints.__all__)  # noqa: F405
__all__.extend(datas.__all__)  # noqa: F405
__all__.extend(energetics.__all__)  # noqa: F405
__all__.extend(engines.__all__)  # noqa: F405
__all__.extend(objectives.__all__)  # noqa: F405
__all__.extend(observables.__all__)  # noqa: F405
__all__.extend(outputs.__all__)  # noqa: F405
__all__.extend(parameters.__all__)  # noqa: F405
__all__.extend(photospheres.__all__)  # noqa: F405
__all__.extend(seds.__all__)
__all__.extend(transforms.__all__)  # noqa: F405


def __getattr__(name):
    """Resolve SED class names on first access without importing ``torch``."""
    if name in seds.__all__:
        return getattr(seds, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))

