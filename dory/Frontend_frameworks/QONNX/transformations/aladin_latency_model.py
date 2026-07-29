try:
    from .aladin_latency import *
    from .aladin_latency import __all__
except ImportError:  # Allows direct use when this folder is on PYTHONPATH.
    from aladin_latency import *
    from aladin_latency import __all__