import numpy as np

def _vectorize(signature=None, included=None):
    """
    Stub for mas.decorators._vectorize.
    Provides numpy vectorization behavior compatible with the slitless codebase usage.
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(**kwargs):
            result = func(**kwargs)
            return np.asarray(result)
        return wrapper
    return decorator
