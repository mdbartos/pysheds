try:
    import numba
    _HAS_NUMBA = True
except ModuleNotFoundError:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from pysheds.sgrid import sGrid as Grid  # noqa: F401
else:
    from pysheds.pgrid import Grid as Grid  # noqa: F401
