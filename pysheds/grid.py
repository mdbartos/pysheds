try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from pysheds.sgrid import sGrid as Grid
else:
    from pysheds.pgrid import Grid as Grid
