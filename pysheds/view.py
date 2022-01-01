try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from pysheds.sview import Raster, ViewFinder, View
else:
    from pysheds.pview import Raster, BaseViewFinder, RegularViewFinder, IrregularViewFinder
    from pysheds.pview import RegularGridViewer, IrregularGridViewer
