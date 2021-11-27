import numpy as np
from scipy import spatial
from scipy import interpolate
from numba import njit, prange
import pyproj
from affine import Affine
from distutils.version import LooseVersion
from pysheds.view import Raster, BaseViewFinder
from pysheds.view import RegularViewFinder, IrregularViewFinder
from pysheds.view import RegularGridViewer, IrregularGridViewer

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

class sRegularGridViewer(RegularGridViewer):
    def __init__(self):
        super().__init__()

    @classmethod
    def _view_affine(cls, data, data_view, target_view, x_tolerance=1e-3, y_tolerance=1e-3):
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata, dtype=data.dtype)
        viewrows, viewcols = target_view.grid_indices()
        _, target_row_ix = ~data_view.affine * np.vstack([np.zeros(target_view.shape[0]), viewrows])
        target_col_ix, _ = ~data_view.affine * np.vstack([viewcols, np.zeros(target_view.shape[1])])
        y_ix = np.around(target_row_ix).astype(int)
        x_ix = np.around(target_col_ix).astype(int)
        y_passed = ((np.abs(y_ix - target_row_ix) < y_tolerance)
                    & (y_ix < data_view.shape[0]) & (y_ix >= 0))
        x_passed = ((np.abs(x_ix - target_col_ix) < x_tolerance)
                    & (x_ix < data_view.shape[1]) & (x_ix >= 0))
        view = _view_fill_numba(data, view, y_ix, x_ix, y_passed, x_passed)
        return view

@njit(parallel=True)
def _view_fill_numba(data, out, y_ix, x_ix, y_passed, x_passed):
    n = x_ix.size
    m = y_ix.size
    for i in prange(m):
        for j in prange(n):
            if (y_passed[i]) & (x_passed[j]):
                out[i, j] = data[y_ix[i], x_ix[j]]
    return out
