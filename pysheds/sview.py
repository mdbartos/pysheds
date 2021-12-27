import numpy as np
from scipy import spatial
from scipy import interpolate
from numba import njit, prange
from numba.types import float64, UniTuple
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

    @classmethod
    def _view_same_crs(cls, data, data_view, target_view, interpolation='nearest'):
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata, dtype=data.dtype)
        y, x = target_view.axes
        inv_affine = tuple(~data_view.affine)
        _, y_ix = affine_map(inv_affine,
                            np.zeros(target_view.shape[0], dtype=np.float64),
                            y)
        x_ix, _ = affine_map(inv_affine,
                            x,
                            np.zeros(target_view.shape[1], dtype=np.float64))
        if interpolation == 'nearest':
            view = _view_fill_by_axes_nearest_numba(data, view, y_ix, x_ix)
        elif interpolation == 'linear':
            view = _view_fill_by_axes_linear_numba(data, view, y_ix, x_ix)
        else:
            raise ValueError('Interpolation method must be one of: `nearest`, `linear`')
        return view

    @classmethod
    def _view_different_crs(cls, data, data_view, target_view, interpolation='nearest'):
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata, dtype=data.dtype)
        y, x = target_view.coords.T
        xt, yt = pyproj.transform(target_view.crs, data_view.crs, x=x, y=y,
                                  errcheck=True, always_xy=True)
        inv_affine = tuple(~data_view.affine)
        x_ix, y_ix = affine_map(inv_affine, xt, yt)
        if interpolation == 'nearest':
            view = _view_fill_by_entries_nearest_numba(data, view, y_ix, x_ix)
        elif interpolation == 'linear':
            view = _view_fill_by_entries_linear_numba(data, view, y_ix, x_ix)
        else:
            raise ValueError('Interpolation method must be one of: `nearest`, `linear`')
        return view

@njit(parallel=True)
def _view_fill_numba(data, out, y_ix, x_ix, y_passed, x_passed):
    # TODO: This is probably inefficient---don't need to iterate over everything
    n = x_ix.size
    m = y_ix.size
    for i in prange(m):
        for j in prange(n):
            if (y_passed[i]) & (x_passed[j]):
                out[i, j] = data[y_ix[i], x_ix[j]]
    return out

@njit(parallel=True)
def _view_fill_by_axes_nearest_numba(data, out, y_ix, x_ix):
    m, n = y_ix.size, x_ix.size
    M, N = data.shape
    # Currently need to use inplace form of round
    y_near = np.empty(m, dtype=np.int64)
    x_near = np.empty(n, dtype=np.int64)
    np.around(y_ix, 0, y_near).astype(np.int64)
    np.around(x_ix, 0, x_near).astype(np.int64)
    y_in_bounds = ((y_near >= 0) & (y_near < M))
    x_in_bounds = ((x_near >= 0) & (x_near < N))
    for i in prange(m):
        for j in prange(n):
            if (y_in_bounds[i]) & (x_in_bounds[j]):
                out[i, j] = data[y_near[i], x_near[j]]
    return out

@njit(parallel=True)
def _view_fill_by_axes_linear_numba(data, out, y_ix, x_ix):
    m, n = y_ix.size, x_ix.size
    M, N = data.shape
    # Find which cells are in bounds
    y_in_bounds = ((y_ix >= 0) & (y_ix < M))
    x_in_bounds = ((x_ix >= 0) & (x_ix < N))
    # Compute upper and lower values of y and x
    y_floor = np.floor(y_ix).astype(np.int64)
    y_ceil = y_floor + 1
    x_floor = np.floor(x_ix).astype(np.int64)
    x_ceil = x_floor + 1
    # Compute fractional distance between adjacent cells
    ty = (y_ix - y_floor)
    tx = (x_ix - x_floor)
    # Handle lower and right boundaries
    lower_boundary = (y_ceil == M)
    right_boundary = (x_ceil == N)
    y_ceil[lower_boundary] = y_floor[lower_boundary]
    x_ceil[right_boundary] = x_floor[right_boundary]
    ty[lower_boundary] = 0.
    tx[right_boundary] = 0.
    for i in prange(m):
        for j in prange(n):
            if (y_in_bounds[i]) & (x_in_bounds[j]):
                ul = data[y_floor[i], x_floor[j]]
                ur = data[y_floor[i], x_ceil[j]]
                ll = data[y_ceil[i], x_floor[j]]
                lr = data[y_ceil[i], x_ceil[j]]
                value = ( ( ( 1 - tx[j] ) * ( 1 - ty[i] ) * ul )
                         + ( tx[j] * ( 1 - ty[i] ) * ur )
                         + ( ( 1 - tx[j] ) * ty[i] * ll )
                         + ( tx[j] * ty[i] * lr ) )
                out[i, j] = value
    return out

@njit(parallel=True)
def _view_fill_by_entries_nearest_numba(data, out, y_ix, x_ix):
    m, n = y_ix.size, x_ix.size
    M, N = data.shape
    # Currently need to use inplace form of round
    y_near = np.empty(m, dtype=np.int64)
    x_near = np.empty(n, dtype=np.int64)
    np.around(y_ix, 0, y_near).astype(np.int64)
    np.around(x_ix, 0, x_near).astype(np.int64)
    y_in_bounds = ((y_near >= 0) & (y_near < M))
    x_in_bounds = ((x_near >= 0) & (x_near < N))
    # x and y indices should be the same size
    assert(n == m)
    for i in prange(n):
        if (y_in_bounds[i]) & (x_in_bounds[i]):
            out.flat[i] = data[y_near[i], x_near[i]]
    return out

@njit(parallel=True)
def _view_fill_by_entries_linear_numba(data, out, y_ix, x_ix):
    m, n = y_ix.size, x_ix.size
    M, N = data.shape
    # Find which cells are in bounds
    y_in_bounds = ((y_ix >= 0) & (y_ix < M))
    x_in_bounds = ((x_ix >= 0) & (x_ix < N))
    # Compute upper and lower values of y and x
    y_floor = np.floor(y_ix).astype(np.int64)
    y_ceil = y_floor + 1
    x_floor = np.floor(x_ix).astype(np.int64)
    x_ceil = x_floor + 1
    # Compute fractional distance between adjacent cells
    ty = (y_ix - y_floor)
    tx = (x_ix - x_floor)
    # Handle lower and right boundaries
    lower_boundary = (y_ceil == M)
    right_boundary = (x_ceil == N)
    y_ceil[lower_boundary] = y_floor[lower_boundary]
    x_ceil[right_boundary] = x_floor[right_boundary]
    ty[lower_boundary] = 0.
    tx[right_boundary] = 0.
    # x and y indices should be the same size
    assert(n == m)
    for i in prange(n):
        if (y_in_bounds[i]) & (x_in_bounds[i]):
            ul = data[y_floor[i], x_floor[i]]
            ur = data[y_floor[i], x_ceil[i]]
            ll = data[y_ceil[i], x_floor[i]]
            lr = data[y_ceil[i], x_ceil[i]]
            value = ( ( ( 1 - tx[i] ) * ( 1 - ty[i] ) * ul )
                    + ( tx[i] * ( 1 - ty[i] ) * ur )
                    + ( ( 1 - tx[i] ) * ty[i] * ll )
                    + ( tx[i] * ty[i] * lr ) )
            out.flat[i] = value
    return out

@njit(UniTuple(float64[:], 2)(UniTuple(float64, 9), float64[:], float64[:]), parallel=True)
def affine_map(affine, x, y):
    a, b, c, d, e, f, _, _, _ = affine
    n = x.size
    new_x = np.zeros(n, dtype=np.float64)
    new_y = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        new_x[i] = x[i] * a + y[i] * b + c
        new_y[i] = x[i] * d + y[i] * e + f
    return new_x, new_y
