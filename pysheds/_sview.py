import numpy as np
from numba import njit, prange
from numba.types import float64, UniTuple

@njit(parallel=True)
def _view_fill_numba(data, out, y_ix, x_ix, y_passed, x_passed):
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
def _affine_map_vec_numba(affine, x, y):
    a, b, c, d, e, f, _, _, _ = affine
    n = x.size
    new_x = np.zeros(n, dtype=np.float64)
    new_y = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        new_x[i] = x[i] * a + y[i] * b + c
        new_y[i] = x[i] * d + y[i] * e + f
    return new_x, new_y

@njit(UniTuple(float64, 2)(UniTuple(float64, 9), float64, float64))
def _affine_map_scalar_numba(affine, x, y):
    a, b, c, d, e, f, _, _, _ = affine
    new_x = x * a + y * b + c
    new_y = x * d + y * e + f
    return new_x, new_y
