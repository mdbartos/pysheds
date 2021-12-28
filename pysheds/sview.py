import numpy as np
from scipy import spatial
from scipy import interpolate
from numba import njit, prange
from numba.types import float64, UniTuple
import pyproj
from affine import Affine
from distutils.version import LooseVersion

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

class Raster(np.ndarray):
    def __new__(cls, input_array, viewfinder, metadata={}):
        obj = np.asarray(input_array).view(cls)
        try:
            assert(isinstance(viewfinder, ViewFinder))
        except:
            raise ValueError("Must initialize with a ViewFinder")
        obj.viewfinder = viewfinder
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.viewfinder = getattr(obj, 'viewfinder', None)
        self.metadata = getattr(obj, 'metadata', None)

    @property
    def bbox(self):
        return self.viewfinder.bbox
    @property
    def coords(self):
        return self.viewfinder.coords
    @property
    def view_shape(self):
        return self.viewfinder.shape
    @property
    def mask(self):
        return self.viewfinder.mask
    @property
    def nodata(self):
        return self.viewfinder.nodata
    @nodata.setter
    def nodata(self, new_nodata):
        self.viewfinder.nodata = new_nodata
    @property
    def crs(self):
        return self.viewfinder.crs
    @property
    def view_size(self):
        return np.prod(self.viewfinder.shape)
    @property
    def extent(self):
        bbox = self.viewfinder.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent
    @property
    def cellsize(self):
        dy, dx = self.dy_dx
        cellsize = (dy + dx) / 2
        return cellsize
    @property
    def affine(self):
        return self.viewfinder.affine
    @property
    def properties(self):
        property_dict = {
            'affine' : self.viewfinder.affine,
            'shape' : self.viewfinder.shape,
            'crs' : self.viewfinder.crs,
            'nodata' : self.viewfinder.nodata,
            'mask' : self.viewfinder.mask
        }
        return property_dict
    @property
    def dy_dx(self):
        return (-self.affine.e, self.affine.a)

class ViewFinder():
    def __init__(self, affine=(1., 0., 0., 0., -1., 0.), shape=(1,1),
                 mask=None, nodata=None, crs=pyproj.Proj(_pyproj_init)):
        self.affine = affine
        self.shape = shape
        self.crs = crs
        if nodata is None:
            self.nodata = np.nan
        else:
            self.nodata = nodata
        if mask is None:
            self.mask = np.ones(shape, dtype=np.bool8)
        else:
            self.mask = mask
        # TODO: Removed x_coord_ix and y_coord_ix---need to double-check

    def __eq__(self, other):
        if isinstance(other, ViewFinder):
            is_eq = True
            is_eq &= (self.affine == other.affine)
            is_eq &= (self.shape[0] == other.shape[0])
            is_eq &= (self.shape[1] == other.shape[1])
            is_eq &= (self.mask == other.mask).all()
            if np.isnan(self.nodata):
                is_eq &= np.isnan(other.nodata)
            else:
                is_eq &= self.nodata == other.nodata
            is_eq &= (self.crs == other.crs)
            return is_eq
        else:
            return False

    @property
    def affine(self):
        return self._affine
    @affine.setter
    def affine(self, new_affine):
        assert(isinstance(new_affine, Affine))
        self._affine = new_affine
    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, new_shape):
        self._shape = new_shape
    @property
    def mask(self):
        return self._mask
    @mask.setter
    def mask(self, new_mask):
        assert (new_mask.shape == self.shape)
        self._mask = new_mask
    @property
    def nodata(self):
        return self._nodata
    @nodata.setter
    def nodata(self, new_nodata):
        self._nodata = new_nodata
    @property
    def crs(self):
        return self._crs
    @crs.setter
    def crs(self, new_crs):
        assert (isinstance(new_crs, pyproj.Proj))
        self._crs = new_crs
    @property
    def size(self):
        return np.prod(self.shape)
    @property
    def bbox(self):
        shape = self.shape
        xmin, ymax = self.affine * (0,0)
        xmax, ymin = self.affine * (shape[1], shape[0])
        _bbox = (xmin, ymin, xmax, ymax)
        return _bbox
    @property
    def extent(self):
        bbox = self.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent
    @property
    def coords(self):
        coordinates = np.meshgrid(*self.grid_indices(), indexing='ij')
        return np.vstack(np.dstack(coordinates))
    @property
    def dy_dx(self):
        return (-self.affine.e, self.affine.a)
    @property
    def properties(self):
        property_dict = {
            'affine' : self.affine,
            'shape' : self.shape,
            'nodata' : self.nodata,
            'crs' : self.crs,
            'mask' : self.mask
        }
        return property_dict
    @property
    def axes(self):
        return self.grid_indices()

    def view(raster):
        data_view = raster.viewfinder
        target_view = self
        return View.view(raster, data_view, target_view, interpolation='nearest')

    def grid_indices(self, affine=None, shape=None, col_ascending=True, row_ascending=False):
        """
        Return row and column coordinates of a bounding box at a
        given cellsize.
 
        Parameters
        ----------
        shape : tuple of ints (length 2)
                The shape of the 2D array (rows, columns). Defaults
                to instance shape.
        precision : int
                    Precision to use when matching geographic coordinates.
        """
        if affine is None:
            affine = self.affine
        if shape is None:
            shape = self.shape
        y_ix = np.arange(shape[0])
        x_ix = np.arange(shape[1])
        if row_ascending:
            y_ix = y_ix[::-1]
        if not col_ascending:
            x_ix = x_ix[::-1]
        x, _ = affine * np.vstack([x_ix, np.zeros(shape[1])])
        _, y = affine * np.vstack([np.zeros(shape[0]), y_ix])
        return y, x

    def move_window(self, dxmin, dymin, dxmax, dymax):
        """
        Move bounding box window by integer indices
        """
        cell_height, cell_width  = self.dy_dx
        nrows_old, ncols_old = self.shape
        xmin_old, ymin_old, xmax_old, ymax_old = self.bbox
        new_bbox = (xmin_old + dxmin*cell_width, ymin_old + dymin*cell_height,
                    xmax_old + dxmax*cell_width, ymax_old + dymax*cell_height)
        new_shape = (nrows_old + dymax - dymin,
                     ncols_old + dxmax - dxmin)
        new_mask = np.ones(new_shape).astype(bool)
        mask_values = self._mask[max(dymin, 0):min(nrows_old + dymax, nrows_old),
                                 max(dxmin, 0):min(ncols_old + dxmax, ncols_old)]
        new_mask[max(0, dymax):max(0, dymax) + mask_values.shape[0],
                 max(0, -dxmin):max(0, -dxmin) + mask_values.shape[1]] = mask_values
        self.bbox = new_bbox
        self.shape = new_shape
        self.mask = new_mask


class View():
    def __init__(self):
        pass

    @classmethod
    def view(cls, data, data_view, target_view, interpolation='nearest',
             apply_input_mask=False, apply_output_mask=True,
             affine=None, shape=None, crs=None, mask=None, nodata=None,
             dtype=None, inherit_metadata=True, new_metadata={}):
        # Override parameters of target view if desired
        target_view = cls._override_target_view(target_view,
                                                affine=affine,
                                                shape=shape,
                                                crs=crs,
                                                mask=mask,
                                                nodata=nodata)
        # Resolve dtype of output Raster
        dtype = cls._override_dtype(data, target_view,
                                    dtype=dtype,
                                    interpolation=interpolation)
        # Mask input data if desired
        if apply_input_mask:
            arr = np.where(data_view.mask, data, target_view.nodata).astype(dtype)
            data = Raster(arr, data.viewfinder, metadata=data.metadata)
        # If data view and target view are the same, return a copy of the data
        if (data_view == target_view):
            out = cls._view_same_viewfinder(data, data_view, target_view, dtype,
                                            apply_output_mask=apply_output_mask)
        # If data view and target view are different...
        else:
            out = cls._view_different_viewfinder(data, data_view, target_view, dtype,
                                                 apply_output_mask=apply_output_mask)
        # Write metadata
        if inherit_metadata:
            out.metadata.update(data.metadata)
        out.metadata.update(new_metadata)
        return out

    @classmethod
    def _override_target_view(cls, target_view, **kwargs):
        new_view = ViewFinder(**target_view.properties)
        for param, value in kwargs.items():
            if (value is not None) and (hasattr(new_view, param)):
                setattr(new_view, param, value)
        return new_view

    @classmethod
    def _override_dtype(cls, data, target_view, dtype=None, interpolation='nearest'):
        if dtype is not None:
            return dtype
        if interpolation == 'nearest':
            # Find minimum type needed to represent nodata
            dtype = max(np.min_scalar_type(target_view.nodata), data.dtype)
            # For matplotlib imshow compatibility, upcast floats to float32
            if issubclass(dtype.type, np.floating):
                dtype = max(dtype, np.dtype(np.float32))
        elif interpolation == 'linear':
            dtype = np.float64
        else:
            raise ValueError('Interpolation method must be one of: `nearest`, `linear`')
        return dtype

    @classmethod
    def _view_same_viewfinder(cls, data, data_view, target_view, dtype,
                              apply_output_mask=True):
        if apply_output_mask:
            out = np.where(target_view.mask, data, target_view.nodata).astype(dtype)
        else:
            out = data.copy().astype(dtype)
        out = Raster(out, target_view)
        return out

    @classmethod
    def _view_different_viewfinder(cls, data, data_view, target_view, dtype,
                                   apply_output_mask=True):
        out = np.full(target_view.shape, target_view.nodata, dtype=dtype)
        if (data_view.crs == target_view.crs):
            out = cls._view_same_crs(out, data, data_view,
                                     target_view, interpolation)
        else:
            out = cls._view_different_crs(out, data, data_view,
                                          target_view, interpolation)
        # Apply mask
        if apply_output_mask:
            np.place(out, ~target_view.mask, target_view.nodata)
        out = Raster(out, target_view, metadata=metadata)
        return out

    @classmethod
    def _view_same_crs(cls, view, data, data_view, target_view, interpolation='nearest'):
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
    def _view_different_crs(cls, view, data, data_view, target_view, interpolation='nearest'):
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
