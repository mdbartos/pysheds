import numpy as np
from scipy import spatial
from scipy import interpolate
import pyproj
from affine import Affine
from distutils.version import LooseVersion

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

class Raster(np.ndarray):
    def __new__(cls, input_array, viewfinder, metadata=None):
        obj = np.asarray(input_array).view(cls)
        try:
            assert(issubclass(type(viewfinder), BaseViewFinder))
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

class BaseViewFinder():
    def __init__(self, shape=None, mask=None, nodata=None,
                 crs=pyproj.Proj(_pyproj_init), y_coord_ix=0, x_coord_ix=1):
        if shape is not None:
            self.shape = shape
        else:
            self.shape = (0,0)
        self.crs = crs
        if nodata is None:
            self.nodata = np.nan
        else:
            self.nodata = nodata
        if mask is None:
            self.mask = np.ones(shape).astype(bool)
        else:
            self.mask = mask
        self.y_coord_ix = y_coord_ix
        self.x_coord_ix = x_coord_ix

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
        self._crs = new_crs
    @property
    def size(self):
        return np.prod(self.shape)

class RegularViewFinder(BaseViewFinder):
    def __init__(self, affine, shape, mask=None, nodata=None,
                 crs=pyproj.Proj(_pyproj_init),
                 y_coord_ix=0, x_coord_ix=1):
        if affine is not None:
            self.affine = affine
        else:
            self.affine = Affine(0,0,0,0,0,0)
        super().__init__(shape=shape, mask=mask, nodata=nodata, crs=crs,
                         y_coord_ix=y_coord_ix, x_coord_ix=x_coord_ix)

    @property
    def bbox(self):
        shape = self.shape
        xmin, ymax = self.affine * (0,0)
        # TODO: I think this is wrong; +1 not needed
        xmax, ymin = self.affine * (shape[1], shape[0])
        _bbox = (xmin, ymin, xmax, ymax)
        return _bbox

    @property
    def extent(self):
        bbox = self.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent

    @property
    def affine(self):
        return self._affine

    @affine.setter
    def affine(self, new_affine):
        assert(isinstance(new_affine, Affine))
        self._affine = new_affine

    @property
    def coords(self):
        coordinates = np.meshgrid(*self.grid_indices(), indexing='ij')
        return np.vstack(np.dstack(coordinates))

    @coords.setter
    def coords(self):
        pass

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

class IrregularViewFinder(BaseViewFinder):
    def __init__(self, coords, shape=None, mask=None, nodata=None,
                 crs=pyproj.Proj(_pyproj_init),
                 y_coord_ix=0, x_coord_ix=1):
        if coords is not None:
            self.coords = coords
        else:
            self.coords = np.asarray([0, 0]).reshape(1, 2)
        if shape is None:
            shape = len(coords)
        super().__init__(shape=shape, mask=mask, nodata=nodata, crs=crs,
                         y_coord_ix=y_coord_ix, x_coord_ix=x_coord_ix)
    @property
    def coords(self):
        return self._coords
    @coords.setter
    def coords(self, new_coords):
        self._coords = new_coords
    @property
    def bbox(self):
        ymin = self.coords[:, self.y_coord_ix].min()
        ymax = self.coords[:, self.y_coord_ix].max()
        xmin = self.coords[:, self.x_coord_ix].min()
        xmax = self.coords[:, self.x_coord_ix].max()
        return xmin, ymin, xmax, ymax
    @bbox.setter
    def bbox(self, new_bbox):
        pass
    @property
    def extent(self):
        bbox = self.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent

class RegularGridViewer():
    def __init__(self):
        pass

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
        view[np.ix_(y_passed, x_passed)] = data[y_ix[y_passed]][:, x_ix[x_passed]]
        return view

    @classmethod
    def _view_rectbivariate(cls, data, data_view, target_view, kx=3, ky=3, s=0,
                            x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        target_dx, target_dy = target_view.affine.a, target_view.affine.e
        data_dx, data_dy = data_view.affine.a, data_view.affine.e
        viewrows, viewcols = target_view.grid_indices(col_ascending=True,
                                                      row_ascending=True)
        rows, cols = data_view.grid_indices(col_ascending=True,
                                            row_ascending=True)
        viewrows += target_dy
        viewcols += target_dx
        rows += data_dy
        cols += data_dx
        row_bool = (rows <= t_ymax + y_tolerance) & (rows >= t_ymin - y_tolerance)
        col_bool = (cols <= t_xmax + x_tolerance) & (cols >= t_xmin - x_tolerance)
        rbs_interpolator = (interpolate.
                            RectBivariateSpline(rows[row_bool],
                                                cols[col_bool],
                                                data[np.ix_(row_bool[::-1], col_bool)],
                                                kx=kx, ky=ky, s=s))
        xy_query = np.vstack(np.dstack(np.meshgrid(viewrows, viewcols, indexing='ij')))
        view = rbs_interpolator.ev(xy_query[:,0], xy_query[:,1]).reshape(target_view.shape)
        return view

    @classmethod
    def _view_rectspherebivariate(cls, data, data_view, target_view, coords_in_radians=False,
                                  kx=3, ky=3, s=0, x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        target_dx, target_dy = target_view.affine.a, target_view.affine.e
        data_dx, data_dy = data_view.affine.a, data_view.affine.e
        viewrows, viewcols = target_view.grid_indices(col_ascending=True,
                                                      row_ascending=True)
        rows, cols = data_view.grid_indices(col_ascending=True,
                                            row_ascending=True)
        viewrows += target_dy
        viewcols += target_dx
        rows += data_dy
        cols += data_dx
        row_bool = (rows <= t_ymax + y_tolerance) & (rows >= t_ymin - y_tolerance)
        col_bool = (cols <= t_xmax + x_tolerance) & (cols >= t_xmin - x_tolerance)
        if not coords_in_radians:
            rows = np.radians(rows) + np.pi/2
            cols = np.radians(cols) + np.pi
            viewrows = np.radians(viewrows) + np.pi/2
            viewcols = np.radians(viewcols) + np.pi
        rsbs_interpolator = (interpolate.
                            RectBivariateSpline(rows[row_bool],
                                                cols[col_bool],
                                                data[np.ix_(row_bool[::-1], col_bool)],
                                                kx=kx, ky=ky, s=s))
        xy_query = np.vstack(np.dstack(np.meshgrid(viewrows, viewcols, indexing='ij')))
        view = rsbs_interpolator.ev(xy_query[:,0], xy_query[:,1]).reshape(target_view.shape)
        return view

class IrregularGridViewer():
    def __init__(self):
        pass

    @classmethod
    def _view_kd_2d(cls, data, data_view, target_view, x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        viewcoords = target_view.coords
        datacoords = data_view.coords
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        row_bool = ((datacoords[:,0] <= t_ymax + y_tolerance) &
                    (datacoords[:,0] >= t_ymin - y_tolerance))
        col_bool = ((datacoords[:,1] <= t_xmax + x_tolerance) &
                    (datacoords[:,1] >= t_xmin - x_tolerance))
        yx_tree = datacoords[row_bool & col_bool]
        tree = spatial.cKDTree(yx_tree)
        yx_dist, yx_ix = tree.query(viewcoords)
        yx_passed = yx_dist <= yx_tolerance
        view.flat[yx_passed] = data.flat[row_bool & col_bool].flat[yx_ix[yx_passed]]
        return view

    @classmethod
    def _view_griddata(cls, data, data_view, target_view, method='nearest',
                       x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        viewcoords = target_view.coords
        datacoords = data_view.coords
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        row_bool = ((datacoords[:,0] <= t_ymax + y_tolerance) &
                    (datacoords[:,0] >= t_ymin - y_tolerance))
        col_bool = ((datacoords[:,1] <= t_xmax + x_tolerance) &
                    (datacoords[:,1] >= t_xmin - x_tolerance))
        yx_grid = datacoords[row_bool & col_bool]
        view = interpolate.griddata(yx_grid,
                                    data.flat[row_bool & col_bool],
                                    viewcoords, method=method,
                                    fill_value=nodata)
        view = view.reshape(target_view.shape)
        return view
