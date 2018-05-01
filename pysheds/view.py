import numpy as np
from scipy import spatial
from scipy import interpolate
import pyproj

class ViewFinder():
    def __init__(self, bbox, shape, mask=None, nodata=None,
                 crs=pyproj.Proj('+init=epsg:4326'), **kwargs):
        self._bbox = bbox
        self._shape = shape
        self._crs = crs
        if nodata is None:
            self._nodata = np.nan
        else:
            self._nodata = nodata
        if mask is None:
            self._mask = np.ones(shape).astype(bool)
        else:
            self._mask = mask
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, new_bbox):
        self._bbox = new_bbox

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

    @property
    def extent(self):
        extent = (self._bbox[0], self._bbox[2], self._bbox[1], self._bbox[3])
        return extent

    @property
    def coords(self):
        coordinates = np.meshgrid(*self.bbox_indices(self.bbox, self.shape), indexing='ij')
        return np.vstack(np.dstack(coordinates))

    def move_window(self, dxmin, dymin, dxmax, dymax):
        """
        Move bounding box window by integer indices
        """
        cell_height, cell_width  = self._dy_dx()
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

    def bbox_indices(self, bbox=None, shape=None, precision=7, col_ascending=True,
                     row_ascending=False):
        """
        Return row and column coordinates of a bounding box at a
        given cellsize.

        Parameters
        ----------
        bbox : tuple of floats or ints (length 4)
               bbox of new data. Defaults to instance bbox.
        shape : tuple of ints (length 2)
                The shape of the 2D array (rows, columns). Defaults
                to instance shape.
        precision : int
                    Precision to use when matching geographic coordinates.
        """
        if bbox is None:
            bbox = self._bbox
        if shape is None:
            shape = self.shape
        rows = np.linspace(bbox[1], bbox[3], shape[0], endpoint=False)
        cols = np.linspace(bbox[0], bbox[2], shape[1], endpoint=False)
        if not row_ascending:
            rows = rows[::-1]
        if not col_ascending:
            cols = cols[::-1]
        rows = np.around(rows, precision)
        cols = np.around(cols, precision)
        return rows, cols

    def _dy_dx(self):
        x0, y0, x1, y1 = self.bbox
        dy = np.abs(y1 - y0) / (self.shape[0])
        dx = np.abs(x1 - x0) / (self.shape[1])
        return dy, dx

class IrregularViewFinder():
    def __init__(self, coords, shape=None, mask=None, nodata=None,
                 crs=pyproj.Proj('+init=epsg:4326'),
                 y_coord_ix=0, x_coord_ix=1, **kwargs):
        self._coords = coords
        if shape is None:
            self._shape = len(coords)
        else:
            self._shape = shape
        self._crs = crs
        if nodata is None:
            self._nodata = np.nan
        else:
            self._nodata = nodata
        if mask is None:
            self._mask = np.ones(shape).astype(bool)
        else:
            self._mask = mask
        self.y_coord_ix = y_coord_ix
        self.x_coord_ix = x_coord_ix
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, new_coords):
        self._coords = new_coords

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

    @property
    def extent(self):
        extent = (self._bbox[0], self._bbox[2], self._bbox[1], self._bbox[3])
        return extent

    @property
    def bbox(self):
        ymin = self.coords[:, self.y_coord_ix].min()
        ymax = self.coords[:, self.y_coord_ix].max()
        xmin = self.coords[:, self.x_coord_ix].min()
        xmax = self.coords[:, self.x_coord_ix].max()
        return xmin, ymin, xmax, ymax

class RegularGridViewer():
    def __init__(self):
        pass

    @classmethod
    def _view_df(cls, data, data_view, target_view, tolerance=0.1):
        nodata = target_view.nodata
        dy, dx = self._dy_dx()
        x_tolerance = dx * tolerance
        y_tolerance = dy * tolerance
        viewrows, viewcols = target_view.bbox_indices(target_view.bbox, target_view.shape)
        rows, cols = data_view.bbox_indices(data_view.bbox, data_view.shape)
        view = (pd.DataFrame(data, index=rows, columns=cols)
                .reindex(selfrows, tolerance=y_tolerance, method='nearest')
                .reindex(selfcols, axis=1, tolerance=x_tolerance,
                         method='nearest')
                .fillna(nodata).values)
        return view

    @classmethod
    def _view_kd(cls, data, data_view, target_view, tolerance=0.1):
        """
        Appropriate if:
            - Grid is regular
            - Data is regular
            - Grid and data have same cellsize OR no interpolation is needed
        """
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        dy, dx = target_view._dy_dx()
        x_tolerance = dx * tolerance
        y_tolerance = dy * tolerance
        viewrows, viewcols = target_view.bbox_indices(target_view.bbox, target_view.shape)
        rows, cols = data_view.bbox_indices(data_view.bbox, data_view.shape)
        ytree = spatial.cKDTree(rows[:, None])
        xtree = spatial.cKDTree(cols[:, None])
        ydist, y_ix = ytree.query(viewrows[:, None])
        xdist, x_ix = xtree.query(viewcols[:, None])
        y_passed = ydist < y_tolerance
        x_passed = xdist < x_tolerance
        view[np.ix_(y_passed, x_passed)] = data[y_ix[y_passed]][:, x_ix[x_passed]]
        return view

    @classmethod
    def _view_searchsorted(cls, data, data_view, target_view, tolerance=0.1):
        """
        Appropriate if:
            - Grid is regular
            - Data is regular
            - Grid and data have same cellsize OR no interpolation is needed
        """
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        dy, dx = target_view._dy_dx()
        x_tolerance = dx * tolerance
        y_tolerance = dy * tolerance
        viewrows, viewcols = target_view.bbox_indices(target_view.bbox,
                                                    target_view.shape,
                                                    col_ascending=True,
                                                    row_ascending=True)
        rows, cols = data_view.bbox_indices(data_view.bbox, data_view.shape,
                                            col_ascending=True,
                                            row_ascending=True)
        y_ix = np.searchsorted(rows, viewrows, side='right')
        x_ix = np.searchsorted(cols, viewcols, side='left')
        y_ix[y_ix > rows.size] = rows.size
        x_ix[x_ix >= cols.size] = cols.size - 1
        y_passed = np.abs(rows[y_ix - 1] - viewrows) < y_tolerance
        x_passed = np.abs(cols[x_ix] - viewcols) < x_tolerance
        y_ix = rows.size - y_ix[y_passed][::-1]
        x_ix = x_ix[x_passed]
        view[np.ix_(y_passed[::-1], x_passed)] = data[y_ix][:, x_ix]
        return view

    @classmethod
    def _view_kd_2d(cls, data, data_view, target_view, tolerance=0.1):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        dy, dx = target_view._dy_dx()
        x_tolerance = dx * tolerance
        y_tolerance = dy * tolerance
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        viewrows, viewcols = target_view.bbox_indices(target_view.bbox, target_view.shape)
        rows, cols = data_view.bbox_indices(data_view.bbox, data_view.shape)
        row_bool = (rows <= t_ymax + tolerance*dy) & (rows >= t_ymin - tolerance*dy)
        col_bool = (cols <= t_xmax + tolerance*dx) & (cols >= t_xmin - tolerance*dx)
        yx_tree = np.vstack(np.dstack(np.meshgrid(rows[row_bool], cols[col_bool], indexing='ij')))
        yx_query = np.vstack(np.dstack(np.meshgrid(viewrows, viewcols, indexing='ij')))
        tree = spatial.cKDTree(yx_tree)
        yx_dist, yx_ix = tree.query(yx_query)
        yx_passed = yx_dist < yx_tolerance
        view.flat[yx_passed] = data[np.ix_(row_bool, col_bool)].flat[yx_ix[yx_passed]]
        return view

    @classmethod
    def _view_rectbivariate(cls, data, data_view, target_view, kx=3, ky=3, s=0, tolerance=0.1):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        dy, dx = target_view._dy_dx()
        x_tolerance = dx * tolerance
        y_tolerance = dy * tolerance
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        viewrows, viewcols = target_view.bbox_indices(target_view.bbox,
                                                      target_view.shape,
                                                      col_ascending=True,
                                                      row_ascending=True)
        rows, cols = data_view.bbox_indices(data_view.bbox,
                                            data_view.shape,
                                            col_ascending=True,
                                            row_ascending=True)
        row_bool = (rows <= t_ymax + tolerance*dy) & (rows >= t_ymin - tolerance*dy)
        col_bool = (cols <= t_xmax + tolerance*dx) & (cols >= t_xmin - tolerance*dx)
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
                                  kx=3, ky=3, s=0, tolerance=0.1):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        dy, dx = target_view._dy_dx()
        x_tolerance = dx * tolerance
        y_tolerance = dy * tolerance
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        viewrows, viewcols = target_view.bbox_indices(target_view.bbox,
                                                      target_view.shape,
                                                      col_ascending=True,
                                                      row_ascending=True)
        rows, cols = data_view.bbox_indices(data_view.bbox,
                                            data_view.shape,
                                            col_ascending=True,
                                            row_ascending=True)
        row_bool = (rows <= t_ymax + tolerance*dy) & (rows >= t_ymin - tolerance*dy)
        col_bool = (cols <= t_xmax + tolerance*dx) & (cols >= t_xmin - tolerance*dx)
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
    def _view_kd_2d(cls, data, data_view, target_view, abs_tolerance=1e-5):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        viewcoords = target_view.coords
        datacoords = data_view.coords
        row_bool = ((datacoords[:,0] <= t_ymax + abs_tolerance) &
                    (datacoords[:,0] >= t_ymin - abs_tolerance))
        col_bool = ((datacoords[:,1] <= t_xmax + abs_tolerance) &
                    (datacoords[:,1] >= t_xmin - abs_tolerance))
        yx_tree = datacoords[row_bool & col_bool]
        tree = spatial.cKDTree(yx_tree)
        yx_dist, yx_ix = tree.query(viewcoords)
        yx_passed = yx_dist <= abs_tolerance
        view.flat[yx_passed] = data.flat[row_bool & col_bool].flat[yx_ix[yx_passed]]
        return view

    @classmethod
    def _view_griddata(cls, data, data_view, target_view, method='nearest',
                       abs_tolerance=1e-5):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        viewcoords = target_view.coords
        datacoords = data_view.coords
        row_bool = ((datacoords[:,0] <= t_ymax + abs_tolerance) &
                    (datacoords[:,0] >= t_ymin - abs_tolerance))
        col_bool = ((datacoords[:,1] <= t_xmax + abs_tolerance) &
                    (datacoords[:,1] >= t_xmin - abs_tolerance))
        yx_grid = datacoords[row_bool & col_bool]
        view = interpolate.griddata(yx_grid,
                                    data.flat[row_bool & col_bool],
                                    viewcoords, method=method,
                                    fill_value=nodata)
        view = view.reshape(target_view.shape)
        return view
