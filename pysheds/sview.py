import copy
import numpy as np
import pyproj
from affine import Affine
from distutils.version import LooseVersion
try:
    import scipy.spatial
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False

import pysheds._sview as _self

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

class Raster(np.ndarray):
    """
    Array-like data structure with a coordinate reference system. A Raster is instantiated
    from an array-like object and a ViewFinder. Optional metadata may also be provided
    as a keyword argument.

    Attributes
    ==========
    viewfinder : Class containing all information about the coordinate system
                 of the Raster object. Includes the `affine`, `shape`, `crs`,
                 `nodata` and `mask` attributes.
    affine : Affine transformation matrix (uses affine module).
    shape : The shape of the raster (number of rows, number of columns).
    crs : The coordinate reference system.
    nodata : The value indicating `no data`.
    mask : A boolean array used to mask raster cells; may be used to indicate
           which cells lie inside a catchment.
    metadata : A dictionary containing optional metadata about the Raster.
    bbox : The bounding box of the raster (xmin, ymin, xmax, ymax).
    extent : The extent of the raster (xmin, xmax, ymin, ymax).
    size : The number of cells in the raster.
    coords : An (N, 2) array indicating the coordinates of the top-left corner
             of each cell in the Raster. Coordinates of cells are list in C order.
    properties : A dict containing the names and values of the essential properties
                 that define the coordinate reference system, including `affine`,
                 `shape`, `mask`, `crs`, and `nodata`.
    dy_dx : Tuple describing the cell size in the y and x directions.

    Methods
    =======
    to_crs : Transforms the Raster to a new coordinate reference system defined
             by a pyproj.Proj object.
    """

    def __new__(cls, input_array, viewfinder=None, metadata={}):
        # Handle case where input is a Raster itself
        if isinstance(input_array, Raster):
            input_array, viewfinder, metadata = cls._handle_raster_input(input_array,
                                                                         viewfinder,
                                                                         metadata)
        # Create a numpy array from the input
        obj = np.asarray(input_array).view(cls)
        # If no viewfinder provided, construct one congruent with the array shape
        if viewfinder is None:
            viewfinder = ViewFinder(shape=obj.shape)
        # If a viewfinder is provided, ensure that it is a viewfinder...
        else:
            try:
                assert(isinstance(viewfinder, ViewFinder))
            except:
                raise ValueError("Must initialize with a ViewFinder.")
            # Ensure that viewfinder shape is correct...
            try:
                assert viewfinder.shape == obj.shape
            except:
                raise ValueError('Viewfinder and array shape must be the same.')
        # Test typing of array
        try:
            assert not np.issubdtype(obj.dtype, np.object_)
            assert not np.issubdtype(obj.dtype, np.flexible)
        except:
            raise TypeError('`object` and `flexible` dtypes not allowed.')
        try:
            assert np.min_scalar_type(viewfinder.nodata) <= obj.dtype
        except:
            raise TypeError('`nodata` value not representable in dtype of array.')
        # Don't allow original viewfinder and metadata to be modified
        viewfinder = viewfinder.copy()
        metadata = metadata.copy()
        # Set attributes of array
        obj._viewfinder = viewfinder
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._viewfinder = getattr(obj, 'viewfinder', None)
        self.metadata = getattr(obj, 'metadata', None)

    @classmethod
    def _handle_raster_input(cls, input_array, viewfinder, metadata):
        if not metadata:
            metadata = input_array.metadata
        # If no viewfinder provided, use viewfinder of input raster
        if viewfinder is None:
            viewfinder = input_array.viewfinder
        # Otherwise, given viewfinder overrides and returns a view
        else:
            if viewfinder != input_array.viewfinder:
                input_array = View.view(data=input_array,
                                        target_view=viewfinder,
                                        apply_input_mask=False,
                                        apply_output_mask=False,
                                        inherit_metadata=False,
                                        new_metadata=metadata)
        return input_array, viewfinder, metadata

    @property
    def viewfinder(self):
        return self._viewfinder

    @viewfinder.setter
    def viewfinder(self, new_viewfinder):
        try:
            assert(isinstance(new_viewfinder, ViewFinder))
        except:
            raise ValueError("Must be a `ViewFinder` object")
        try:
            assert new_viewfinder.shape == self.shape
        except:
            raise ValueError('viewfinder and raster array must have the same shape.')
        self._viewfinder = new_viewfinder

    @property
    def affine(self):
        return self.viewfinder.affine

    @affine.setter
    def affine(self, new_affine):
        self.viewfinder.affine = new_affine

    @property
    def mask(self):
        return self.viewfinder.mask

    @mask.setter
    def mask(self, new_mask):
        try:
            assert (new_mask.shape == self.shape)
        except:
            raise ValueError('`mask` must have same dimensions as Raster.')
        self.viewfinder.mask = new_mask

    @property
    def nodata(self):
        return self.viewfinder.nodata

    @nodata.setter
    def nodata(self, new_nodata):
        self.viewfinder.nodata = new_nodata

    @property
    def crs(self):
        return self.viewfinder.crs

    @crs.setter
    def crs(self, new_crs):
        self.viewfinder.crs = new_crs

    @property
    def bbox(self):
        return self.viewfinder.bbox

    @property
    def coords(self):
        return self.viewfinder.coords

    @property
    def axes(self):
        return self.viewfinder.axes

    @property
    def extent(self):
        bbox = self.viewfinder.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent

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
        return (abs(self.affine.e), abs(self.affine.a))

    def to_crs(self, new_crs, **kwargs):
        """
        Transforms and resamples the Raster in a new coordinate reference system.
        A new ViewFinder is generated such that all points in the old Raster are
        contained within the new transformed Raster.

        Parameters
        ----------
        new_crs : pyproj.Proj
                  New coordinate reference system.

        Additional keyword arguments (**kwargs) are passed to View.view.

        Returns
        -------
        new_raster : Raster
                     Raster transformed to the new coordinate reference system
        """
        old_crs = self.crs
        dx = self.affine.a
        dy = self.affine.e
        m, n = self.shape
        Y, X = np.mgrid[0:m, 0:n]
        top = np.column_stack([X[0, :], Y[0, :]])
        bottom = np.column_stack([X[-1, :], Y[-1, :]])
        left = np.column_stack([X[:, 0], Y[:, 0]])
        right = np.column_stack([X[:, -1], Y[:, -1]])
        boundary = np.vstack([top, bottom, left, right])
        xi, yi = boundary[:,0], boundary[:,1]
        xb, yb = View.affine_transform(self.affine, xi, yi)
        xb_p, yb_p = pyproj.transform(old_crs, new_crs, xb, yb,
                                    errcheck=True, always_xy=True)
        x0_p = xb_p.min() if (dx > 0) else xb_p.max()
        y0_p = yb_p.min() if (dy > 0) else yb_p.max()
        xn_p = xb_p.max() if (dx > 0) else xb_p.min()
        yn_p = yb_p.max() if (dy > 0) else yb_p.min()
        a = (xn_p - x0_p) / n
        e = (yn_p - y0_p) / m
        new_affine = Affine(a, 0., x0_p, 0., e, y0_p)
        new_viewfinder = ViewFinder(affine=new_affine, shape=self.shape,
                                    nodata=self.nodata, mask=self.mask,
                                    crs=new_crs)
        new_raster = View.view(self, target_view=new_viewfinder,
                            data_view=self.viewfinder, **kwargs)
        return new_raster

class ViewFinder():
    """
    Class that defines a spatial reference system for a Raster or Grid instance.
    The spatial reference is completely defined by an affine transformation matrix (affine),
    a desired shape (shape), a coordinate reference system (crs), a boolean mask (mask),
    and a sentinel value indicating `no data` (nodata).

    Attributes
    ==========
    affine : Affine transformation matrix (uses affine module).
    shape : The shape of the raster (number of rows, number of columns). Note: if
            the shape of the provided `mask` differs, the `mask` shape will be used.
    crs : The coordinate reference system.
    nodata : The value indicating `no data`.
    mask : A boolean array used to mask raster cells; may be used to indicate
           which cells lie inside a catchment. Note: if the shape of the provided
           `mask` is different from `shape`, the shape of `mask` will be used.
    bbox : The bounding box of the raster (xmin, ymin, xmax, ymax).
    extent : The extent of the raster (xmin, xmax, ymin, ymax).
    size : The number of cells in the raster.
    coords : An (N, 2) array indicating the coordinates of the top-left corner
             of each cell in the Raster. Coordinates of cells are list in C order.
    axes : Tuple of arrays indicating the y and x axes (i.e. the coordinates of the
           top-left corners of the leftmost and upper edges of the dataset, respectively).
    properties : A dict containing the names and values of the essential properties
                 that define the coordinate reference system, including `affine`,
                 `shape`, `mask`, `crs`, and `nodata`.
    dy_dx : Tuple describing the cell size in the y and x directions.
    """
    def __init__(self, affine=Affine(1., 0., 0., 0., 1., 0.), shape=(1,1),
                 nodata=0, mask=None, crs=pyproj.Proj(_pyproj_init)):
        self.affine = affine
        self.crs = crs
        self.nodata = nodata
        if mask is None:
            self.mask = np.ones(shape, dtype=np.bool8)
        else:
            self.mask = mask

    def __eq__(self, other):
        if isinstance(other, ViewFinder):
            is_eq = True
            is_eq &= (self.affine == other.affine)
            is_eq &= (self.shape[0] == other.shape[0])
            is_eq &= (self.shape[1] == other.shape[1])
            is_eq &= (self.crs == other.crs)
            is_eq &= (self.mask == other.mask).all()
            if np.isnan(self.nodata):
                is_eq &= np.isnan(other.nodata)
            else:
                is_eq &= self.nodata == other.nodata
            return is_eq
        else:
            return False

    def __repr__(self):
        repr_str = '\n'.join([repr(k) + ' : ' + repr(v)
                              for k, v in self.properties.items()])
        return repr_str

    @property
    def affine(self):
        return self._affine

    @affine.setter
    def affine(self, new_affine):
        try:
            assert(isinstance(new_affine, Affine))
        except:
            raise TypeError('Affine transformation must be an `Affine` object')
        self._affine = new_affine

    @property
    def shape(self):
        return self.mask.shape

    @shape.setter
    def shape(self, new_shape):
        raise AttributeError('Viewfinder shape cannot be changed after instantiation.')

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, new_mask):
        try:
            assert (np.min_scalar_type(new_mask) <= np.dtype(np.bool8))
        except:
            raise TypeError('`mask` must be of boolean type')
        new_mask = np.asarray(new_mask).astype(np.bool8)
        self._mask = new_mask

    @property
    def nodata(self):
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata):
        try:
            assert not (np.min_scalar_type(new_nodata) == np.dtype('O'))
        except:
            raise TypeError('`nodata` value must be a numeric type.')
        self._nodata = new_nodata

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, new_crs):
        try:
            assert (isinstance(new_crs, pyproj.Proj))
        except:
            raise TypeError('`crs` must be a `pyproj.Proj` object.')
        self._crs = new_crs

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def bbox(self):
        shape = self.shape
        xmin, ymax = View.affine_transform(self.affine, 0, 0)
        xmax, ymin = View.affine_transform(self.affine, shape[1], shape[0])
        _bbox = (xmin, ymin, xmax, ymax)
        return _bbox

    @property
    def extent(self):
        bbox = self.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent

    @property
    def coords(self):
        coordinates = np.meshgrid(*self.axes, indexing='ij')
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
        return View.axes(self.affine, self.shape)

    def is_congruent_with(self, other):
        if isinstance(other, ViewFinder):
            is_congruent = True
            is_congruent &= (self.affine == other.affine)
            is_congruent &= (self.shape[0] == other.shape[0])
            is_congruent &= (self.shape[1] == other.shape[1])
            is_congruent &= (self.crs == other.crs)
            return is_congruent
        else:
            return False

    def copy(self):
        new_view = copy.deepcopy(self)
        return new_view

    def view(self, raster, **kwargs):
        data_view = raster.viewfinder
        target_view = self
        return View.view(raster, data_view, target_view, **kwargs)

class View():
    """
    Class containing methods for manipulating views of gridded datasets.

    Methods
    ==========
    view : View a Raster in a different spatial reference system.
    affine_transform : Apply an affine transformation to a point or set of points.
    nearest_cell : Find the nearest cell to a set of x, y coordinates.
    trim_zeros : Clip a raster to the bounding box defined by its non-null values.
    clip_to_mask : Clip a raster to a pre-defined Raster mask.
    """

    def __init__(self):
        raise NotImplementedError('The View class is used for classmethods '
                                  'and is not meant to be instantiated.')

    @classmethod
    def view(cls, data, target_view, data_view=None, interpolation='nearest',
             apply_input_mask=False, apply_output_mask=True, inherit_nodata=True,
             affine=None, shape=None, crs=None, mask=None, nodata=None,
             dtype=None, inherit_metadata=True, new_metadata={}):
        """
        Return a copy of a gridded dataset `data` transformed to the spatial reference
        system defined by `target_view`.

        Parameters
        ----------
        data : Raster
               A Raster object containing the gridded data and its spatial reference system
               (as defined by its ViewFinder).
        target_view : ViewFinder
                      The desired spatial reference system.
        data_view : ViewFinder
                    The spatial reference system of the data. Defaults to the Raster dataset's
                    `viewfinder` attribute.
        interpolation : 'nearest', 'linear'
                        Interpolation method to be used if spatial reference systems
                        are not congruent.
        apply_input_mask : bool
                           If True, mask the input Raster according to data.mask.
        apply_output_mask : bool
                           If True, mask the output Raster according to grid.mask.
        inherit_nodata : bool
                         If True, output Raster inherits `nodata` value from `data_view`.
                         If False, output Raster uses `nodata` value from `target_view`.
        affine : affine.Affine
                 Affine transformation matrix (overrides target_view.affine)
        shape : tuple of ints (length 2)
                Shape of desired Raster (overrides target_view.shape)
        crs : pyproj.Proj
              Coordinate reference system (overrides target_view.crs)
        mask : np.ndarray or Raster
               Boolean array to mask output (overrides target_view.mask)
        nodata : int or float
                 Value indicating no data in output Raster (overrides target_view.nodata)
        dtype : numpy datatype
                Desired datatype of the output array.
        inherit_metadata : bool
                           If True, output Raster inherits metadata from input data.
        new_metadata : dict
                       Optional metadata to add to output Raster.

        Returns
        -------
        out : Raster
              View of the input Raster at the provided target view.
        """
        # If no data view given, use data's view
        if data_view is None:
            try:
                assert(isinstance(data, Raster))
            except:
                raise TypeError('`data` must be a Raster instance.')
            data_view = data.viewfinder
        # By default, use dataset's `nodata` value
        if nodata is None:
            if inherit_nodata:
                nodata = data_view.nodata
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
        if data_view.is_congruent_with(target_view):
            out = cls._view_same_viewfinder(data, data_view, target_view, dtype,
                                            apply_output_mask=apply_output_mask)
        # If data view and target view are different...
        else:
            out = cls._view_different_viewfinder(data, data_view, target_view, dtype,
                                                 apply_output_mask=apply_output_mask,
                                                 interpolation=interpolation)
        # Write metadata
        if inherit_metadata:
            out.metadata.update(data.metadata)
        out.metadata.update(new_metadata)
        return out

    @classmethod
    def affine_transform(cls, affine, x, y):
        """
        Basic affine transformation of a point (x, y) or set of points (x, y).

        Parameters
        ----------
        affine : affine.Affine
                 An affine transformation.
        x : float or np.ndarray
            An x-coordinate or array of x-coordinates.
        y : float or np.ndarray
            A y-coordinate or array of y-coordinates.

        Returns
        -------
        x_t, y_t : tuple
                   A set of transformed x and y coordinates
        """
        # Check affine input type
        try:
            assert isinstance(affine, Affine)
            affine = tuple(affine)
        except:
            raise TypeError('`affine` must be an Affine instance')
        # Vector case
        if hasattr(x, '__len__'):
            if hasattr(y, '__len__'):
                x = np.asarray(x).astype(np.float64)
                y = np.asarray(y).astype(np.float64)
                x_t, y_t = _self._affine_map_vec_numba(affine, x, y)
            else:
                raise TypeError('If `x` is a sequence, `y` must also be a sequence')
        # Scalar case
        else:
            x = float(x)
            y = float(y)
            x_t, y_t = _self._affine_map_scalar_numba(affine, x, y)
        return x_t, y_t

    @classmethod
    def nearest_cell(cls, x, y, affine, snap='corner'):
        """
        Returns the index of the cell (column, row) closest
        to a given geographical coordinate.

        Parameters
        ----------
        x : int or float
            x coordinate.
        y : int or float
            y coordinate.
        affine : affine.Affine
                 Affine transformation that defines the translation between
                 geographic x/y coordinate and array row/column coordinate.
        snap : str
               Indicates the cell indexing method. If "corner", will resolve to
               snapping the (x,y) geometry to the index of the nearest top-left
               cell corner. If "center", will return the index of the cell that
               the geometry falls within.

        Returns
        -------
        col, row : tuple of ints
                   Column index and row index
        """
        try:
            assert isinstance(affine, Affine)
        except:
            raise TypeError('affine must be an Affine instance.')
        snap_dict = {'corner': np.around, 'center': np.floor}
        xi, yi = cls.affine_transform(~affine, x, y)
        col, row = snap_dict[snap]((xi, yi)).astype(int)
        return col, row

    @classmethod
    def axes(cls, affine, shape):
        """
        Return row and column coordinates of axes, such that the cartesian product
        of the two axis vectors uniquely addresses each grid cell.

        Parameters
        ----------
        affine : affine.Affine
                 Affine transformation
        shape : tuple of ints (length 2)
                The shape of the 2D array (rows, columns).

        Returns
        -------
        y, x : tuple
               y- and x-coordinates of axes
        """
        y_ix = np.arange(shape[0])
        x_ix = np.arange(shape[1])
        x_null = np.zeros(shape[0])
        y_null = np.zeros(shape[1])
        x, _ = cls.affine_transform(affine, x_ix, y_null)
        _, y = cls.affine_transform(affine, x_null, y_ix)
        return y, x

    @classmethod
    def snap_to_mask(cls, mask, xy, affine=None, return_dist=False, **kwargs):
        """
        Snap a set of coordinates (given by `xy`) to the nearest nonzero cells in a
        boolean raster (given by `mask`). (Note that the mask raster is first mapped to the
        grid's ViewFinder using self.view).

        Parameters
        ----------
        mask : Raster or np.ndarray
               A Raster or array dataset with nonzero elements indicating cells to match to (e.g:
               a flow accumulation grid with ones indicating cells above a certain threshold).
        xy : np.ndarray-like with shape (N, 2)
             Points to match (example: gage location coordinates).
        affine : affine.Affine
                 Affine transformation. If None given, defaults to `mask.affine`
                 if mask is a Raster.
        return_dist : If true, return the distances from xy to the nearest matched point in mask.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        xy_new : np.ndarray with shape (N, 2)
                 Coordinates of nearest points where mask is nonzero.
        dist : np.ndarray with shape (N,), (optional)
               Distances from points in xy to xy_new
        """
        if not _HAS_SCIPY:
            raise ImportError('Requires scipy.spatial module')
        if affine is None:
            try:
                assert isinstance(mask, Raster)
            except:
                raise TypeError('If no affine transform given, mask must be a raster')
            affine = mask.affine
        yi, xi = np.where(mask)
        x, y = cls.affine_transform(affine, xi, yi)
        tree_xy = np.column_stack([x, y])
        tree = scipy.spatial.cKDTree(tree_xy)
        dist, ix = tree.query(xy)
        if return_dist:
            return tree_xy[ix], dist
        else:
            return tree_xy[ix]

    @classmethod
    def trim_zeros(cls, data, pad=(0,0,0,0)):
        """
        Clip a Raster to the smallest area that contains all non-null data.

        Parameters
        ----------
        data : Raster
               A Raster dataset.
        pad : tuple of int (length 4)
              Apply padding to edges of new view (left, bottom, right, top). A pad of
              (1,1,1,1), for instance, will add a one-cell rim around the new view.

        Returns
        -------
        out : Raster
              A Raster dataset clipped to the bounding box of its non-null values.
        """
        try:
            for value in pad:
                assert (isinstance(value, int))
                assert (value >= 0)
        except:
            raise ValueError('Pad values must be non-negative integers')
        try:
            assert isinstance(data, Raster)
        except:
            raise TypeError('`data` must be a Raster instance.')
        if np.isnan(data.nodata):
            mask = (~np.isnan(data))
        else:
            mask = (data != data.nodata)
        return cls.clip_to_mask(data, mask=mask, pad=pad)

    @classmethod
    def clip_to_mask(cls, data, mask=None, pad=(0,0,0,0)):
        """
        Clip a Raster to the smallest area that contains all nonzero entries for a
        given boolean mask.

        Parameters
        ----------
        data : Raster
               A Raster dataset.
        mask : Raster
               A Raster dataset representing a boolean mask. Defaults to data.mask.
        pad : tuple of int (length 4)
              Apply padding to edges of new view (left, bottom, right, top). A pad of
              (1,1,1,1), for instance, will add a one-cell rim around the new view.

        Returns
        -------
        out : Raster
              A Raster dataset clipped to the bounding box of the non-null entries
              in the given mask.
        """
        try:
            for value in pad:
                assert (isinstance(value, int))
                assert (value >= 0)
        except:
            raise ValueError('Pad values must be non-negative integers')
        try:
            assert isinstance(data, Raster)
        except:
            raise TypeError('`data` must be a Raster instance.')
        if mask is None:
            mask = data.mask
        else:
            try:
                assert (data.shape == mask.shape)
            except:
                raise ValueError('Shape of `data` and `mask` must be the same')
        vert_pad = (pad[3], pad[1])
        horiz_pad = (pad[0], pad[2])
        nz_r, nz_c = np.nonzero(mask)
        yi_min = nz_r.min()
        yi_max = nz_r.max()
        xi_min = nz_c.min()
        xi_max = nz_c.max()
        xul, yul = View.affine_transform(data.affine,
                                         xi_min - pad[0],
                                         yi_min - pad[3])
        new_affine = Affine(data.affine.a, data.affine.b, xul,
                            data.affine.d, data.affine.e, yul)
        out = data[yi_min:yi_max + 1, xi_min:xi_max + 1]
        out = np.pad(out, (vert_pad, horiz_pad),
                    mode='constant', constant_values=data.nodata)
        out_mask = mask[yi_min:yi_max + 1, xi_min:xi_max + 1]
        out_mask = np.pad(out_mask, (vert_pad, horiz_pad),
                          mode='constant', constant_values=False)
        new_viewfinder = ViewFinder(affine=new_affine, shape=out.shape,
                                    nodata=data.nodata, crs=data.crs,
                                    mask=out_mask)
        out = Raster(out, viewfinder=new_viewfinder, metadata=data.metadata)
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
        try:
            assert not np.issubdtype(dtype, np.object_)
            assert not np.issubdtype(dtype, np.flexible)
        except:
            raise TypeError('`object` and `flexible` dtypes not allowed.')
        return dtype

    @classmethod
    def _view_same_viewfinder(cls, data, data_view, target_view, dtype,
                              apply_output_mask=True):
        if apply_output_mask:
            out = np.where(target_view.mask, data, target_view.nodata).astype(dtype)
        else:
            out = np.asarray(data.copy(), dtype=dtype)
        out = Raster(out, target_view)
        return out

    @classmethod
    def _view_different_viewfinder(cls, data, data_view, target_view, dtype,
                                   apply_output_mask=True, interpolation='nearest'):
        out = np.full(target_view.shape, target_view.nodata, dtype=dtype)
        if (data_view.crs == target_view.crs):
            out = cls._view_same_crs(out, data, data_view,
                                     target_view, interpolation)
        else:
            out = cls._view_different_crs(out, data, data_view,
                                          target_view, interpolation)
        if apply_output_mask:
            np.place(out, ~target_view.mask, target_view.nodata)
        out = Raster(out, target_view)
        return out

    @classmethod
    def _view_same_crs(cls, view, data, data_view, target_view, interpolation='nearest'):
        y, x = target_view.axes
        inv_affine = ~data_view.affine
        _, y_ix = cls.affine_transform(inv_affine,
                                       np.zeros(target_view.shape[0],
                                                dtype=np.float64), y)
        x_ix, _ = cls.affine_transform(inv_affine, x,
                                       np.zeros(target_view.shape[1],
                                                dtype=np.float64))
        if interpolation == 'nearest':
            view = _self._view_fill_by_axes_nearest_numba(data, view, y_ix, x_ix)
        elif interpolation == 'linear':
            view = _self._view_fill_by_axes_linear_numba(data, view, y_ix, x_ix)
        else:
            raise ValueError('Interpolation method must be one of: `nearest`, `linear`')
        return view

    @classmethod
    def _view_different_crs(cls, view, data, data_view, target_view, interpolation='nearest'):
        y, x = target_view.coords.T
        xt, yt = pyproj.transform(target_view.crs, data_view.crs, x=x, y=y,
                                  errcheck=True, always_xy=True)
        inv_affine = ~data_view.affine
        x_ix, y_ix = cls.affine_transform(inv_affine, xt, yt)
        if interpolation == 'nearest':
            view = _self._view_fill_by_entries_nearest_numba(data, view, y_ix, x_ix)
        elif interpolation == 'linear':
            view = _self._view_fill_by_entries_linear_numba(data, view, y_ix, x_ix)
        else:
            raise ValueError('Interpolation method must be one of: `nearest`, `linear`')
        return view

