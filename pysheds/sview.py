import numpy as np
import pyproj
from affine import Affine
from distutils.version import LooseVersion

import pysheds._sview as _self

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

# TODO: Need to make sure this can handle Raster inputs as well
class Raster(np.ndarray):
    def __new__(cls, input_array, viewfinder=None, metadata={}):
        obj = np.asarray(input_array).view(cls)
        if viewfinder is None:
            affine = Affine(1., 0., 0., 0., 1., 0.)
            shape = input_array.shape
            viewfinder = Viewfinder(affine=affine, shape=shape)
        else:
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

    def to_crs(self, new_crs, **kwargs):
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
        xb, yb = self.affine * boundary.T
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
                                    mask=self.mask, crs=new_crs)
        new_raster = View.view(self, target_view=new_viewfinder,
                            data_view=self.viewfinder, **kwargs)
        return new_raster

class ViewFinder():
    def __init__(self, affine=Affine(1., 0., 0., 0., 1., 0.), shape=(1,1),
                 nodata=0, mask=None, crs=pyproj.Proj(_pyproj_init)):
        self.affine = affine
        self.shape = shape
        self.crs = crs
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
            is_eq &= (self.crs == other.crs)
            # TODO: May want to double-check this...
            # is_eq &= (self.mask == other.mask).all()
            # if np.isnan(self.nodata):
            #     is_eq &= np.isnan(other.nodata)
            # else:
            #     is_eq &= self.nodata == other.nodata
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

    def grid_indices(self, affine=None, shape=None):
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
        x, _ = affine * np.vstack([x_ix, np.zeros(shape[1])])
        _, y = affine * np.vstack([np.zeros(shape[0]), y_ix])
        return y, x

class View():
    def __init__(self):
        pass

    @classmethod
    def view(cls, data, target_view, data_view=None, interpolation='nearest',
             apply_input_mask=False, apply_output_mask=True,
             affine=None, shape=None, crs=None, mask=None, nodata=None,
             dtype=None, inherit_metadata=True, new_metadata={}):
        # If no data view given, use data's view
        if data_view is None:
            try:
                assert(isinstance(data, Raster))
            except:
                raise TypeError('`data` must be a Raster instance.')
            data_view = data.viewfinder
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
                                                 apply_output_mask=apply_output_mask,
                                                 interpolation=interpolation)
        # Write metadata
        if inherit_metadata:
            out.metadata.update(data.metadata)
        out.metadata.update(new_metadata)
        return out

    @classmethod
    def affine_transform(cls, affine, x, y):
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
    def trim_zeros(cls, data, pad=(0,0,0,0)):
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
        Clip grid to bbox representing the smallest area that contains all
        non-null data for a given dataset. If inplace is True, will set
        self.bbox to the bbox generated by this method.
 
        Parameters
        ----------
        data_name : str
                    Name of attribute to base the clip on.
        precision : int
                    Precision to use when matching geographic coordinates.
        inplace : bool
                  If True, update current view (self.affine and self.shape) to
                  conform to clip.
        apply_mask : bool
                     If True, update self.mask based on nonzero values of <data_name>.
        pad : tuple of int (length 4)
              Apply padding to edges of new view (left, bottom, right, top). A pad of
              (1,1,1,1), for instance, will add a one-cell rim around the new view.
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
        nz_r, nz_c = np.nonzero(mask)
        yi_min = nz_r.min()
        yi_max = nz_r.max()
        xi_min = nz_c.min()
        xi_max = nz_c.max()
        xul, yul = data.affine * (xi_min - pad[0], yi_min - pad[3])
        new_affine = Affine(data.affine.a, data.affine.b, xul,
                            data.affine.d, data.affine.e, yul)
        out = data[yi_min:yi_max + 1, xi_min:xi_max + 1]
        vert_pad = (pad[3], pad[1])
        horiz_pad = (pad[0], pad[2])
        out = np.pad(out, (vert_pad, horiz_pad),
                    mode='constant', constant_values=data.nodata)
        new_viewfinder = ViewFinder(affine=new_affine, shape=out.shape,
                                    nodata=data.nodata, crs=data.crs)
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

