import sys
import ast
import copy
import warnings
import pyproj
import numpy as np
import pandas as pd
import geojson
from affine import Affine
from distutils.version import LooseVersion
try:
    import scipy.spatial
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False
try:
    import skimage.measure
    _HAS_SKIMAGE = True
except:
    _HAS_SKIMAGE = False
try:
    import rasterio
    import rasterio.features
    _HAS_RASTERIO = True
except:
    _HAS_RASTERIO = False

print('TEST')

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_crs = lambda Proj: Proj.crs if not _OLD_PYPROJ else Proj
_pyproj_crs_is_geographic = 'is_latlong' if _OLD_PYPROJ else 'is_geographic'
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

# Import input/output functions
import pysheds.io

# Import viewing functions
from pysheds.sview import Raster
from pysheds.sview import View, ViewFinder

# Import numba functions
import pysheds._sgrid as _self

class sGrid():
    """
    Container class for holding and manipulating gridded data.

    Attributes
    ==========
    affine : Affine transformation matrix (uses affine module)
    shape : The shape of the grid (number of rows, number of columns).
    bbox : The geographical bounding box of the current view of the gridded data
           (xmin, ymin, xmax, ymax).
    mask : A boolean array used to mask certain grid cells in the bbox;
           may be used to indicate which cells lie inside a catchment.

    Methods
    =======
        --------
        File I/O
        --------
        add_gridded_data : Add a gridded dataset (dem, flowdir, accumulation)
                           to Grid instance (generic method).
        read_ascii : Read an ascii grid from a file and add it to a
                     Grid instance.
        read_raster : Read a raster file and add the data to a Grid
                      instance.
        from_ascii : Initializes Grid from an ascii file.
        from_raster : Initializes Grid from a raster file.
        to_ascii : Writes current "view" of gridded dataset(s) to ascii file.
        ----------
        Hydrologic
        ----------
        flowdir : Generate a flow direction grid from a given digital elevation
                  dataset (dem). Does not currently handle flats.
        catchment : Delineate the watershed for a given pour point (x, y)
                    or (column, row).
        accumulation : Compute the number of cells upstream of each cell.
        flow_distance : Compute the distance (in cells) from each cell to the
                        outlet.
        extract_river_network : Extract river segments from a catchment.
        fraction : Generate the fractional contributing area for a coarse
                   scale flow direction grid based on a fine-scale flow
                   direction grid.
        ---------------
        Data Processing
        ---------------
        view : Returns a "view" of a dataset defined by an affine transformation
               self.affine (can optionally be masked with self.mask).
        set_bbox : Sets the bbox of the current "view" (self.bbox).
        set_nodata : Sets the nodata value for a given dataset.
        grid_indices : Returns arrays containing the geographic coordinates
                       of the grid's rows and columns for the current "view".
        nearest_cell : Returns the index (column, row) of the cell closest
                       to a given geographical coordinate (x, y).
        clip_to : Clip the bbox to the smallest area containing all non-
                  null gridcells for a provided dataset.
    """

    def __init__(self, viewfinder=None):
        if viewfinder is not None:
            try:
                assert isinstance(new_viewfinder, ViewFinder)
            except:
                raise TypeError('viewfinder must be an instance of ViewFinder.')
            self._viewfinder = viewfinder
        else:
            self._viewfinder = ViewFinder(**self.defaults)

    @property
    def viewfinder(self):
        return self._viewfinder

    @viewfinder.setter
    def viewfinder(self, new_viewfinder):
        try:
            assert isinstance(new_viewfinder, ViewFinder)
        except:
            raise TypeError('viewfinder must be an instance of ViewFinder.')
        self._viewfinder = new_viewfinder

    @property
    def defaults(self):
        props = {
            'affine' : Affine(1.,0.,0.,0.,1.,0.),
            'shape' : (1,1),
            'nodata' : 0,
            'crs' : pyproj.Proj(_pyproj_init),
        }
        return props

    @property
    def affine(self):
        return self.viewfinder.affine

    @property
    def shape(self):
        return self.viewfinder.shape

    @property
    def nodata(self):
        return self.viewfinder.nodata

    @property
    def crs(self):
        return self.viewfinder.crs

    @property
    def mask(self):
        return self.viewfinder.mask

    @affine.setter
    def affine(self, new_affine):
        self.viewfinder.affine = new_affine

    @shape.setter
    def shape(self, new_shape):
        self.viewfinder.shape = new_shape

    @nodata.setter
    def nodata(self, new_nodata):
        self.viewfinder.nodata = new_nodata

    @crs.setter
    def crs(self, new_crs):
        self.viewfinder.crs = new_crs

    @mask.setter
    def mask(self, new_mask):
        self.viewfinder.mask = new_mask

    @property
    def bbox(self):
        return self.viewfinder.bbox

    @property
    def size(self):
        return self.viewfinder.size

    @property
    def extent(self):
        bbox = self.bbox
        extent = (self.bbox[0], self.bbox[2], self.bbox[1], self.bbox[3])
        return extent

    def read_ascii(self, data, skiprows=6, mask=None,
                   crs=pyproj.Proj(_pyproj_init), xll='lower', yll='lower',
                   metadata={}, **kwargs):
        return pysheds.io.read_ascii(data, skiprows=skiprows, mask=mask,
                                     crs=crs, xll=xll, yll=yll, metadata=metadata,
                                     **kwargs)

    def read_raster(self, data, band=1, window=None, window_crs=None,
                    metadata={}, mask_geometry=False, **kwargs):
        return pysheds.io.read_raster(data=data, band=band, window=window,
                                      window_crs=window_crs, metadata=metadata,
                                      mask_geometry=mask_geometry, **kwargs)

    def to_ascii(self, data, file_name, target_view=None, delimiter=' ', fmt=None,
                interpolation='nearest', apply_input_mask=False,
                apply_output_mask=True, affine=None, shape=None, crs=None,
                mask=None, nodata=None, dtype=None, **kwargs):
        if target_view is None:
            target_view = self.viewfinder
        return pysheds.io.to_ascii(data, file_name, target_view=target_view,
                                   delimiter=delimiter, fmt=fmt, interpolation=interpolation,
                                   apply_input_mask=apply_input_mask,
                                   apply_output_mask=apply_output_mask,
                                   affine=affine, shape=shape, crs=crs,
                                   mask=mask, nodata=nodata,
                                   dtype=dtype, **kwargs)

    def to_raster(self, data, file_name, target_view=None, profile=None, view=True,
                blockxsize=256, blockysize=256, interpolation='nearest',
                apply_input_mask=False, apply_output_mask=True, affine=None,
                shape=None, crs=None, mask=None, nodata=None, dtype=None,
                **kwargs):
        if target_view is None:
            target_view = self.viewfinder
        return pysheds.io.to_raster(data, file_name, target_view=target_view,
                                    profile=profile, view=view,
                                    blockxsize=blockxsize,
                                    blockysize=blockysize,
                                    interpolation=interpolation,
                                    apply_input_mask=apply_input_mask,
                                    apply_output_mask=apply_output_mask,
                                    affine=affine, shape=shape, crs=crs,
                                    mask=mask, nodata=nodata, dtype=dtype,
                                    **kwargs)

    @classmethod
    def from_ascii(cls, path, **kwargs):
        newinstance = cls()
        data = newinstance.read_ascii(path, **kwargs)
        newinstance.viewfinder = data.viewfinder
        return newinstance

    @classmethod
    def from_raster(cls, path, **kwargs):
        newinstance = cls()
        data = newinstance.read_raster(path, **kwargs)
        newinstance.viewfinder = data.viewfinder
        return newinstance

    def view(self, data, data_view=None, target_view=None, interpolation='nearest',
             apply_input_mask=False, apply_output_mask=True,
             affine=None, shape=None, crs=None, mask=None, nodata=None,
             dtype=None, inherit_metadata=True, new_metadata={}, **kwargs):
        """
        Return a copy of a gridded dataset clipped to the current "view". The view is determined by
        an affine transformation which describes the bounding box and cellsize of the grid.
        The view will also optionally mask grid cells according to the boolean array self.mask.

        Parameters
        ----------
        data : str or Raster
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        data_view : RegularViewFinder or IrregularViewFinder
                    The view at which the data is defined (based on an affine
                    transformation and shape). Defaults to the Raster dataset's
                    viewfinder attribute.
        target_view : RegularViewFinder or IrregularViewFinder
                      The desired view (based on an affine transformation and shape)
                      Defaults to a viewfinder based on self.affine and self.shape.
        apply_mask : bool
               If True, "mask" the view using self.mask.
        nodata : int or float
                 Value indicating no data in output array.
                 Defaults to the `nodata` attribute of the input dataset.
        interpolation: 'nearest', 'linear', 'cubic', 'spline'
                       Interpolation method to be used. If both the input data
                       view and output data view can be defined on a regular grid,
                       all interpolation methods are available. If one
                       of the datasets cannot be defined on a regular grid, or the
                       datasets use a different CRS, only 'nearest', 'linear' and
                       'cubic' are available.
        as_crs: pyproj.Proj
                Projection at which to view the data (overrides self.crs).
        return_coords: bool
                       If True, return the coordinates corresponding to each value
                       in the output array.
        kx, ky: int
                Degrees of the bivariate spline, if 'spline' interpolation is desired.
        s : float
            Smoothing factor of the bivariate spline, if 'spline' interpolation is desired.
        tolerance: float
                   Maximum tolerance when matching coordinates. Data coordinates
                   that cannot be matched to a target coordinate within this
                   tolerance will be masked with the nodata value in the output array.
        dtype: numpy datatype
               Desired datatype of the output array.
        """
        # Check input type
        try:
            assert isinstance(data, Raster)
        except:
            raise TypeError("data must be a Raster instance")
        # Check interpolation method
        try:
            interpolation = interpolation.lower()
            assert(interpolation in {'nearest', 'linear'})
        except:
            raise ValueError("Interpolation method must be one of: "
                             "'nearest', 'linear'")
        # If no target view is provided, use grid's viewfinder
        if target_view is None:
            target_view = self.viewfinder
        out = View.view(data, target_view, data_view=data_view,
                        interpolation=interpolation,
                        apply_input_mask=apply_input_mask,
                        apply_output_mask=apply_output_mask,
                        affine=affine, shape=shape,
                        crs=crs, mask=mask, nodata=nodata,
                        dtype=dtype,
                        inherit_metadata=inherit_metadata,
                        new_metadata=new_metadata)
        # Return output
        return out

    def nearest_cell(self, x, y, affine=None, snap='corner'):
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
                 Defaults to self.affine.
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
        if not affine:
            affine = self.affine
        return View.nearest_cell(x, y, affine=affine, snap=snap)

    def clip_to(self, data, pad=(0,0,0,0)):
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
        # get class attributes
        new_raster = View.trim_zeros(data, pad=pad)
        self.viewfinder = new_raster.viewfinder

    def flowdir(self, dem, routing='d8', flats=-1, pits=-2, nodata_out=None,
                dirmap=(64, 128, 1, 2, 4, 8, 16, 32), **kwargs):
        """
        Generates a flow direction grid from a DEM grid.

        Parameters
        ----------
        data : str or Raster
               DEM data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new flow direction array.
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        pits : int
               Value to indicate pits in output array.
        flats : int
                Value to indicate flat areas in output array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        as_crs : pyproj.Proj instance
                 CRS projection to use when computing slopes.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and crs.
        """
        default_metadata = {'dirmap' : dirmap, 'flats' : flats, 'pits' : pits}
        input_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        kwargs.update(input_overrides)
        dem = self._input_handler(dem, **kwargs)
        nodata_cells = self._get_nodata_cells(dem)
        if routing.lower() == 'd8':
            if nodata_out is None:
                nodata_out = 0
            fdir = self._d8_flowdir(dem=dem, nodata_cells=nodata_cells,
                                    nodata_out=nodata_out, flats=flats,
                                    pits=pits, dirmap=dirmap)
        elif routing.lower() == 'dinf':
            if nodata_out is None:
                nodata_out = np.nan
            fdir = self._dinf_flowdir(dem=dem, nodata_cells=nodata_cells,
                                      nodata_out=nodata_out, flats=flats,
                                      pits=pits, dirmap=dirmap)
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        fdir.metadata.update(default_metadata)
        return fdir


    def _d8_flowdir(self, dem, nodata_cells, nodata_out=0, flats=-1, pits=-2,
                    dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
        # Make sure nothing flows to the nodata cells
        dem[nodata_cells] = dem.max() + 1
        # Get cell spans and heights
        dx = abs(dem.affine.a)
        dy = abs(dem.affine.e)
        # Compute D8 flow directions
        fdir = _self._d8_flowdir_numba(dem, dx, dy, dirmap, nodata_cells,
                                       nodata_out, flat=flats, pit=pits)
        return self._output_handler(data=fdir, viewfinder=dem.viewfinder,
                                    metadata=dem.metadata, nodata=nodata_out)

    def _dinf_flowdir(self, dem, nodata_cells, nodata_out=np.nan, flats=-1, pits=-2,
                      dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
        # Make sure nothing flows to the nodata cells
        dem[nodata_cells] = dem.max() + 1
        dx = abs(dem.affine.a)
        dy = abs(dem.affine.e)
        fdir = _self._dinf_flowdir_numba(dem, dx, dy, nodata_out, flat=flats, pit=pits)
        return self._output_handler(data=fdir, viewfinder=dem.viewfinder,
                                    metadata=dem.metadata, nodata=nodata_out)

    def catchment(self, x, y, fdir, pour_value=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                  nodata_out=False, xytype='coordinate', routing='d8', snap='corner', **kwargs):
        """
        Delineates a watershed from a given pour point (x, y).

        Parameters
        ----------
        x : int or float
            x coordinate of pour point
        y : int or float
            y coordinate of pour point
        data : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        pour_value : int or None
                     If not None, value to represent pour point in catchment
                     grid (required by some programs).
        out_name : string
                   Name of attribute containing new catchment array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                    Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        xytype : 'index' or 'label'
                 How to interpret parameters 'x' and 'y'.
                     'index' : x and y represent the column and row
                               indices of the pour point.
                     'label' : x and y represent geographic coordinates
                               (will be passed to self.nearest_cell).
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions
        recursionlimit : int
                         Recursion limit--may need to be raised if
                         recursion limit is reached.
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and crs.
        snap : str
               Function to use on array for indexing:
               'corner' : numpy.around()
               'center' : numpy.floor()
        """
        if routing.lower() == 'd8':
            input_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        elif routing.lower() == 'dinf':
            input_overrides = {'dtype' : np.float64, 'nodata' : fdir.nodata}
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        kwargs.update(input_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        xmin, ymin, xmax, ymax = fdir.bbox
        if xytype in {'label', 'coordinate'}:
            if (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax):
                raise ValueError('Pour point ({}, {}) is out of bounds for dataset with bbox {}.'
                                .format(x, y, (xmin, ymin, xmax, ymax)))
        elif xytype == 'index':
            if (x < 0) or (y < 0) or (x >= fdir.shape[1]) or (y >= fdir.shape[0]):
                raise ValueError('Pour point ({}, {}) is out of bounds for dataset with shape {}.'
                                .format(x, y, fdir.shape))
        if routing.lower() == 'd8':
            catch = self._d8_catchment(x, y, fdir=fdir, pour_value=pour_value, dirmap=dirmap,
                                       nodata_out=nodata_out, xytype=xytype, snap=snap)
        elif routing.lower() == 'dinf':
            catch = self._dinf_catchment(x, y, fdir=fdir, pour_value=pour_value, dirmap=dirmap,
                                         nodata_out=nodata_out, xytype=xytype, snap=snap)
        return catch

    def _d8_catchment(self, x, y, fdir, pour_value=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                      nodata_out=False, xytype='coordinate', snap='corner'):
        # Pad the rim
        left, right, top, bottom = self._pop_rim(fdir, nodata=0)
        # If xytype is 'coordinate', delineate catchment based on cell nearest
        # to given geographic coordinate
        if xytype in {'label', 'coordinate'}:
            x, y = self.nearest_cell(x, y, fdir.affine, snap)
        # Delineate the catchment
        catch = _self._d8_catchment_numba(fdir, (y, x), dirmap)
        if pour_value is not None:
            catch[r, c] = pour_value
        catch = self._output_handler(data=catch, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return catch

    def _dinf_catchment(self, x, y, fdir, pour_value=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                        nodata_out=False, xytype='coordinate', snap='corner'):
        # Find nodata cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split dinf flowdir
        fdir_0, fdir_1, prop_0, prop_1 = _self._angle_to_d8_numba(fdir, dirmap, nodata_cells)
        # Pad the rim
        left_0, right_0, top_0, bottom_0 = self._pop_rim(fdir_0, nodata=0)
        left_1, right_1, top_1, bottom_1 = self._pop_rim(fdir_1, nodata=0)
        # Valid if the dataset is a view.
        if xytype in {'label', 'coordinate'}:
            x, y = self.nearest_cell(x, y, fdir.affine, snap)
        # Delineate the catchment
        catch = _self._dinf_catchment_numba(fdir_0, fdir_1, (y, x), dirmap)
        # if pour point needs to be a special value, set it
        if pour_value is not None:
            catch[r, c] = pour_value
        catch = self._output_handler(data=catch, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return catch

    def accumulation(self, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                     nodata_out=0., efficiency=None, routing='d8', cycle_size=1, **kwargs):
        """
        Generates an array of flow accumulation, where cell values represent
        the number of upstream cells.

        Parameters
        ----------
        data : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        weights: numpy ndarray
-                 Array of weights to be applied to each accumulation cell. Must
-                 be same size as data.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        efficiency: numpy ndarray
                 transport efficiency, relative correction factor applied to the
                 outflow of each cell
                 nodata will be set to 1, i.e. no correction
                 Must be same size as data.
        nodata_in : int or float
                    Value to indicate nodata in input array. If using a named dataset, will
                    default to the 'nodata' value of the named dataset. If using an ndarray,
                    will default to 0.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        out_name : string
                   Name of attribute containing new accumulation array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        pad : bool
              If True, pad the rim of the input array with zeros. Else, ignore
              the outer rim of cells in the computation.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and crs.
        """
        if routing.lower() == 'd8':
            fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        elif routing.lower() == 'dinf':
            fdir_overrides = {'dtype' : np.float64, 'nodata' : fdir.nodata}
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        kwargs.update(fdir_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        if weights is not None:
            weights_overrides = {'dtype' : np.float64, 'nodata' : weights.nodata}
            kwargs.update(weights_overrides)
            weights = self._input_handler(weights, **kwargs)
        if efficiency is not None:
            efficiency_overrides = {'dtype' : np.float64, 'nodata' : efficiency.nodata}
            kwargs.update(efficiency_overrides)
            efficiency = self._input_handler(efficiency, **kwargs)
        if routing.lower() == 'd8':
            acc = self._d8_accumulation(fdir, weights=weights, dirmap=dirmap,
                                        nodata_out=nodata_out,
                                        efficiency=efficiency)
        elif routing.lower() == 'dinf':
            acc = self._dinf_accumulation(fdir, weights=weights, dirmap=dirmap,
                                          nodata_out=nodata_out,
                                          efficiency=efficiency,
                                          cycle_size=cycle_size)
        return acc

    def _d8_accumulation(self, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                         nodata_out=0., efficiency=None, **kwargs):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        # Start and end nodes
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes = _self._flatten_fdir_numba(fdir, dirmap).reshape(fdir.shape)
        # Initialize accumulation array to weights, if using weights
        if weights is not None:
            acc = weights.astype(np.float64).reshape(fdir.shape)
        # Otherwise, initialize accumulation array to ones where valid cells exist
        else:
            acc = (~nodata_cells).astype(np.float64).reshape(fdir.shape)
        # If using efficiency, initialize array
        if efficiency is not None:
            eff = efficiency.astype(np.float64).reshape(fdir.shape)
        # Find indegree of all cells
        indegree = np.bincount(endnodes.ravel(), minlength=fdir.size).astype(np.uint8)
        # Set starting nodes to those with no predecessors
        startnodes = startnodes[(indegree == 0)]
        # Compute accumulation
        if efficiency is None:
            acc = _self._d8_accumulation_numba(acc, endnodes, indegree, startnodes)
        else:
            acc = _self._d8_accumulation_eff_numba(acc, endnodes, indegree, startnodes, eff)
        acc = self._output_handler(data=acc, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return acc

    def _dinf_accumulation(self, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                           nodata_out=0., efficiency=None, cycle_size=1, **kwargs):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split d-infinity grid
        fdir_0, fdir_1, prop_0, prop_1 = _self._angle_to_d8_numba(fdir, dirmap, nodata_cells)
        # Get matching of start and end nodes
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes_0 = _self._flatten_fdir_numba(fdir_0, dirmap).reshape(fdir.shape)
        endnodes_1 = _self._flatten_fdir_numba(fdir_1, dirmap).reshape(fdir.shape)
        # Remove cycles
        _self._dinf_fix_cycles_numba(endnodes_0, endnodes_1, cycle_size)
        # Initialize accumulation array to weights, if using weights
        if weights is not None:
            acc = weights.reshape(fdir.shape).astype(np.float64)
        # Otherwise, initialize accumulation array to ones where valid cells exist
        else:
            acc = (~nodata_cells).reshape(fdir.shape).astype(np.float64)
        if efficiency is not None:
            eff = efficiency.reshape(fdir.shape).astype(np.float64)
        # Find indegree of all cells
        indegree_0 = np.bincount(endnodes_0.ravel(), minlength=fdir.size)
        indegree_1 = np.bincount(endnodes_1.ravel(), minlength=fdir.size)
        indegree = (indegree_0 + indegree_1).astype(np.uint8)
        # Set starting nodes to those with no predecessors
        startnodes = startnodes[(indegree == 0)]
        # Compute accumulation
        if efficiency is None:
            acc = _self._dinf_accumulation_numba(acc, endnodes_0, endnodes_1, indegree,
                                           startnodes, prop_0, prop_1)
        else:
            acc = _self._dinf_accumulation_eff_numba(acc, endnodes_0, endnodes_1, indegree,
                                               startnodes, prop_0, prop_1, eff)
        acc = self._output_handler(data=acc, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return acc

    def flow_distance(self, x, y, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                      nodata_out=np.nan, routing='d8', method='shortest',
                      xytype='coordinate', snap='corner', **kwargs):
        """
        Generates an array representing the topological distance from each cell
        to the outlet.

        Parameters
        ----------
        x : int or float
            x coordinate of pour point
        y : int or float
            y coordinate of pour point
        data : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        weights: numpy ndarray
                 Weights (distances) to apply to link edges.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                    Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        out_name : string
                   Name of attribute containing new flow distance array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        xytype : 'index' or 'label'
                 How to interpret parameters 'x' and 'y'.
                     'index' : x and y represent the column and row
                               indices of the pour point.
                     'label' : x and y represent geographic coordinates
                               (will be passed to self.nearest_cell).
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        snap : str
               Function to use on array for indexing:
               'corner' : numpy.around()
               'center' : numpy.floor()
        """
        if routing.lower() == 'd8':
            input_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        elif routing.lower() == 'dinf':
            input_overrides = {'dtype' : np.float64, 'nodata' : fdir.nodata}
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        kwargs.update(input_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        if weights is not None:
            weights_overrides = {'dtype' : np.float64, 'nodata' : weights.nodata}
            kwargs.update(weights_overrides)
            weights = self._input_handler(weights, **kwargs)
        xmin, ymin, xmax, ymax = fdir.bbox
        if xytype in {'label', 'coordinate'}:
            if (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax):
                raise ValueError('Pour point ({}, {}) is out of bounds for dataset with bbox {}.'
                                .format(x, y, (xmin, ymin, xmax, ymax)))
        elif xytype == 'index':
            if (x < 0) or (y < 0) or (x >= fdir.shape[1]) or (y >= fdir.shape[0]):
                raise ValueError('Pour point ({}, {}) is out of bounds for dataset with shape {}.'
                                .format(x, y, fdir.shape))
        if routing.lower() == 'd8':
            dist = self._d8_flow_distance(x=x, y=y, fdir=fdir, weights=weights,
                                          dirmap=dirmap, nodata_out=nodata_out,
                                          method=method, xytype=xytype,
                                          snap=snap)
        elif routing.lower() == 'dinf':
            dist = self._dinf_flow_distance(x=x, y=y, fdir=fdir, weights=weights,
                                            dirmap=dirmap, nodata_out=nodata_out,
                                            method=method, xytype=xytype,
                                            snap=snap)
        return dist

    def _d8_flow_distance(self, x, y, fdir, weights=None,
                          dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                          nodata_out=np.nan, method='shortest',
                          xytype='coordinate', snap='corner', **kwargs):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        if xytype in {'label', 'coordinate'}:
            x, y = self.nearest_cell(x, y, fdir.affine, snap)
        if weights is None:
            weights = (~nodata_cells).reshape(fdir.shape).astype(np.float64)
        dist = _self._d8_flow_distance_numba(fdir, weights, (y, x), dirmap)
        dist = self._output_handler(data=dist, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return dist

    def _dinf_flow_distance(self, x, y, fdir, weights=None,
                          dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                          nodata_out=np.nan, method='shortest',
                          xytype='coordinate', snap='corner', **kwargs):
        # Find nodata cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split d-infinity grid
        fdir_0, fdir_1, prop_0, prop_1 = _self._angle_to_d8_numba(fdir, dirmap, nodata_cells)
        if xytype in {'label', 'coordinate'}:
            x, y = self.nearest_cell(x, y, fdir.affine, snap)
        if weights is not None:
            weights_0 = weights
            weights_1 = weights
        else:
            weights_0 = (~nodata_cells).reshape(fdir.shape).astype(np.float64)
            weights_1 = weights_0
        if method.lower() == 'shortest':
            dist = _self._dinf_flow_distance_numba(fdir_0, fdir_1, weights_0,
                                                   weights_1, (y, x), dirmap)
        else:
            raise NotImplementedError("Only implemented for shortest path distance.")
        # Prepare output
        dist = self._output_handler(data=dist, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return dist

    def compute_hand(self, fdir, dem, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                     nodata_out=None, routing='d8', return_index=False, **kwargs):
        """
        Computes the height above nearest drainage (HAND), based on a flow direction grid,
        a digital elevation grid, and a grid containing the locations of drainage channels.

        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        dem : str or Raster
              Digital elevation data.
              If str: name of the dataset to be viewed.
              If Raster: a Raster instance (see pysheds.view.Raster)
        drainage_mask : str or Raster
                        Boolean raster or ndarray with nonzero elements indicating
                        locations of drainage channels.
                        If str: name of the dataset to be viewed.
                        If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new catchment array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in_fdir : int or float
                         Value to indicate nodata in flow direction input array.
        nodata_in_dem : int or float
                        Value to indicate nodata in digital elevation input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions (not implemented)
        recursionlimit : int
                         Recursion limit--may need to be raised if
                         recursion limit is reached.
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and crs.
        """
        if routing.lower() == 'd8':
            fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        elif routing.lower() == 'dinf':
            fdir_overrides = {'dtype' : np.float64, 'nodata' : fdir.nodata}
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        dem_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        mask_overrides = {'dtype' : np.bool8, 'nodata' : False}
        kwargs.update(fdir_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        kwargs.update(dem_overrides)
        dem = self._input_handler(dem, **kwargs)
        kwargs.update(mask_overrides)
        mask = self._input_handler(mask, **kwargs)
        # Set default nodata for hand index and hand
        if nodata_out is None:
            if return_index:
                nodata_out = -1
            else:
                nodata_out = dem.nodata
        # Compute height above nearest drainage
        if routing.lower() == 'd8':
            hand = self._d8_compute_hand(fdir=fdir, mask=mask,
                                         dirmap=dirmap, nodata_out=nodata_out)
        elif routing.lower() == 'dinf':
            hand = self._dinf_compute_hand(fdir=fdir, mask=mask,
                                           nodata_out=nodata_out)
        # If index is not desired, return heights
        if not return_index:
            hand = _self._assign_hand_heights_numba(hand, dem, nodata_out)
        return hand

    def _d8_compute_hand(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                         nodata_out=-1):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        # TODO: Need to check validity of fdir
        dirleft, dirright, dirtop, dirbottom = self._pop_rim(fdir, nodata=0)
        maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=False)
        hand = _self._d8_hand_iter_numba(fdir, mask, dirmap)
        hand = self._output_handler(data=hand, viewfinder=fdir.viewfinder,
                                    metadata=fdir.metadata, nodata=nodata_out)
        return hand

    def _dinf_compute_hand(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                         nodata_out=-1):
        # Get nodata cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split dinf flowdir
        fdir_0, fdir_1, prop_0, prop_1 = _self._angle_to_d8_numba(fdir, dirmap, nodata_cells)
        # Pad the rim
        dirleft_0, dirright_0, dirtop_0, dirbottom_0 = self._pop_rim(fdir_0,
                                                                     nodata=0)
        dirleft_1, dirright_1, dirtop_1, dirbottom_1 = self._pop_rim(fdir_1,
                                                                     nodata=0)
        maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=False)
        hand = _self._dinf_hand_iter_numba(fdir_0, fdir_1, mask, dirmap)
        hand = self._output_handler(data=hand, viewfinder=fdir.viewfinder,
                                    metadata=fdir.metadata, nodata=nodata_out)
        return hand

    def resolve_flats(self, data, nodata_out=None, eps=1e-5, max_iter=1000, **kwargs):
        """
        Resolve flats in a DEM using the modified method of Barnes et al. (2015).
        See: https://arxiv.org/abs/1511.04433

        Parameters
        ----------
        data : str or Raster
               DEM data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new flow direction array.
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        input_overrides = {'dtype' : np.float64}
        kwargs.update(input_overrides)
        dem = self._input_handler(data, **kwargs)
        # Find no data cells
        # TODO: Should these be used?
        nodata_cells = self._get_nodata_cells(dem)
        # Get inside indices
        inside = np.arange(dem.size, dtype=np.int64).reshape(dem.shape)[1:-1, 1:-1].ravel()
        # Find (i) cells in flats, (ii) cells with flow directions defined
        # and (iii) cells with at least one higher neighbor
        flats, fdirs_defined, higher_cells = _self._par_get_candidates_numba(dem, inside)
        # Label all flats
        labels, numlabels = skimage.measure.label(flats, return_num=True)
        # Get high-edge cells
        hec = _self._par_get_high_edge_cells_numba(inside, fdirs_defined, higher_cells, labels)
        # Get low-edge cells
        lec = _self._par_get_low_edge_cells_numba(inside, dem, fdirs_defined, labels, numlabels)
        # Construct gradient from higher terrain
        grad_from_higher = _self._grad_from_higher_numba(hec, flats, labels, numlabels, max_iter)
        # Construct gradient towards lower terrain
        grad_towards_lower = _self._grad_towards_lower_numba(lec, flats, dem, max_iter)
        # Construct a gradient that is guaranteed to drain
        new_drainage_grad = (2 * grad_towards_lower + grad_from_higher)
        # Create a flat-removed DEM by applying drainage gradient
        inflated_dem = dem + eps * new_drainage_grad
        inflated_dem = self._output_handler(data=inflated_dem,
                                            viewfinder=dem.viewfinder,
                                            metadata=dem.metadata)
        return inflated_dem

    def extract_river_network(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                              routing='d8', **kwargs):
        """
        Generates river segments from accumulation and flow_direction arrays.

        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        mask : np.ndarray or Raster
               Boolean array indicating channelized regions
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.

        Returns
        -------
        geo : geojson.FeatureCollection
              A geojson feature collection of river segments. Each array contains the cell
              indices of junctions in the segment.
        """
        if routing.lower() == 'd8':
            fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        else:
            raise NotImplementedError('Only implemented for D8 routing.')
        mask_overrides = {'dtype' : np.bool8, 'nodata' : False}
        kwargs.update(fdir_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        kwargs.update(mask_overrides)
        mask = self._input_handler(mask, **kwargs)
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=False)
        masked_fdir = np.where(mask, fdir, 0).astype(np.int64)
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes = _self._flatten_fdir_numba(masked_fdir, dirmap).reshape(fdir.shape)
        indegree = np.bincount(endnodes.ravel(), minlength=fdir.size).astype(np.uint8)
        orig_indegree = np.copy(indegree)
        startnodes = startnodes[(indegree == 0)]
        profiles = _self._d8_stream_network_numba(endnodes, indegree, orig_indegree, startnodes)
        # Fill geojson dict with profiles
        featurelist = []
        for index, profile in enumerate(profiles):
            yi, xi = np.unravel_index(list(profile), fdir.shape)
            x, y = self.affine * (xi, yi)
            line = geojson.LineString(np.column_stack([x, y]).tolist())
            featurelist.append(geojson.Feature(geometry=line, id=index))
            geo = geojson.FeatureCollection(featurelist)
        return geo

    def stream_order(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                     nodata_out=0, routing='d8', **kwargs):
        """
        Generates river segments from accumulation and flow_direction arrays.

        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        mask : np.ndarray or Raster
               Boolean array indicating channelized regions
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.

        Returns
        -------
        geo : geojson.FeatureCollection
              A geojson feature collection of river segments. Each array contains the cell
              indices of junctions in the segment.
        """
        if routing.lower() == 'd8':
            fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        else:
            raise NotImplementedError('Only implemented for D8 routing.')
        mask_overrides = {'dtype' : np.bool8, 'nodata' : False}
        kwargs.update(fdir_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        kwargs.update(mask_overrides)
        mask = self._input_handler(mask, **kwargs)
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=False)
        masked_fdir = np.where(mask, fdir, 0).astype(np.int64)
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes = _self._flatten_fdir_numba(masked_fdir, dirmap).reshape(fdir.shape)
        indegree = np.bincount(endnodes.ravel()).astype(np.uint8)
        orig_indegree = np.copy(indegree)
        startnodes = startnodes[(indegree == 0)]
        min_order = np.full(fdir.shape, np.iinfo(np.int64).max, dtype=np.int64)
        max_order = np.ones(fdir.shape, dtype=np.int64)
        order = np.where(mask, 1, 0).astype(np.int64).reshape(fdir.shape)
        order = _self._d8_streamorder_numba(min_order, max_order, order, endnodes,
                                        indegree, orig_indegree, startnodes)
        order = self._output_handler(data=order, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return order

    def reverse_distance(self, fdir, mask, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                         nodata_out=0, routing='d8', **kwargs):
        """
        Generates river segments from accumulation and flow_direction arrays.

        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        mask : np.ndarray or Raster
               Boolean array indicating channelized regions
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.

        Returns
        -------
        geo : geojson.FeatureCollection
              A geojson feature collection of river segments. Each array contains the cell
              indices of junctions in the segment.
        """
        if routing.lower() == 'd8':
            fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        else:
            raise NotImplementedError('Only implemented for D8 routing.')
        mask_overrides = {'dtype' : np.bool8, 'nodata' : False}
        kwargs.update(fdir_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        kwargs.update(mask_overrides)
        mask = self._input_handler(mask, **kwargs)
        if weights is not None:
            weights_overrides = {'dtype' : np.float64, 'nodata' : weights.nodata}
            kwargs.update(weights_overrides)
            weights = self._input_handler(weights, **kwargs)
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        if weights is None:
            weights = (~nodata_cells).reshape(fdir.shape).astype(np.float64)
        maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=0)
        masked_fdir = np.where(mask, fdir, 0).astype(np.int64)
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes = _self._flatten_fdir_numba(masked_fdir, dirmap).reshape(fdir.shape)
        indegree = np.bincount(endnodes.ravel()).astype(np.uint8)
        orig_indegree = np.copy(indegree)
        startnodes = startnodes[(indegree == 0)]
        min_order = np.full(fdir.shape, np.iinfo(np.int64).max, dtype=np.int64)
        max_order = np.ones(fdir.shape, dtype=np.int64)
        rdist = np.zeros(fdir.shape, dtype=np.float64)
        rdist = _self._d8_reverse_distance_numba(min_order, max_order, rdist,
                                            endnodes, indegree, startnodes, weights)
        rdist = self._output_handler(data=rdist, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return rdist

    def cell_dh(self, dem, fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                nodata_out=np.nan, routing='d8', **kwargs):
        """
        Generates an array representing the elevation difference from each cell to its
        downstream neighbor.

        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        dem : str or Raster
              DEM data.
              If str: name of the dataset to be viewed.
              If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new cell elevation difference array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        if routing.lower() == 'd8':
            fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        elif routing.lower() == 'dinf':
            fdir_overrides = {'dtype' : np.float64, 'nodata' : fdir.nodata}
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        dem_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        kwargs.update(fdir_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        kwargs.update(dem_overrides)
        dem = self._input_handler(dem, **kwargs)
        if routing.lower() == 'd8':
            dh = self._d8_cell_dh(dem=dem, fdir=fdir, dirmap=dirmap,
                                  nodata_out=nodata_out)
        elif routing.lower() == 'dinf':
            dh = self._dinf_cell_dh(dem=dem, fdir=fdir, dirmap=dirmap,
                                    nodata_out=nodata_out)
        return dh

    def _d8_cell_dh(self, dem, fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                    nodata_out=np.nan):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        dirleft, dirright, dirtop, dirbottom = self._pop_rim(fdir, nodata=0)
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes = _self._flatten_fdir_numba(fdir, dirmap).reshape(fdir.shape)
        dh = _self._d8_cell_dh_numba(startnodes, endnodes, dem)
        dh = self._output_handler(data=dh, viewfinder=fdir.viewfinder,
                                  metadata=fdir.metadata, nodata=nodata_out)
        return dh

    def _dinf_cell_dh(self, dem, fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                         nodata_out=np.nan):
        # Get nodata cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split dinf flowdir
        fdir_0, fdir_1, prop_0, prop_1 = _self._angle_to_d8_numba(fdir, dirmap, nodata_cells)
        # Pad the rim
        dirleft_0, dirright_0, dirtop_0, dirbottom_0 = self._pop_rim(fdir_0,
                                                                     nodata=0)
        dirleft_1, dirright_1, dirtop_1, dirbottom_1 = self._pop_rim(fdir_1,
                                                                     nodata=0)
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes_0 = _self._flatten_fdir_numba(fdir_0, dirmap).reshape(fdir.shape)
        endnodes_1 = _self._flatten_fdir_numba(fdir_1, dirmap).reshape(fdir.shape)
        dh = _self._dinf_cell_dh_numba(startnodes, endnodes_0, endnodes_1, prop_0, prop_1, dem)
        dh = self._output_handler(data=dh, viewfinder=fdir.viewfinder,
                                  metadata=fdir.metadata, nodata=nodata_out)
        return dh

    def cell_distances(self, fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32), nodata_out=np.nan,
                       routing='d8', **kwargs):
        """
        Generates an array representing the distance from each cell to its downstream neighbor.

        Parameters
        ----------
        data : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new cell distance array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        as_crs : pyproj.Proj
                 CRS at which to compute the distance from each cell to its downstream neighbor.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        if routing.lower() == 'd8':
            fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        elif routing.lower() == 'dinf':
            fdir_overrides = {'dtype' : np.float64, 'nodata' : fdir.nodata}
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        kwargs.update(fdir_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        if routing.lower() == 'd8':
            cdist = self._d8_cell_distances(fdir=fdir, dirmap=dirmap,
                                         nodata_out=nodata_out)
        elif routing.lower() == 'dinf':
            cdist = self._dinf_cell_distances(fdir=fdir, dirmap=dirmap,
                                           nodata_out=nodata_out)
        return cdist

    def _d8_cell_distances(self, fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                           nodata_out=np.nan):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        dx = abs(fdir.affine.a)
        dy = abs(fdir.affine.e)
        cdist = _self._d8_cell_distances_numba(fdir, dirmap, dx, dy)
        cdist = self._output_handler(data=cdist, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return cdist

    def _dinf_cell_distances(self, fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                             nodata_out=np.nan):
        # Get nodata cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split dinf flowdir
        fdir_0, fdir_1, prop_0, prop_1 = _self._angle_to_d8_numba(fdir, dirmap, nodata_cells)
        # Pad the rim
        dirleft_0, dirright_0, dirtop_0, dirbottom_0 = self._pop_rim(fdir_0,
                                                                     nodata=0)
        dirleft_1, dirright_1, dirtop_1, dirbottom_1 = self._pop_rim(fdir_1,
                                                                     nodata=0)
        dx = abs(fdir.affine.a)
        dy = abs(fdir.affine.e)
        cdist = _self._dinf_cell_distances_numba(fdir_0, fdir_1, prop_0, prop_1,
                                           dirmap, dx, dy)
        cdist = self._output_handler(data=cdist, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return cdist

    def cell_slopes(self, dem, fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32), nodata_out=np.nan,
                    routing='d8', **kwargs):
        """
        Generates an array representing the distance from each cell to its downstream neighbor.

        Parameters
        ----------
        data : str or Raster
               Flow direction data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new cell distance array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        as_crs : pyproj.Proj
                 CRS at which to compute the distance from each cell to its downstream neighbor.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        if routing.lower() == 'd8':
            fdir_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        elif routing.lower() == 'dinf':
            fdir_overrides = {'dtype' : np.float64, 'nodata' : fdir.nodata}
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        dem_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        kwargs.update(fdir_overrides)
        fdir = self._input_handler(fdir, **kwargs)
        kwargs.update(dem_overrides)
        dem = self._input_handler(dem, **kwargs)
        dh = self.cell_dh(dem, fdir, dirmap=dirmap, nodata_out=np.nan,
                          routing=routing, **kwargs)
        cdist = self.cell_distances(fdir, dirmap=dirmap, nodata_out=np.nan,
                                    routing=routing, **kwargs)
        slopes = _self._cell_slopes_numba(dh, cdist)
        return slopes

    def fill_pits(self, dem, nodata_out=None, **kwargs):
        """
        Fill pits in a DEM. Raises pits to same elevation as lowest neighbor.

        Parameters
        ----------
        data : str or Raster
               DEM data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new filled pit array.
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value indicating no data in output array.
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        input_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        kwargs.update(input_overrides)
        dem = self._input_handler(dem, **kwargs)
        # Find no data cells
        nodata_cells = self._get_nodata_cells(dem)
        # Make sure nothing flows to the nodata cells
        dem[nodata_cells] = dem.max() + 1
        # Get indices of inner cells
        inside = np.arange(dem.size, dtype=np.int64).reshape(dem.shape)[1:-1, 1:-1].ravel()
        # Find pits in input DEM
        pits = _self._find_pits_numba(dem, inside)
        pit_indices = np.flatnonzero(pits).astype(np.int64)
        # Create new array to hold pit-filled dem
        pit_filled_dem = dem.copy().astype(np.float64)
        # Fill pits
        _self._fill_pits_numba(pit_filled_dem, pit_indices)
        # Set output nodata value
        if nodata_out is None:
            nodata_out = dem.nodata
        # Ensure nodata cells propagate to pit-filled dem
        pit_filled_dem[nodata_cells] = nodata_out
        pit_filled_dem = self._output_handler(data=pit_filled_dem,
                                              viewfinder=dem.viewfinder,
                                              metadata=dem.metadata)
        return pit_filled_dem

    def detect_pits(self, dem, **kwargs):
        """
        Detect pits in a DEM.

        Parameters
        ----------
        data : str or Raster
               DEM data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new filled pit array.
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value indicating no data in output array.
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.

        Returns
        -------
        pits : numpy ndarray
               Boolean array indicating locations of pits.
        """
        input_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        kwargs.update(input_overrides)
        dem = self._input_handler(dem, **kwargs)
        # Find no data cells
        nodata_cells = self._get_nodata_cells(dem)
        # Make sure nothing flows to the nodata cells
        dem[nodata_cells] = dem.max() + 1
        # Get indices of inner cells
        inside = np.arange(dem.size, dtype=np.int64).reshape(dem.shape)[1:-1, 1:-1].ravel()
        # Find pits
        pits = _self._find_pits_numba(dem, inside)
        pits = self._output_handler(data=pits, viewfinder=dem.viewfinder,
                                    metadata=dem.metadata, nodata=None)
        return pits

    def detect_depressions(self, dem, **kwargs):
        """
        Fill depressions in a DEM. Raises depressions to same elevation as lowest neighbor.

        Parameters
        ----------
        data : str or Raster
               DEM data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new filled depressions array.
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value indicating no data in output array.
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        if not _HAS_SKIMAGE:
            raise ImportError('detect_depressions requires skimage.morphology module')
        input_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        kwargs.update(input_overrides)
        dem = self._input_handler(dem, **kwargs)
        filled_dem = self.fill_depressions(dem, **kwargs)
        depressions = np.zeros(filled_dem.shape, dtype=np.bool8)
        depressions[dem != filled_dem] = True
        depressions[np.isnan(dem) | np.isnan(filled_dem)] = False
        depressions = self._output_handler(data=depressions,
                                           viewfinder=filled_dem.viewfinder,
                                           metadata=filled_dem.metadata,
                                           nodata=False)
        return depressions

    def fill_depressions(self, dem, nodata_out=np.nan, **kwargs):
        """
        Fill depressions in a DEM. Raises depressions to same elevation as lowest neighbor.

        Parameters
        ----------
        data : str or Raster
               DEM data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new filled depressions array.
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value indicating no data in output array.
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        if not _HAS_SKIMAGE:
            raise ImportError('resolve_flats requires skimage.morphology module')
        input_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        kwargs.update(input_overrides)
        dem = self._input_handler(dem, **kwargs)
        dem_mask = self._get_nodata_cells(dem)
        dem_mask[0, :] = True
        dem_mask[-1, :] = True
        dem_mask[:, 0] = True
        dem_mask[:, -1] = True
        # Make sure nothing flows to the nodata cells
        seed = np.copy(dem)
        seed[~dem_mask] = np.nanmax(dem)
        dem_out = skimage.morphology.reconstruction(seed, dem, method='erosion')
        dem_out = self._output_handler(data=dem_out, viewfinder=dem.viewfinder,
                                     metadata=dem.metadata, nodata=nodata_out)
        return dem_out

    def detect_flats(self, dem, **kwargs):
        """
        Detect flats in a DEM.

        Parameters
        ----------
        data : str or Raster
               DEM data.
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new flow direction array.
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        inplace : bool
                  If True, write output array to self.<out_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.

        Returns
        -------
        flats : numpy ndarray
                Boolean array indicating locations of flats.
        """
        input_overrides = {'dtype' : np.float64, 'nodata' : dem.nodata}
        kwargs.update(input_overrides)
        dem = self._input_handler(dem, **kwargs)
        # Find no data cells
        nodata_cells = self._get_nodata_cells(dem)
        # Make sure nothing flows to the nodata cells
        dem[nodata_cells] = dem.max() + 1
        # Get indices of inner cells
        inside = np.arange(dem.size, dtype=np.int64).reshape(dem.shape)[1:-1, 1:-1].ravel()
        # handle nodata values in dem
        flats, _, _ = _self._par_get_candidates_numba(dem, inside)
        flats = self._output_handler(data=flats, viewfinder=dem.viewfinder,
                                     metadata=dem.metadata, nodata=None)
        return flats

    def polygonize(self, data=None, mask=None, connectivity=4, transform=None):
        """
        Yield (polygon, value) for each set of adjacent pixels of the same value.
        Wrapper around rasterio.features.shapes

        From rasterio documentation:

        Parameters
        ----------
        data : numpy ndarray
        mask : numpy ndarray
               Values of False or 0 will be excluded from feature generation.
        connectivity : 4 or 8 (int)
                       Use 4 or 8 pixel connectivity.
        transform : affine.Affine
                    Transformation from pixel coordinates of `image` to the
                    coordinate system of the input `shapes`.
        """
        if not _HAS_RASTERIO:
            raise ImportError('Requires rasterio module')
        if data is None:
            data = self.mask.astype(np.uint8)
        if mask is None:
            mask = self.mask
        if transform is None:
            transform = self.affine
        shapes = rasterio.features.shapes(data, mask=mask, connectivity=connectivity,
                                          transform=transform)
        return shapes

    def rasterize(self, shapes, out_shape=None, fill=0, out=None, transform=None,
                  all_touched=False, default_value=1, dtype=None):
        """
        Return an image array with input geometries burned in.
        Wrapper around rasterio.features.rasterize

        From rasterio documentation:

        Parameters
        ----------
        shapes : iterable of (geometry, value) pairs or iterable over
                 geometries.
        out_shape : tuple or list
                    Shape of output numpy ndarray.
        fill : int or float, optional
               Fill value for all areas not covered by input geometries.
        out : numpy ndarray
              Array of same shape and data type as `image` in which to store
              results.
        transform : affine.Affine
                    Transformation from pixel coordinates of `image` to the
                    coordinate system of the input `shapes`.
        all_touched : boolean, optional
                      If True, all pixels touched by geometries will be burned in.  If
                      false, only pixels whose center is within the polygon or that
                      are selected by Bresenham's line algorithm will be burned in.
        default_value : int or float, optional
                        Used as value for all geometries, if not provided in `shapes`.
        dtype : numpy data type
                Used as data type for results, if `out` is not provided.
        """
        if not _HAS_RASTERIO:
            raise ImportError('Requires rasterio module')
        if out_shape is None:
            out_shape = self.shape
        if transform is None:
            transform = self.affine
        raster = rasterio.features.rasterize(shapes, out_shape=out_shape, fill=fill,
                                             out=out, transform=transform,
                                             all_touched=all_touched,
                                             default_value=default_value, dtype=dtype)
        return raster

    def snap_to_mask(self, mask, xy, return_dist=False, **kwargs):
        """
        Snap a set of xy coordinates (xy) to the nearest nonzero cells in a raster (mask)
        TODO: Behavior has changed here---now coerces to grid's viewfinder

        Parameters
        ----------
        mask: numpy ndarray-like with shape (M, K)
              A raster dataset with nonzero elements indicating cells to match to (e.g:
              a flow accumulation grid with ones indicating cells above a certain threshold).
        xy: numpy ndarray-like with shape (N, 2)
            Points to match (example: gage location coordinates).
        return_dist: If true, return the distances from xy to the nearest matched point in mask.
        """

        if not _HAS_SCIPY:
            raise ImportError('Requires scipy.spatial module')
        try:
            assert isinstance(mask, Raster)
        except:
            raise TypeError('`mask` must be a Raster instance.')
        mask_overrides = {'dtype' : np.bool8, 'nodata' : False}
        kwargs.update(mask_overrides)
        mask = self._input_handler(mask, **kwargs)
        affine = mask.affine
        yi, xi = np.where(mask)
        xiyi = np.vstack([xi, yi])
        x, y = affine * xiyi
        tree_xy = np.column_stack([x, y])
        tree = scipy.spatial.cKDTree(tree_xy)
        dist, ix = tree.query(xy)
        if return_dist:
            return tree_xy[ix], dist
        else:
            return tree_xy[ix]

    def _input_handler(self, data, **kwargs):
        try:
            assert (isinstance(data, Raster))
        except:
            raise TypeError('Data must be a Raster.')
        dataset = self.view(data, data_view=data.viewfinder, target_view=self.viewfinder,
                            **kwargs)
        return dataset

    def _output_handler(self, data, viewfinder, metadata={}, **kwargs):
        new_view = ViewFinder(**viewfinder.properties)
        for param, value in kwargs.items():
            if (value is not None) and (hasattr(new_view, param)):
                setattr(new_view, param, value)
        dataset = Raster(data, new_view, metadata=metadata)
        return dataset

    def _get_nodata_cells(self, data):
        try:
            assert (isinstance(data, Raster))
        except:
            raise TypeError('Data must be a Raster.')
        nodata = data.nodata
        if np.isnan(nodata):
            nodata_cells = np.isnan(data).astype(np.bool8)
        else:
            nodata_cells = (data == nodata).astype(np.bool8)
        return nodata_cells

    def _pop_rim(self, data, nodata=0):
        left, right, top, bottom = (data[:,0].copy(), data[:,-1].copy(),
                                    data[0,:].copy(), data[-1,:].copy())
        data[:,0] = nodata
        data[:,-1] = nodata
        data[0,:] = nodata
        data[-1,:] = nodata
        return left, right, top, bottom

    def _replace_rim(self, data, left, right, top, bottom):
        data[:,0] = left
        data[:,-1] = right
        data[0,:] = top
        data[-1,:] = bottom
        return None
