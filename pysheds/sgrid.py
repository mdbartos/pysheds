import sys
import ast
import copy
import warnings
import pyproj
import numpy as np
import pandas as pd
from numba import njit, prange
from numba.types import float64, int64, uint32, uint16, uint8, boolean, UniTuple, Tuple, List, void
import geojson
from affine import Affine
from distutils.version import LooseVersion
try:
    import scipy.sparse
    import scipy.spatial
    from scipy.sparse import csgraph
    import scipy.interpolate
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False
try:
    import skimage.measure
    import skimage.transform
    import skimage.morphology
    _HAS_SKIMAGE = True
except:
    _HAS_SKIMAGE = False
try:
    import rasterio
    import rasterio.features
    _HAS_RASTERIO = True
except:
    _HAS_RASTERIO = False
from pysheds.pgrid import Grid

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_crs = lambda Proj: Proj.crs if not _OLD_PYPROJ else Proj
_pyproj_crs_is_geographic = 'is_latlong' if _OLD_PYPROJ else 'is_geographic'
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

from pysheds.sview import Raster
from pysheds.view import RegularViewFinder, IrregularViewFinder
from pysheds.view import IrregularGridViewer
from pysheds.sview import View, ViewFinder

class sGrid(Grid):
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

    def read_ascii(self, data, skiprows=6, mask=None, crs=pyproj.Proj(_pyproj_init),
                   xll='lower', yll='lower', metadata={}, **kwargs):
        """
        Reads data from an ascii file into a named attribute of Grid
        instance (name of attribute determined by 'data_name').
 
        Parameters
        ----------
        data : str
               File name or path.
        data_name : str
                    Name of dataset. Will determine the name of the attribute
                    representing the gridded data.
        skiprows : int (optional)
                   The number of rows taken up by the header (defaults to 6).
        crs : pyroj.Proj
              Coordinate reference system of ascii data.
        xll : 'lower' or 'center' (str)
              Whether XLLCORNER or XLLCENTER is used.
        yll : 'lower' or 'center' (str)
              Whether YLLCORNER or YLLCENTER is used.
        metadata : dict
                   Other attributes describing dataset, such as direction
                   mapping for flow direction files. e.g.:
                   metadata={'dirmap' : (64, 128, 1, 2, 4, 8, 16, 32),
                             'routing' : 'd8'}
 
        Additional keyword arguments are passed to numpy.loadtxt()
        """
        with open(data) as header:
            ncols = int(header.readline().split()[1])
            nrows = int(header.readline().split()[1])
            xll = ast.literal_eval(header.readline().split()[1])
            yll = ast.literal_eval(header.readline().split()[1])
            cellsize = ast.literal_eval(header.readline().split()[1])
            nodata = ast.literal_eval(header.readline().split()[1])
            shape = (nrows, ncols)
        data = np.loadtxt(data, skiprows=skiprows, **kwargs)
        nodata = data.dtype.type(nodata)
        affine = Affine(cellsize, 0., xll, 0., -cellsize, yll + nrows * cellsize)
        viewfinder = ViewFinder(affine=affine, shape=shape, mask=mask, nodata=nodata, crs=crs)
        out = Raster(data, viewfinder, metadata=metadata)
        return out

    def read_raster(self, data, band=1, window=None, window_crs=None,
                    metadata={}, mask_geometry=False, **kwargs):
        """
        Reads data from a raster file into a named attribute of Grid
        (name of attribute determined by keyword 'data_name').
 
        Parameters
        ----------
        data : str
               File name or path.
        data_name : str
                    Name of dataset. Will determine the name of the attribute
                    representing the gridded data.
        band : int
               The band number to read if multiband.
        window : tuple
                 If using windowed reading, specify window (xmin, ymin, xmax, ymax).
        window_crs : pyproj.Proj instance
                     Coordinate reference system of window. If None, assume it's in raster's crs.
        mask_geometry : iterable object
                        The values must be a GeoJSON-like dict or an object that implements
                        the Python geo interface protocol (such as a Shapely Polygon).
        metadata : dict
                   Other attributes describing dataset, such as direction
                   mapping for flow direction files. e.g.:
                   metadata={'dirmap' : (64, 128, 1, 2, 4, 8, 16, 32),
                             'routing' : 'd8'}
 
        Additional keyword arguments are passed to rasterio.open()
        """
        # read raster file
        if not _HAS_RASTERIO:
            raise ImportError('Requires rasterio module')
        mask = None
        with rasterio.open(data, **kwargs) as f:
            crs = pyproj.Proj(f.crs, preserve_units=True)
            if window is None:
                shape = f.shape
                if len(f.indexes) > 1:
                    data = np.ma.filled(f.read_band(band))
                else:
                    data = np.ma.filled(f.read())
                affine = f.transform
                data = data.reshape(shape)
            else:
                if window_crs is not None:
                    if window_crs.srs != crs.srs:
                        xmin, ymin, xmax, ymax = window
                        if _OLD_PYPROJ:
                            extent = pyproj.transform(window_crs, crs, (xmin, xmax),
                                                    (ymin, ymax))
                        else:
                            extent = pyproj.transform(window_crs, crs, (xmin, xmax),
                                                      (ymin, ymax), errcheck=True,
                                                      always_xy=True)
                        window = (extent[0][0], extent[1][0], extent[0][1], extent[1][1])
                # If window crs not specified, assume it's in raster crs
                ix_window = f.window(*window)
                if len(f.indexes) > 1:
                    data = np.ma.filled(f.read_band(band, window=ix_window))
                else:
                    data = np.ma.filled(f.read(window=ix_window))
                affine = f.window_transform(ix_window)
                data = np.squeeze(data)
                shape = data.shape
            if mask_geometry:
                mask = rasterio.features.geometry_mask(mask_geometry, shape, affine, invert=True)
                if not mask.any():  # no mask was applied if all False, out of bounds
                    warnings.warn('mask_geometry does not fall within the bounds of the raster!')
                    mask = ~mask  # return mask to all True and deliver warning
            nodata = f.nodatavals[0]
        if nodata is not None:
            nodata = data.dtype.type(nodata)
        viewfinder = ViewFinder(affine=affine, shape=shape, mask=mask, nodata=nodata, crs=crs)
        out = Raster(data, viewfinder, metadata=metadata)
        return out

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
        # If no data view is provided, use dataset's viewfinder
        if data_view is None:
            data_view = data.viewfinder
        # If no target view is provided, use grid's viewfinder
        if target_view is None:
            target_view = self.viewfinder
        out = View.view(data, data_view, target_view,
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
        fdir = _d8_flowdir_numba(dem, dx, dy, dirmap, nodata_cells,
                                 nodata_out, flat=flats, pit=pits)
        return self._output_handler(data=fdir, viewfinder=dem.viewfinder,
                                    metadata=dem.metadata, nodata=nodata_out)

    def _dinf_flowdir(self, dem, nodata_cells, nodata_out=np.nan, flats=-1, pits=-2,
                      dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
        # Make sure nothing flows to the nodata cells
        dem[nodata_cells] = dem.max() + 1
        dx = abs(dem.affine.a)
        dy = abs(dem.affine.e)
        fdir = _dinf_flowdir_numba(dem, dx, dy, nodata_out, flat=flats, pit=pits)
        return self._output_handler(data=fdir, viewfinder=dem.viewfinder,
                                    metadata=dem.metadata, nodata=nodata_out)

    def catchment(self, x, y, fdir, pour_value=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                  nodata_out=None, xytype='coordinate', routing='d8', snap='corner', **kwargs):
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
                      nodata_out=None, xytype='coordinate', snap='corner'):
        # Pad the rim
        left, right, top, bottom = self._pop_rim(fdir, nodata=0)
        # If xytype is 'coordinate', delineate catchment based on cell nearest
        # to given geographic coordinate
        if xytype in {'label', 'coordinate'}:
            c, r = self.nearest_cell(x, y, fdir.affine, snap)
        # Delineate the catchment
        catch = _d8_catchment_numba(fdir, (r, c), dirmap)
        if pour_value is not None:
            catch[r, c] = pour_value
        catch = self._output_handler(data=catch, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return catch

    def _dinf_catchment(self, x, y, fdir, pour_value=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                        nodata_out=None, xytype='coordinate', snap='corner'):
        # Find nodata cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split dinf flowdir
        fdir_0, fdir_1, prop_0, prop_1 = _angle_to_d8(fdir, dirmap, nodata_cells)
        # Pad the rim
        left_0, right_0, top_0, bottom_0 = self._pop_rim(fdir_0, nodata=0)
        left_1, right_1, top_1, bottom_1 = self._pop_rim(fdir_1, nodata=0)
        # Valid if the dataset is a view.
        if xytype in {'label', 'coordinate'}:
            c, r = self.nearest_cell(x, y, fdir.affine, snap)
        # Delineate the catchment
        catch = _dinf_catchment_numba(fdir_0, fdir_1, (r, c), dirmap)
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
            input_overrides = {'dtype' : np.int64, 'nodata' : fdir.nodata}
        elif routing.lower() == 'dinf':
            input_overrides = {'dtype' : np.float64, 'nodata' : fdir.nodata}
        else:
            raise ValueError('Routing method must be one of: `d8`, `dinf`')
        kwargs.update(input_overrides)
        fdir = self._input_handler(fdir, **kwargs)
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
        endnodes = _flatten_fdir(fdir, dirmap).reshape(fdir.shape)
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
            acc = _d8_accumulation_numba(acc, endnodes, indegree, startnodes)
        else:
            acc = _d8_accumulation_eff_numba(acc, endnodes, indegree, startnodes, eff)
        acc = self._output_handler(data=acc, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return acc

    def _dinf_accumulation(self, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                           nodata_out=0., efficiency=None, cycle_size=1, **kwargs):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split d-infinity grid
        fdir_0, fdir_1, prop_0, prop_1 = _angle_to_d8(fdir, dirmap, nodata_cells)
        # Get matching of start and end nodes
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes_0 = _flatten_fdir(fdir_0, dirmap).reshape(fdir.shape)
        endnodes_1 = _flatten_fdir(fdir_1, dirmap).reshape(fdir.shape)
        # Remove cycles
        _dinf_fix_cycles_numba(endnodes_0, endnodes_1, cycle_size)
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
            acc = _dinf_accumulation_numba(acc, endnodes_0, endnodes_1, indegree,
                                           startnodes, prop_0, prop_1)
        else:
            acc = _dinf_accumulation_eff_numba(acc, endnodes_0, endnodes_1, indegree,
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
            c, r = self.nearest_cell(x, y, fdir.affine, snap)
        if weights is not None:
            weights = weights.reshape(fdir.shape).astype(np.float64)
        else:
            weights = (~nodata_cells).reshape(fdir.shape).astype(np.float64)
        dist = _d8_flow_distance_numba(fdir, weights, (r, c), dirmap)
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
        fdir_0, fdir_1, prop_0, prop_1 = _angle_to_d8(fdir, dirmap, nodata_cells)
        if xytype in {'label', 'coordinate'}:
            c, r = self.nearest_cell(x, y, fdir.affine, snap)
        if weights is not None:
            if isinstance(weights, list) or isinstance(weights, tuple):
                weights_0 = weights[0].reshape(fdir.shape).astype(np.float64)
                weights_1 = weights[1].reshape(fdir.shape).astype(np.float64)
            elif isinstance(weights, np.ndarray):
                weights_0 = weights[:,0].reshape(fdir.shape).astype(np.float64)
                weights_1 = weights[:,1].reshape(fdir.shape).astype(np.float64)
        else:
            weights_0 = (~nodata_cells).reshape(fdir.shape).astype(np.float64)
            weights_1 = weights_0
        if method.lower() == 'shortest':
            dist = _dinf_flow_distance_numba(fdir_0, fdir_1, weights_0,
                                                weights_1, (r, c), dirmap)
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
            hand = _assign_hand_heights_numba(hand, dem, nodata_out)
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
        hand = _d8_hand_iter_numba(fdir, mask, dirmap)
        hand = self._output_handler(data=hand, viewfinder=fdir.viewfinder,
                                    metadata=fdir.metadata, nodata=nodata_out)
        return hand

    def _dinf_compute_hand(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                         nodata_out=-1):
        # Get nodata cells
        nodata_cells = self._get_nodata_cells(fdir)
        # Split dinf flowdir
        fdir_0, fdir_1, prop_0, prop_1 = _angle_to_d8(fdir, dirmap, nodata_cells)
        # Pad the rim
        dirleft_0, dirright_0, dirtop_0, dirbottom_0 = self._pop_rim(fdir_0,
                                                                     nodata=0)
        dirleft_1, dirright_1, dirtop_1, dirbottom_1 = self._pop_rim(fdir_1,
                                                                     nodata=0)
        maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=False)
        hand = _dinf_hand_iter_numba(fdir_0, fdir_1, mask, dirmap)
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
        flats, fdirs_defined, higher_cells = _par_get_candidates(dem, inside)
        # Label all flats
        labels, numlabels = skimage.measure.label(flats, return_num=True)
        # Get high-edge cells
        hec = _par_get_high_edge_cells(inside, fdirs_defined, higher_cells, labels)
        # Get low-edge cells
        lec = _par_get_low_edge_cells(inside, dem, fdirs_defined, labels, numlabels)
        # Construct gradient from higher terrain
        grad_from_higher = _grad_from_higher(hec, flats, labels, numlabels, max_iter)
        # Construct gradient towards lower terrain
        grad_towards_lower = _grad_towards_lower(lec, flats, dem, max_iter)
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
        endnodes = _flatten_fdir(masked_fdir, dirmap).reshape(fdir.shape)
        indegree = np.bincount(endnodes.ravel(), minlength=fdir.size).astype(np.uint8)
        orig_indegree = np.copy(indegree)
        startnodes = startnodes[(indegree == 0)]
        profiles = _d8_stream_network_numba(endnodes, indegree, orig_indegree, startnodes)
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
        endnodes = _flatten_fdir(masked_fdir, dirmap).reshape(fdir.shape)
        indegree = np.bincount(endnodes.ravel()).astype(np.uint8)
        orig_indegree = np.copy(indegree)
        startnodes = startnodes[(indegree == 0)]
        min_order = np.full(fdir.shape, np.iinfo(np.int64).max, dtype=np.int64)
        max_order = np.ones(fdir.shape, dtype=np.int64)
        order = np.where(mask, 1, 0).astype(np.int64).reshape(fdir.shape)
        order = _d8_streamorder_numba(min_order, max_order, order, endnodes,
                                        indegree, orig_indegree, startnodes)
        order = self._output_handler(data=order, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return order

    def reverse_distance(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
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
        maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=0)
        masked_fdir = np.where(mask, fdir, 0).astype(np.int64)
        startnodes = np.arange(fdir.size, dtype=np.int64)
        endnodes = _flatten_fdir(masked_fdir, dirmap).reshape(fdir.shape)
        indegree = np.bincount(endnodes.ravel()).astype(np.uint8)
        orig_indegree = np.copy(indegree)
        startnodes = startnodes[(indegree == 0)]
        min_order = np.full(fdir.shape, np.iinfo(np.int64).max, dtype=np.int64)
        max_order = np.ones(fdir.shape, dtype=np.int64)
        # TODO: Weights not implemented
        rdist = np.zeros(fdir.shape, dtype=np.float64)
        rdist = _d8_reverse_distance_numba(min_order, max_order, rdist,
                                            endnodes, indegree, startnodes)
        rdist = self._output_handler(data=rdist, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return rdist

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
        pits = _find_pits_numba(dem, inside)
        pit_indices = np.flatnonzero(pits).astype(np.int64)
        # Create new array to hold pit-filled dem
        pit_filled_dem = dem.copy().astype(np.float64)
        # Fill pits
        _fill_pits_numba(pit_filled_dem, pit_indices)
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
        pits = _find_pits_numba(dem, inside)
        pits = self._output_handler(data=pits, viewfinder=dem.viewfinder,
                                    metadata=dem.metadata, nodata=None)
        return pits

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
        flats, _, _ = _par_get_candidates(dem, inside)
        flats = self._output_handler(data=flats, viewfinder=dem.viewfinder,
                                     metadata=dem.metadata, nodata=None)
        return flats

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

    def _sanitize_fdir(self, fdir):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        return fdir

# Functions for 'flowdir'

@njit(int64[:,:](float64[:,:], float64, float64, UniTuple(int64, 8), boolean[:,:],
                 int64, int64, int64),
      parallel=True,
      cache=True)
def _d8_flowdir_numba(dem, dx, dy, dirmap, nodata_cells, nodata_out, flat=-1, pit=-2):
    fdir = np.zeros(dem.shape, dtype=np.int64)
    m, n = dem.shape
    dd = np.sqrt(dx**2 + dy**2)
    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([dy, dd, dx, dd, dy, dd, dx, dd])
    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
            if nodata_cells[i, j]:
                fdir[i, j] = nodata_out
            else:
                elev = dem[i, j]
                max_slope = -np.inf
                for k in range(8):
                    row_offset = row_offsets[k]
                    col_offset = col_offsets[k]
                    distance = distances[k]
                    slope = (elev - dem[i + row_offset, j + col_offset]) / distance
                    if slope > max_slope:
                        fdir[i, j] = dirmap[k]
                        max_slope = slope
                if max_slope == 0:
                    fdir[i, j] = flat
                elif max_slope < 0:
                    fdir[i, j] = pit
    return fdir

@njit(int64[:,:](float64[:,:], float64[:,:], float64[:,:], UniTuple(int64, 8), boolean[:,:],
                 int64, int64, int64),
      parallel=True,
      cache=True)
def _d8_flowdir_irregular_numba(dem, x_arr, y_arr, dirmap, nodata_cells,
                                nodata_out, flat=-1, pit=-2):
    fdir = np.zeros(dem.shape, dtype=np.int64)
    m, n = dem.shape
    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
            if nodata_cells[i, j]:
                fdir[i, j] = nodata_out
            else:
                elev = dem[i, j]
                x_center = x_arr[i, j]
                y_center = y_arr[i, j]
                max_slope = -np.inf
                for k in range(8):
                    row_offset = row_offsets[k]
                    col_offset = col_offsets[k]
                    dh = elev - dem[i + row_offset, j + col_offset]
                    dx = np.abs(x_center - x_arr[i + row_offset, j + col_offset])
                    dy = np.abs(y_center - y_arr[i + row_offset, j + col_offset])
                    distance = np.sqrt(dx**2 + dy**2)
                    slope = dh / distance
                    if slope > max_slope:
                        fdir[i, j] = dirmap[k]
                        max_slope = slope
                if max_slope == 0:
                    fdir[i, j] = flat
                elif max_slope < 0:
                    fdir[i, j] = pit
    return fdir

@njit(UniTuple(float64, 2)(float64, float64, float64, float64, float64),
      cache=True)
def _facet_flow(e0, e1, e2, d1=1., d2=1.):
    s1 = (e0 - e1) / d1
    s2 = (e1 - e2) / d2
    r = np.arctan2(s2, s1)
    s = np.hypot(s1, s2)
    diag_angle    = np.arctan2(d2, d1)
    diag_distance = np.hypot(d1, d2)
    b0 = (r < 0)
    b1 = (r > diag_angle)
    if b0:
        r = 0
        s = s1
    if b1:
        r = diag_angle
        s = (e0 - e2) / diag_distance
    return r, s

@njit(float64[:,:](float64[:,:], float64, float64, float64, float64, float64),
      parallel=True,
      cache=True)
def _dinf_flowdir_numba(dem, x_dist, y_dist, nodata, flat=-1., pit=-2.):
    m, n = dem.shape
    e1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    e2s = np.array([1, 1, 3, 3, 5, 5, 7, 7])
    d1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    d2s = np.array([2, 0, 4, 2, 6, 4, 0, 6])
    ac = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    af = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    angle = np.full(dem.shape, nodata, dtype=np.float64)
    diag_dist = np.sqrt(x_dist**2 + y_dist**2)
    cell_dists = np.array([x_dist, diag_dist, y_dist, diag_dist,
                           x_dist, diag_dist, y_dist, diag_dist])
    row_offsets = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    col_offsets = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
            e0 = dem[i, j]
            s_max = -np.inf
            k_max = 8
            r_max = 0.
            for k in prange(8):
                edge_1 = e1s[k]
                edge_2 = e2s[k]
                row_offset_1 = row_offsets[edge_1]
                row_offset_2 = row_offsets[edge_2]
                col_offset_1 = col_offsets[edge_1]
                col_offset_2 = col_offsets[edge_2]
                e1 = dem[i + row_offset_1, j + col_offset_1]
                e2 = dem[i + row_offset_2, j + col_offset_2]
                distance_1 = d1s[k]
                distance_2 = d2s[k]
                d1 = cell_dists[distance_1]
                d2 = cell_dists[distance_2]
                r, s = _facet_flow(e0, e1, e2, d1, d2)
                if s > s_max:
                    s_max = s
                    k_max = k
                    r_max = r
            if s_max < 0:
                angle[i, j] = pit
            elif s_max == 0:
                angle[i, j] = flat
            else:
                flow_angle = (af[k_max] * r_max) + (ac[k_max] * np.pi / 2)
                flow_angle = flow_angle % (2 * np.pi)
                angle[i, j] = flow_angle
    return angle

@njit(float64[:,:](float64[:,:], float64[:,:], float64[:,:], float64, float64, float64),
      parallel=True,
      cache=True)
def _dinf_flowdir_irregular_numba(dem, x_arr, y_arr, nodata, flat=-1., pit=-2.):
    m, n = dem.shape
    e1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    e2s = np.array([1, 1, 3, 3, 5, 5, 7, 7])
    d1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    d2s = np.array([2, 0, 4, 2, 6, 4, 0, 6])
    ac = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    af = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    angle = np.full(dem.shape, nodata, dtype=np.float64)
    row_offsets = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    col_offsets = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
            e0 = dem[i, j]
            x0 = x_arr[i, j]
            y0 = y_arr[i, j]
            s_max = -np.inf
            k_max = 8
            r_max = 0.
            for k in prange(8):
                edge_1 = e1s[k]
                edge_2 = e2s[k]
                row_offset_1 = row_offsets[edge_1]
                row_offset_2 = row_offsets[edge_2]
                col_offset_1 = col_offsets[edge_1]
                col_offset_2 = col_offsets[edge_2]
                e1 = dem[i + row_offset_1, j + col_offset_1]
                e2 = dem[i + row_offset_2, j + col_offset_2]
                x1 = x_arr[i + row_offset_1, j + col_offset_1]
                x2 = x_arr[i + row_offset_2, j + col_offset_2]
                y1 = y_arr[i + row_offset_1, j + col_offset_1]
                y2 = y_arr[i + row_offset_2, j + col_offset_2]
                d1 = np.sqrt(x1**2 + y1**2)
                d2 = np.sqrt(x2**2 + y2**2)
                r, s = _facet_flow(e0, e1, e2, d1, d2)
                if s > s_max:
                    s_max = s
                    k_max = k
                    r_max = r
            if s_max < 0:
                angle[i, j] = pit
            elif s_max == 0:
                angle[i, j] = flat
            else:
                flow_angle = (af[k_max] * r_max) + (ac[k_max] * np.pi / 2)
                flow_angle = flow_angle % (2 * np.pi)
                angle[i, j] = flow_angle
    return angle

@njit(Tuple((int64[:,:], int64[:,:], float64[:,:], float64[:,:]))
      (float64[:,:], UniTuple(int64, 8), boolean[:,:]),
      parallel=True,
      cache=True)
def _angle_to_d8(angles, dirmap, nodata_cells):
    n = angles.size
    min_angle = 0.
    max_angle = 2 * np.pi
    mod = np.pi / 4
    c0_order = np.array([2, 1, 0, 7, 6, 5, 4, 3])
    c1_order = np.array([1, 0, 7, 6, 5, 4, 3, 2])
    c0 = np.zeros(8, dtype=np.uint8)
    c1 = np.zeros(8, dtype=np.uint8)
    # Need to watch typing of fdir_0 and fdir_1
    fdirs_0 = np.zeros(angles.shape, dtype=np.int64)
    fdirs_1 = np.zeros(angles.shape, dtype=np.int64)
    props_0 = np.zeros(angles.shape, dtype=np.float64)
    props_1 = np.zeros(angles.shape, dtype=np.float64)
    for i in range(8):
        c0[i] = dirmap[c0_order[i]]
        c1[i] = dirmap[c1_order[i]]
    for i in prange(n):
        angle = angles.flat[i]
        nodata = nodata_cells.flat[i]
        if np.isnan(angle) or nodata:
            zfloor = 8
            prop_0 = 0
            prop_1 = 0
            fdir_0 = 0
            fdir_1 = 0
        elif (angle < min_angle) or (angle > max_angle):
            zfloor = 8
            prop_0 = 0
            prop_1 = 0
            fdir_0 = 0
            fdir_1 = 0
        else:
            zmod = angle % mod
            zfloor = int(angle // mod)
            prop_1 = (zmod / mod)
            prop_0 = 1 - prop_1
            fdir_0 = c0[zfloor]
            fdir_1 = c1[zfloor]
        # Handle case where flow proportion is zero in either direction
        if (prop_0 == 0):
            fdir_0 = fdir_1
            prop_0 = 0.5
            prop_1 = 0.5
        elif (prop_1 == 0):
            fdir_1 = fdir_0
            prop_0 = 0.5
            prop_1 = 0.5
        fdirs_0.flat[i] = fdir_0
        fdirs_1.flat[i] = fdir_1
        props_0.flat[i] = prop_0
        props_1.flat[i] = prop_1
    return fdirs_0, fdirs_1, props_0, props_1

# Functions for 'catchment'

@njit(void(int64, boolean[:,:], int64[:,:], int64[:], int64[:]),
      cache=True)
def _d8_catchment_recursion(ix, catch, fdir, offsets, r_dirmap):
    visited = catch.flat[ix]
    if not visited:
        catch.flat[ix] = True
        neighbors = offsets + ix
        for k in range(8):
            neighbor = neighbors[k]
            points_to = (fdir.flat[neighbor] == r_dirmap[k])
            if points_to:
                _d8_catchment_recursion(neighbor, catch, fdir, offsets, r_dirmap)

@njit(boolean[:,:](int64[:,:], UniTuple(int64, 2), UniTuple(int64, 8)),
      cache=True)
def _d8_catchment_numba(fdir, pour_point, dirmap):
    catch = np.zeros(fdir.shape, dtype=np.bool8)
    offset = fdir.shape[1]
    i, j = pour_point
    ix = (i * offset) + j
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset,
                        offset, - 1 + offset, - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])
    _d8_catchment_recursion(ix, catch, fdir, offsets, r_dirmap)
    return catch

@njit(void(int64, boolean[:,:], int64[:,:], int64[:,:], int64[:], int64[:]),
      cache=True)
def _dinf_catchment_recursion(ix, catch, fdir_0, fdir_1, offsets, r_dirmap):
    visited = catch.flat[ix]
    if not visited:
        catch.flat[ix] = True
        neighbors = offsets + ix
        for k in range(8):
            neighbor = neighbors[k]
            points_to_0 = (fdir_0.flat[neighbor] == r_dirmap[k])
            points_to_1 = (fdir_1.flat[neighbor] == r_dirmap[k])
            points_to = points_to_0 or points_to_1
            if points_to:
                _dinf_catchment_recursion(neighbor, catch, fdir_0, fdir_1, offsets, r_dirmap)

@njit(boolean[:,:](int64[:,:], int64[:,:], UniTuple(int64, 2), UniTuple(int64, 8)),
      cache=True)
def _dinf_catchment_numba(fdir_0, fdir_1, pour_point, dirmap):
    catch = np.zeros(fdir_0.shape, dtype=np.bool8)
    dirmap = np.array(dirmap)
    offset = fdir_0.shape[1]
    i, j = pour_point
    ix = (i * offset) + j
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])
    _dinf_catchment_recursion(ix, catch, fdir_0, fdir_1, offsets, r_dirmap)
    return catch

# Functions for 'accumulation'

@njit(void(int64, int64, float64[:,:], int64[:,:], uint8[:]),
      cache=True)
def _d8_accumulation_recursion(startnode, endnode, acc, fdir, indegree):
    acc.flat[endnode] += acc.flat[startnode]
    indegree[endnode] -= 1
    if (indegree[endnode] == 0):
        new_startnode = endnode
        new_endnode = fdir.flat[endnode]
        _d8_accumulation_recursion(new_startnode, new_endnode, acc, fdir, indegree)

@njit(float64[:,:](float64[:,:], int64[:,:], uint8[:], int64[:]),
      cache=True)
def _d8_accumulation_numba(acc, fdir, indegree, startnodes):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes[k]
        endnode = fdir.flat[startnode]
        _d8_accumulation_recursion(startnode, endnode, acc, fdir, indegree)
    return acc

@njit(void(int64, int64, float64[:,:], int64[:,:], uint8[:], float64[:,:]),
      cache=True)
def _d8_accumulation_eff_recursion(startnode, endnode, acc, fdir, indegree, eff):
    acc.flat[endnode] += (acc.flat[startnode] * eff.flat[startnode])
    indegree[endnode] -= 1
    if (indegree[endnode] == 0):
        new_startnode = endnode
        new_endnode = fdir.flat[endnode]
        _d8_accumulation_eff_recursion(new_startnode, new_endnode, acc, fdir, indegree, eff)

@njit(float64[:,:](float64[:,:], int64[:,:], uint8[:], int64[:], float64[:,:]),
      cache=True)
def _d8_accumulation_eff_numba(acc, fdir, indegree, startnodes, eff):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes[k]
        endnode = fdir.flat[startnode]
        _d8_accumulation_eff_recursion(startnode, endnode, acc, fdir, indegree, eff)
    return acc

@njit(void(int64, int64, float64[:,:], int64[:,:], int64[:,:], uint8[:], float64,
           boolean[:,:], float64[:,:], float64[:,:]),
      cache=True)
def _dinf_accumulation_recursion(startnode, endnode, acc, fdir_0, fdir_1,
                                indegree, prop, visited, props_0, props_1):
    acc.flat[endnode] += (prop * acc.flat[startnode])
    indegree.flat[endnode] -= 1
    visited.flat[startnode] = True
    if (indegree.flat[endnode] == 0):
        new_startnode = endnode
        new_endnode_0 = fdir_0.flat[new_startnode]
        new_endnode_1 = fdir_1.flat[new_startnode]
        prop_0 = props_0.flat[new_startnode]
        prop_1 = props_1.flat[new_startnode]
        _dinf_accumulation_recursion(new_startnode, new_endnode_0, acc, fdir_0, fdir_1,
                                     indegree, prop_0, visited, props_0, props_1)
        _dinf_accumulation_recursion(new_startnode, new_endnode_1, acc, fdir_0, fdir_1,
                                     indegree, prop_1, visited, props_0, props_1)

@njit(float64[:,:](float64[:,:], int64[:,:], int64[:,:], uint8[:], int64[:],
                   float64[:,:], float64[:,:]),
      cache=True)
def _dinf_accumulation_numba(acc, fdir_0, fdir_1, indegree, startnodes,
                             props_0, props_1):
    n = startnodes.size
    visited = np.zeros(acc.shape, dtype=np.bool8)
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode_0 = fdir_0.flat[startnode]
        endnode_1 = fdir_1.flat[startnode]
        prop_0 = props_0.flat[startnode]
        prop_1 = props_1.flat[startnode]
        _dinf_accumulation_recursion(startnode, endnode_0, acc, fdir_0, fdir_1,
                                     indegree, prop_0, visited, props_0, props_1)
        _dinf_accumulation_recursion(startnode, endnode_1, acc, fdir_0, fdir_1,
                                     indegree, prop_1, visited, props_0, props_1)
        # TODO: Needed?
        visited.flat[startnode] = True
    return acc

@njit(void(int64, int64, float64[:,:], int64[:,:], int64[:,:], uint8[:], float64,
           boolean[:,:], float64[:,:], float64[:,:], float64[:,:]),
      cache=True)
def _dinf_accumulation_eff_recursion(startnode, endnode, acc, fdir_0, fdir_1,
                                     indegree, prop, visited, props_0, props_1, eff):
    acc.flat[endnode] += (prop * acc.flat[startnode] * eff.flat[startnode])
    indegree.flat[endnode] -= 1
    visited.flat[startnode] = True
    if (indegree.flat[endnode] == 0):
        new_startnode = endnode
        new_endnode_0 = fdir_0.flat[new_startnode]
        new_endnode_1 = fdir_1.flat[new_startnode]
        prop_0 = props_0.flat[new_startnode]
        prop_1 = props_1.flat[new_startnode]
        _dinf_accumulation_eff_recursion(new_startnode, new_endnode_0, acc, fdir_0, fdir_1,
                                         indegree, prop_0, visited, props_0, props_1, eff)
        _dinf_accumulation_eff_recursion(new_startnode, new_endnode_1, acc, fdir_0, fdir_1,
                                         indegree, prop_1, visited, props_0, props_1, eff)

@njit(float64[:,:](float64[:,:], int64[:,:], int64[:,:], uint8[:], int64[:],
                   float64[:,:], float64[:,:], float64[:,:]),
      cache=True)
def _dinf_accumulation_eff_numba(acc, fdir_0, fdir_1, indegree, startnodes,
                                 props_0, props_1, eff):
    n = startnodes.size
    visited = np.zeros(acc.shape, dtype=np.bool8)
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode_0 = fdir_0.flat[startnode]
        endnode_1 = fdir_1.flat[startnode]
        prop_0 = props_0.flat[startnode]
        prop_1 = props_1.flat[startnode]
        _dinf_accumulation_eff_recursion(startnode, endnode_0, acc, fdir_0, fdir_1,
                                         indegree, prop_0, visited, props_0, props_1, eff)
        _dinf_accumulation_eff_recursion(startnode, endnode_1, acc, fdir_0, fdir_1,
                                         indegree, prop_1, visited, props_0, props_1, eff)
        # TODO: Needed?
        visited.flat[startnode] = True
    return acc

# Functions for 'flow_distance'

@njit(void(int64, int64[:,:], boolean[:,:], float64[:,:], float64[:,:],
           int64[:], float64, int64[:]),
      cache=True)
def _d8_flow_distance_recursion(ix, fdir, visits, dist, weights, r_dirmap,
                                inc, offsets):
    visited = visits.flat[ix]
    if not visited:
        visits.flat[ix] = True
        dist.flat[ix] = inc
        neighbors = offsets + ix
        for k in range(8):
            neighbor = neighbors[k]
            points_to = (fdir.flat[neighbor] == r_dirmap[k])
            if points_to:
                next_inc = inc + weights.flat[neighbor]
                _d8_flow_distance_recursion(neighbor, fdir, visits, dist, weights,
                                            r_dirmap, next_inc, offsets)

@njit(float64[:,:](int64[:,:], float64[:,:], UniTuple(int64, 2), UniTuple(int64, 8)),
      cache=True)
def _d8_flow_distance_numba(fdir, weights, pour_point, dirmap):
    visits = np.zeros(fdir.shape, dtype=np.bool8)
    dist = np.full(fdir.shape, np.inf, dtype=np.float64)
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])
    m, n = fdir.shape
    offsets = np.array([-n, 1 - n, 1,
                        1 + n, n, - 1 + n,
                        - 1, - 1 - n])
    i, j = pour_point
    ix = (i * n) + j
    _d8_flow_distance_recursion(ix, fdir, visits, dist, weights,
                                r_dirmap, 0., offsets)
    return dist

@njit(void(int64, int64[:,:], int64[:,:], boolean[:,:], float64[:,:],
           float64[:,:], float64[:,:], int64[:], float64, int64[:]),
      cache=True)
def _dinf_flow_distance_recursion(ix, fdir_0, fdir_1, visits, dist,
                                  weights_0, weights_1, r_dirmap, inc, offsets):
    current_dist = dist.flat[ix]
    if (inc < current_dist):
        dist.flat[ix] = inc
        neighbors = offsets + ix
        for k in range(8):
            neighbor = neighbors[k]
            points_to_0 = (fdir_0.flat[neighbor] == r_dirmap[k])
            points_to_1 = (fdir_1.flat[neighbor] == r_dirmap[k])
            if points_to_0:
                next_inc = inc + weights_0.flat[neighbor]
                _dinf_flow_distance_recursion(neighbor, fdir_0, fdir_1, visits, dist,
                                              weights_0, weights_1, r_dirmap, next_inc,
                                              offsets)
            elif points_to_1:
                next_inc = inc + weights_1.flat[neighbor]
                _dinf_flow_distance_recursion(neighbor, fdir_0, fdir_1, visits, dist,
                                              weights_0, weights_1, r_dirmap, next_inc,
                                              offsets)

@njit(float64[:,:](int64[:,:], int64[:,:], float64[:,:], float64[:,:],
                   UniTuple(int64, 2), UniTuple(int64, 8)),
      cache=True)
def _dinf_flow_distance_numba(fdir_0, fdir_1, weights_0, weights_1,
                              pour_point, dirmap):
    visits = np.zeros(fdir_0.shape, dtype=np.bool8)
    dist = np.full(fdir_0.shape, np.inf, dtype=np.float64)
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])
    m, n = fdir_0.shape
    offsets = np.array([-n, 1 - n, 1,
                        1 + n, n, - 1 + n,
                        - 1, - 1 - n])
    i, j = pour_point
    ix = (i * n) + j
    _dinf_flow_distance_recursion(ix, fdir_0, fdir_1, visits, dist,
                                  weights_0, weights_1, r_dirmap, 0., offsets)
    return dist

@njit(void(int64, int64, int64[:,:], int64[:,:], float64[:,:], int64[:,:], uint8[:]),
      cache=True)
def _d8_reverse_distance_recursion(startnode, endnode, min_order, max_order,
                                   rdist, fdir, indegree):
    min_order.flat[endnode] = min(min_order.flat[endnode], rdist.flat[startnode])
    max_order.flat[endnode] = max(max_order.flat[endnode], rdist.flat[startnode])
    indegree.flat[endnode] -= 1
    if indegree.flat[endnode] == 0:
        rdist.flat[endnode] = max_order.flat[endnode] + 1
        new_startnode = endnode
        new_endnode = fdir.flat[new_startnode]
        _d8_reverse_distance_recursion(new_startnode, new_endnode, min_order,
                                       max_order, rdist, fdir, indegree)

@njit(float64[:,:](int64[:,:], int64[:,:], float64[:,:], int64[:,:], uint8[:], int64[:]),
      cache=True)
def _d8_reverse_distance_numba(min_order, max_order, rdist, fdir, indegree, startnodes):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        _d8_reverse_distance_recursion(startnode, endnode, min_order, max_order,
                                       rdist, fdir, indegree)
    return rdist

# Functions for 'resolve_flats'

@njit(UniTuple(boolean[:,:], 3)(float64[:,:], int64[:]),
      parallel=True,
      cache=True)
def _par_get_candidates(dem, inside):
    n = inside.size
    offset = dem.shape[1]
    fdirs_defined = np.zeros(dem.shape, dtype=np.bool8)
    flats = np.zeros(dem.shape, dtype=np.bool8)
    higher_cells = np.zeros(dem.shape, dtype=np.bool8)
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    for i in prange(n):
        k = inside[i]
        inner_neighbors = (k + offsets)
        fdir_defined = False
        is_pit = True
        higher_cell = False
        same_elev_cell = False
        for j in prange(8):
            neighbor = inner_neighbors[j]
            diff = dem.flat[k] - dem.flat[neighbor]
            fdir_defined |= (diff > 0)
            is_pit &= (diff < 0)
            higher_cell |= (diff < 0)
        is_flat = (~fdir_defined & ~is_pit)
        fdirs_defined.flat[k] = fdir_defined
        flats.flat[k] = is_flat
        higher_cells.flat[k] = higher_cell
    fdirs_defined[0, :] = True
    fdirs_defined[:, 0] = True
    fdirs_defined[-1, :] = True
    fdirs_defined[:, -1] = True
    return flats, fdirs_defined, higher_cells

@njit(uint32[:,:](int64[:], boolean[:,:], boolean[:,:], int64[:,:]),
      parallel=True,
      cache=True)
def _par_get_high_edge_cells(inside, fdirs_defined, higher_cells, labels):
    n = inside.size
    high_edge_cells = np.zeros(fdirs_defined.shape, dtype=np.uint32)
    for i in range(n):
        k = inside[i]
        fdir_defined = fdirs_defined.flat[k]
        higher_cell = higher_cells.flat[k]
        # Find high-edge cells
        is_high_edge_cell = (~fdir_defined & higher_cell)
        if is_high_edge_cell:
            high_edge_cells.flat[k] = labels.flat[k]
    return high_edge_cells

@njit(uint32[:,:](int64[:], float64[:,:], boolean[:,:], int64[:,:], int64),
      parallel=True,
      cache=True)
def _par_get_low_edge_cells(inside, dem, fdirs_defined, labels, numlabels):
    n = inside.size
    offset = dem.shape[1]
    low_edge_cells = np.zeros(dem.shape, dtype=np.uint32)
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    for i in prange(n):
        k = inside[i]
        # Find low-edge cells
        inner_neighbors = (k + offsets)
        fdir_defined = fdirs_defined.flat[k]
        if (~fdir_defined):
            for j in range(8):
                neighbor = inner_neighbors[j]
                diff = dem.flat[k] - dem.flat[neighbor]
                is_same_elev = (diff == 0)
                neighbor_direction_defined = (fdirs_defined.flat[neighbor])
                neighbor_is_low_edge_cell = (is_same_elev) & (neighbor_direction_defined)
                if neighbor_is_low_edge_cell:
                    label = labels.flat[k]
                    low_edge_cells.flat[neighbor] = label
    return low_edge_cells

@njit(uint16[:,:](uint32[:,:], boolean[:,:], int64[:,:], int64, int64),
      cache=True)
def _grad_from_higher(hec, flats, labels, numlabels, max_iter=1000):
    offset = flats.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    z = np.zeros(flats.shape, dtype=np.uint16)
    n = z.size
    cur_queue = []
    next_queue = []
    # Increment gradient
    for i in range(n):
        if hec.flat[i]:
            z.flat[i] = 1
            cur_queue.append(i)
    for i in range(2, max_iter + 1):
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            neighbors = offsets + k
            for j in range(8):
                neighbor = neighbors[j]
                if (flats.flat[neighbor]) & (z.flat[neighbor] == 0):
                    z.flat[neighbor] = i
                    next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    # Invert gradient
    max_incs = np.zeros(numlabels + 1)
    for i in range(n):
        label = labels.flat[i]
        inc = z.flat[i]
        max_incs[label] = max(max_incs[label], inc)
    for i in range(n):
        if z.flat[i]:
            label = labels.flat[i]
            z.flat[i] = max_incs[label] - z.flat[i]
    return z

@njit(uint16[:,:](uint32[:,:], boolean[:,:], float64[:,:], int64),
      cache=True)
def _grad_towards_lower(lec, flats, dem, max_iter=1000):
    offset = flats.shape[1]
    size = flats.size
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    z = np.zeros(flats.shape, dtype=np.uint16)
    cur_queue = []
    next_queue = []
    for i in range(size):
        label = lec.flat[i]
        if label:
            z.flat[i] = 1
            cur_queue.append(i)
    for i in range(2, max_iter + 1):
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            on_left = ((k % offset) == 0)
            on_right = (((k + 1) % offset) == 0)
            on_top = (k < offset)
            on_bottom = (k > (size - offset - 1))
            on_boundary = (on_left | on_right | on_top | on_bottom)
            neighbors = offsets + k
            for j in range(8):
                if on_boundary:
                    if (on_left) & ((j == 5) | (j == 6) | (j == 7)):
                        continue
                    if (on_right) & ((j == 1) | (j == 2) | (j == 3)):
                        continue
                    if (on_top) & ((j == 0) | (j == 1) | (j == 7)):
                        continue
                    if (on_bottom) & ((j == 3) | (j == 4) | (j == 5)):
                        continue
                neighbor = neighbors[j]
                neighbor_is_flat = flats.flat[neighbor]
                not_visited = z.flat[neighbor] == 0
                same_elev = dem.flat[neighbor] == dem.flat[k]
                if (neighbor_is_flat & not_visited & same_elev):
                    z.flat[neighbor] = i
                    next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    return z

# Functions for 'compute_hand'

@njit(int64[:,:](int64[:,:], boolean[:,:], UniTuple(int64, 8)),
      cache=True)
def _d8_hand_iter_numba(fdir, mask, dirmap):
    offset = fdir.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])
    hand = -np.ones(fdir.shape, dtype=np.int64)
    cur_queue = []
    next_queue = []
    for i in range(hand.size):
        if mask.flat[i]:
            hand.flat[i] = i
            cur_queue.append(i)
    while True:
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            neighbors = offsets + k
            for j in range(8):
                neighbor = neighbors[j]
                points_to = (fdir.flat[neighbor] == r_dirmap[j])
                not_visited = (hand.flat[neighbor] < 0)
                if points_to and not_visited:
                    hand.flat[neighbor] = hand.flat[k]
                    next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    return hand

@njit(void(int64, int64, int64[:,:], int64[:], int64[:], int64[:,:]),
      cache=True)
def _d8_hand_recursion(child, parent, hand, offsets, r_dirmap, fdir):
    neighbors = offsets + child
    for k in range(8):
        neighbor = neighbors[k]
        points_to = (fdir.flat[neighbor] == r_dirmap[k])
        not_visited = (hand.flat[neighbor] == -1)
        if points_to and not_visited:
            hand.flat[neighbor] = parent
            _d8_hand_recursion(neighbor, parent, hand, offsets, r_dirmap, fdir)

@njit(int64[:,:](int64[:], int64[:,:], UniTuple(int64, 8)),
      cache=True)
def _d8_hand_recursive_numba(parents, fdir, dirmap):
    n = parents.size
    offset = fdir.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])
    hand = -np.ones(fdir.shape, dtype=np.int64)
    for i in range(n):
        parent = parents[i]
        hand.flat[parent] = parent
    for i in range(n):
        parent = parents[i]
        _d8_hand_recursion(parent, parent, hand, offsets, r_dirmap, fdir)
    return hand

@njit(int64[:,:](int64[:,:], int64[:,:], boolean[:,:], UniTuple(int64, 8)),
      cache=True)
def _dinf_hand_iter_numba(fdir_0, fdir_1, mask, dirmap):
    offset = fdir_0.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])
    hand = -np.ones(fdir_0.shape, dtype=np.int64)
    cur_queue = []
    next_queue = []
    for i in range(hand.size):
        if mask.flat[i]:
            hand.flat[i] = i
            cur_queue.append(i)
    while True:
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            neighbors = offsets + k
            for j in range(8):
                neighbor = neighbors[j]
                points_to = ((fdir_0.flat[neighbor] == r_dirmap[j]) |
                             (fdir_1.flat[neighbor] == r_dirmap[j]))
                not_visited = (hand.flat[neighbor] < 0)
                if points_to and not_visited:
                    hand.flat[neighbor] = hand.flat[k]
                    next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    return hand

@njit(void(int64, int64, int64[:,:], int64[:], int64[:], int64[:,:], int64[:,:]),
      cache=True)
def _dinf_hand_recursion(child, parent, hand, offsets, r_dirmap, fdir_0, fdir_1):
    neighbors = offsets + child
    for k in range(8):
        neighbor = neighbors[k]
        points_to = ((fdir_0.flat[neighbor] == r_dirmap[k]) |
                     (fdir_1.flat[neighbor] == r_dirmap[k]))
        not_visited = (hand.flat[neighbor] == -1)
        if points_to and not_visited:
            hand.flat[neighbor] = parent
            _dinf_hand_recursion(neighbor, parent, hand, offsets, r_dirmap, fdir_0, fdir_1)

@njit(int64[:,:](int64[:], int64[:,:], int64[:,:], UniTuple(int64, 8)),
      cache=True)
def _dinf_hand_recursive_numba(parents, fdir_0, fdir_1, dirmap):
    n = parents.size
    offset = fdir_0.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])
    hand = -np.ones(fdir_0.shape, dtype=np.int64)
    for i in range(n):
        parent = parents[i]
        hand.flat[parent] = parent
    for i in range(n):
        parent = parents[i]
        _dinf_hand_recursion(parent, parent, hand, offsets, r_dirmap, fdir_0, fdir_1)
    return hand

@njit(float64[:,:](int64[:,:], float64[:,:], float64),
      parallel=True,
      cache=True)
def _assign_hand_heights_numba(hand_idx, dem, nodata_out=np.nan):
    n = hand_idx.size
    hand = np.zeros(dem.shape, dtype=np.float64)
    for i in prange(n):
        j = hand_idx.flat[i]
        if j == -1:
            hand.flat[i] = np.nan
        else:
            hand.flat[i] = dem.flat[i] - dem.flat[j]
    return hand

# Functions for 'streamorder'

@njit(void(int64, int64, int64[:,:], int64[:,:], int64[:,:], int64[:,:], uint8[:], uint8[:]),
      cache=True)
def _d8_streamorder_recursion(startnode, endnode, min_order, max_order,
                              order, fdir, indegree, orig_indegree):
    min_order.flat[endnode] = min(min_order.flat[endnode], order.flat[startnode])
    max_order.flat[endnode] = max(max_order.flat[endnode], order.flat[startnode])
    indegree.flat[endnode] -= 1
    if indegree.flat[endnode] == 0:
        if (min_order.flat[endnode] == max_order.flat[endnode]) and (orig_indegree.flat[endnode] > 1):
            order.flat[endnode] = max_order.flat[endnode] + 1
        else:
            order.flat[endnode] = max_order.flat[endnode]
        new_startnode = endnode
        new_endnode = fdir.flat[new_startnode]
        _d8_streamorder_recursion(new_startnode, new_endnode, min_order,
                                  max_order, order, fdir, indegree, orig_indegree)

@njit(int64[:,:](int64[:,:], int64[:,:], int64[:,:], int64[:,:], uint8[:], uint8[:], int64[:]),
      cache=True)
def _d8_streamorder_numba(min_order, max_order, order, fdir,
                          indegree, orig_indegree, startnodes):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        _d8_streamorder_recursion(startnode, endnode, min_order, max_order, order,
                                 fdir, indegree, orig_indegree)
    return order

@njit(void(int64, int64, int64[:,:], uint8[:], uint8[:], List(List(int64)), List(int64)),
      cache=True)
def _d8_stream_network_recursion(startnode, endnode, fdir, indegree,
                                 orig_indegree, profiles, profile):
    profile.append(endnode)
    if (orig_indegree[endnode] > 1):
        profiles.append(profile)
    indegree.flat[endnode] -= 1
    if (indegree.flat[endnode] == 0):
        if (orig_indegree[endnode] > 1):
            profile = [endnode]
        new_startnode = endnode
        new_endnode = fdir.flat[new_startnode]
        _d8_stream_network_recursion(new_startnode, new_endnode, fdir, indegree,
                                     orig_indegree, profiles, profile)

@njit(List(List(int64))(int64[:,:], uint8[:], uint8[:], int64[:]),
      cache=True)
def _d8_stream_network_numba(fdir, indegree, orig_indegree, startnodes):
    n = startnodes.size
    profiles = [[0]]
    _ = profiles.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        profile = [startnode]
        _d8_stream_network_recursion(startnode, endnode, fdir, indegree,
                                     orig_indegree, profiles, profile)
    return profiles

@njit(parallel=True)
def _d8_cell_dh(startnodes, endnodes, dem):
    n = startnodes.size
    dh = np.zeros_like(dem)
    for k in prange(n):
        startnode = startnodes.flat[k]
        endnode = endnodes.flat[k]
        dh.flat[k] = dem.flat[startnode] - dem.flat[endnode]
    return dh

@njit(parallel=True)
def _dinf_cell_dh(startnodes, endnodes_0, endnodes_1, props_0, props_1, dem):
    n = startnodes.size
    dh = np.zeros(dem.shape, dtype=np.float64)
    for k in prange(n):
        startnode = startnodes.flat[k]
        endnode_0 = endnodes_0.flat[k]
        endnode_1 = endnodes_1.flat[k]
        prop_0 = props_0.flat[k]
        prop_1 = props_1.flat[k]
        dh.flat[k] = (prop_0 * (dem.flat[startnode] - dem.flat[endnode_0]) +
                      prop_1 * (dem.flat[startnode] - dem.flat[endnode_1]))
    return dh

@njit(void(int64, int64[:,:], int64[:,:], int64, int64, int64, boolean[:,:]),
      cache=True)
def _dinf_fix_cycles_recursion(node, fdir_0, fdir_1, ancestor,
                               depth, max_cycle_size, visited):
    if visited.flat[node]:
        return None
    if depth > max_cycle_size:
        return None
    left = fdir_0.flat[node]
    right = fdir_1.flat[node]
    if left == ancestor:
        fdir_0.flat[node] = right
        return None
    else:
        _dinf_fix_cycles_recursion(left, fdir_0, fdir_1, ancestor,
                                   depth + 1, max_cycle_size, visited)
    if right == ancestor:
        fdir_1.flat[node] = left
        return None
    else:
        _dinf_fix_cycles_recursion(right, fdir_0, fdir_1, ancestor,
                                   depth + 1, max_cycle_size, visited)

@njit(void(int64[:,:], int64[:,:], int64),
      cache=True)
def _dinf_fix_cycles_numba(fdir_0, fdir_1, max_cycle_size):
    n = fdir_0.size
    visited = np.zeros(fdir_0.shape, dtype=np.bool8)
    depth = 0
    for node in range(n):
        _dinf_fix_cycles_recursion(node, fdir_0, fdir_1, node,
                                   depth, max_cycle_size, visited)
        visited.flat[node] = True

# TODO: Assumes pits and flats are removed
@njit(int64[:,:](int64[:,:], UniTuple(int64, 8)),
      parallel=True,
      cache=True)
def _flatten_fdir(fdir, dirmap):
    r, c = fdir.shape
    n = fdir.size
    flat_fdir = np.zeros((r, c), dtype=np.int64)
    offsets = ( 0 - c,
                1 - c,
                1 + 0,
                1 + c,
                0 + c,
               -1 + c,
               -1 + 0,
               -1 - c
              )
    offset_map = {0 : 0}
    left_map = {0 : 0}
    right_map = {0 : 0}
    top_map = {0 : 0}
    bottom_map = {0 : 0}
    for i in range(8):
        # Inside cells
        offset_map[dirmap[i]] = offsets[i]
        # Left boundary
        if i in {5, 6, 7}:
            left_map[dirmap[i]] = 0
        else:
            left_map[dirmap[i]] = offsets[i]
        # Right boundary
        if i in {1, 2, 3}:
            right_map[dirmap[i]] = 0
        else:
            right_map[dirmap[i]] = offsets[i]
        # Top boundary
        if i in {7, 0, 1}:
            top_map[dirmap[i]] = 0
        else:
            top_map[dirmap[i]] = offsets[i]
        # Bottom boundary
        if i in {3, 4, 5}:
            bottom_map[dirmap[i]] = 0
        else:
            bottom_map[dirmap[i]] = offsets[i]
    for k in prange(n):
        cell_dir = fdir.flat[k]
        on_left = ((k % c) == 0)
        on_right = (((k + 1) % c) == 0)
        on_top = (k < c)
        on_bottom = (k > (n - c - 1))
        on_boundary = (on_left | on_right | on_top | on_bottom)
        if on_boundary:
            if on_left:
                offset = left_map[cell_dir]
            if on_right:
                offset = right_map[cell_dir]
            if on_top:
                offset = top_map[cell_dir]
            if on_bottom:
                offset = bottom_map[cell_dir]
        else:
            offset = offset_map[cell_dir]
        flat_fdir.flat[k] = k + offset
    return flat_fdir

@njit(int64[:,:](int64[:,:], UniTuple(int64, 8)),
      parallel=True,
      cache=True)
def _flatten_fdir_no_boundary(fdir, dirmap):
    r, c = fdir.shape
    n = fdir.size
    flat_fdir = np.zeros((r, c), dtype=np.int64)
    offsets = ( 0 - c,
                1 - c,
                1 + 0,
                1 + c,
                0 + c,
               -1 + c,
               -1 + 0,
               -1 - c
              )
    offset_map = {0 : 0}
    for i in range(8):
        offset_map[dirmap[i]] = offsets[i]
    for k in prange(n):
        cell_dir = fdir.flat[k]
        offset = offset_map[cell_dir]
        flat_fdir.flat[k] = k + offset
    return flat_fdir

@njit
def _construct_matching(fdir, dirmap):
    n = fdir.size
    startnodes = np.arange(n, dtype=np.int64)
    endnodes = _flatten_fdir(fdir, dirmap).ravel()
    return startnodes, endnodes

@njit(boolean[:,:](float64[:,:], int64[:]),
      parallel=True,
      cache=True)
def _find_pits_numba(dem, inside):
    n = inside.size
    offset = dem.shape[1]
    pits = np.zeros(dem.shape, dtype=np.bool8)
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    for i in prange(n):
        k = inside[i]
        inner_neighbors = (k + offsets)
        is_pit = True
        for j in prange(8):
            neighbor = inner_neighbors[j]
            diff = dem.flat[k] - dem.flat[neighbor]
            is_pit &= (diff < 0)
        pits.flat[k] = is_pit
    return pits

@njit(float64[:,:](float64[:,:], int64[:]),
      parallel=True,
      cache=True)
def _fill_pits_numba(dem, pit_indices):
    n = pit_indices.size
    offset = dem.shape[1]
    pits_filled = np.copy(dem).astype(np.float64)
    max_diff = dem.max() - dem.min()
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    for i in prange(n):
        k = pit_indices[i]
        inner_neighbors = (k + offsets)
        adjustment = max_diff
        for j in prange(8):
            neighbor = inner_neighbors[j]
            diff = dem.flat[neighbor] - dem.flat[k]
            adjustment = min(diff, adjustment)
        pits_filled.flat[k] += (adjustment)
    return pits_filled
