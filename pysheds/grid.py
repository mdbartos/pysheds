import sys
import ast
import copy
import warnings
import pyproj
import numpy as np
import pandas as pd
import geojson
from affine import Affine
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
    _HAS_SKIMAGE = True
except:
    _HAS_SKIMAGE = False
try:
    import rasterio
    import rasterio.features
    _HAS_RASTERIO = True
except:
    _HAS_RASTERIO = False

from pysheds.view import Raster
from pysheds.view import BaseViewFinder, RegularViewFinder, IrregularViewFinder
from pysheds.view import RegularGridViewer, IrregularGridViewer

class Grid(object):
    """
    Container class for holding and manipulating gridded data.
 
    Attributes
    ==========
    affine : Affine transformation matrix (uses affine module)
    shape : The shape of the grid (number of rows, number of columns).
    bbox : The geographical bounding box of the current view of the gridded data
           (xmin, ymin, xmax, ymax).
    cellsize : The length/width of each grid cell (assumed to be square).
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

    def __init__(self, affine=Affine(0,0,0,0,0,0), shape=(1,1), nodata=0,
                 crs=pyproj.Proj('+init=epsg:4326'),
                 mask=None):
        self.affine = affine
        self.shape = shape
        self.nodata = nodata
        self.crs = crs
        if mask is None:
            self.mask = np.ones(shape)
        self.grids = []

    @property
    def defaults(self):
        props = {
            'affine' : Affine(0,0,0,0,0,0),
            'shape' : (1,1),
            'nodata' : 0,
            'crs' : pyproj.Proj('+init=epsg:4326'),
        }
        return props

    def add_gridded_data(self, data, data_name, affine=None, shape=None, crs=None,
                         nodata=None, mask=None, metadata={}):
        """
        A generic method for adding data into a Grid instance.
        Inserts data into a named attribute of Grid (name of attribute
        determined by keyword 'data_name').
 
        Parameters
        ----------
        data : numpy ndarray
               Data to be inserted into Grid instance.
        data_name : str
                    Name of dataset. Will determine the name of the attribute
                    representing the gridded data.
        affine : affine.Affine
                 Affine transformation matrix defining the cell size and bounding
                 box (see the affine module for more information).
        shape : tuple of int (length 2)
                Shape (rows, columns) of data.
        crs : dict
              Coordinate reference system of gridded data.
        nodata : int or float
                 Value indicating no data in the input array.
        metadata : dict
                   Other attributes describing dataset, such as direction
                   mapping for flow direction files. e.g.:
                   metadata={'dirmap' : (1, 2, 3, 4, 5, 6, 7, 8),
                             'routing' : 'd8'}
        """
        if mask is None:
            mask = np.ones(shape, dtype=np.bool)
        if not isinstance(data, np.ndarray):
            raise TypeError('Input data must be ndarray')
        # if there are no datasets, initialize bbox, shape,
        # cellsize and crs based on incoming data
        if len(self.grids) < 1:
            # check validity of shape
            if ((hasattr(shape, "__len__")) and (not isinstance(shape, str))
                    and (len(shape) == 2) and (isinstance(sum(shape), int))):
                shape = tuple(shape)
            else:
                raise TypeError('shape must be a tuple of ints of length 2.')
            if crs is not None:
                if isinstance(crs, pyproj.Proj):
                    pass
                elif isinstance(crs, dict) or isinstance(crs, str):
                    crs = pyproj.Proj(crs)
                else:
                    raise TypeError('Valid crs required')
            if isinstance(affine, Affine):
                pass
            else:
                raise TypeError('affine transformation matrix required')

            # initialize instance metadata
            self.affine = affine
            self.shape = shape
            self.crs = crs
            self.nodata = nodata
            self.mask = mask
        # assign new data to attribute; record nodata value
        viewfinder = RegularViewFinder(affine=affine, shape=shape, mask=mask, nodata=nodata,
                                       crs=crs)
        data = Raster(data, viewfinder, metadata=metadata)
        self.grids.append(data_name)
        setattr(self, data_name, data)

    def read_ascii(self, data, data_name, skiprows=6, crs=pyproj.Proj('+init=epsg:4326'),
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
                   metadata={'dirmap' : (1, 2, 3, 4, 5, 6, 7, 8),
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
        affine = Affine(cellsize, 0, xll, 0, -cellsize, yll + nrows * cellsize)
        self.add_gridded_data(data=data, data_name=data_name, affine=affine, shape=shape,
                              crs=crs, nodata=nodata, metadata=metadata)

    def read_raster(self, data, data_name, band=1, window=None, window_crs=None,
                    metadata={}, **kwargs):
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
               The band number to read.
        window : tuple
                 If using windowed reading, specify window (xmin, ymin, xmax, ymax).
        window_crs : pyproj.Proj instance
                     Coordinate reference system of window. If None, assume it's in raster's crs.
        metadata : dict
                   Other attributes describing dataset, such as direction
                   mapping for flow direction files. e.g.:
                   metadata={'dirmap' : (1, 2, 3, 4, 5, 6, 7, 8),
                             'routing' : 'd8'}
 
        Additional keyword arguments are passed to rasterio.open()
        """
        # read raster file
        if not _HAS_RASTERIO:
            raise ImportError('Requires rasterio module')
        with rasterio.open(data, **kwargs) as f:
            crs = pyproj.Proj(f.crs, preserve_units=True)
            if window is None:
                shape = f.shape
                if len(f.indexes) > 1:
                    data = np.ma.filled(f.read_band(band))
                else:
                    data = np.ma.filled(f.read())
                affine = f.affine
            else:
                if window_crs is not None:
                    if window_crs.srs != crs.srs:
                        xmin, ymin, xmax, ymax = window
                        extent = pyproj.transform(window_crs, crs, (xmin, xmax),
                                                  (ymin, ymax))
                        window = (extent[0][0], extent[1][0], extent[0][1], extent[1][1])
                # If window crs not specified, assume it's in raster crs
                ix_window = f.window(*window)
                shape = (ix_window[0][1] - ix_window[0][0],
                         ix_window[1][1] - ix_window[1][0])
                if len(f.indexes) > 1:
                    data = np.ma.filled(f.read_band(band, window=ix_window))
                else:
                    data = np.ma.filled(f.read(window=ix_window))
                affine = f.window_transform(ix_window)
            nodata = f.nodatavals[0]
            data = data.reshape(shape)
        if nodata is not None:
            nodata = data.dtype.type(nodata)
        self.add_gridded_data(data=data, data_name=data_name, affine=affine, shape=shape,
                              crs=crs, nodata=nodata, metadata=metadata)

    @classmethod
    def from_ascii(cls, path, data_name, **kwargs):
        newinstance = cls()
        newinstance.read_ascii(path, data_name, **kwargs)
        return newinstance

    @classmethod
    def from_raster(cls, path, data_name, **kwargs):
        newinstance = cls()
        newinstance.read_raster(path, data_name, **kwargs)
        return newinstance

    def grid_indices(self, affine=None, shape=None, col_ascending=True, row_ascending=False):
        """
        Return row and column coordinates of the grid based on an affine transformation and
        a grid shape.
 
        Parameters
        ----------
        affine: affine.Affine
                Affine transformation matrix. Defualts to self.affine.
        shape : tuple of ints (length 2)
                The shape of the 2D array (rows, columns). Defaults
                to self.shape.
        col_ascending : bool
                        If True, return column coordinates in ascending order.
        row_ascending : bool
                        If True, return row coordinates in ascending order.
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

    def view(self, data, data_view=None, target_view=None, apply_mask=True,
             nodata=None, interpolation='nearest', as_crs=None, return_coords=False,
             kx=3, ky=3, s=0, tolerance=1e-3, dtype=None, metadata={}):
        """
        Return a copy of a gridded dataset clipped to the current "view". The view is determined by
        an affine transformation which describes the bounding box and cellsize of the grid.
        The view will also optionally mask grid cells according to the boolean array self.mask.
 
        Parameters
        ----------
        data : str or Raster
               If string: name of the dataset to be viewed.
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
        # Check interpolation method
        try:
            interpolation = interpolation.lower()
            assert(interpolation in ('nearest', 'linear', 'cubic', 'spline'))
        except:
            raise ValueError("Interpolation method must be one of: "
                             "'nearest', 'linear', 'cubic', 'spline'")
        # Parse data
        if isinstance(data, str):
            data = getattr(self, data)
            if nodata is None:
                nodata = data.nodata
            if data_view is None:
                data_view = data.viewfinder
            metadata.update(data.metadata)
        elif isinstance(data, Raster):
            if nodata is None:
                nodata = data.nodata
            if data_view is None:
                data_view = data.viewfinder
            metadata.update(data.metadata)
        else:
            # If not using a named dataset, make sure the data and view are properly defined
            try:
                assert(isinstance(data, np.ndarray))
            except:
                raise
            # TODO: Should convert array to dataset here
            if nodata is None:
                nodata = data_view.nodata
        # If no target view provided, construct one based on grid parameters
        if target_view is None:
            target_view = RegularViewFinder(affine=self.affine, shape=self.shape,
                                            mask=self.mask, crs=self.crs, nodata=nodata)
        # If viewing at a different crs, convert coordinates
        if as_crs is not None:
            assert(isinstance(as_crs, pyproj.Proj))
            target_coords = target_view.coords
            new_x, new_y = pyproj.transform(target_view.crs, as_crs,
                                            target_coords[:,1], target_coords[:,0])
            # TODO: In general, crs conversion will yield irregular grid (though not necessarily)
            target_view = IrregularViewFinder(coords=np.column_stack([new_y, new_x]),
                                            shape=target_view.shape, crs=as_crs,
                                            nodata=target_view.nodata)
        # Specify mask
        mask = target_view.mask
        # Make sure views are ViewFinder instances
        assert(issubclass(type(data_view), BaseViewFinder))
        assert(issubclass(type(target_view), BaseViewFinder))
        same_crs = target_view.crs.srs == data_view.crs.srs
        # If crs does not match, convert coords of data array to target array
        if not same_crs:
            data_coords = data_view.coords
            # TODO: x and y order might be different
            new_x, new_y = pyproj.transform(data_view.crs, target_view.crs,
                                            data_coords[:,1], data_coords[:,0])
            # TODO: In general, crs conversion will yield irregular grid (though not necessarily)
            data_view = IrregularViewFinder(coords=np.column_stack([new_y, new_x]),
                                            shape=data_view.shape, crs=target_view.crs,
                                            nodata=data_view.nodata)
        # Check if data can be described by regular grid
        data_is_grid = isinstance(data_view, RegularViewFinder)
        view_is_grid = isinstance(target_view, RegularViewFinder)
        # If data is on a grid, use the following speedup
        if data_is_grid and view_is_grid:
            # If doing nearest neighbor search, use fast sorted search
            if interpolation == 'nearest':
                array_view = RegularGridViewer._view_affine(data, data_view, target_view)
            # If spline interpolation is needed, use RectBivariate
            elif interpolation == 'spline':
                # If latitude/longitude, use RectSphereBivariate
                if target_view.crs.is_latlong():
                    array_view = RegularGridViewer._view_rectspherebivariate(data, data_view,
                                                                             target_view,
                                                                             x_tolerance=tolerance,
                                                                             y_tolerance=tolerance,
                                                                             kx=kx, ky=ky, s=s)
                # If not latitude/longitude, use RectBivariate
                else:
                    array_view = RegularGridViewer._view_rectbivariate(data, data_view,
                                                                       target_view,
                                                                       x_tolerance=tolerance,
                                                                       y_tolerance=tolerance,
                                                                       kx=kx, ky=ky, s=s)
            # If some other interpolation method is needed, use griddata
            else:
                array_view = IrregularGridViewer._view_griddata(data, data_view, target_view,
                                                                method=interpolation)
        # If either view is irregular, use griddata
        else:
            array_view = IrregularGridViewer._view_griddata(data, data_view, target_view,
                                                            method=interpolation)
        # TODO: This could be dangerous if it returns an irregular view
        array_view = Raster(array_view, target_view, metadata=metadata)
        # Ensure masking is safe by checking datatype
        if dtype is None:
            dtype = max(np.min_scalar_type(nodata), np.min_scalar_type(array_view.max()),
                        np.min_scalar_type(array_view.min()))
            # For matplotlib imshow compatibility
            if issubclass(dtype.type, np.floating):
                dtype = max(dtype, np.dtype(np.float32))
        array_view = array_view.astype(dtype)
        # Apply mask
        if apply_mask:
            np.place(array_view, ~mask, nodata)
        # Return output
        if return_coords:
            return array_view, target_view.coords
        else:
            return array_view

    def resize(self, data, new_shape, out_suffix='_resized', inplace=True,
               nodata_in=None, nodata_out=np.nan, apply_mask=False, ignore_metadata=True, **kwargs):
        """
        Resize a gridded dataset to a different shape (uses skimage.transform.resize).
        data : str or Raster
               If string: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        new_shape: tuple of int (length 2)
                   Desired array shape.
        out_suffix: str
                    If writing to a named attribute, the suffix to apply to the output name.
        inplace : bool
                  If True, resized array will be written to '<data_name>_<out_suffix>'.
                  Otherwise, return the output array.
        nodata_in : int or float
                    Value indicating no data in input array.
                    Defaults to the `nodata` attribute of the input dataset.
        nodata_out : int or float
                     Value indicating no data in output array.
                     Defaults to np.nan.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and crs.
        """
        nodata_in = self._check_nodata_in(data, nodata_in)
        if isinstance(data, str):
            out_name = '{0}{1}'.format(data, out_suffix)
        else:
            out_name = 'data_{1}'.format(out_suffix)
        grid_props = {'nodata' : nodata_out}
        metadata = {}
        data = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata,
                                   metadata=metadata)
        data = skimage.transform.resize(data, new_shape, **kwargs)
        return self._output_handler(data=data, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def nearest_cell(self, x, y, affine=None, shape=None):
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
        shape : tuple of int (length 2)
                Shape of the gridded data.
                Defaults to self.shape.
        """
        if not affine:
            bbox = self.affine
        if not shape:
            shape = self.shape
        col, row = np.around(~affine * (x, y)).astype(int)
        return col, row

    def flowdir(self, data=None, out_name='dir', nodata_in=None, nodata_out=0,
                pits=-1, flats=-1, dirmap=(1, 2, 3, 4, 5, 6, 7, 8), inplace=True,
                apply_mask=False, ignore_metadata=False, **kwargs):
        """
        Generates a flow direction grid from a DEM grid.
 
        Parameters
        ----------
        data : str or Raster
               DEM data.
               If string: name of the dataset to be viewed.
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
        inplace : bool
                  If True, write output array to self.<data_name>.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and crs.
        """
        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out}
        metadata = {'dirmap' : dirmap}
        dem = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                  properties=grid_props, ignore_metadata=ignore_metadata,
                                  **kwargs)
        if nodata_in is None:
            dem_mask = np.array([]).astype(int)
        else:
            if np.isnan(nodata_in):
                dem_mask = np.where(np.isnan(dem.ravel()))[0]
            else:
                dem_mask = np.where(dem.ravel() == nodata_in)[0]
        # Make sure nothing flows to the nodata cells
        dem.flat[dem_mask] = dem.max() + 1
        try:
            a = np.arange(dem.size)
            top = np.arange(dem.shape[1])[1:-1]
            left = np.arange(0, dem.size, dem.shape[1])
            right = np.arange(dem.shape[1] - 1, dem.size + 1, dem.shape[1])
            bottom = np.arange(dem.size - dem.shape[1], dem.size)[1:-1]
            exclude = np.unique(np.concatenate([top, left, right, bottom, dem_mask]))
            inside = np.delete(a, exclude)
            inner_neighbors, diff, fdir_defined = self._d8_diff(dem, inside)
            fdir = np.where(fdir_defined, np.argmax(diff, axis=0), -1) + 1
            if pits != flats:
                pits_bool = (diff < 0).all(axis=0)
                flats_bool = (~fdir_defined & ~pits)
                fdir[pits_bool] = pits
                fdir[flats_bool] = flats
            else:
                fdir[~fdir_defined] = flats
            # If direction numbering isn't default, convert values of output array.
            if dirmap != (1, 2, 3, 4, 5, 6, 7, 8):
                dir_d = dict(zip((1, 2, 3, 4, 5, 6, 7, 8), dirmap))
                for k, v in dir_d.items():
                    fdir[fdir == k] = v
            fdir_out = np.full(dem.shape, nodata_out)
            fdir_out.flat[inside] = fdir
        except:
            raise
        finally:
            if nodata_in is not None:
                dem.flat[dem_mask] = nodata_in
        return self._output_handler(data=fdir_out, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def catchment(self, x, y, data=None, pour_value=None, out_name='catch', dirmap=None,
                  nodata_in=None, nodata_out=0, xytype='index', recursionlimit=15000,
                  inplace=True, apply_mask=False, ignore_metadata=False, **kwargs):
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
               If string: name of the dataset to be viewed.
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
        recursionlimit : int
                         Recursion limit--may need to be raised if
                         recursion limit is reached.
        inplace : bool
                  If True, catchment will be written to attribute 'catch'.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and crs.
        """
        # Vectorized Recursive algorithm:
        # for each cell j, recursively search through grid to determine
        # if surrounding cells are in the contributing area, then add
        # flattened indices to self.collect

        def catchment_search(cells):
            nonlocal collect
            nonlocal cdir
            collect.extend(cells)
            selection = self._select_surround_ravel(cells, padshape)
            next_idx = selection[np.where(cdir[selection] == r_dirmap)]
            if next_idx.any():
                return catchment_search(next_idx)

        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out}
        metadata = {'dirmap' : dirmap}
        # initialize array to collect catchment cells
        cdir = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata,
                                   **kwargs)
        # Pad the rim
        left, right, top, bottom = self._pop_rim(cdir, nodata=nodata_in)
        try:
            # get shape of padded flow direction array, then flatten
            padshape = cdir.shape
            cdir = cdir.ravel()
            # if xytype is 'label', delineate catchment based on cell nearest
            # to given geographic coordinate
            # TODO: This relies on the bbox of the grid instance, not the dataset
            # Valid if the dataset is a view.
            if xytype == 'label':
                x, y = self.nearest_cell(x, y, cdir.affine, padshape)
            # get the flattened index of the pour point
            pour_point = np.ravel_multi_index(np.array([y, x]),
                                              padshape)
            # reorder direction mapping to work with select_surround_ravel()
            r_dirmap = np.array(dirmap)[[4, 5, 6, 7, 0, 1, 2, 3]].tolist()
            pour_point = np.array([pour_point])
            # set recursion limit (needed for large datasets)
            sys.setrecursionlimit(recursionlimit)
            # call catchment search starting at the pour point
            collect = []
            catchment_search(pour_point)
            # initialize output array
            outcatch = np.zeros(padshape, dtype=int)
            # if nodata is not 0, replace 0 with nodata value in output array
            if nodata_out != 0:
                np.place(outcatch, outcatch == 0, nodata_out)
            # set values of output array based on 'collected' cells
            outcatch.flat[collect] = cdir[collect]
            # if pour point needs to be a special value, set it
            if pour_value is not None:
                outcatch[y, x] = pour_value
        except:
            raise
        finally:
            # reset recursion limit
            sys.setrecursionlimit(1000)
            cdir = cdir.reshape(padshape)
            self._replace_rim(cdir, left, right, top, bottom)
        return self._output_handler(data=outcatch, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def fraction(self, other, nodata=0, out_name='frac', inplace=True):
        """
        Generates a grid representing the fractional contributing area for a
        coarse-scale flow direction grid.
 
        Parameters
        ----------
        other : Grid instance
                Another Grid instance containing fine-scale flow direction
                data. The ratio of self.cellsize/other.cellsize must be a
                positive integer. Grid cell boundaries must have some overlap.
                Must have attributes 'dir' and 'catch' (i.e. must have a flow
                direction grid, along with a delineated catchment).
        nodata : int or float
                 Value to indicate no data in output array.
        inplace : bool (optional)
                  If True, appends fraction grid to attribute 'frac'.
        """
        # check for required attributes in self and other
        assert hasattr(self, 'dir')
        assert hasattr(other, 'dir')
        assert hasattr(other, 'catch')
        # set scale ratio
        raw_ratio = self.cellsize / other.cellsize
        if np.allclose(int(round(raw_ratio)), raw_ratio):
            cell_ratio = int(round(raw_ratio))
        else:
            raise ValueError('Ratio of cell sizes must be an integer')
        # create DataFrames for self and other with geographic coordinates
        # as row and column labels. entries in selfdf represent cell indices.
        selfdf = pd.DataFrame(
            np.arange(self.view('dir', apply_mask=False).size).reshape(self.shape),
            index=np.linspace(self.bbox[1], self.bbox[3],
                              self.shape[0], endpoint=False)[::-1],
            columns=np.linspace(self.bbox[0], self.bbox[2],
                                self.shape[1], endpoint=False)
                )
        otherrows, othercols = self.grid_indices(other.affine, other.shape)
        # reindex self to other based on column labels and fill nulls with
        # nearest neighbor
        result = (selfdf.reindex(otherrows, method='nearest')
                  .reindex(othercols, axis=1, method='nearest'))
        initial_counts = np.bincount(result.values.ravel(),
                                     minlength=selfdf.size).astype(float)
        # mask cells not in catchment of 'other'
        result = result.values[np.where(other.view('catch') !=
            other.grid_props['catch']['nodata'], True, False)]
        final_counts = np.bincount(result, minlength=selfdf.size).astype(float)
        # count remaining indices and divide by the original number of indices
        result = (final_counts / initial_counts).reshape(selfdf.shape)
        # take care of nans
        if np.isnan(result).any():
            result = pd.DataFrame(result).fillna(0).values.astype(float)
        # replace 0 with nodata value
        if nodata != 0:
            np.place(result, result == 0, nodata)
        private_props = {'nodata' : nodata}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(result, inplace, out_name=out_name, **grid_props)

    def accumulation(self, data=None, weights=None, dirmap=None, nodata_in=None, nodata_out=0,
                     out_name='acc', inplace=True, pad=False, apply_mask=False,
                     ignore_metadata=False, **kwargs):
        """
        Generates an array of flow accumulation, where cell values represent
        the number of upstream cells.
 
        Parameters
        ----------
        data : str or Raster
               Flow direction data.
               If string: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        weights: numpy ndarray
                 Array of weights to be applied to each accumulation cell. Must
                 be same size as data.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                    Value to indicate nodata in input array. If using a named dataset, will
                    default to the 'nodata' value of the named dataset. If using an ndarray,
                    will default to 0.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        out_name : string
                   Name of attribute containing new accumulation array.
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        pad : bool
              If True, pad the rim of the input array with zeros. Else, ignore
              the outer rim of cells in the computation.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and crs.
        """
        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out}
        metadata = {}
        fdir = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props,
                                   ignore_metadata=ignore_metadata, **kwargs)
        # Pad the rim
        if pad:
            fdir = np.pad(fdir, (1,1), mode='constant', constant_values=0)
        else:
            left, right, top, bottom = self._pop_rim(fdir, nodata=0)
        fdir_orig_type = fdir.dtype
        # Construct flat index onto flow direction array
        flat_idx = np.arange(fdir.size)
        try:
            if nodata_in is None:
                nodata_cells = np.zeros_like(fdir).astype(bool)
            else:
                if np.isnan(nodata_in):
                    nodata_cells = (np.isnan(fdir))
                else:
                    nodata_cells = (fdir == nodata_in)
            # Ensure consistent types
            mintype = np.min_scalar_type(fdir.size)
            fdir = fdir.astype(mintype)
            flat_idx = flat_idx.astype(mintype)
            # Set nodata cells to zero
            fdir[nodata_cells] = 0
            # Get matching of start and end nodes
            startnodes, endnodes = self._construct_matching(fdir, flat_idx,
                                                            dirmap=dirmap)
            if weights is not None:
                assert(weights.size == fdir.size)
                acc = weights.flatten()
            else:
                acc = (~nodata_cells).ravel().astype(int)
            indegree = np.bincount(endnodes)
            level_0 = (indegree == 0)
            indegree = indegree.reshape(acc.shape).astype(np.uint8)
            startnodes = startnodes[level_0]
            endnodes = endnodes[level_0]
            for _ in range(fdir.size):
                if endnodes.any():
                    np.add.at(acc, endnodes, acc[startnodes])
                    np.subtract.at(indegree, endnodes, 1)
                    startnodes = np.unique(endnodes)
                    startnodes = startnodes[indegree[startnodes] == 0]
                    endnodes = fdir.flat[startnodes]
                else:
                    break
            # TODO: Hacky: should probably fix this
            acc[0] = 1
            # Reshape and offset accumulation
            acc = np.reshape(acc, fdir.shape)
            if pad:
                acc = acc[1:-1, 1:-1]
        except:
            raise
        finally:
        # Clean up
            self._unflatten_fdir(fdir, flat_idx, dirmap)
            fdir = fdir.astype(fdir_orig_type)
            if nodata_in is not None:
                fdir[nodata_cells] = nodata_in
            if pad:
                fdir = fdir[1:-1, 1:-1]
            else:
                self._replace_rim(fdir, left, right, top, bottom)
        return self._output_handler(data=acc, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def flow_distance(self, x, y, data, weights=None, dirmap=None, nodata_in=None,
                      nodata_out=0, out_name='dist', inplace=True,
                      xytype='index', apply_mask=True, ignore_metadata=False, **kwargs):
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
               If string: name of the dataset to be viewed.
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
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
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
        """
        if not _HAS_SCIPY:
            raise ImportError('flow_distance requires scipy.sparse module')
        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out}
        metadata = {}
        fdir = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata,
                                   **kwargs)
        # Construct flat index onto flow direction array
        flat_idx = np.arange(fdir.size)
        fdir_orig_type = fdir.dtype
        if nodata_in is None:
            nodata_cells = np.zeros_like(fdir).astype(bool)
        else:
            if np.isnan(nodata_in):
                nodata_cells = (np.isnan(fdir))
            else:
                nodata_cells = (fdir == nodata_in)
        try:
            mintype = np.min_scalar_type(fdir.size)
            fdir = fdir.astype(mintype)
            flat_idx = flat_idx.astype(mintype)
            startnodes, endnodes = self._construct_matching(fdir, flat_idx,
                                                            dirmap=dirmap)
            if xytype == 'label':
                x, y = self.nearest_cell(x, y, fdir.affine, fdir.shape)
            # TODO: Currently the size of weights is hard to understand
            if weights is not None:
                weights = weights.ravel()
                assert(weights.size == startnodes.size)
                assert(weights.size == endnodes.size)
            else:
                assert(startnodes.size == endnodes.size)
                weights = (~nodata_cells).ravel().astype(int)
            C = scipy.sparse.lil_matrix((fdir.size, fdir.size))
            for i,j,w in zip(startnodes, endnodes, weights):
                C[i,j] = w
            C = C.tocsr()
            xyindex = np.ravel_multi_index((y, x), fdir.shape)
            dist = csgraph.shortest_path(C, indices=[xyindex], directed=False)
            dist[~np.isfinite(dist)] = nodata_out
            dist = dist.ravel()
            dist = dist.reshape(fdir.shape)
        except:
            raise
        finally:
            self._unflatten_fdir(fdir, flat_idx, dirmap)
            fdir = fdir.astype(fdir_orig_type)
        # Prepare output
        return self._output_handler(data=dist, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def cell_area(self, out_name='area', nodata_out=0, inplace=True, as_crs=None):
        """
        Generates an array representing the area of each cell to the outlet.
 
        Parameters
        ----------
        out_name : string
                   Name of attribute containing new cell area array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        as_crs : pyproj.Proj
                 CRS at which to compute the area of each cell.
        """
        if as_crs is None:
            if self.crs.is_latlong():
                warnings.warn(('CRS is geographic. Area will not have meaningful '
                            'units.'))
        else:
            if as_crs.is_latlong():
                warnings.warn(('CRS is geographic. Area will not have meaningful '
                            'units.'))
        indices = np.vstack(np.dstack(np.meshgrid(*self.grid_indices(),
                                                  indexing='ij')))
        # TODO: Add to_crs conversion here
        if as_crs:
            indices = self._convert_grid_indices_crs(indices, self.crs, as_crs)
        dyy, dyx = np.gradient(indices[:, 0].reshape(self.shape))
        dxy, dxx = np.gradient(indices[:, 1].reshape(self.shape))
        dy = np.sqrt(dyy**2 + dyx**2)
        dx = np.sqrt(dxy**2 + dxx**2)
        area = dx * dy
        metadata = {}
        private_props = {'nodata' : nodata_out}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(data=area, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def cell_distances(self, data, out_name='cdist', dirmap=None, nodata_in=None, nodata_out=0,
                       inplace=True, as_crs=None, apply_mask=True, ignore_metadata=False):
        """
        Generates an array representing the distance from each cell to its downstream neighbor.
 
        Parameters
        ----------
        data : str or Raster
               Flow direction data.
               If string: name of the dataset to be viewed.
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
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        as_crs : pyproj.Proj
                 CRS at which to compute the distance from each cell to its downstream neighbor.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        if as_crs is None:
            if self.crs.is_latlong():
                warnings.warn(('CRS is geographic. Area will not have meaningful '
                            'units.'))
        else:
            if as_crs.is_latlong():
                warnings.warn(('CRS is geographic. Area will not have meaningful '
                            'units.'))
        indices = np.vstack(np.dstack(np.meshgrid(*self.grid_indices(),
                                                  indexing='ij')))
        if as_crs:
            indices = self._convert_grid_indices_crs(indices, self.crs, as_crs)
        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out}
        metadata = {}
        fdir = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata)
        dyy, dyx = np.gradient(indices[:, 0].reshape(self.shape))
        dxy, dxx = np.gradient(indices[:, 1].reshape(self.shape))
        dy = np.sqrt(dyy**2 + dyx**2)
        dx = np.sqrt(dxy**2 + dxx**2)
        ddiag = np.sqrt(dy**2 + dx**2)
        cdist = np.zeros(self.shape)
        for i, direction in enumerate(dirmap):
            if i in (0, 4):
                cdist[fdir == direction] = dy[fdir == direction]
            elif i in (2, 6):
                cdist[fdir == direction] = dx[fdir == direction]
            else:
                cdist[fdir == direction] = ddiag[fdir == direction]
        # Prepare output
        return self._output_handler(data=cdist, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def cell_dh(self, fdir, dem, out_name='dh', inplace=True, dirmap=None, nodata_in=None,
                nodata_out=np.nan, apply_mask=True, ignore_metadata=False):
        """
        Generates an array representing the elevation difference from each cell to its
        downstream neighbor.
 
        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If string: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        dem : str or Raster
              DEM data.
              If string: name of the dataset to be viewed.
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
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        nodata_in = self._check_nodata_in(fdir, nodata_in)
        fdir_props = {'nodata' : nodata_out}
        fdir = self._input_handler(fdir, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=fdir_props, ignore_metadata=ignore_metadata)
        nodata_in = self._check_nodata_in(dem, nodata_in)
        dem_props = {'nodata' : nodata_out}
        metadata = {}
        dem = self._input_handler(dem, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=dem_props, ignore_metadata=ignore_metadata)
        dirmap = self._set_dirmap(dirmap, fdir)
        flat_idx = np.arange(fdir.size)
        fdir_orig_type = fdir.dtype
        if nodata_in is None:
            nodata_cells = np.zeros_like(fdir).astype(bool)
        else:
            if np.isnan(nodata_in):
                nodata_cells = (np.isnan(fdir))
            else:
                nodata_cells = (fdir == nodata_in)
        try:
            mintype = np.min_scalar_type(fdir.size)
            fdir = fdir.astype(mintype)
            flat_idx = flat_idx.astype(mintype)
            startnodes, endnodes = self._construct_matching(fdir, flat_idx, dirmap)
            startelev = dem.ravel()[startnodes].astype(np.float64)
            endelev = dem.ravel()[endnodes].astype(np.float64)
            dh = (startelev - endelev).reshape(self.shape)
            dh[nodata_cells] = nodata_out
        except:
            raise
        finally:
            self._unflatten_fdir(fdir, flat_idx, dirmap)
            fdir = fdir.astype(fdir_orig_type)
        # Prepare output
        private_props = {'nodata' : nodata_out}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(data=dh, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def cell_slopes(self, fdir, dem, out_name='slopes', dirmap=None, nodata_in=None,
                    nodata_out=np.nan, as_crs=None, inplace=True, apply_mask=True,
                    ignore_metadata=False):
        """
        Generates an array representing the slope from each cell to its downstream neighbor.
 
        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If string: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        dem : str or Raster
              DEM data.
              If string: name of the dataset to be viewed.
              If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new cell slope array.
        as_crs : pyproj.Proj
                 CRS at which to compute the distance from each cell to its downstream neighbor.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        dh = self.cell_dh(fdir, dem, out_name, inplace=False,
                          nodata_out=nodata_out, dirmap=dirmap)
        cdist = self.cell_distances(fdir, inplace=False, as_crs=as_crs)
        if apply_mask:
            slopes = np.where(self.mask, dh/cdist, nodata_out)
        else:
            slopes = dh/cdist
        # Prepare output
        metadata = {}
        private_props = {'nodata' : nodata_out}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(data=slopes, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)

    def _check_nodata_in(self, data, nodata_in, override=None):
        if nodata_in is None:
            if isinstance(data, str):
                try:
                    nodata_in = getattr(self, data).viewfinder.nodata
                except:
                    raise NameError("nodata value for '{0}' not found in instance."
                                    .format(data))
        if override is not None:
            nodata_in = override
        return nodata_in

    def _input_handler(self, data, apply_mask=True, nodata_view=None, properties={},
                       ignore_metadata=False, inherit_metadata=True,  metadata={}, **kwargs):
        required_params = ('affine', 'shape', 'nodata', 'crs')
        defaults = self.defaults
        # Handle raw data
        if (isinstance(data, np.ndarray) or isinstance(data, Raster)):
            for param in required_params:
                if not param in properties:
                    if param in kwargs:
                        properties[param] = kwargs[param]
                    elif ignore_metadata:
                        properties[param] = defaults[param]
                    else:
                        raise KeyError("Missing required parameter: {0}"
                                       .format(param))
            if isinstance(data, Raster):
                if inherit_metadata:
                    metadata.update(data.metadata)
            viewfinder = RegularViewFinder(**properties)
            dataset = Raster(data, viewfinder, metadata=metadata)
            return dataset
        # Handle named dataset
        elif isinstance(data, str):
            for param in required_params:
                if not param in properties:
                    if param in kwargs:
                        properties[param] = kwargs[param]
                    elif hasattr(self, param):
                        properties[param] = getattr(self, param)
                    elif ignore_metadata:
                        properties[param] = defaults[param]
                    else:
                        raise KeyError("Missing required parameter: {0}"
                                        .format(param))
            viewfinder = RegularViewFinder(**properties)
            data = self.view(data, apply_mask=apply_mask, nodata=nodata_view)
            if inherit_metadata:
                metadata.update(data.metadata)
            dataset = Raster(data, viewfinder, metadata=metadata)
            return dataset
        else:
            raise TypeError('Data must be a numpy ndarray or name string.')

    def _output_handler(self, data, out_name, properties, inplace, metadata={}):
        # TODO: Should this be rolled into add_data?
        viewfinder = RegularViewFinder(**properties)
        dataset = Raster(data, viewfinder, metadata=metadata)
        if inplace:
            setattr(self, out_name, dataset)
            self.grids.append(out_name)
        else:
            return dataset

    def _generate_grid_props(self, **kwargs):
        properties = {}
        required = ('affine', 'shape', 'nodata', 'crs')
        properties.update(kwargs)
        for param in required:
            properties[param] = properties.setdefault(param,
                                                      getattr(self, param))
        return properties

    def _pop_rim(self, data, nodata=0):
        # TODO: Does this default make sense?
        if nodata is None:
            nodata = 0
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

    def _dy_dx(self):
        x0, y0, x1, y1 = self.bbox
        dy = np.abs(y1 - y0) / (self.shape[0]) #TODO: Should this be shape - 1?
        dx = np.abs(x1 - x0) / (self.shape[1]) #TODO: Should this be shape - 1?
        return dy, dx

    def _convert_bbox_crs(self, bbox, old_crs, new_crs):
        # TODO: Won't necessarily work in every case as ur might be lower than
        # ul
        x1 = np.asarray((bbox[0], bbox[2]))
        y1 = np.asarray((bbox[1], bbox[3]))
        x2, y2 = pyproj.transform(old_crs, new_crs,
                                  x1, y1)
        new_bbox = (x2[0], y2[0], x2[1], y2[1])
        return new_bbox

    def _convert_grid_indices_crs(self, affine, shape, old_crs, new_crs):
        y1, x1 = self.grid_indices(affine=affine, shape=shape)
        yx1 = np.vstack(np.dstack(np.meshgrid(y1, x1, indexing='ij')))
        yx2 = self._convert_grid_indices_crs(yx1, old_crs, new_crs)
        return yx2

    def _convert_grid_indices_crs(self, grid_indices, old_crs, new_crs):
        x2, y2 = pyproj.transform(old_crs, new_crs, grid_indices[:,1],
                                  grid_indices[:,0])
        yx2 = np.column_stack([y2, x2])
        return yx2

    def _convert_outer_indices_crs(self, affine, shape, old_crs, new_crs):
        y1, x1 = self.grid_indices(affine=affine, shape=shape)
        lx, _ = pyproj.transform(old_crs, new_crs,
                                  x1, np.repeat(y1[0], len(x1)))
        rx, _ = pyproj.transform(old_crs, new_crs,
                                  x1, np.repeat(y1[-1], len(x1)))
        __, by = pyproj.transform(old_crs, new_crs,
                                  np.repeat(x1[0], len(y1)), y1)
        __, uy = pyproj.transform(old_crs, new_crs,
                                  np.repeat(x1[-1], len(y1)), y1)
        return by, uy, lx, rx

    def _flatten_fdir(self, fdir, flat_idx, dirmap, copy=False):
        # WARNING: This modifies fdir in place if copy is set to False!
        if copy:
            fdir = fdir.copy()
        shape = fdir.shape
        go_to = (
             0 - shape[1],
             1 - shape[1],
             1 + 0,
             1 + shape[1],
             0 + shape[1],
            -1 + shape[1],
            -1 + 0,
            -1 - shape[1]
            )
        gotomap = dict(zip(dirmap, go_to))
        for k, v in gotomap.items():
            fdir[fdir == k] = v
        fdir.flat[flat_idx] += flat_idx

    def _unflatten_fdir(self, fdir, flat_idx, dirmap):
        shape = fdir.shape
        go_to = (
             0 - shape[1],
             1 - shape[1],
             1 + 0,
             1 + shape[1],
             0 + shape[1],
            -1 + shape[1],
            -1 + 0,
            -1 - shape[1]
            )
        gotomap = dict(zip(go_to, dirmap))
        fdir.flat[flat_idx] -= flat_idx
        for k, v in gotomap.items():
            fdir[fdir == k] = v

    def _construct_matching(self, fdir, flat_idx, dirmap, fdir_flattened=False):
        # TODO: Maybe fdir should be flattened outside this function
        if not fdir_flattened:
            self._flatten_fdir(fdir, flat_idx, dirmap)
        startnodes = flat_idx
        endnodes = fdir.flat[flat_idx]
        return startnodes, endnodes

    def clip_to(self, data_name, precision=7, inplace=True, apply_mask=True, pad=(0,0,0,0),
                **kwargs):
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
 
        Other keyword arguments are passed to self.set_bbox
        """
        # get class attributes
        data = getattr(self, data_name)
        nodata = data.nodata
        # get bbox of nonzero entries
        if np.isnan(data.nodata):
            mask = (~np.isnan(data))
            nz = np.nonzero(mask)
        else:
            mask = (data != nodata)
            nz = np.nonzero(mask)
        # TODO: Something is messed up with the padding
        yi_min = nz[0].min() - pad[1]
        yi_max = nz[0].max() + pad[3]
        xi_min = nz[1].min() - pad[0]
        xi_max = nz[1].max() + pad[2]
        xul, yul = data.affine * (xi_min, yi_min)
        xlr, ylr = data.affine * (xi_max + 1, yi_max + 1)
        # if inplace is True, clip all grids to new bbox and set self.bbox
        if inplace:
            new_affine = Affine(data.affine.a, data.affine.b, xul,
                                data.affine.d, data.affine.e, yul)
            ncols, nrows = ~new_affine * (xlr, ylr)
            np.testing.assert_almost_equal(nrows, round(nrows), decimal=precision)
            np.testing.assert_almost_equal(ncols, round(ncols), decimal=precision)
            ncols, nrows = np.around([ncols, nrows]).astype(int)
            self.affine = new_affine
            self.shape = (nrows, ncols)
            self.crs = data.crs
            if apply_mask:
                mask = np.pad(mask, ((pad[1], pad[3]),(pad[0], pad[2])), mode='constant',
                              constant_values=0).astype(bool)
                self.mask = mask[yi_min + pad[1] : yi_max + pad[3] + 1,
                                 xi_min + pad[0] : xi_max + pad[2] + 1]
            else:
                self.mask = np.ones((nrows, ncols)).astype(bool)
        else:
            # if inplace is False, return the clipped data
            # TODO: This will fail if there is padding because of negative index
            return data[yi_min:yi_max+1, xi_min:xi_max+1]

    @property
    def bbox(self):
        shape = self.shape
        xmin, ymax = self.affine * (0,0)
        xmax, ymin = self.affine * (shape[1] + 1, shape[0] + 1)
        _bbox = (xmin, ymin, xmax, ymax)
        return _bbox

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def extent(self):
        bbox = self.bbox
        extent = (self.bbox[0], self.bbox[2], self.bbox[1], self.bbox[3])
        return extent

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, new_crs):
        assert isinstance(new_crs, pyproj.Proj)
        self._crs = new_crs

    @property
    def affine(self):
        return self._affine

    @affine.setter
    def affine(self, new_affine):
        assert isinstance(new_affine, Affine)
        self._affine = new_affine

    @property
    def cellsize(self):
        dy, dx = self._dy_dx()
        # TODO: Assuming square cells
        cellsize = (dy + dx) / 2
        return cellsize

    def set_nodata(self, data_name, new_nodata, old_nodata=None):
        """
        Change nodata value of a dataset.
 
        Parameters
        ----------
        data_name : string
                    Attribute name of dataset to change.
        new_nodata : int or float
                     New nodata value to use.
        old_nodata : int or float (optional)
                     If none provided, defaults to
                     self.grid_props[data_name]['nodata']
        """
        if old_nodata is None:
            old_nodata = getattr(self, data_name).nodata
        data = getattr(self, data_name)
        if np.isnan(old_nodata):
            np.place(data, np.isnan(data), new_nodata)
        else:
            np.place(data, data == old_nodata, new_nodata)
        data.nodata = new_nodata

    def to_ascii(self, data_name=None, file_name=None, view=True, apply_mask=False, delimiter=' ',
                 **kwargs):
        """
        Writes current "view" of grid data to ascii grid files.
 
        Parameters
        ----------
        data_name : string or list-like (optional)
                    Attribute name(s) of datasets to write. Defaults to all
                    grid dataset names.
        file_name : string or list-like (optional)
                    Name(s) of file(s) to write to (defaults to attribute
                    name).
        view : bool
               If True, writes the "view" of the dataset. Otherwise, writes the
               entire dataset.
        apply_mask : bool
               If True, write the "masked" view of the dataset.
        delimiter : string (optional)
                    Delimiter to use in output file (defaults to ' ')
 
        Additional keyword arguments are passed to numpy.savetxt
        """
        if data_name is None:
            data_name = self.grids
        if file_name is None:
            file_name = self.grids
        if isinstance(data_name, str):
            data_name = [data_name]
        if isinstance(file_name, str):
            file_name = [file_name]
        header_space = 9*' '
        for in_name, out_name in zip(data_name, file_name):
            nodata = getattr(self, in_name).nodata
            if view:
                shape = self.shape
                bbox = self.bbox
                # TODO: This breaks if cells are not square; issue with ASCII
                # format
                cellsize = self.cellsize
            else:
                shape = getattr(self, in_name).view_shape
                bbox = getattr(self, in_name).bbox
                cellsize = getattr(self, in_name).cellsize
            header = (("ncols{0}{1}\nnrows{0}{2}\nxllcorner{0}{3}\n"
                      "yllcorner{0}{4}\ncellsize{0}{5}\nNODATA_value{0}{6}")
                      .format(header_space,
                              shape[1],
                              shape[0],
                              bbox[0],
                              bbox[1],
                              cellsize,
                              nodata))
            if view:
                np.savetxt(out_name, self.view(in_name, apply_mask=apply_mask), delimiter=delimiter,
                        header=header, comments='', **kwargs)
            else:
                np.savetxt(out_name, getattr(self, in_name), delimiter=delimiter,
                        header=header, comments='', **kwargs)

    def extract_river_network(self, fdir, acc, threshold=100,
                              dirmap=None, nodata_in=None, apply_mask=True,
                              ignore_metadata=False, **kwargs):
        """
        Generates river segments from accumulation and flow_direction arrays.
 
        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If string: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        acc : str or Raster
              Accumulation data.
              If string: name of the dataset to be viewed.
              If Raster: a Raster instance (see pysheds.view.Raster)
        threshold : int or float
                    Minimum allowed cell accumulation needed for inclusion in
                    river network.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
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
        # TODO: If two "forks" are directly connected, it can introduce a gap
        nodata_in = self._check_nodata_in(fdir, nodata_in)
        fdir_props = {}
        acc_props = {}
        fdir = self._input_handler(fdir, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=fdir_props,
                                   ignore_metadata=ignore_metadata, **kwargs)
        acc = self._input_handler(acc, apply_mask=apply_mask, nodata_view=nodata_in,
                                  properties=acc_props,
                                  ignore_metadata=ignore_metadata, **kwargs)
        dirmap = self._set_dirmap(dirmap, fdir)
        flat_idx = np.arange(fdir.size)
        fdir_orig_type = fdir.dtype
        def _get_spurious_indexes(branches):
            branch_starts = np.asarray([branch[0] for branch in branches])
            branch_ends = np.asarray([branch[-1] for branch in branches])
            sc = pd.Series(branch_starts).value_counts()
            ec = pd.Series(branch_ends).value_counts()
            e_in_s = sc.reindex(ec.index.values).dropna().astype(int)
            spurious_branch_ends = (e_in_s == 1) & (ec.reindex(e_in_s.index) == 1)
            spurious_ixes = np.where(np.in1d(branch_ends,
                                             spurious_branch_ends
                                             .index.values[spurious_branch_ends.values]))[0]
            return spurious_ixes, branch_starts, branch_ends
        try:
            mintype = np.min_scalar_type(fdir.size)
            fdir = fdir.astype(mintype)
            flat_idx = flat_idx.astype(mintype)
            startnodes, endnodes = self._construct_matching(fdir, flat_idx,
                                                            dirmap=dirmap)
            start = startnodes[acc.flat[startnodes] > threshold]
            end = fdir.flat[start]
            # Find nodes with indegree > 1 and sever them
            indegree = (np.bincount(end)).astype(np.uint8)
            forks_end = np.where(indegree > 1)[0]
            no_fork = ~np.in1d(end, forks_end)
            # Find connected components with forks severed
            A = scipy.sparse.lil_matrix((fdir.size, fdir.size))
            for i,j in zip(start[no_fork], end[no_fork]):
                A[i,j] = 1
            n_components, labels = csgraph.connected_components(A)
            u, inverse, c = np.unique(labels, return_inverse=True, return_counts=True)
            idx_vals_repeated = np.where(c > 1)[0]
            # Get shortest paths to sort nodes in each branch
            C = scipy.sparse.lil_matrix((fdir.size, fdir.size))
            for i,j in zip(start, end):
                C[i,j] = 1
            C = C.tocsr()
            outlet = np.argmax(acc)
            y, x = np.unravel_index(outlet, acc.shape)
            xyindex = np.ravel_multi_index((y, x), fdir.shape)
            dist = csgraph.shortest_path(C, indices=[xyindex], directed=False)
            dist = dist.ravel()
            noninf = np.where(np.isfinite(dist))[0]
            sorted_dists = np.argsort(dist)
            sorted_dists = sorted_dists[np.in1d(sorted_dists, noninf)][::-1]
            # Construct branches
            branches = []
            for val in idx_vals_repeated:
                branch = np.where(labels == val)[0]
                # Ensure no self-loops
                branch = branch[branch != val]
                # Sort indices by distance to outlet
                branch = branch[np.argsort(dist[branch])].tolist()
                fork = fdir.flat[branch[0]]
                branch = [fork] + branch
                branches.append(branch)
            # Handle case where two adjacent forks are connected
            after_fork = fdir.flat[forks_end]
            second_fork = np.unique(after_fork[np.in1d(after_fork, forks_end)])
            second_fork_start = start[np.in1d(end, second_fork)]
            second_fork_end = fdir.flat[second_fork_start]
            for fork_start, fork_end in zip(second_fork_start, second_fork_end):
                branches.append([fork_end, fork_start])
            # TODO: Experimental
            # Take care of spurious segments
            spurious_ixes, branch_starts, branch_ends = _get_spurious_indexes(branches)
            spurious_starts = np.asarray([branch_starts[ix] for ix in spurious_ixes])
            spurious_ends = np.asarray([branch_ends[ix] for ix in spurious_ixes])
            double_joints_ds = np.in1d(spurious_starts, spurious_ends)
            if double_joints_ds.any():
                double_joints_start = []
                double_joints_end = []
                for joint in spurious_starts[double_joints_ds]:
                    double_joints_start.append(np.asscalar(np.where(spurious_starts == joint)[0]))
                    double_joints_end.append(np.asscalar(np.where(spurious_ends == joint)[0]))
                spurious_double_end = [spurious_ixes[ix] for ix in double_joints_start]
                spurious_double_start = [spurious_ixes[ix] for ix in double_joints_end]
                for starts, ends in zip(spurious_double_start, spurious_double_end):
                    ds_seg = branches[ends][1:]
                    branches[starts].extend(ds_seg)
                for us_seg in sorted(spurious_double_end)[::-1]:
                    del branches[us_seg]
                spurious_ixes, branch_starts, branch_ends = _get_spurious_indexes(branches)
            branch_starts_s = pd.Series(np.arange(len(branch_starts)), index=branch_starts)
            branch_ends_s = pd.Series(np.arange(len(branch_ends)), index=branch_ends)
            upstream_ixes = [branch_starts_s[branches[ix][-1]] for ix in spurious_ixes]
            for ix in upstream_ixes:
                branch = branches[ix]
                upstream_joint = branch.pop(0)
                downstream_ix = branch_ends_s[upstream_joint]
                branches[downstream_ix].extend(branch)
            for ix in np.sort(upstream_ixes)[::-1]:
                del branches[ix]
            # Get x, y coordinates for plotting
            yx = np.vstack(np.dstack(
                        np.meshgrid(*self.grid_indices(), indexing='ij')))
            xy = yx[:, [1,0]]
            featurelist = []
            for index, branch in enumerate(branches):
                line = geojson.LineString(xy[branch].tolist())
                featurelist.append(geojson.Feature(geometry=line, id=index))
            geo = geojson.FeatureCollection(featurelist)
        except:
            raise
        finally:
            self._unflatten_fdir(fdir, flat_idx, dirmap)
            fdir = fdir.astype(fdir_orig_type)
        return geo

    def check_cycles(self, fdir, max_cycle_size=50, dirmap=None, nodata_in=0, nodata_out=-1,
                     apply_mask=True, ignore_metadata=False, **kwargs):
        """
        Checks for cycles in flow direction array.
 
        Parameters
        ----------
        fdir : str or Raster
               Flow direction data.
               If string: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        max_cycle_size: int
                        Max depth of cycle to search for.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value indicating no data in output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        dirmap = self._set_dirmap(dirmap, fdir)
        nodata_in = self._check_nodata_in(fdir, nodata_in)
        grid_props = {'nodata' : nodata_out}
        metadata = {}
        fdir = self._input_handler(fdir, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props,
                                   ignore_metadata=ignore_metadata, **kwargs)
        if np.isnan(nodata_in):
            in_catch = ~np.isnan(fdir.ravel())
        else:
            in_catch = (fdir.ravel() != nodata_in)
        ix = np.where(in_catch)[0]
        flat_idx = np.arange(fdir.size)
        fdir_orig_type = fdir.dtype
        try:
            mintype = np.min_scalar_type(fdir.size)
            fdir = fdir.astype(mintype)
            flat_idx = flat_idx.astype(mintype)
            startnodes, endnodes = self._construct_matching(fdir, flat_idx, dirmap)
            startnodes = startnodes[ix]
            endnodes = endnodes[ix]
            z = np.zeros(fdir.size).astype(int)
            for n in range(max_cycle_size):
                check = (startnodes == endnodes)
                check_ix = np.where(check)
                z[check_ix] += n
                startnodes = startnodes[~check]
                endnodes = endnodes[~check]
                if not startnodes.any():
                    break
                endnodes = fdir.flat[endnodes]
            z.flat[~in_catch] = nodata_out
        except:
            raise
        finally:
            self._unflatten_fdir(fdir, flat_idx, dirmap)
            fdir = fdir.astype(fdir_orig_type)
        z = z.reshape(fdir.shape)
        return z

    def _select_surround(self, i, j):
        """
        Select the eight indices surrounding a given index.
        """
        return ([i - 1, i - 1, i + 0, i + 1, i + 1, i + 1, i + 0, i - 1],
                [j + 0, j + 1, j + 1, j + 1, j + 0, j - 1, j - 1, j - 1])

    def _select_edge_sur(self, edges, k):
        """
        Select the five cell indices surrounding each edge cell.
        """
        i, j = edges[k]['k']
        if k == 'n':
            return ([i + 0, i + 1, i + 1, i + 1, i + 0],
                    [j + 1, j + 1, j + 0, j - 1, j - 1])
        elif k == 'e':
            return ([i - 1, i + 1, i + 1, i + 0, i - 1],
                    [j + 0, j + 0, j - 1, j - 1, j - 1])
        elif k == 's':
            return ([i - 1, i - 1, i + 0, i + 0, i - 1],
                    [j + 0, j + 1, j + 1, j - 1, j - 1])
        elif k == 'w':
            return ([i - 1, i - 1, i + 0, i + 1, i + 1],
                    [j + 0, j + 1, j + 1, j + 1, j + 0])

    def _select_surround_ravel(self, i, shape):
        """
        Select the eight indices surrounding a flattened index.
        """
        offset = shape[1]
        return np.array([i + 0 - offset,
                         i + 1 - offset,
                         i + 1 + 0,
                         i + 1 + offset,
                         i + 0 + offset,
                         i - 1 + offset,
                         i - 1 + 0,
                         i - 1 - offset]).T

    def _set_dirmap(self, dirmap, data, default_dirmap=(1, 2, 3, 4, 5, 6, 7, 8)):
        # TODO: Is setting a default dirmap even a good idea?
        if dirmap is None:
            if isinstance(data, str):
                if data in self.grids:
                    try:
                        dirmap = getattr(self, data).metadata['dirmap']
                    except:
                        dirmap = default_dirmap
                else:
                    raise KeyError("{0} not found in grid instance"
                                   .format(direction_name))
            elif isinstance(data, Raster):
                try:
                    dirmap = data.metadata['dirmap']
                except:
                    dirmap = default_dirmap
            else:
                dirmap = default_dirmap
        if len(dirmap) != 8:
            raise AssertionError('dirmap must be a sequence of length 8')
        try:
            assert(not 0 in dirmap)
        except:
            raise ValueError("Directional mapping cannot contain '0' (reserved value)")
        return dirmap

    def _grad_from_higher(self, high_edge_cells, inner_neighbors, diff,
                          fdir_defined, in_bounds, labels, numlabels, crosswalk):
        print("Constructing gradient from higher terrain")
        z = np.zeros_like(labels)
        max_iter = np.bincount(labels.ravel())[1:].max()
        u = high_edge_cells.copy()
        z[1:-1, 1:-1].flat[u] = 1
        for i in range(2, max_iter):
            # Select neighbors of high edge cells
            hec_neighbors = inner_neighbors[:, u]
            # Get neighbors with same elevation that are in bounds
            u = np.unique(np.where((diff[:, u] == 0) & (in_bounds.flat[hec_neighbors] == 1), hec_neighbors, 0))
            # Filter out entries that have already been incremented
            not_got = (z.flat[u] == 0)
            u = u[not_got]
            # Get indices of inner cells from raw index
            u = crosswalk.flat[u]
            # Filter out neighbors that are in low edge_cells
            u = u[(~fdir_defined[u])]
            # Increment neighboring cells
            z[1:-1, 1:-1].flat[u] = i
            if u.size <= 1:
                break
        z[1:-1,1:-1].flat[0] = 0
        # Flip increments
        d = {}
        for i in range(1, z.max()):
            label = labels[z == i]
            label = label[label != 0]
            label = np.unique(label)
            d.update({i : label})
        max_incs = np.zeros(numlabels + 1)
        for i in range(1, z.max()):
            max_incs[d[i]] = i
        max_incs = max_incs[labels.ravel()].reshape(labels.shape)
        grad_from_higher = max_incs - z
        return grad_from_higher

    def _grad_towards_lower(self, low_edge_cells, inner_neighbors, diff,
                          fdir_defined, in_bounds, labels, numlabels, crosswalk):
        print("Constructing gradient towards lower terrain")
        x = np.zeros_like(labels)
        u = low_edge_cells.copy()
        x[1:-1, 1:-1].flat[u] = 1
        max_iter = np.bincount(labels.ravel())[1:].max()

        for i in range(2, max_iter):
            # Select neighbors of high edge cells
            lec_neighbors = inner_neighbors[:, u]
            # Get neighbors with same elevation that are in bounds
            u = np.unique(
                np.where((diff[:, u] == 0) & (in_bounds.flat[lec_neighbors] == 1),
                         lec_neighbors, 0))
            # Filter out entries that have already been incremented
            not_got = (x.flat[u] == 0)
            u = u[not_got]
            # Get indices of inner cells from raw index
            u = crosswalk.flat[u]
            u = u[~fdir_defined.flat[u]]
            # Increment neighboring cells
            x[1:-1, 1:-1].flat[u] = i
            if u.size == 0:
                break
        x[1:-1,1:-1].flat[0] = 0
        grad_towards_lower = x
        return grad_towards_lower

    def _drainage_gradient(self, dem, inside):
        if not _HAS_SKIMAGE:
            raise ImportError('resolve_flats requires skimage.measure module')
        inner_neighbors, diff, fdir_defined = self._d8_diff(dem, inside)
        higher_cell = (diff < 0).any(axis=0)
        same_elev_cell = (diff == 0).any(axis=0)
        # High edge cells are defined as:
        # (a) Flow direction is not defined
        # (b) Has at least one neighboring cell at a higher elevation
        print('Determining high edge cells')
        high_edge_cells_bool = (~fdir_defined & higher_cell)
        high_edge_cells = np.where(high_edge_cells_bool)[0]
        # Low edge cells are defined as:
        # (a) Flow direction is defined
        # (b) Has at least one neighboring cell, n, at the same elevation
        # (c) The flow direction for this cell n is undefined
        # Need to check if neighboring cell has fdir undefined
        print('Determining low edge cells')
        low_edge_cell_candidates = (fdir_defined & same_elev_cell)
        fdir_def_all = -1 * np.ones(dem.shape)
        fdir_def_all[1:-1, 1:-1] = fdir_defined.reshape(dem.shape[0] - 2, dem.shape[1] - 2)
        fdir_def_neighbors = fdir_def_all.flat[inner_neighbors[:, low_edge_cell_candidates]]
        same_elev_neighbors = ((diff[:, low_edge_cell_candidates]) == 0)
        low_edge_cell_passed = (fdir_def_neighbors == 0) & (same_elev_neighbors == 1)
        low_edge_cells = (np.where(low_edge_cell_candidates)[0]
                          [low_edge_cell_passed.any(axis=0)])
        # Get flats to label
        tolabel = (fdir_def_all == 0)
        labels, numlabels = skimage.measure.label(tolabel, return_num=True)
        # Make sure cells stay in bounds
        in_bounds = np.ones_like(labels)
        in_bounds[0, :] = 0
        in_bounds[:, 0] = 0
        in_bounds[-1, :] = 0
        in_bounds[:, -1] = 0
        crosswalk = np.zeros_like(labels)
        crosswalk[1:-1, 1:-1] = (np.arange(inside.size)
                                 .reshape(dem.shape[0] - 2, dem.shape[1] - 2))
        grad_from_higher = self._grad_from_higher(high_edge_cells, inner_neighbors, diff,
                          fdir_defined, in_bounds, labels, numlabels, crosswalk)
        grad_towards_lower = self._grad_towards_lower(low_edge_cells, inner_neighbors, diff,
                          fdir_defined, in_bounds, labels, numlabels, crosswalk)
        drainage_grad = (2*grad_towards_lower + grad_from_higher).astype(int)
        return drainage_grad, high_edge_cells, low_edge_cells

    def _d8_diff(self, dem, inside):
        inner_neighbors = self._select_surround_ravel(inside, dem.shape).T
        inner_neighbors_elev = dem.flat[inner_neighbors]
        diff = np.subtract(dem.flat[inside], inner_neighbors_elev)
        fdir_defined = (diff > 0).any(axis=0)
        return inner_neighbors, diff, fdir_defined

    def resolve_flats(self, data=None, out_name='flats_dir', nodata_in=None, nodata_out=0,
                      pits=-1, flats=-1, dirmap=(1, 2, 3, 4, 5, 6, 7, 8), inplace=True,
                      apply_mask=False, ignore_metadata=False, **kwargs):
        """
        Resolve flats in a DEM using the modified method of Garbrecht and Martz (1997).
 
        Parameters
        ----------
        data : str or Raster
               DEM data.
               If string: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        out_name : string
                   Name of attribute containing new flow direction array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_in : int or float
                     Value to indicate nodata in input array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        pits : int
               Value to indicate pits in output array.
        flats : int
                Value to indicate flat areas in output array.
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        apply_mask : bool
               If True, "mask" the output using self.mask.
        ignore_metadata : bool
                          If False, require a valid affine transform and CRS.
        """
        if len(dirmap) != 8:
            raise AssertionError('dirmap must be a sequence of length 8')
        # if data not provided, use self.dem
        # handle nodata values in dem
        if nodata_in is None:
            if isinstance(data, str):
                try:
                    nodata_in = getattr(self, data).nodata
                except:
                    raise NameError("nodata value for '{0}' not found in instance."
                                    .format(data))
            else:
                raise KeyError("No 'nodata' value specified.")
        grid_props = {'nodata' : nodata_out}
        metadata = {'dirmap' : dirmap}
        dem = self._input_handler(data, apply_mask=apply_mask, properties=grid_props,
                                  ignore_metadata=ignore_metadata, metadata=metadata, **kwargs)
        # TODO: Note that this won't work for nans
        dem_mask = np.where(dem.ravel() == nodata_in)[0]
        # TODO: This is repeated from flowdir
        a = np.arange(dem.size)
        top = np.arange(dem.shape[1])[1:-1]
        left = np.arange(0, dem.size, dem.shape[1])
        right = np.arange(dem.shape[1] - 1, dem.size + 1, dem.shape[1])
        bottom = np.arange(dem.size - dem.shape[1], dem.size)[1:-1]
        exclude = np.unique(np.concatenate([top, left, right, bottom, dem_mask]))
        inside = np.delete(a, exclude)
        drainage_grad, high_edge_cells, low_edge_cells = self._drainage_gradient(dem, inside)
        sub_props = copy.deepcopy(grid_props)
        sub_props.update({'nodata_in' : 0, 'nodata_out' : nodata_out})
        fdir_flats = self.flowdir(data=drainage_grad, inplace=False, **sub_props)
        fdir_flats[1:-1, 1:-1].flat[low_edge_cells] = nodata_out
        return self._output_handler(data=fdir_flats, out_name=out_name, inplace=inplace,
                                    metadata=metadata, **grid_props)

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

    def snap_to_mask(self, mask, xy, return_dist=True):
        """
        Snap a set of xy coordinates (xy) to the nearest nonzero cells in a raster (mask)

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
        if isinstance(mask, Raster):
            affine = mask.viewfinder.affine
        elif isinstance(mask, 'str'):
            affine = getattr(self, mask).viewfinder.affine
        mask_ix = np.where(mask.ravel())[0]
        yi, xi = np.unravel_index(mask_ix, mask.shape)
        xiyi = np.vstack([xi, yi])
        x, y = affine * xiyi
        tree_xy = np.column_stack([x, y])
        tree = scipy.spatial.cKDTree(tree_xy)
        dist, ix = tree.query(xy)
        if return_dist:
            return tree_xy[ix], dist
        else:
            return tree_xy[ix]
