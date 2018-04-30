import pyproj
import warnings
import numpy as np
import pandas as pd
import sys
import ast
import copy
try:
    import scipy.sparse
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
    _HAS_RASTERIO = True
except:
    _HAS_RASTERIO = False
from pysheds.view import ViewFinder, IrregularViewFinder, RegularGridViewer, IrregularGridViewer

class Grid(object):
    """
    Container class for holding and manipulating gridded data.
 
    Attributes
    ==========
    bbox : The geographical bounding box for viewing the gridded data
           (xmin, ymin, xmax, ymax).
    shape : The shape of the bbox (nrows, ncolumns) at the given cellsize.
    cellsize : The length/width of each grid cell (assumed to be square).
    grid_props : dict containing metadata for each gridded dataset.
    mask : A boolean array used to mask certain grid cells in the bbox;
           may be used to indicate which cells lie inside a catchment.
 
    Methods
    =======
        --------
        File I/O
        --------
        add_data : Add a gridded dataset (dem, flowdir, accumulation)
                   to Grid instance (generic method).
        read_ascii : Read an ascii grid from a file and add it to a
                     Grid instance.
        read_raster : Read a raster file and add the data to a Grid
                      instance.
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
        view : Returns a "view" of a dataset within the bounding box specified
               by self.bbox (can optionally be masked with self.mask).
        set_bbox : Sets the bbox of the current "view" (self.bbox).
        set_nodata : Sets the nodata value for a given dataset.
        bbox_indices : Returns arrays containing the geographic coordinates
                       of the grid's rows and columns for the current "view".
        nearest_cell : Returns the index (column, row) of the cell closest
                       to a given geographical coordinate (x, y).
        clip_to : Clip the bbox to the smallest area containing all non-
                  null gridcells for a provided dataset (defaults to
                  self.catch).
        catchment_mask : Updates self.mask to mask all gricells not inside the
                         catchment (given by self.catch).
 
    Default Dataset Names
    ======================
    dem : digital elevation grid
    dir : flow direction grid
    acc : flow accumulation grid
    catch : Catchment delineated from 'dir' and a given pour point
    frac : fractional contributing area grid
    """

    def __init__(self, bbox=(0,0,0,0), shape=(0,0), nodata=0,
                 crs=pyproj.Proj('+init=epsg:4326')):
        self._bbox = bbox
        self.shape = shape
        self.nodata = nodata
        self._crs = crs
        self.grid_props = {}

    @property
    def defaults(self):
        props = {
            'bbox' : (0,0,0,0),
            'shape' : (0,0),
            'nodata' : 0,
            'crs' : pyproj.Proj('+init=epsg:4326'),
        }
        return props

    def add_gridded_data(self, data, data_name, bbox=None, shape=None, cellsize=None,
                         crs=None, nodata=None, mask=None, data_attrs={}):
        """
        A generic method for adding data into a Grid instance.
        Inserts data into a named attribute of Grid (name of attribute
        determined by keyword 'data_name').
 
        Parameters
        ----------
        data : numpy ndarray
               Data to be inserted into Grid instance.
        data_name : string
                     Name of dataset. Will determine the name of the attribute
                     representing the gridded data. Default values are used
                     internally by some class methods:
                         'dem' : digital elevation data
                         'dir' : flow direction data
                         'acc' : flow accumulation (upstream area) data
                         'catch' : catchment grid
                         'frac' : fractional contributing area
        bbox : tuple (length 4)
               Bounding box of data.
        shape : tuple of ints (length 2)
                Shape (rows, columns) of data.
        cellsize : float or int
                   Cellsize of gridded data.
        crs : dict
              Coordinate reference system of gridded data.
        nodata : int or float
                 Value indicating no data.
        data_attrs : dict
                     Other attributes describing dataset, such as direction
                     mapping for flow direction files. e.g.:
                     data_attrs={'dirmap' : (1, 2, 3, 4, 5, 6, 7, 8),
                                 'routing' : 'd8'}
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('Input data must be ndarray')
        # if there are no datasets, initialize bbox, shape,
        # cellsize and crs based on incoming data
        if len(self.grid_props) < 1:
            # check validity of bbox
            if ((hasattr(bbox, "__len__")) and (not isinstance(bbox, str))
                    and (len(bbox) == 4)):
                bbox = tuple(bbox)
            else:
                raise TypeError('bbox must be a tuple of length 4.')
            # check validity of shape
            if ((hasattr(shape, "__len__")) and (not isinstance(shape, str))
                    and (len(shape) == 2) and (isinstance(sum(shape), int))):
                shape = tuple(shape)
            else:
                raise TypeError('shape must be a tuple of ints of length 2.')
            # check validity of cellsize
            if not isinstance(cellsize, (int, float)):
                raise TypeError('cellsize must be an int or float.')
            if crs is not None:
                if isinstance(crs, pyproj.Proj):
                    pass
                if isinstance(crs, dict) or isinstance(crs, str):
                    crs = pyproj.Proj(crs)
            # initialize instance metadata
            self._bbox = bbox
            self.shape = shape
            self._crs = crs
            self.nodata = nodata
            self.mask = np.ones(self.shape, dtype=np.bool)
            self.shape_min = np.min_scalar_type(max(self.shape))
            self.size_min = np.min_scalar_type(data.size)
        # assign new data to attribute; record nodata value
        self.grid_props.update({data_name : {}})
        self.grid_props[data_name].update({'bbox' : bbox})
        self.grid_props[data_name].update({'shape' : shape})
        self.grid_props[data_name].update({'cellsize' : cellsize})
        self.grid_props[data_name].update({'nodata' : nodata})
        self.grid_props[data_name].update({'crs' : crs})
        view = ViewFinder(bbox=bbox, shape=shape, mask=mask, nodata=nodata, crs=crs)
        self.grid_props[data_name].update({'view' : view})
        for other_name, other_value in data_attrs.items():
            self.grid_props[data_name].update({other_name : other_value})
        setattr(self, data_name, data)

    def read_ascii(self, data, data_name, skiprows=6, crs=None, data_attrs={}, **kwargs):
        """
        Reads data from an ascii file into a named attribute of Grid
        instance (name of attribute determined by 'data_name').
 
        Parameters
        ----------
        data : string
               File name or path.
        data_name : string
                     Name of dataset. Will determine the name of the attribute
                     representing the gridded data. Default values are used
                     internally by some class methods:
                         'dem' : digital elevation data
                         'dir' : flow direction data
                         'acc' : flow accumulation (upstream area) data
                         'catch' : catchment grid
                         'frac' : fractional contributing area
        skiprows : int (optional)
                   The number of rows taken up by the header (defaults to 6).
        crs : dict (optional)
              Coordinate reference system of ascii data.
        data_attrs : dict
                     Other attributes describing dataset, such as direction
                     mapping for flow direction files. e.g.:
                     data_attrs={'dirmap' : (1, 2, 3, 4, 5, 6, 7, 8),
                                 'routing' : 'd8'}
 
        Additional keyword arguments are passed to numpy.loadtxt()
        """
        # TODO: Consider setting default crs to geographic
        with open(data) as header:
            ncols = int(header.readline().split()[1])
            nrows = int(header.readline().split()[1])
            xll = ast.literal_eval(header.readline().split()[1])
            yll = ast.literal_eval(header.readline().split()[1])
            cellsize = ast.literal_eval(header.readline().split()[1])
            nodata = ast.literal_eval(header.readline().split()[1])
            shape = (nrows, ncols)
            bbox = (xll, yll, xll + ncols * cellsize, yll + nrows * cellsize)
        data = np.loadtxt(data, skiprows=skiprows, **kwargs)
        nodata = data.dtype.type(nodata)
        self.add_gridded_data(data, data_name, bbox, shape, cellsize, crs, nodata,
                              data_attrs=data_attrs)

    def read_raster(self, data, data_name, band=1, window=None, window_crs=None,
                    data_attrs={}, **kwargs):
        """
        Reads data from a raster file into a named attribute of Grid
        (name of attribute determined by keyword 'data_name').
 
        Parameters
        ----------
        data : string
               File name or path.
        data_name : string
                     Name of dataset. Will determine the name of the attribute
                     representing the gridded data. Default values are used
                     internally by some class methods:
                         'dem' : digital elevation data
                         'dir' : flow direction data
                         'acc' : flow accumulation (upstream area) data
                         'catch' : catchment grid
                         'frac' : fractional contributing area
        band : int
               The band number to read.
        window : tuple
                 If using windowed reading, specify window (xmin, ymin, xmax, ymax).
        window_crs : pyproj.Proj instance
                     Coordinate reference system of window. If None, assume it's in raster's crs.
        data_attrs : dict
                     Other attributes describing dataset, such as direction
                     mapping for flow direction files. e.g.:
                     data_attrs={'dirmap' : (1, 2, 3, 4, 5, 6, 7, 8),
                                 'routing' : 'd8'}
 
        Additional keyword arguments are passed to rasterio.open()
        """
        # read raster file
        if not _HAS_RASTERIO:
            raise ImportError('Requires rasterio module')
        with rasterio.open(data, **kwargs) as f:
            crs = pyproj.Proj(f.crs, preserve_units=True)
            if window is None:
                bbox = tuple(f.bounds)
                shape = f.shape
                if len(f.indexes) > 1:
                    data = np.ma.filled(f.read_band(band))
                else:
                    data = np.ma.filled(f.read())
            else:
                if window_crs is not None:
                    if window_crs.srs != crs.srs:
                        xmin, ymin, xmax, ymax = window
                        extent = pyproj.transform(window_crs, crs, (xmin, xmax),
                                                  (ymin, ymax))
                        window = (extent[0][0], extent[1][0], extent[0][1], extent[1][1])
                # If window crs not specified, assume it's in raster crs
                ix_window = f.window(*window)
                bbox = tuple(window)
                shape = (ix_window[0][1] - ix_window[0][0],
                         ix_window[1][1] - ix_window[1][0])
                if len(f.indexes) > 1:
                    data = np.ma.filled(f.read_band(band, window=ix_window))
                else:
                    data = np.ma.filled(f.read(window=ix_window))
            cellsize = f.affine[0]
            nodata = f.nodatavals[0]
            data = data.reshape(shape)
        if nodata is not None:
            nodata = data.dtype.type(nodata)
        self.add_gridded_data(data, data_name, bbox, shape, cellsize, crs, nodata,
                              data_attrs=data_attrs)

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

    def view(self, data, data_view=None, target_view=None, apply_mask=True,
             nodata=None, interpolation='nearest', as_crs=None, return_coords=False,
             kx=3, ky=3, s=0, tolerance=0.01):
        """
        Return a copy of a gridded dataset clipped to the bounding box
        (self.bbox) with cells outside the catchment mask (self.mask)
        optionally displayed as 'nodata' (self.grid_props[data_name]['nodata'])
 
        Parameters
        ----------
        data_name : string
                    Name of the dataset to be viewed.
        mask : bool
               If True, "mask" the view using self.mask.
        nodata : int of float
                 Value indicating no data. Defaults to
                 self.grid_props[data_name]['nodata']
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
            if nodata is None:
                nodata = self.grid_props[data]['nodata']
            if data_view is None:
                data_view = self.grid_props[data]['view']
            data = getattr(self, data)
        else:
            # If not using a named dataset, make sure the data and view are properly defined
            try:
                assert(isinstance(data, np.ndarray))
            except:
                raise
            if nodata is None:
                nodata = data_view.nodata
        # Set default target crs to grid crs
        if as_crs is None:
            as_crs = self.crs
        # If no target view provided, construct one based on grid parameters
        if target_view is None:
            target_view = ViewFinder(bbox=self.bbox, shape=self.shape,
                                     mask=self.mask, crs=as_crs,
                                     nodata=nodata)
        # Make sure views are ViewFinder instances
        assert(isinstance(data_view, ViewFinder) or isinstance(data_view, IrregularViewFinder))
        assert(isinstance(target_view, ViewFinder) or isinstance(target_view, IrregularViewFinder))
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
        data_is_grid = isinstance(data_view, ViewFinder)
        view_is_grid = isinstance(target_view, ViewFinder)
        # If data is on a grid, use the following speedup
        if data_is_grid and view_is_grid:
            # If doing nearest neighbor search, use fast sorted search
            if interpolation == 'nearest':
                array_view = RegularGridViewer._view_searchsorted(data, data_view, target_view,
                                                                  apply_mask=apply_mask)
            # If spline interpolation is needed, use RectBivariate
            elif interpolation == 'spline':
                # If latitude/longitude, use RectSphereBivariate
                if target_view.crs.is_latlong():
                    array_view = RegularGridViewer._view_rectspherebivariate(data, data_view,
                                                                             target_view,
                                                                             tolerance=tolerance,
                                                                             kx=kx, ky=ky, s=s,
                                                                             apply_mask=apply_mask)
                # If not latitude/longitude, use RectBivariate
                else:
                    array_view = RegularGridViewer._view_rectbivariate(data, data_view,
                                                                       target_view,
                                                                       tolerance=tolerance,
                                                                       kx=kx, ky=ky, s=s,
                                                                       apply_mask=apply_mask)
            # If some other interpolation method is needed, use griddata
            else:
                array_view = IrregularGridViewer._view_griddata(data, data_view, target_view,
                                                                method=interpolation,
                                                                apply_mask=apply_mask)
        # If either view is irregular, use griddata
        else:
            array_view = IrregularGridViewer._view_griddata(data, data_view, target_view,
                                                            method=interpolation,
                                                            apply_mask=apply_mask)
        if return_coords:
            return array_view, data_view.coords
        else:
            return array_view

    def resize(self, data, new_shape, out_suffix='_resized', inplace=True,
               nodata_in=None, nodata_out=np.nan, apply_mask=False, ignore_metadata=True, **kwargs):
        nodata_in = self._check_nodata_in(data, nodata_in)
        if isinstance(data, str):
            out_name = '{0}{1}'.format(data, out_suffix)
        else:
            out_name = 'data_{1}'.format(out_suffix)
        grid_props = {'nodata' : nodata_out}
        data = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props,
                                  ignore_metadata=ignore_metadata, **kwargs)
        data = skimage.transform.resize(data, new_shape, **kwargs)
        return self._output_handler(data, inplace, out_name=out_name, **grid_props)

    def nearest_cell(self, x, y, bbox=None, shape=None):
        """
        Returns the index of the cell (column, row) closest
        to a given geographical coordinate.
 
        Parameters
        ----------
        x : int or float
            x coordinate.
        y : int or float
            y coordinate.
        """
        if not bbox:
            bbox = self._bbox
        if not shape:
            shape = self.shape
        dy, dx = self._dy_dx()
        # Note: this speedup assumes grid cells are square
        y_ix, x_ix = self.bbox_indices(self._bbox, self.shape)
        y_ix += dy / 2.0
        x_ix += dx / 2.0
        desired_y = np.argmin(np.abs(y_ix - y))
        desired_x = np.argmin(np.abs(x_ix - x))
        return desired_x, desired_y

    def flowdir(self, data=None, out_name='dir', nodata_in=None, nodata_out=0,
                pits=-1, flats=-1, dirmap=(1, 2, 3, 4, 5, 6, 7, 8), inplace=True,
                apply_mask=False, ignore_metadata=False, **kwargs):
        """
        Generates a flow direction grid from a DEM grid.
 
        Parameters
        ----------
        data : numpy ndarray
               Array of DEM data (overrides dem_name constructor)
        dem_name : string
                    Name of attribute containing dem data.
        out_name : string
                   Name of attribute containing new flow direction array.
        include_edges : bool
                        If True, include outer rim of grid.
        nodata_in : int
                     Value to indicate nodata in input array.
        nodata_out : int
                     Value to indicate nodata in output array.
        flat : int
               Value to indicate flat areas in output array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        inplace : bool
                  If True, write output array to self.<data_name>.
                  Otherwise, return the output array.
        """
        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out, 'dirmap' : dirmap}
        dem = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in, properties=grid_props,
                                  ignore_metadata=ignore_metadata, **kwargs)
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
        return self._output_handler(fdir_out, inplace, out_name=out_name, **grid_props)

    def catchment(self, x, y, data=None, pour_value=None, out_name='catch', dirmap=None,
                  nodata_in=None, nodata_out=0, xytype='index', recursionlimit=15000,
                  inplace=True, pad=True, apply_mask=False, ignore_metadata=False, **kwargs):
        """
        Delineates a watershed from a given pour point (x, y).
 
        Parameters
        ----------
        x : int or float
            x coordinate of pour point
        y : int or float
            y coordinate of pour point
        data : numpy ndarray
               Array of flow direction data (overrides direction_name constructor)
        pour_value : int or None
                     If not None, value to represent pour point in catchment
                     grid (required by some programs).
        direction_name : string
                         Name of attribute containing flow direction data.
        out_name : string
                   Name of attribute containing new catchment array.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int
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
        grid_props = {'nodata' : nodata_out, 'dirmap' : dirmap}
        # initialize array to collect catchment cells
        cdir = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata,
                                   **kwargs)
        bbox = grid_props['bbox']
        # Pad the rim
        if pad:
            cdir = np.pad(cdir, (1,1), mode='constant')
            offset = 1
        else:
            left, right, top, bottom = self._pop_rim(fdir, nodata=nodata_in)
            offset = 0
        try:
            # get shape of padded flow direction array, then flatten
            padshape = cdir.shape
            cdir = cdir.ravel()
            # if xytype is 'label', delineate catchment based on cell nearest
            # to given geographic coordinate
            # TODO: This relies on the bbox of the grid instance, not the dataset
            # Valid if the dataset is a view.
            if xytype == 'label':
                x, y = self.nearest_cell(x, y, bbox,
                                        (padshape[0] - offset,
                                         padshape[1] - offset))
            # get the flattened index of the pour point
            pour_point = np.ravel_multi_index(np.array([y + offset, x + offset]),
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
            # remove outer rim, delete temporary arrays
            if pad:
                outcatch = outcatch[1:-1, 1:-1]
            # if pour point needs to be a special value, set it
            if pour_value is not None:
                outcatch[y, x] = pour_value
        except:
            raise
        finally:
            # reset recursion limit
            sys.setrecursionlimit(1000)
            cdir = cdir.reshape(padshape)
            if pad:
                cdir = cdir[1:-1, 1:-1]
            else:
                self._replace_rim(cdir, left, right, top, bottom)
        return self._output_handler(outcatch, inplace, out_name=out_name, **grid_props)

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
        otherrows, othercols = self.bbox_indices(other.bbox, other.shape)
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
                     out_name='acc', inplace=True, pad=False, apply_mask=False, ignore_metadata=False,
                     **kwargs):
        """
        Generates an array of flow accumulation, where cell values represent
        the number of upstream cells.
 
        Parameters
        ----------
        data : numpy ndarray
               Array of flow direction data (overrides direction_name constructor)
        weights: numpy ndarray
                 Array of weights to be applied to each accumulation cell. Must
                 be same size as data.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        direction_name : string
                         Name of attribute containing flow direction data.
        nodata_in : int
                    Value to indicate nodata in input array. If using a named dataset, will
                    default to the 'nodata' value of the named dataset. If using an ndarray,
                    will default to 0.
        nodata_out : int
                     Value to indicate nodata in output array.
        out_name : string
                   Name of attribute containing new accumulation array.
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        pad : bool
              If True, pad the rim of the input array with zeros. Else, ignore
              the outer rim of cells in the computation.
        """
        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out}
        fdir = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in, properties=grid_props,
                                   ignore_metadata=ignore_metadata, **kwargs)
        # Pad the rim
        if pad:
            fdir = np.pad(fdir, (1,1), mode='constant', constant_values=0)
        else:
            left, right, top, bottom = self._pop_rim(fdir, nodata=0)
        fdir_orig_type = fdir.dtype
        try:
            # Construct flat index onto flow direction array
            flat_idx = np.arange(fdir.size)
            if nodata_in is None:
                nodata_cells = np.zeros_like(fdir).astype(bool)
            else:
                if np.isnan(nodata_in):
                    nodata_cells = (np.isnan(fdir))
                else:
                    nodata_cells = (fdir == nodata_in)
            # Set nodata cells to zero
            fdir[nodata_cells] = 0
            # Ensure consistent types
            mintype = np.min_scalar_type(fdir.size)
            fdir = fdir.astype(mintype)
            flat_idx = flat_idx.astype(mintype)
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
            if nodata_in is not None:
                fdir[nodata_cells] = nodata_in
            if pad:
                fdir = fdir[1:-1, 1:-1]
            else:
                self._replace_rim(fdir, left, right, top, bottom)
            fdir = fdir.astype(fdir_orig_type)
        return self._output_handler(acc, inplace, out_name=out_name, **grid_props)

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
        data : str or numpy ndarray
               Named dataset or array of flow direction data.
        weights: numpy ndarray
                 Weights (distances) to apply to link edges.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        direction_name : string
                         Name of attribute containing flow direction data.
        nodata_out : int
                 Value to indicate nodata in output array.
        out_name : string
                   Name of attribute containing new flow distance array.
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        """
        if not _HAS_SCIPY:
            raise ImportError('flow_distance requires scipy.sparse module')
        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out}
        fdir = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata,
                                   **kwargs)
        bbox = grid_props['bbox']
        # Construct flat index onto flow direction array
        flat_idx = np.arange(fdir.size)
        if nodata_in is None:
            nodata_cells = np.zeros_like(fdir).astype(bool)
        else:
            if np.isnan(nodata_in):
                nodata_cells = (np.isnan(fdir))
            else:
                nodata_cells = (fdir == nodata_in)
        try:
            startnodes, endnodes = self._construct_matching(fdir, flat_idx,
                                                            dirmap=dirmap)
            if xytype == 'label':
                x, y = self.nearest_cell(x, y, bbox, fdir.shape)
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
        # Prepare output
        return self._output_handler(dist, inplace, out_name=out_name, **grid_props)

    def cell_area(self, out_name='area', nodata_out=0, inplace=True, as_crs=None):
        if as_crs is None:
            if self.crs.is_latlong():
                warnings.warn(('CRS is geographic. Area will not have meaningful '
                            'units.'))
        else:
            if as_crs.is_latlong():
                warnings.warn(('CRS is geographic. Area will not have meaningful '
                            'units.'))
        indices = np.vstack(np.dstack(np.meshgrid(*self.bbox_indices(),
                                                  indexing='ij')))
        # TODO: Add to_crs conversion here
        if as_crs:
            indices = self._convert_grid_indices_crs(indices, self.crs, as_crs)
        dyy, dyx = np.gradient(indices[:, 0].reshape(self.shape))
        dxy, dxx = np.gradient(indices[:, 1].reshape(self.shape))
        dy = np.sqrt(dyy**2 + dyx**2)
        dx = np.sqrt(dxy**2 + dxx**2)
        area = dx * dy
        private_props = {'nodata' : nodata_out}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(area, inplace, out_name=out_name, **grid_props)

    def cell_distances(self, data, out_name='cdist', nodata_in=None, nodata_out=0,
                       inplace=True, as_crs=None, ignore_metadata=False):
        if as_crs is None:
            if self.crs.is_latlong():
                warnings.warn(('CRS is geographic. Area will not have meaningful '
                            'units.'))
        else:
            if as_crs.is_latlong():
                warnings.warn(('CRS is geographic. Area will not have meaningful '
                            'units.'))
        indices = np.vstack(np.dstack(np.meshgrid(*self.bbox_indices(),
                                                  indexing='ij')))
        if as_crs:
            indices = self._convert_grid_indices_crs(indices, self.crs, as_crs)
        dirmap = self._set_dirmap(dirmap, data)
        nodata_in = self._check_nodata_in(data, nodata_in)
        grid_props = {'nodata' : nodata_out}
        fdir = self._input_handler(data, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata,
                                   **kwargs)
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
        return self._output_handler(cdist, inplace, out_name=out_name, **grid_props)

    def cell_dh(self, fdir, dem, out_name='dh', inplace=True, nodata_in=None,
                nodata_out=np.nan, dirmap=None):
        nodata_in = self._check_nodata_in(data, nodata_in)
        fdir_props = {'nodata' : nodata_out}
        fdir = self._input_handler(fdir, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata,
                                   **kwargs)
        dem_props = {'nodata' : nodata_out}
        dem = self._input_handler(dem, apply_mask=apply_mask, nodata_view=nodata_in,
                                   properties=grid_props, ignore_metadata=ignore_metadata,
                                   **kwargs)
        dirmap = self._set_dirmap(dirmap, fdir)
        flat_idx = np.arange(fdir.size)
        if nodata_in is None:
            nodata_cells = np.zeros_like(fdir).astype(bool)
        else:
            if np.isnan(nodata_in):
                nodata_cells = (np.isnan(fdir))
            else:
                nodata_cells = (fdir == nodata_in)
        try:
            startnodes, endnodes = self._construct_matching(fdir, flat_idx, dirmap)
            startelev = dem.ravel()[startnodes].astype(np.float64)
            endelev = dem.ravel()[endnodes].astype(np.float64)
            dh = (startelev - endelev).reshape(self.shape)
            dh[nodata_cells] = nodata_out
        except:
            raise
        finally:
            self._unflatten_fdir(fdir, flat_idx, dirmap)
        # Prepare output
        private_props = {'nodata' : nodata_out}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(dh, inplace, out_name=out_name, **grid_props)

    def cell_slopes(self, fdir, dem, out_name='slopes',
                    inplace=True, nodata_in=None, nodata_out=np.nan, dirmap=None, as_crs=None):
        nodata_in = self._check_nodata_in(data, nodata_in)
        dh = self.cell_dh(direction_name, dem_name, out_name, inplace=False,
                          nodata_out=nodata_out, dirmap=dirmap)
        cdist = self.cell_distances(direction_name, inplace=False, as_crs=as_crs)
        slopes = np.where(self.mask, dh/cdist, nodata_out)
        # Prepare output
        private_props = {'nodata' : nodata_out}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(slopes, inplace, out_name=out_name, **grid_props)

    def _check_nodata_in(self, data, nodata_in, override=None):
        if nodata_in is None:
            if isinstance(data, str):
                try:
                    nodata_in = self.grid_props[data]['nodata']
                except:
                    raise NameError("nodata value for '{0}' not found in instance."
                                    .format(data))
        if override is not None:
            nodata_in = override
        return nodata_in

    def _input_handler(self, data, apply_mask=True, nodata_view=None, properties={},
                       ignore_metadata=False, **kwargs):
        required_params = ('bbox', 'shape', 'nodata', 'crs')
        defaults = self.defaults
        # Handle raw data
        if isinstance(data, np.ndarray):
            for param in required_params:
                if not param in properties:
                    if param in kwargs:
                        properties[param] = kwargs[param]
                    elif ignore_metadata:
                        properties[param] = defaults[param]
                    else:
                        raise KeyError("Missing required parameter: {0}"
                                       .format(param))
            viewfinder = ViewFinder(**properties)
            properties.update({'view': viewfinder})
            return data
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
            viewfinder = ViewFinder(**properties)
            properties.update({'view': viewfinder})
            data = self.view(data, apply_mask=apply_mask, nodata=nodata_view)
            return data
        else:
            raise TypeError('Data must be a numpy ndarray or name string.')

    def _output_handler(self, data, inplace, out_name, **kwargs):
        # TODO: Should this be rolled into add_data?
        if inplace:
            setattr(self, out_name, data)
            self.grid_props.update({out_name : {}})
            self.grid_props[out_name].update(kwargs)
        else:
            return data

    def _generate_grid_props(self, **kwargs):
        properties = {}
        required = ('bbox', 'shape', 'nodata', 'crs')
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

    def _convert_bbox_indices_crs(self, bbox, shape, old_crs, new_crs):
        y1, x1 = self.bbox_indices(bbox=bbox, shape=shape)
        yx1 = np.vstack(np.dstack(np.meshgrid(y1, x1, indexing='ij')))
        yx2 = self._convert_grid_indices_crs(yx1, old_crs, new_crs)
        return yx2

    def _convert_grid_indices_crs(self, grid_indices, old_crs, new_crs):
        x2, y2 = pyproj.transform(old_crs, new_crs, grid_indices[:,1],
                                  grid_indices[:,0])
        yx2 = np.column_stack([y2, x2])
        return yx2

    def _convert_outer_indices_crs(self, bbox, shape, old_crs, new_crs):
        y1, x1 = self.bbox_indices(bbox=bbox, shape=shape)
        lx, _ = pyproj.transform(old_crs, new_crs,
                                  x1, np.repeat(y1[0], len(x1)))
        rx, _ = pyproj.transform(old_crs, new_crs,
                                  x1, np.repeat(y1[-1], len(x1)))
        __, by = pyproj.transform(old_crs, new_crs,
                                  np.repeat(x1[0], len(y1)), y1)
        __, uy = pyproj.transform(old_crs, new_crs,
                                  np.repeat(x1[-1], len(y1)), y1)
        return by, uy, lx, rx

    # Not a good idea to do it this way
    # def to_crs(self, new_crs, old_crs=None, to_regular=False, preserve_units=False):
    #     old_bbox = self.bbox
    #     if old_crs is None:
    #         old_crs = self.crs
    #     if (isinstance(new_crs, str) or isinstance(new_crs, dict)):
    #         new_crs = pyproj.Proj(new_crs, preserve_units=preserve_units)
    #     # TODO: Should test for regularity instead
    #     self.is_regular = to_regular
    #     self._grid_indices = self._convert_bbox_indices_crs(old_bbox,
    #                                                         self.shape,
    #                                                         old_crs,
    #                                                         new_crs)
    #     ymin = self._grid_indices[:, 0].min()
    #     ymax = self._grid_indices[:, 0].max()
    #     xmin = self._grid_indices[:, 1].min()
    #     xmax = self._grid_indices[:, 1].max()
    #     self._bounds = (xmin, ymin, xmax, ymax)
    #     self._bbox = self._convert_bbox_crs(self.bbox, old_crs, new_crs)
    #     self.crs = new_crs

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

    def clip_to(self, data_name, precision=7, inplace=True, **kwargs):
        """
        Clip grid to bbox representing the smallest area that contains all
        non-null data for a given dataset. If inplace is True, will set
        self.bbox to the bbox generated by this method.
 
        Parameters
        ----------
        data_name : numpy ndarray
                    Name of attribute to base the clip on.
        inplace : bool
                  If True, update bbox to conform to clip.
        precision : int
                    Precision to use when matching geographic coordinates.
 
        Other keyword arguments are passed to self.set_bbox
        """
        # get class attributes
        data = getattr(self, data_name)
        nodata = self.grid_props[data_name]['nodata']
        # get bbox of nonzero entries
        # TODO: This won't work for nans
        nz = np.nonzero(data != nodata)
        nz_ix = (nz[0].min() - 1, nz[0].max(), nz[1].min(), nz[1].max() + 1)
        # if inplace is True, clip all grids to new bbox and set self.bbox
        if inplace:
            selfrows, selfcols = \
                    self.bbox_indices(self.grid_props[data_name]['bbox'],
                                      self.grid_props[data_name]['shape'],
                                      precision=precision)
            new_bbox = (selfcols[nz_ix[2]], selfrows[nz_ix[1]],
                        selfcols[nz_ix[3]], selfrows[nz_ix[0]])
            # set self.bbox to clipped bbox
            self.set_bbox(new_bbox, **kwargs)
        else:
            # if inplace is False, return the clipped data
            return data[nz_ix[0]:nz_ix[1], nz_ix[2]:nz_ix[3]]

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, new_bbox):
        self.set_bbox(new_bbox)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def extent(self):
        extent = (self._bbox[0], self.bbox[2], self._bbox[1], self._bbox[3])
        return extent

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, new_crs):
        assert isinstance(new_crs, pyproj.Proj)
        self._crs = new_crs

    @property
    def cellsize(self):
        dy, dx = self._dy_dx()
        # TODO: Assuming square cells
        cellsize = (dy + dx) / 2
        return cellsize

    def set_bbox(self, new_bbox, precision=7):
        """
        Set the bounding box of the class instance (self.bbox). If the new
        bbox is not alignable to self.cellsize, each entry is automatically
        rounded such that the bbox is alignable.
 
        Parameters
        ----------
        new_bbox : tuple
                   New bbox to use (xmin, ymin, xmax, ymax).
        precision : int
                    Precision to use when matching geographic coordinates.
        """
        # check validity of new bbox
        if ((hasattr(new_bbox, "__len__")) and (not isinstance(new_bbox, str))
                and (len(new_bbox) == 4)):
            new_bbox = tuple(new_bbox)
        else:
            raise TypeError('new_bbox must be a tuple of length 4.')
        # check if alignable; if not, round unaligned bbox entries to nearest
        dy, dx = self._dy_dx()
        new_bbox = np.asarray(new_bbox)
        err = np.abs(new_bbox)
        err[[1, 3]] = err[[1, 3]] % dy
        err[[0, 2]] = err[[0, 2]] % dx
        try:
            np.testing.assert_almost_equal(err, np.zeros(len(new_bbox)),
                                           decimal=precision)
        except AssertionError:
            err_bbox = new_bbox
            direction = np.where(new_bbox > 0.0, 1, -1)
            new_bbox = new_bbox - (err * direction)
            print('Unalignable bbox provided: {0}.\nRounding to {1}'.format(err_bbox,
                  new_bbox))
        # construct arrays representing old bbox coords
        selfrows, selfcols = self.bbox_indices(self.bbox, self.shape)
        # construct arrays representing coordinates of new grid
        nrows = ((new_bbox[3] - new_bbox[1]) / dy)
        ncols = ((new_bbox[2] - new_bbox[0]) / dx)
        np.testing.assert_almost_equal(nrows, round(nrows), decimal=precision)
        np.testing.assert_almost_equal(ncols, round(ncols), decimal=precision)
        rows = np.linspace(new_bbox[1], new_bbox[3],
                           round(nrows), endpoint=False)
        cols = np.linspace(new_bbox[0], new_bbox[2],
                           round(ncols), endpoint=False)
        # set class attributes
        self._bbox = tuple(new_bbox)
        self.shape = tuple([len(rows), len(cols)])
        if hasattr(self, 'catch'):
            self.catchment_mask()
        else:
            self.mask = np.ones(self.shape, dtype=np.bool)

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
            old_nodata = self.grid_props[data_name]['nodata']
        data = getattr(self, data_name)
        np.place(data, data == old_nodata, new_nodata)
        self.grid_props[data_name]['nodata'] = new_nodata

    def catchment_mask(self, mask_source='catch'):
        """
        Masks grid cells not included in catchment. The catchment mask is saved
        to self.mask.
 
        Parameters
        ----------
        to_mask : string
                  Name of dataset to mask
        mask_source : string (optional)
                      Dataset on which mask is based (defaults to 'catch')
        """
        self.mask = (self.view(mask_source, apply_mask=False) !=
                     self.grid_props[mask_source]['nodata'])

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
        mask : bool
               If True, write the "masked" view of the dataset.
        delimiter : string (optional)
                    Delimiter to use in output file (defaults to ' ')
 
        Additional keyword arguments are passed to numpy.savetxt
        """
        if data_name is None:
            data_name = self.grid_props.keys()
        if file_name is None:
            file_name = self.grid_props.keys()
        if isinstance(data_name, str):
            data_name = [data_name]
        if isinstance(file_name, str):
            file_name = [file_name]
        header_space = 9*' '
        for in_name, out_name in zip(data_name, file_name):
            nodata = self.grid_props[in_name]['nodata']
            if view:
                shape = self.shape
                bbox = self.bbox
                # TODO: This breaks if cells are not square; issue with ASCII
                # format
                cellsize = self.cellsize
            else:
                shape = self.grid_props[in_name]['shape']
                bbox = self.grid_props[in_name]['bbox']
                cellsize = self.grid_props[in_name]['cellsize']
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
        fdir : numpy ndarray
               Array of flow direction data (overrides catchment_name constructor)
        acc : numpy ndarray
              Array of flow accumulation data (overrides accumulation_name constructor)
        threshold : int or float
                    Minimum allowed cell accumulation needed for inclusion in
                    river network.
        catchment_name : string
                         Name of attribute containing flow direction data. Must
                         be a catchment (all cells drain to a common point).
        accumulation_name : string
                         Name of attribute containing flow accumulation data.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
 
        Returns
        -------
        branches : list of numpy ndarray
                   A list of river segments. Each array contains the cell
                   indices of junctions in the segment.
        yx : numpy ndarray
             Ordered y and x coordinates of each cell.
        The x and y coordinates of each river segment can be obtained as
        follows:
 
        ```
        for branch in branches:
            coords = yx[branch]
            y, x = coords[:,0], coords[:,1]
        ```
        """
        # TODO: If two "forks" are directly connected, it can introduce a gap
        nodata_in = self._check_nodata_in(fdir, nodata_in)
        fdir_props = {}
        acc_props = {}
        fdir = self._input_handler(fdir, apply_mask=apply_mask, nodata_view=nodata_in, properties=fdir_props,
                                   ignore_metadata=ignore_metadata, **kwargs)
        acc = self._input_handler(acc, apply_mask=apply_mask, nodata_view=nodata_in, properties=acc_props,
                                   ignore_metadata=ignore_metadata, **kwargs)
        dirmap = self._set_dirmap(dirmap, fdir)
        flat_idx = np.arange(fdir.size)
        try:
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
                branch = branch[np.argsort(dist[branch])].tolist()
                fork = fdir.flat[branch[0]]
                branch = [fork] + branch
                branches.append(np.asarray(branch))
            # Handle case where two adjacent forks are connected
            after_fork = fdir.flat[forks_end]
            second_fork = np.unique(after_fork[np.in1d(after_fork, forks_end)])
            second_fork_start = start[np.in1d(end, second_fork)]
            second_fork_end = fdir.flat[second_fork_start]
            for fork_start, fork_end in zip(second_fork_start, second_fork_end):
                branches.append([fork_start, fork_end])
            # Get x, y coordinates for plotting
            yx = np.vstack(np.dstack(
                        np.meshgrid(*self.bbox_indices(self.bbox, self.shape), indexing='ij')))
        except:
            raise
        finally:
            self._unflatten_fdir(fdir, flat_idx, dirmap)
        return branches, yx

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

    def _set_dirmap(self, dirmap, direction_name, default_dirmap=(1, 2, 3, 4, 5, 6, 7, 8)):
        if dirmap is None:
            if isinstance(direction_name, str):
                if direction_name in self.grid_props:
                    dirmap = self.grid_props[direction_name].setdefault(
                        'dirmap', default_dirmap)
                else:
                    raise KeyError("{0} not found in grid instance"
                                   .format(direction_name))
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
        if len(dirmap) != 8:
            raise AssertionError('dirmap must be a sequence of length 8')
        # if data not provided, use self.dem
        # handle nodata values in dem
        if nodata_in is None:
            if isinstance(data, str):
                try:
                    nodata_in = self.grid_props[data]['nodata']
                except:
                    raise NameError("nodata value for '{0}' not found in instance."
                                    .format(data))
            else:
                raise KeyError("No 'nodata' value specified.")
        grid_props = {'nodata' : nodata_out, 'dirmap' : dirmap}
        dem = self._input_handler(data, apply_mask=apply_mask, properties=grid_props,
                                  ignore_metadata=ignore_metadata, **kwargs)
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
        return self._output_handler(fdir_flats, inplace, out_name=out_name, **grid_props)

