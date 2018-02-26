import pyproj
import warnings
import numpy as np
import pandas as pd
import sys
import ast
try:
    import scipy.sparse
    from scipy.sparse import csgraph
    import scipy.interpolate
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False
try:
    import rasterio
    _HAS_RASTERIO = True
except:
    _HAS_RASTERIO = False


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


    def __init__(self):
        self.grid_props = {}

    def add_data(self, data, data_name, bbox=None, shape=None, cellsize=None,
            crs=None, nodata=None, is_regular=None, data_attrs={}):
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
            self.is_regular = is_regular
            if is_regular:
                self._bounds = bbox
            else:
                try:
                    self._bounds = data_attrs['bounds']
                except:
                    self._bounds = bbox
                    warnings.warn("No bounds set. Assuming equal to bbox.")
        # assign new data to attribute; record nodata value
        self.grid_props.update({data_name : {}})
        self.grid_props[data_name].update({'bbox' : bbox})
        self.grid_props[data_name].update({'shape' : shape})
        self.grid_props[data_name].update({'cellsize' : cellsize})
        self.grid_props[data_name].update({'nodata' : nodata})
        self.grid_props[data_name].update({'crs' : crs})
        self.grid_props[data_name].update({'is_regular' : is_regular})
        for other_name, other_value in data_attrs.items():
            self.grid_props[data_name].update({other_name : other_value})
        if (is_regular) and (not 'bounds' in data_attrs):
            self.grid_props[data_name].update({'bounds' : bbox})
        else:
            if not 'bounds' in self.grid_props[data_name]:
                warnings.warn("No bounds set for dataset.")
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
        self.add_data(data, data_name, bbox, shape, cellsize, crs, nodata,
                      is_regular=True, data_attrs=data_attrs)

    def read_raster(self, data, data_name, band=1, data_attrs={}, **kwargs):
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
        f = rasterio.open(data, **kwargs)
        crs = pyproj.Proj(f.crs)
        bbox = tuple(f.bounds)
        shape = f.shape
        cellsize = f.affine[0]
        nodata = f.nodatavals[0]
        if len(f.indexes) > 1:
            data = np.ma.filled(f.read_band(band))
        else:
            data = np.ma.filled(f.read())
            f.close()
            data = data.reshape(shape)
        nodata = data.dtype.type(nodata)
        self.add_data(data, data_name, bbox, shape, cellsize, crs, nodata,
                      is_regular=True, data_attrs=data_attrs)

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

    def bbox_indices(self, bbox=None, shape=None, precision=7):
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
        rows = np.around(np.linspace(bbox[1], bbox[3],
               shape[0], endpoint=False)[::-1], precision)
        cols = np.around(np.linspace(bbox[0], bbox[2],
               shape[1], endpoint=False), precision)
        return rows, cols

    def view(self, data_name, mask=True, nodata=None, method='nearest',
             return_coords=False, tolerance=0.01):
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
        data_crs = self.grid_props[data_name]['crs']
        data_regular = self.grid_props[data_name]['is_regular']
        same_crs = self.crs.srs == data_crs.srs
        if self.is_regular and data_regular and same_crs:
            return self._regular_view(data_name, mask, nodata, return_coords, tolerance)
        else:
            return self._irregular_view(data_name, mask, nodata, method,
                                        return_coords, tolerance=tolerance)

    def _regular_view(self, data_name, mask=True, nodata=None,
                      return_coords=False, tolerance=0.01):
        data_bbox = self.grid_props[data_name]['bbox']
        data_shape = self.grid_props[data_name]['shape']
        dy, dx = self._dy_dx()
        x_tolerance = dx * tolerance
        y_tolerance = dy * tolerance
        if nodata is None:
            nodata = self.grid_props[data_name]['nodata']
        selfrows, selfcols = self.bbox_indices(self.bbox, self.shape)
        rows, cols = self.bbox_indices(data_bbox,
                                       data_shape)
        outview = (pd.DataFrame(getattr(self, data_name),
                                index=rows, columns=cols)
                   .reindex(selfrows, tolerance=y_tolerance, method='nearest')
                   .reindex(selfcols, axis=1, tolerance=x_tolerance,
                            method='nearest')
                   .fillna(nodata).values)
        if mask:
            outview = np.where(self.mask, outview, nodata)
        if return_coords:
            coords = np.vstack(np.dstack(np.meshgrid(selfrows, selfcols,
                                                     indexing='ij')))
            return outview, coords
        else:
            return outview

    def _irregular_view(self, data_name, mask=True, nodata=None,
                        method='nearest', return_coords=False, tolerance=0.01):
        data_bbox = self.grid_props[data_name]['bbox']
        data_shape = self.grid_props[data_name]['shape']
        data_crs = self.grid_props[data_name]['crs']
        data_regular = self.grid_props[data_name]['is_regular']
        if nodata is None:
            nodata = self.grid_props[data_name]['nodata']
        xmin = self.bounds[0]
        ymin = self.bounds[1]
        xmax = self.bounds[2]
        ymax = self.bounds[3]
        # If data is defined on a regular grid, run a pre-filter
        if data_regular:
            # Filter data by master grid bbox
            by, uy, lx, rx = self._convert_outer_indices_crs(data_bbox, data_shape,
                                                             data_crs, self.crs)
            # TODO: Should use max of cornerpoints instead of bbox
            by_bool = (by >= ymin) & (by <= ymax)
            uy_bool = (uy >= ymin) & (uy <= ymax)
            lx_bool = (lx >= xmin) & (lx <= xmax)
            rx_bool = (rx >= xmin) & (rx <= xmax)
            y_bool = (by_bool | uy_bool)
            x_bool = (lx_bool | rx_bool)
            # Ensure contiguous range
            y_bool[np.nonzero(y_bool)[0].min() : np.nonzero(y_bool)[0].max()] = 1
            x_bool[np.nonzero(x_bool)[0].min() : np.nonzero(x_bool)[0].max()] = 1
            data_y, data_x = self.bbox_indices(data_bbox, data_shape)
            data_y = data_y[y_bool]
            data_x = data_x[x_bool]
            yx_data = np.vstack(np.dstack(np.meshgrid(data_y, data_x, indexing='ij')))
            search_grid = getattr(self, data_name)[y_bool][:, x_bool].ravel()
            if self.crs.srs != data_crs.srs:
                yx_data = self._convert_grid_indices_crs(yx_data, data_crs, self.crs)
        else:
            #TODO: Is it possible to prefilter irregular data?
            yx_data = self.grid_props[data_name]['grid_indices']
            if self.crs.srs != data_crs.srs:
                yx_data = self._convert_grid_indices_crs(yx_data, data_crs, self.crs)
            y_bool = (yx_data[:, 0] >= ymin) & (yx_data[:, 0] <= ymax)
            x_bool = (yx_data[:, 1] >= xmin) & (yx_data[:, 1] <= xmax)
            yx_bool = (y_bool & x_bool)
            yx_data = yx_data[yx_bool]
            search_grid = getattr(self, data_name).ravel()[yx_bool]
        # Prepare master grid
        if not self.is_regular:
            yx_grid = self._grid_indices
        else:
            yx_grid = np.vstack(np.dstack(
                np.meshgrid(*self.bbox_indices(), indexing='ij')))
        outview = scipy.interpolate.griddata(yx_data, search_grid,
                                             yx_grid,
                                             method='nearest').reshape(self.shape)
        if mask:
            outview = np.where(self.mask, outview, nodata)
        if return_coords:
            return outview, yx_grid
        else:
            return outview

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

    def _input_handler(self, data, view=True, mask=True, **kwargs):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, str):
            if view:
                data = self.view(data, mask=mask, **kwargs)
            else:
                data = getattr(self, data)
            return data
        else:
            raise TypeError('Data must be a numpy ndarray or name string.')

    def flowdir(self, data=None, dem_name='dem', out_name='dir',
                include_edges=True, nodata_in=None, nodata_out=0, flat=-1,
            dirmap=(1, 2, 3, 4, 5, 6, 7, 8), inplace=True):
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
        if len(dirmap) != 8:
            raise AssertionError('dirmap must be a sequence of length 8')
        # if data not provided, use self.dem
        dem = self._input_handler(data, mask=False)
        # generate grid of indices
        indices = np.indices(dem.shape, dtype=np.min_scalar_type(dem.shape))
        # handle nodata values in dem
        if nodata_in is None:
            try:
                nodata_in = self.grid_props[dem_name]['nodata']
            except:
                raise NameError("nodata value for '{0}' not found in instance."
                                .format(dem_name))
        dem_mask = (dem == nodata_in)
        np.place(dem, dem_mask, np.iinfo(dem.dtype.type).max)
        # initialize indices of corners
        corners = {
        'nw' : {'k' : tuple(indices[:, 0, 0]),
                'v' : [[0, 1, 1],  [1, 1, 0]],
                'pad': np.array([3, 4, 5])},
        'ne' : {'k' : tuple(indices[:, 0, -1]),
                'v' : [[1, 1, 0],  [-1, -2, -2]],
                'pad': np.array([5, 6, 7])},
        'sw' : {'k' : tuple(indices[:, -1, 0]),
                'v' : [[-2, -2, -1],  [0, 1, 1]],
                'pad': np.array([1, 2, 3])},
        'se' : {'k' : tuple(indices[:, -1, -1]),
                'v' : [[-1, -2, -2],  [-2, -2, -1]],
                'pad': np.array([7, 8, 1])}
        }
        # initialize indices of edges
        edges = {
        'n' : {'k' : tuple(indices[:, 0, 1:-1]),
               'pad' : np.array([3, 4, 5, 6, 7])},
        'w' : {'k' : tuple(indices[:, 1:-1, 0]),
               'pad' : np.array([1, 2, 3, 4, 5])},
        'e' : {'k' : tuple(indices[:, 1:-1, -1]),
               'pad' : np.array([1, 5, 6, 7, 8])},
        's' : {'k' : tuple(indices[:, -1, 1:-1]),
               'pad' : np.array([1, 2, 3, 7, 8])}
        }
        # initialize indices of body (all cells except edges and corners)
        body = indices[:, 1:-1, 1:-1]
        # initialize output array
        min_dir_dtype = np.min_scalar_type(min(dirmap))
        max_dir_dtype = np.min_scalar_type(max(dirmap))
        nodata_dtype = np.min_scalar_type(nodata_out)
        min_out_dtype = np.find_common_type([], [min_dir_dtype, max_dir_dtype,
                                            nodata_dtype])
        outmap = np.full(self.shape, nodata_out, dtype=min_out_dtype)
        # for each entry in "body" determine flow direction based
        # on steepest neighboring slope
        for i, j in np.nditer(tuple(body), flags=['external_loop']):
            dat = dem[i, j]
            sur = dem[self._select_surround(i, j)]
            a = ((dat - sur) > 0).any(axis=0)
            b = np.argmax((dat - sur), axis=0) + 1
            c = flat
            outmap[i, j] = np.where(a, b, c)
        # determine flow direction for edges and corners, if desired
        if include_edges:
            # fill corners
            for corner in corners.keys():
                dat = dem[corners[corner]['k']]
                sur = dem[corners[corner]['v']]
                if ((dat - sur) > 0).any():
                    outmap[corners[corner]['k']] = \
                            corners[corner]['pad'][np.argmax(dat - sur)]
                else:
                    outmap[corners[corner]['k']] = flat
            # fill edges
            for edge in edges.keys():
                dat = dem[edges[edge]['k']]
                sur = dem[self._select_edge_sur(edges, edge)]
                a = ((dat - sur) > 0).any(axis=0)
                b = edges[edge]['pad'][np.argmax((dat - sur), axis=0)]
                c = flat
                outmap[edges[edge]['k']] = np.where(a, b, c)
        # If direction numbering isn't default, convert values of output array.
        if dirmap != (1, 2, 3, 4, 5, 6, 7, 8):
            dir_d = dict(zip((1, 2, 3, 4, 5, 6, 7, 8), dirmap))
            for k, v in dir_d.items():
                outmap[outmap == k] = v
            # outmap = (pd.DataFrame(outmap)
                      # .apply(lambda x: x.map(dir_d), axis=1).values)
        np.place(outmap, dem_mask, nodata_out)
        np.place(dem, dem_mask, nodata_in)
        is_regular = self.grid_props[dem_name].setdefault('is_regular', None)
        private_props = {'nodata' : nodata_out, 'dirmap' : dirmap,
                         'is_regular' : is_regular}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(outmap, inplace, out_name=out_name, **grid_props)

    def catchment(self, x, y, data=None, pour_value=None, direction_name='dir',
                  out_name='catch', dirmap=None,
                  nodata=0, xytype='index', bbox=None, recursionlimit=15000, inplace=True):
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
        nodata : int
                 Value to indicate nodata in output array.
        xytype : 'index' or 'label'
                 How to interpret parameters 'x' and 'y'.
                     'index' : x and y represent the column and row
                               indices of the pour point.
                     'label' : x and y represent geographic coordinates
                               (will be passed to self.nearest_cell).
        bbox :  tuple (length 4)
                Bounding box of flow direction array, if different from
                instance bbox.
        recursionlimit : int
                         Recursion limit--may need to be raised if
                         recursion limit is reached.
        inplace : bool
                  If True, catchment will be written to attribute 'catch'.
                  Otherwise, return the output array.
        """
        # TODO: No nodata_in attribute. Inconsistent.
        dirmap = self._set_dirmap(dirmap, direction_name)
        # initialize array to collect catchment cells
        self.collect = []
        # if data not provided, use self.dir
        # pad the flow direction grid with a rim of 'nodata' cells
        # easy way to prevent catchment search from going out of bounds
        # TODO: Need better way of doing this
        if data is not None:
            self.cdir = np.pad(data, 1, mode='constant')
        else:
            try:
                self.cdir = np.pad(self.view(direction_name, mask=False),
                    1, mode='constant',
                    constant_values=np.asscalar(self.grid_props['dir']['nodata']))
            except ValueError:
                self.cdir = np.pad(self.view(direction_name, mask=False),
                    1, mode='constant')
            except NameError:
                raise NameError("Flow direction grid '{0}' not found in instance."
                                .format(direction_name))
        # get shape of padded flow direction array, then flatten
        padshape = self.cdir.shape
        self.cdir = self.cdir.ravel()
        # if xytype is 'label', delineate catchment based on cell nearest
        # to given geographic coordinate
        # TODO: This relies on the bbox of the grid instance, not the dataset
        if xytype == 'label':
            x, y = self.nearest_cell(x, y, bbox,
                                     (padshape[0] - 1, padshape[1] - 1))
        # get the flattened index of the pour point
        pour_point = np.ravel_multi_index(np.array([y + 1, x + 1]), padshape)
        # reorder direction mapping to work with select_surround_ravel()
        r_dirmap = np.array(dirmap)[[4, 5, 6, 7, 0, 1, 2, 3]].tolist()
        pour_point = np.array([pour_point])
        # for each cell j, recursively search through grid to determine
        # if surrounding cells are in the contributing area, then add
        # flattened indices to self.collect
        def catchment_search(j):
            # self.collect = np.append(self.collect, j)
            self.collect.extend(j)
            selection = self._select_surround_ravel(j, padshape)
            next_idx = selection[np.where(self.cdir[selection] == r_dirmap)]
            if next_idx.any():
                return catchment_search(next_idx)
        try:
            # set recursion limit (needed for large datasets)
            sys.setrecursionlimit(recursionlimit)
            # call catchment search starting at the pour point
            catchment_search(pour_point)
            # initialize output array
            outcatch = np.zeros(padshape, dtype=int)
            # if nodata is not 0, replace 0 with nodata value in output array
            if nodata != 0:
                np.place(outcatch, outcatch == 0, nodata)
            # set values of output array based on 'collected' cells
            outcatch.flat[self.collect] = self.cdir[self.collect]
            # remove outer rim, delete temporary arrays
            outcatch = outcatch[1:-1, 1:-1]
            del self.cdir
            del self.collect
            # if pour point needs to be a special value, set it
            if pour_value is not None:
                outcatch[y, x] = pour_value
            # reset recursion limit
        except:
            raise
        finally:
            sys.setrecursionlimit(1000)
        is_regular = self.grid_props[direction_name].setdefault('is_regular', None)
        private_props = {'nodata' : nodata, 'dirmap' : dirmap,
                         'is_regular' : is_regular}
        grid_props = self._generate_grid_props(**private_props)
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
            np.arange(self.view('dir', mask=False).size).reshape(self.shape),
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

    def accumulation(self, data=None, weights=None, dirmap=None, direction_name='dir',
                     nodata=0, out_name='acc', inplace=True, pad=False):
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
        nodata : int
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
        fdir = self._input_handler(data, mask=False)
        # Pad the rim
        if pad:
            fdir = np.pad(fdir, (1,1), mode='constant')
        else:
            left, right, top, bottom = self._pop_rim(fdir)
        fdir_orig_type = fdir.dtype
        try:
            # Construct flat index onto flow direction array
            flat_idx = np.arange(fdir.size)
            # Ensure consistent types
            mintype = np.min_scalar_type(fdir.size)
            fdir = fdir.astype(mintype)
            flat_idx = flat_idx.astype(mintype)
            # Get matching of start and end nodes
            startnodes, endnodes = self._construct_matching(fdir, flat_idx,
                                                            dirmap=dirmap)
            if weights:
                assert(weights.size == fdir.size)
                acc = weights.ravel()
            else:
                acc = np.ones(fdir.shape).astype(int).ravel()
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
            # TODO: Should subtract weights if weighted?
            if weights:
                acc -= weights
            else:
                acc -= 1
            acc = np.reshape(acc, fdir.shape)
            if pad:
                acc = acc[1:-1, 1:-1]
        except:
            raise
        # Clean up
        finally:
            if pad:
                fdir = fdir[1:-1, 1:-1]
            else:
                self._replace_rim(fdir, left, right, top, bottom)
            fdir = fdir.astype(fdir_orig_type)
        is_regular = self.grid_props[direction_name].setdefault('is_regular', None)
        private_props = {'nodata' : nodata, 'is_regular' : is_regular}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(acc, inplace, out_name=out_name, **grid_props)

    def flow_distance(self, x, y, data, weights=None, dirmap=None, nodata_in=0,
                      nodata_out=0, out_name='dist', inplace=True, pad_inplace=True):
        """
        Generates an array representing the topological distance from each cell
        to the outlet.
 
        Parameters
        ----------
        x : int or float
            x coordinate of pour point
        y : int or float
            y coordinate of pour point
        data : numpy ndarray
               Array of flow direction data (overrides direction_name constructor)
        weights: numpy ndarray
                 Weights (distances) to apply to link edges.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        direction_name : string
                         Name of attribute containing flow direction data.
        nodata : int
                 Value to indicate nodata in output array.
        out_name : string
                   Name of attribute containing new flow distance array.
        inplace : bool
                  If True, accumulation will be written to attribute 'acc'.
                  Otherwise, return the output array.
        pad_inplace : bool
                  If True, do not include the edges of the flow direction array in the
                  accumulation computation. Otherwise, create a copy of the flow direction
                  array with the edges padded (this is more expensive in terms of
                  computational resources).
        """
        # TODO: Currently only accepts index-based x, y coords
        if not _HAS_SCIPY:
            raise ImportError('flow_distance requires scipy.sparse module')
        dirmap = self._set_dirmap(dirmap, data)
        fdir = self._input_handler(data, mask=True)
        # Construct flat index onto flow direction array
        flat_idx = np.arange(fdir.size)
        startnodes, endnodes = self._construct_matching(fdir, flat_idx,
                                                        dirmap=dirmap)
        # TODO: Currently the size of weights is hard to understand
        if weights:
            weights = weights.ravel()
            assert(weights.size == startnodes.size)
            assert(weights.size == endnodes.size)
        else:
            assert(startnodes.size == endnodes.size)
            weights = np.where(fdir == nodata_in, 0, 1).ravel().astype(int)
        C = scipy.sparse.lil_matrix((fdir.size, fdir.size))
        for i,j,w in zip(startnodes, endnodes, weights):
            C[i,j] = w
        C = C.tocsr()
        xyindex = np.ravel_multi_index((y, x), fdir.shape)
        dist = csgraph.shortest_path(C, indices=[xyindex], directed=False)
        dist[~np.isfinite(dist)] = np.nan
        dist = dist.ravel()
        dist = dist.reshape(fdir.shape)
        # Prepare output
        private_props = {'nodata' : nodata_out}
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(dist, inplace, out_name=out_name, **grid_props)

    def cell_area(self, out_name='area', nodata=0, inplace=True, as_crs=None):
        is_regular = self.is_regular
        if self.crs.is_latlong():
            warnings.warn(('CRS is geographic. Area will not have meaningful'
                           'units.'))
        if is_regular:
            indices = np.vstack(np.dstack(np.meshgrid(*self.bbox_indices,
                                                      indexing='ij')))
        else:
            indices = self._grid_indices
        # TODO: Add to_crs conversion here
        if as_crs:
            indices = self._convert_grid_indices(self.crs, as_crs,
                                                 indices[:,1], indices[:,0])
        dyy, dyx = np.gradient(indices[:, 0].reshape(self.shape))
        dxy, dxx = np.gradient(indices[:, 1].reshape(self.shape))
        dy = np.sqrt(dyy**2 + dyx**2)
        dx = np.sqrt(dxy**2 + dxx**2)
        area = dx * dy
        private_props = {'nodata' : nodata, 'is_regular' : is_regular}
        if not is_regular:
            private_props.update({'grid_indices' : self._grid_indices})
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(area, inplace, out_name=out_name, **grid_props)

    def cell_distances(self, direction_name, out_name='cdist',
                       inplace=True, as_crs=None):
        is_regular = self.is_regular
        if self.crs.is_latlong():
            warnings.warn(('CRS is geographic. Area will not have meaningful'
                           'units.'))
        if self.is_regular:
            indices = np.vstack(np.dstack(np.meshgrid(*self.bbox_indices,
                                                      indexing='ij')))
        else:
            indices = self._grid_indices
        if as_crs:
            indices = self._convert_grid_indices(self.crs, as_crs,
                                                 indices[:,1], indices[:,0])
        dyy, dyx = np.gradient(indices[:, 0].reshape(self.shape))
        dxy, dxx = np.gradient(indices[:, 1].reshape(self.shape))
        dy = np.sqrt(dyy**2 + dyx**2)
        dx = np.sqrt(dxy**2 + dxx**2)
        ddiag = np.sqrt(dy**2 + dx**2)
        cdist = np.zeros(self.shape)
        fdir = self.view(direction_name)
        dirmap = self.grid_props[direction_name]['dirmap']
        nodata = self.grid_props[direction_name]['nodata']
        for i, direction in enumerate(dirmap):
            if i in (0, 4):
                cdist[fdir == direction] = dy[fdir == direction]
            if i in (2, 6):
                cdist[fdir == direction] = dx[fdir == direction]
            else:
                cdist[fdir == direction] = ddiag[fdir == direction]
        # Prepare output
        private_props = {'nodata' : nodata, 'is_regular' : is_regular}
        if not is_regular:
            private_props.update({'grid_indices' : self._grid_indices})
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(cdist, inplace, out_name=out_name, **grid_props)

    def cell_dh(self, direction_name, dem_name, out_name='dh',
                    inplace=True, nodata_out=None, dirmap=None):
        if nodata_out is None:
            nodata_out = np.nan
        is_regular = self.is_regular
        dem = self.view(dem_name, nodata=np.nan)
        fdir = self.view(direction_name)
        dirmap = self._set_dirmap(dirmap, direction_name)
        flat_idx = np.arange(fdir.size)
        startnodes, endnodes = self._construct_matching(fdir, flat_idx, dirmap)
        startelev = dem.ravel()[startnodes].astype(np.float64)
        endelev = dem.ravel()[endnodes].astype(np.float64)
        dh = (startelev - endelev).reshape(self.shape)
        dh[np.isnan(dh)] = nodata_out
        # Prepare output
        private_props = {'nodata' : nodata_out, 'is_regular' : is_regular}
        if not is_regular:
            private_props.update({'grid_indices' : self._grid_indices})
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(dh, inplace, out_name=out_name, **grid_props)

    def cell_slopes(self, direction_name, dem_name, out_name='slopes',
                    inplace=True, nodata_out=None, dirmap=None, as_crs=None):
        if nodata_out is None:
            nodata_out = np.nan
        is_regular = self.is_regular
        dh = self.cell_dh(direction_name, dem_name, out_name, inplace=False,
                          nodata_out=nodata_out, dirmap=dirmap)
        cdist = self.cell_distances(direction_name, inplace=False, as_crs=as_crs)
        slopes = np.where(self.mask, dh/cdist, nodata_out)
        # Prepare output
        private_props = {'nodata' : nodata_out, 'is_regular' : is_regular}
        if not is_regular:
            private_props.update({'grid_indices' : self._grid_indices})
        grid_props = self._generate_grid_props(**private_props)
        return self._output_handler(slopes, inplace, out_name=out_name, **grid_props)

    def _output_handler(self, data, inplace, out_name, **kwargs):
        # TODO: Should this be rolled into add_data?
        if inplace:
            setattr(self, out_name, data)
            self.grid_props.update({out_name : {}})
            self.grid_props[out_name].update(kwargs)
        else:
            return data

    def _generate_grid_props(self, **kwargs):
        grid_props = {}
        required = ('bbox', 'shape', 'cellsize', 'nodata', 'crs', 'bounds')
        grid_props.update(kwargs)
        for param in required:
            grid_props[param] = grid_props.setdefault(param,
                                                      getattr(self, param))
        return grid_props

    def _pop_rim(self, data):
        left, right, top, bottom = (data[:,0].copy(), data[:,-1].copy(),
                                    data[0,:].copy(), data[-1,:].copy())
        data[:,0] = 0
        data[:,-1] = 0
        data[0,:] = 0
        data[-1,:] = 0
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

    def to_crs(self, new_crs, old_crs=None, to_regular=False, preserve_units=False):
        old_bbox = self.bbox
        if old_crs is None:
            old_crs = self.crs
        if (isinstance(new_crs, str) or isinstance(new_crs, dict)):
            new_crs = pyproj.Proj(new_crs, preserve_units=preserve_units)
        # TODO: Should test for regularity instead
        self.is_regular = to_regular
        self._grid_indices = self._convert_bbox_indices_crs(old_bbox,
                                                            self.shape,
                                                            old_crs,
                                                            new_crs)
        ymin = self._grid_indices[:, 0].min()
        ymax = self._grid_indices[:, 0].max()
        xmin = self._grid_indices[:, 1].min()
        xmax = self._grid_indices[:, 1].max()
        self._bounds = (xmin, ymin, xmax, ymax)
        self._bbox = self._convert_bbox_crs(self.bbox, old_crs, new_crs)
        self.crs = new_crs

    def _flatten_fdir(self, fdir, flat_idx, dirmap):
        # WARNING: This modifies fdir in place!
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
    def bounds(self):
        return self._bounds

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
        self.mask = (self.view(mask_source, mask=False) !=
                     self.grid_props[mask_source]['nodata'])

    def to_ascii(self, data_name=None, file_name=None, view=True, mask=False, delimiter=' ',
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
                np.savetxt(out_name, self.view(in_name, mask=mask), delimiter=delimiter,
                        header=header, comments='', **kwargs)
            else:
                np.savetxt(out_name, getattr(self, in_name), delimiter=delimiter,
                        header=header, comments='', **kwargs)

    def extract_river_network(self, fdir=None, acc=None, threshold=100,
                              catchment_name='catch', accumulation_name='acc',
                              dirmap=None):
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
        if fdir is None:
            try:
                fdir = self.view(catchment_name, mask=True)
            except:
                raise NameError("Flow direction grid '{0}' not found in instance."
                                .format(catchment_name))
        if acc is None:
            try:
                acc = self.view(accumulation_name, mask=True)
            except:
                raise NameError("Accumulation grid '{0}' not found in instance."
                                .format(accumulation_name))
        dirmap = self._set_dirmap(dirmap, catchment_name)
        flat_idx = np.arange(fdir.size)
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

    def _set_dirmap(self, dirmap, direction_name):
        # TODO: For transparency, default dirmap should be in kwargs
        default_dirmap = (1, 2, 3, 4, 5, 6, 7, 8)
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
        return dirmap

