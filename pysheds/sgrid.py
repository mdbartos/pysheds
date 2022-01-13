import sys
import ast
import copy
import pyproj
import numpy as np
import pandas as pd
import geojson
from affine import Affine
from distutils.version import LooseVersion
try:
    import skimage.measure
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
    Container class for holding, aligning, and manipulating gridded data.

    Attributes
    ==========
    viewfinder : Class containing all information about the coordinate system
                 of the grid object. Includes the `affine`, `shape`, `crs`,
                 `nodata` and `mask` attributes.
    affine : Affine transformation matrix (uses affine module).
    shape : The shape of the grid (number of rows, number of columns).
    crs : The coordinate reference system.
    nodata : The value indicating `no data`.
    mask : A boolean array used to mask grid cells; may be used to indicate
           which cells lie inside a catchment.
    bbox : The bounding box of the grid (xmin, ymin, xmax, ymax).
    extent : The extent of the grid (xmin, xmax, ymin, ymax).
    size : The number of cells in the grid.

    Methods
    =======
        --------
        File I/O
        --------
        read_ascii : Read an ascii grid from a file and return a Raster object.
        read_raster : Read a raster image file and return a Raster object.
        from_ascii : Initializes Grid from an ascii file and return a new Grid instance.
        from_raster : Initializes Grid from a raster image file or Raster object and
                      return a new Grid instance.
        to_ascii : Writes current "view" of a gridded dataset to an ascii file.
        to_raster : Writes current "view" of a gridded dataset to a raster image file.
        ----------
        Hydrologic
        ----------
        flowdir : Generate a flow direction grid from a given digital elevation dataset.
        catchment : Delineate the watershed for a given pour point (x, y).
        accumulation : Compute the number of cells upstream of each cell; if weights are
                       given, compute the sum of weighted cells upstream of each cell.
        distance_to_outlet : Compute the (weighted) distance from each cell to a given
                             pour point, moving downstream.
        distance_to_ridge : Compute the (weighted) distance from each cell to its originating
                            drainage divide, moving upstream.
        compute_hand : Compute the height above nearest drainage (HAND).
        stream_order : Compute the (strahler) stream order.
        extract_river_network : Extract river segments from a catchment and return a geojson
                                object.
        cell_dh : Compute the drop in elevation from each cell to its downstream neighbor.
        cell_distances : Compute the distance from each cell to its downstream neighbor.
        cell_slopes : Compute the slope between each cell and its downstream neighbor.
        fill_pits : Fill single-celled pits in a digital elevation dataset.
        fill_depressions : Fill multi-celled depressions in a digital elevation dataset.
        resolve_flats : Remove flats from a digital elevation dataset.
        detect_pits : Detect single-celled pits in a digital elevation dataset.
        detect_depressions : Detect multi-celled depressions in a digital elevation dataset.
        detect_flats : Detect flats in a digital elevation dataset.
        ---------------
        Data Processing
        ---------------
        view : Returns a "view" of a dataset defined by the grid's viewfinder.
        clip_to : Clip the viewfinder to the smallest area containing all non-
                  null gridcells for a provided dataset.
        nearest_cell : Returns the index (column, row) of the cell closest
                       to a given geographical coordinate (x, y).
        snap_to_mask : Snaps a set of points to the nearest nonzero cell in a boolean mask;
                       useful for finding pour points from an accumulation raster.
    """

    def __init__(self, viewfinder=None):
        if viewfinder is not None:
            try:
                assert isinstance(viewfinder, ViewFinder)
            except:
                raise TypeError('viewfinder must be an instance of ViewFinder.')
            self._viewfinder = viewfinder
        else:
            self._viewfinder = ViewFinder(**self.defaults)

    def __repr__(self):
        return repr(self.viewfinder)

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
        """
        Reads data from an ascii file and returns a Raster.

        Parameters
        ----------
        data : str
            File name or path.
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

        Additional keyword arguments (**kwargs) are passed to numpy.loadtxt()

        Returns
        -------
        out : Raster
            Raster object containing loaded data.
        """
        return pysheds.io.read_ascii(data, skiprows=skiprows, mask=mask,
                                     crs=crs, xll=xll, yll=yll, metadata=metadata,
                                     **kwargs)

    def read_raster(self, data, band=1, window=None, window_crs=None,
                    metadata={}, mask_geometry=False, **kwargs):
        """
        Reads data from a raster file and returns a Raster object.

        Parameters
        ----------
        data : str
            File name or path.
        band : int
            The band number to read if multiband.
        window : tuple
                If using windowed reading, specify window (xmin, ymin, xmax, ymax).
        window_crs : pyproj.Proj instance
                    Coordinate reference system of window. If None, use the raster file's crs.
        mask_geometry : iterable object
                        Geometries indicating where data should be read. The values must be a
                        GeoJSON-like dict or an object that implements the Python geo interface
                        protocol (such as a Shapely Polygon).
        metadata : dict
                    Other attributes describing dataset, such as direction
                    mapping for flow direction files. e.g.:
                    metadata={'dirmap' : (64, 128, 1, 2, 4, 8, 16, 32),
                                'routing' : 'd8'}

        Additional keyword arguments are passed to rasterio.open()

        Returns
        -------
        out : Raster
            Raster object containing loaded data.
        """
        return pysheds.io.read_raster(data=data, band=band, window=window,
                                      window_crs=window_crs, metadata=metadata,
                                      mask_geometry=mask_geometry, **kwargs)

    def to_ascii(self, data, file_name, target_view=None, delimiter=' ',
                 fmt=None, interpolation='nearest', apply_input_mask=False,
                 apply_output_mask=True, inherit_nodata=True, affine=None,
                 shape=None, crs=None, mask=None, nodata=None, dtype=None,
                 **kwargs):
        """
        Writes a Raster object to a formatted ascii text file.

        Parameters
        ----------
        data: Raster
            Raster dataset to write.
        file_name : str
                    Name of file or path to write to.
        target_view : ViewFinder
                    ViewFinder to use when writing data. Defaults to self.viewfinder.
        delimiter : string (optional)
                    Delimiter to use in output file (defaults to ' ')
        fmt : str
                Formatting for numeric data. Passed to np.savetxt.
        interpolation : 'nearest', 'linear'
                        Interpolation method to be used if spatial reference systems
                        are not congruent.
        apply_input_mask : bool
                            If True, mask the input Raster according to self.mask.
        apply_output_mask : bool
                            If True, mask the output Raster according to target_view.mask.
        inherit_nodata : bool
                         If True, output ascii inherits `nodata` value from `data`.
                         If False, output ascii uses `nodata` value from grid's viewfinder.
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

        Additional keyword arguments (**kwargs) are passed to np.savetxt
        """
        if target_view is None:
            target_view = self.viewfinder
        return pysheds.io.to_ascii(data, file_name, target_view=target_view,
                                   delimiter=delimiter, fmt=fmt, interpolation=interpolation,
                                   apply_input_mask=apply_input_mask,
                                   apply_output_mask=apply_output_mask,
                                   inherit_nodata=inherit_nodata,
                                   affine=affine, shape=shape, crs=crs,
                                   mask=mask, nodata=nodata,
                                   dtype=dtype, **kwargs)

    def to_raster(self, data, file_name, target_view=None, profile=None,
                  blockxsize=256, blockysize=256, interpolation='nearest',
                  apply_input_mask=False, apply_output_mask=True,
                  inherit_nodata=True, affine=None, shape=None, crs=None,
                  mask=None, nodata=None, dtype=None, **kwargs):
        """
        Writes gridded data to a raster.

        Parameters
        ----------
        data: Raster
            Raster dataset to write.
        file_name : str
                    Name of file or path to write to.
        target_view : ViewFinder
                    ViewFinder to use when writing data. Defaults to self.viewfinder.
        profile : dict
                    Profile of driver for writing data. See rasterio documentation.
        blockxsize : int
                        Size of blocks in horizontal direction. See rasterio documentation.
        blockysize : int
                        Size of blocks in vertical direction. See rasterio documentation.
        interpolation : 'nearest', 'linear'
                        Interpolation method to be used if spatial reference systems
                        are not congruent.
        apply_input_mask : bool
                            If True, mask the input Raster according to self.mask.
        apply_output_mask : bool
                            If True, mask the output Raster according to target_view.mask.
        inherit_nodata : bool
                         If True, output Raster inherits `nodata` value from `data`.
                         If False, output Raster uses `nodata` value from grid's viewfinder.
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
        """
        if target_view is None:
            target_view = self.viewfinder
        return pysheds.io.to_raster(data, file_name, target_view=target_view,
                                    profile=profile, blockxsize=blockxsize,
                                    blockysize=blockysize,
                                    interpolation=interpolation,
                                    apply_input_mask=apply_input_mask,
                                    apply_output_mask=apply_output_mask,
                                    inherit_nodata=inherit_nodata,
                                    affine=affine, shape=shape, crs=crs,
                                    mask=mask, nodata=nodata, dtype=dtype,
                                    **kwargs)

    @classmethod
    def from_ascii(cls, data, **kwargs):
        """
        Instantiates grid from an ascii text file.

        Parameters
        ----------
        data: str
              File path of ascii text file.

        Additional keyword arguments (**kwargs) are passed to self.read_ascii.

        Returns
        -------
        new_grid : Grid
                   A new Grid instance with its ViewFinder defined by the ascii file.
        """
        newinstance = cls()
        data = newinstance.read_ascii(data, **kwargs)
        newinstance.viewfinder = data.viewfinder
        return newinstance

    @classmethod
    def from_raster(cls, data, **kwargs):
        """
        Instantiates grid from a raster object or raster file.

        Parameters
        ----------
        data: Raster or str representing file path
              Raster data to use for instantiation.

        Additional keyword arguments (**kwargs) are passed to self.read_raster if
        data is a file path.

        Returns
        -------
        new_grid : Grid
                   A new Grid instance with its ViewFinder defined by the input raster.
        """
        newinstance = cls()
        if isinstance(data, Raster):
            newinstance.viewfinder = data.viewfinder
            return newinstance
        elif isinstance(data, str):
            data = newinstance.read_raster(data, **kwargs)
            newinstance.viewfinder = data.viewfinder
            return newinstance
        else:
            raise TypeError('`data` must be a Raster or str.')

    def view(self, data, data_view=None, target_view=None, interpolation='nearest',
             apply_input_mask=False, apply_output_mask=True, inherit_nodata=True,
             affine=None, shape=None, crs=None, mask=None, nodata=None,
             dtype=None, inherit_metadata=True, new_metadata={}, **kwargs):
        """
        Return a copy of a gridded dataset transformed to a new spatial reference system. The
        spatial reference system is determined by a ViewFinder instance, and is completely
        defined by an affine transformation matrix (affine), a desired shape (shape),
        a coordinate reference system (crs), a boolean mask (mask), and a sentinel value
        indicating `no data` (nodata). The target spatial reference system defaults to the
        `viewfinder` attribute of the Grid instance.

        Parameters
        ----------
        data : Raster
               A Raster object containing the gridded data and its spatial reference system
               (as defined by its ViewFinder).
        data_view : ViewFinder
                    The spatial reference system of the data. Defaults to the Raster dataset's
                    `viewfinder` attribute.
        target_view : ViewFinder
                      The desired spatial reference system. Defaults the the Grid instance's
                      `viewfinder` attribute.
        interpolation : 'nearest', 'linear'
                        Interpolation method to be used if spatial reference systems
                        are not congruent.
        apply_input_mask : bool
                           If True, mask the input Raster according to data.mask.
        apply_output_mask : bool
                           If True, mask the output Raster according to grid.mask.
        inherit_nodata : bool
                         If True, output Raster inherits `nodata` value from `data` or `data_view`.
                         If False, output Raster uses `nodata` value from grid's viewfinder.
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
                           If True, output Raster inherits metadata from input Raster.
        new_metadata : dict
                       Optional metadata to add to output Raster.

        Returns
        -------
        out : Raster
              View of the input Raster at the provided target view.
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
                        inherit_nodata=inherit_nodata,
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
        non-null data for a given dataset.

        Parameters
        ----------
        data : Raster
               Raster dataset to clip to.
        pad : tuple of ints (length 4)
              Apply padding to edges of new view (left, bottom, right, top). A pad of
              (1,1,1,1), for instance, will add a one-cell rim around the new view.
        """
        # get class attributes
        new_raster = View.trim_zeros(data, pad=pad)
        self.viewfinder = new_raster.viewfinder

    def flowdir(self, dem, routing='d8', flats=-1, pits=-2, nodata_out=None,
                dirmap=(64, 128, 1, 2, 4, 8, 16, 32), **kwargs):
        """
        Generates a flow direction raster from a DEM grid. Both d8 and d-infinity routing
        are supported.

        Parameters
        ----------
        dem : Raster
              Digital elevation model data.
        flats : int
                Value to indicate flat areas in output array.
        pits : int
               Value to indicate pits in output array.
        nodata_out : int or float
                     Value to indicate nodata in output array.
                       - If d8 routing is used, defaults to 0
                       - If dinf routing is used, defaults to np.nan
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        fdir : Raster
               Raster indicating flow directions.
               - If d8 routing is used, dtype is int64. Each cell indicates the flow
                 direction defined by dirmap.
               - If dinf routing is used, dtype is float64. Each cell indicates the flow
                 angle (from 0 to 2 pi radians).
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
                  nodata_out=False, xytype='coordinate', routing='d8', snap='corner',
                  algorithm='iterative', **kwargs):
        """
        Delineates a watershed from a given pour point (x, y).

        Parameters
        ----------
        x : float or int
            x coordinate (or index) of pour point
        y : float or int
            y coordinate (or index) of pour point
        fdir : Raster
               Flow direction data.
        pour_value : int or None
                     If not None, value to represent pour point in catchment
                     grid.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int or float
                     Value to indicate `no data` in output array.
        xytype : str
                 How to interpret parameters 'x' and 'y'.
                     'coordinate' : x and y represent geographic coordinates
                                    (will be passed to self.nearest_cell).
                     'index' : x and y represent the column and row
                               indices of the pour point.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions
        snap : str
               Function to use for self.nearest_cell:
               'corner' : numpy.around()
               'center' : numpy.floor()
        algorithm : str
                    Algorithm type to use:
                    'iterative' : Use an iterative algorithm (recommended).
                    'recursive' : Use a recursive algorithm.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        catch : Raster
                Raster indicating cells that lie in the catchment. The dtype will be
                np.bool8, unless `pour_value` is specified, in which case the dtype will
                be the smallest dtype capable of representing the pour value.
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
                                       nodata_out=nodata_out, xytype=xytype, snap=snap,
                                       algorithm=algorithm)
        elif routing.lower() == 'dinf':
            catch = self._dinf_catchment(x, y, fdir=fdir, pour_value=pour_value, dirmap=dirmap,
                                         nodata_out=nodata_out, xytype=xytype, snap=snap,
                                         algorithm=algorithm)
        return catch

    def _d8_catchment(self, x, y, fdir, pour_value=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                      nodata_out=False, xytype='coordinate', snap='corner',
                      algorithm='iterative'):
        # Pad the rim
        left, right, top, bottom = self._pop_rim(fdir, nodata=0)
        # If xytype is 'coordinate', delineate catchment based on cell nearest
        # to given geographic coordinate
        if xytype in {'label', 'coordinate'}:
            x, y = self.nearest_cell(x, y, fdir.affine, snap)
        # Delineate the catchment
        if algorithm.lower() == 'iterative':
            catch = _self._d8_catchment_iter_numba(fdir, (y, x), dirmap)
        elif algorithm.lower() == 'recursive':
            catch = _self._d8_catchment_recur_numba(fdir, (y, x), dirmap)
        else:
            raise ValueError('Algorithm must be `iterative` or `recursive`.')
        if pour_value is not None:
            catch[y, x] = pour_value
        catch = self._output_handler(data=catch, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return catch

    def _dinf_catchment(self, x, y, fdir, pour_value=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                        nodata_out=False, xytype='coordinate', snap='corner',
                        algorithm='iterative'):
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
        if algorithm.lower() == 'iterative':
            catch = _self._dinf_catchment_iter_numba(fdir_0, fdir_1, (y, x), dirmap)
        elif algorithm.lower() == 'recursive':
            catch = _self._dinf_catchment_recur_numba(fdir_0, fdir_1, (y, x), dirmap)
        else:
            raise ValueError('Algorithm must be `iterative` or `recursive`.')
        # if pour point needs to be a special value, set it
        if pour_value is not None:
            catch[y, x] = pour_value
        catch = self._output_handler(data=catch, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return catch

    def accumulation(self, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                     nodata_out=0., efficiency=None, routing='d8', cycle_size=1,
                     algorithm='iterative', **kwargs):
        """
        Generates a flow accumulation raster. If no weights are provided, the value of each cell
        is equal to the number of upstream cells. If weights are provided, the value of each cell
        is the sum of upstream weights.

        Parameters
        ----------
        fdir : Raster
               Flow direction data.
        weights: Raster
                 Weights to be applied to each accumulation cell. Defaults to the
                 vector of all ones.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        efficiency: Raster
                    Transport efficiency, relative correction factor applied to the
                    outflow of each cell. Nodata will be set to 1, i.e. no correction.
        nodata_out : int or float
                     Value to indicate nodata in output raster.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions
        cycle_size : int
                     Maximum length of cycles to check for in d-infinity grids. (Note
                     that d-infinity routing can generate cycles that will cause
                     the accumulation algorithm to abort. These cycles are removed prior
                     to running the d-infinity accumulation algorithm.)
        algorithm : str
                    Algorithm type to use:
                    'iterative' : Use an iterative algorithm (recommended).
                    'recursive' : Use a recursive algorithm.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        --------
        acc : Raster
              Raster indicating the (weighted) accumulation at each cell.
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
                                        efficiency=efficiency, algorithm=algorithm)
        elif routing.lower() == 'dinf':
            acc = self._dinf_accumulation(fdir, weights=weights, dirmap=dirmap,
                                          nodata_out=nodata_out,
                                          efficiency=efficiency,
                                          cycle_size=cycle_size, algorithm=algorithm)
        return acc

    def _d8_accumulation(self, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                         nodata_out=0., efficiency=None, algorithm='iterative', **kwargs):
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
        acc = np.asarray(acc)
        # If using efficiency, initialize array
        if efficiency is not None:
            eff = efficiency.astype(np.float64).reshape(fdir.shape)
            eff = np.asarray(eff)
        # Find indegree of all cells
        indegree = np.bincount(endnodes.ravel(), minlength=fdir.size).astype(np.uint8)
        # Set starting nodes to those with no predecessors
        startnodes = startnodes[(indegree == 0)]
        # Compute accumulation for no efficiency case
        if efficiency is None:
            if algorithm.lower() == 'iterative':
                acc = _self._d8_accumulation_iter_numba(acc, endnodes, indegree, startnodes)
            elif algorithm.lower() == 'recursive':
                acc = _self._d8_accumulation_recur_numba(acc, endnodes, indegree, startnodes)
            else:
                raise ValueError('Algorithm must be `iterative` or `recursive`.')
        # Compute accumulation for efficiency case
        else:
            if algorithm.lower() == 'iterative':
                acc = _self._d8_accumulation_eff_iter_numba(acc, endnodes, indegree,
                                                            startnodes, eff)
            elif algorithm.lower() == 'recursive':
                acc = _self._d8_accumulation_eff_recur_numba(acc, endnodes, indegree,
                                                             startnodes, eff)
            else:
                raise ValueError('Algorithm must be `iterative` or `recursive`.')
        acc = self._output_handler(data=acc, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return acc

    def _dinf_accumulation(self, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                           nodata_out=0., efficiency=None, cycle_size=1, algorithm='iterative',
                           **kwargs):
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
        acc = np.asarray(acc)
        if efficiency is not None:
            eff = efficiency.reshape(fdir.shape).astype(np.float64)
            eff = np.asarray(eff)
        # Find indegree of all cells
        indegree_0 = np.bincount(endnodes_0.ravel(), minlength=fdir.size)
        indegree_1 = np.bincount(endnodes_1.ravel(), minlength=fdir.size)
        indegree = (indegree_0 + indegree_1).astype(np.uint8)
        # Set starting nodes to those with no predecessors
        startnodes = startnodes[(indegree == 0)]
        # Compute accumulation for no efficiency case
        if efficiency is None:
            if algorithm.lower() == 'iterative':
                acc = _self._dinf_accumulation_iter_numba(acc, endnodes_0, endnodes_1,
                                                          indegree, startnodes, prop_0,
                                                          prop_1)
            elif algorithm.lower() == 'recursive':
                acc = _self._dinf_accumulation_recur_numba(acc, endnodes_0, endnodes_1,
                                                           indegree, startnodes, prop_0,
                                                           prop_1)
            else:
                raise ValueError('Algorithm must be `iterative` or `recursive`.')
        # Compute accumulation for efficiency case
        else:
            if algorithm.lower() == 'iterative':
                acc = _self._dinf_accumulation_eff_iter_numba(acc, endnodes_0, endnodes_1,
                                                            indegree, startnodes, prop_0,
                                                            prop_1, eff)
            elif algorithm.lower() == 'recursive':
                acc = _self._dinf_accumulation_eff_recur_numba(acc, endnodes_0, endnodes_1,
                                                               indegree, startnodes, prop_0,
                                                               prop_1, eff)
            else:
                raise ValueError('Algorithm must be `iterative` or `recursive`.')
        acc = self._output_handler(data=acc, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return acc

    def distance_to_outlet(self, x, y, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                           nodata_out=np.nan, routing='d8', method='shortest',
                           xytype='coordinate', snap='corner', algorithm='iterative', **kwargs):
        """
        Generates a raster representing the (weighted) topological distance from each cell
        to the outlet, moving downstream.

        Parameters
        ----------
        x : float or int
            x coordinate (or index) of pour point
        y : float or int
            y coordinate (or index) of pour point
        fdir : Raster
               Flow direction data.
        weights: Raster
                 Weights (distances) to apply to link edges. Defaults to the vector of
                 all ones.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int or float
                     Value to indicate nodata in output array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions
        xytype : 'coordinate' or 'index'
                 How to interpret parameters 'x' and 'y'.
                     'coordinate' : x and y represent geographic coordinates
                                    (will be passed to self.nearest_cell).
                     'index' : x and y represent the column and row
                               indices of the pour point.
        method : str
                 Method to use for distance calculation when multiple paths exist.
                 Currently, only shortest path distance is supported.
        snap : str
               Function to use on array for indexing:
               'corner' : numpy.around()
               'center' : numpy.floor()
        algorithm : str
                    Algorithm type to use:
                    'iterative' : Use an iterative algorithm (recommended).
                    'recursive' : Use a recursive algorithm.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        --------
        dist : Raster
               Raster indicating the (possibly weighted) distance from each cell to the outlet.
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
                                          snap=snap, algorithm=algorithm)
        elif routing.lower() == 'dinf':
            dist = self._dinf_flow_distance(x=x, y=y, fdir=fdir, weights=weights,
                                            dirmap=dirmap, nodata_out=nodata_out,
                                            method=method, xytype=xytype,
                                            snap=snap, algorithm=algorithm)
        return dist

    def _d8_flow_distance(self, x, y, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                          nodata_out=np.nan, method='shortest', xytype='coordinate',
                          snap='corner', algorithm='iterative', **kwargs):
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
        if algorithm.lower() == 'iterative':
            dist = _self._d8_flow_distance_iter_numba(fdir, weights, (y, x), dirmap)
        elif algorithm.lower() == 'recursive':
            dist = _self._d8_flow_distance_recur_numba(fdir, weights, (y, x), dirmap)
        else:
            raise ValueError('Algorithm must be `iterative` or `recursive`.')
        dist = self._output_handler(data=dist, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return dist

    def _dinf_flow_distance(self, x, y, fdir, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                            nodata_out=np.nan, method='shortest', xytype='coordinate',
                            snap='corner', algorithm='iterative', **kwargs):
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
            if algorithm.lower() == 'iterative':
                dist = _self._dinf_flow_distance_iter_numba(fdir_0, fdir_1, weights_0,
                                                            weights_1, (y, x), dirmap)
            elif algorithm.lower() == 'recursive':
                dist = _self._dinf_flow_distance_recur_numba(fdir_0, fdir_1, weights_0,
                                                             weights_1, (y, x), dirmap)
            else:
                raise ValueError('Algorithm must be `iterative` or `recursive`.')
        else:
            raise NotImplementedError("Only implemented for shortest path distance.")
        # Prepare output
        dist = self._output_handler(data=dist, viewfinder=fdir.viewfinder,
                                   metadata=fdir.metadata, nodata=nodata_out)
        return dist

    def compute_hand(self, fdir, dem, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                     nodata_out=None, routing='d8', return_index=False, algorithm='iterative',
                     **kwargs):
        """
        Computes the height above nearest drainage (HAND), based on a flow direction grid,
        a digital elevation grid, and a grid containing the locations of drainage channels.

        Parameters
        ----------
        fdir : Raster
               Flow direction data.
        dem : Raster
              Digital elevation data.
        mask : Raster
               Boolean raster with nonzero elements indicating
               locations of drainage channels.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int or float
                     Value to indicate nodata in output array.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions (not implemented)
        return_index : bool
                       Boolean value indicating desired output.
                       - If True, return a Raster where each cell indicates the index
                         of the (topologically) nearest channel cell.
                       - If False, return a Raster where each cell indicates the elevation
                         above the (topologically) nearest channel cell.
        algorithm : str
                    Algorithm type to use:
                    'iterative' : Use an iterative algorithm (recommended).
                    'recursive' : Use a recursive algorithm.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        hand : Raster
               Raster indicating either the index of the nearest channel cell, or the height
               above nearest drainage, depending on the value of the `return_index` parameter.
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
                nodata_out = np.nan
        # Compute height above nearest drainage
        if routing.lower() == 'd8':
            hand = self._d8_compute_hand(fdir=fdir, mask=mask, dirmap=dirmap,
                                         nodata_out=nodata_out,
                                         algorithm=algorithm)
        elif routing.lower() == 'dinf':
            hand = self._dinf_compute_hand(fdir=fdir, mask=mask,
                                           nodata_out=nodata_out,
                                           algorithm=algorithm)
        # If index is not desired, return heights
        if not return_index:
            hand = _self._assign_hand_heights_numba(hand, dem, nodata_out)
            hand = self._output_handler(data=hand, viewfinder=fdir.viewfinder,
                                        metadata=fdir.metadata, nodata=nodata_out)
        return hand

    def _d8_compute_hand(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                         nodata_out=-1, algorithm='iterative'):
        # Find nodata cells and invalid cells
        nodata_cells = self._get_nodata_cells(fdir)
        invalid_cells = ~np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)
        # Set nodata cells to zero
        fdir[nodata_cells] = 0
        fdir[invalid_cells] = 0
        dirleft, dirright, dirtop, dirbottom = self._pop_rim(fdir, nodata=0)
        maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=False)
        if algorithm.lower() == 'iterative':
            hand = _self._d8_hand_iter_numba(fdir, mask, dirmap)
        elif algorithm.lower() == 'recursive':
            hand = _self._d8_hand_recur_numba(fdir, mask, dirmap)
        else:
            raise ValueError('Algorithm must be `iterative` or `recursive`.')
        hand = self._output_handler(data=hand, viewfinder=fdir.viewfinder,
                                    metadata=fdir.metadata, nodata=-1)
        return hand

    def _dinf_compute_hand(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                           nodata_out=-1, algorithm='iterative'):
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
        if algorithm.lower() == 'iterative':
            hand = _self._dinf_hand_iter_numba(fdir_0, fdir_1, mask, dirmap)
        elif algorithm.lower() == 'recursive':
            hand = _self._dinf_hand_recur_numba(fdir_0, fdir_1, mask, dirmap)
        else:
            raise ValueError('Algorithm must be `iterative` or `recursive`.')
        hand = self._output_handler(data=hand, viewfinder=fdir.viewfinder,
                                    metadata=fdir.metadata, nodata=-1)
        return hand

    def extract_river_network(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                              routing='d8', algorithm='iterative', **kwargs):
        """
        Generates river segments from accumulation and flow_direction arrays.

        Parameters
        ----------
        fdir : Raster
               Flow direction data.
        mask : Raster
               Boolean raster indicating channelized regions
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        algorithm : str
                    Algorithm type to use:
                    'iterative' : Use an iterative algorithm (recommended).
                    'recursive' : Use a recursive algorithm.

        Additional keyword arguments (**kwargs) are passed to self.view.

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
        if algorithm.lower() == 'iterative':
            profiles = _self._d8_stream_network_iter_numba(endnodes, indegree,
                                                           orig_indegree, startnodes)
        elif algorithm.lower() == 'recursive':
            profiles = _self._d8_stream_network_recur_numba(endnodes, indegree,
                                                            orig_indegree, startnodes)
        else:
            raise ValueError('Algorithm must be `iterative` or `recursive`.')
        # Fill geojson dict with profiles
        featurelist = []
        for index, profile in enumerate(profiles):
            yi, xi = np.unravel_index(list(profile), fdir.shape)
            x, y = View.affine_transform(self.affine, xi, yi)
            line = geojson.LineString(np.column_stack([x, y]).tolist())
            featurelist.append(geojson.Feature(geometry=line, id=index))
        geo = geojson.FeatureCollection(featurelist)
        return geo

    def stream_order(self, fdir, mask, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                     nodata_out=0, routing='d8', algorithm='iterative', **kwargs):
        """
        Computes the Strahler stream order.

        Parameters
        ----------
        fdir : Raster
               Flow direction data.
        mask : Raster
               Boolean Raster indicating channelized regions
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int or float
                     Value to indicate nodata in output Raster.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        algorithm : str
                    Algorithm type to use:
                    'iterative' : Use an iterative algorithm (recommended).
                    'recursive' : Use a recursive algorithm.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        order : Raster
                Raster indicating Strahler stream order of each cell
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
        if algorithm.lower() == 'iterative':
            order = _self._d8_streamorder_iter_numba(min_order, max_order, order, endnodes,
                                                     indegree, orig_indegree, startnodes)
        elif algorithm.lower() == 'recursive':
            order = _self._d8_streamorder_recur_numba(min_order, max_order, order, endnodes,
                                                      indegree, orig_indegree, startnodes)
        else:
            raise ValueError('Algorithm must be `iterative` or `recursive`.')
        order = self._output_handler(data=order, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return order

    def distance_to_ridge(self, fdir, mask, weights=None, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                          nodata_out=0, routing='d8', algorithm='iterative', **kwargs):
        """
        Generates a raster representing the (weighted) topological distance from each cell
        to its originating drainage divide, moving upstream.

        Parameters
        ----------
        fdir : Raster
               Flow direction data.
        mask : Raster
               Boolean raster indicating channelized regions
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int or float
                     Value to indicate nodata in output raster.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
        algorithm : str
                    Algorithm type to use:
                    'iterative' : Use an iterative algorithm (recommended).
                    'recursive' : Use a recursive algorithm.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        rdist : Raster
                Raster indicating the (weighted) distance from each cell to its furthest
                upstream parent.
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
        if algorithm.lower() == 'iterative':
            rdist = _self._d8_reverse_distance_iter_numba(min_order, max_order, rdist,
                                                        endnodes, indegree, startnodes,
                                                        weights)
        elif algorithm.lower() == 'recursive':
            rdist = _self._d8_reverse_distance_recur_numba(min_order, max_order, rdist,
                                                           endnodes, indegree, startnodes,
                                                           weights)
        else:
            raise ValueError('Algorithm must be `iterative` or `recursive`.')
        rdist = self._output_handler(data=rdist, viewfinder=fdir.viewfinder,
                                     metadata=fdir.metadata, nodata=nodata_out)
        return rdist

    def cell_dh(self, dem, fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                nodata_out=np.nan, routing='d8', **kwargs):
        """
        Generates an array representing the elevation difference from each cell to its
        downstream neighbor(s).

        Parameters
        ----------
        dem : Raster
              Digital elevation dataset.
        fdir : Raster
               Flow direction data.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int or float
                     Value to indicate nodata in output raster.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions (not implemented)

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        dh : Raster
             Raster indicating elevation drop from each cell to its downstream neighbor(s).
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
        Generates an array representing the distance from each cell to its downstream neighbor(s).

        Parameters
        ----------
        fdir : Raster
               Flow direction data.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int or float
                     Value to indicate nodata in output raster.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions (not implemented)

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        cdist : Raster
                Raster indicating the distance from each cell to its downstream neighbor(s).

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
        Generates an array representing the slope between each cell and
        its downstream neighbor(s).

        Parameters
        ----------
        dem : Raster
              Digital elevation data.
        fdir : Raster
               Flow direction data.
        dirmap : list or tuple (length 8)
                 List of integer values representing the following
                 cardinal and intercardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        nodata_out : int or float
                     Value to indicate nodata in output raster.
        routing : str
                  Routing algorithm to use:
                  'd8'   : D8 flow directions
                  'dinf' : D-infinity flow directions (not implemented)

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        slopes : Raster
                 Raster indicating the slope between each cell and
                 its downstream neighbor(s).
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
        slopes = self._output_handler(data=slopes, viewfinder=dem.viewfinder,
                                      metadata=dem.metadata, nodata=nodata_out)
        return slopes

    def detect_pits(self, dem, **kwargs):
        """
        Detect single-celled pits in a digital elevation model.

        Parameters
        ----------
        dem : Raster
              Digital elevation data.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        pits : Raster
               Boolean Raster indicating locations of pits.
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
                                    metadata=dem.metadata, nodata=False)
        return pits

    def fill_pits(self, dem, nodata_out=None, **kwargs):
        """
        Fill single-celled pits in a digital elevation model. Raises pits to same elevation
        as lowest neighbor.

        Parameters
        ----------
        dem : Raster
              Digital elevation data.
        nodata_out : int or float
                     Value indicating no data in output raster.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        pit_filled_dem : Raster
                         Raster of digital elevation data with pits removed.
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
        pit_filled_dem = _self._fill_pits_numba(pit_filled_dem, pit_indices)
        # Set output nodata value
        if nodata_out is None:
            nodata_out = dem.nodata
        # Ensure nodata cells propagate to pit-filled dem
        pit_filled_dem[nodata_cells] = nodata_out
        pit_filled_dem = self._output_handler(data=pit_filled_dem,
                                              viewfinder=dem.viewfinder,
                                              metadata=dem.metadata)
        return pit_filled_dem

    def detect_depressions(self, dem, **kwargs):
        """
        Detect multi-celled depressions in a DEM.

        Parameters
        ----------
        dem : Raster
              Digital elevation data

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        depressions : Raster
                      Boolean Raster indicating locations of depressions.
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
        Fill multi-celled depressions in a DEM. Raises depressions to same elevation
        as lowest neighbor.

        Parameters
        ----------
        dem : Raster
              Digital elevation data
        nodata_out : int or float
                     Value indicating no data in output raster.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        flooded_dem : Raster
                      Raster representing digital elevation data with multi-celled
                      depressions removed.
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
        Detect flats in a digital elevation dataset.

        Parameters
        ----------
        dem : Raster
              Digital elevation data

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        flats : Raster
                Boolean Raster indicating locations of flats.
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
                                     metadata=dem.metadata, nodata=False)
        return flats

    def resolve_flats(self, dem, nodata_out=None, eps=1e-5, max_iter=1000, **kwargs):
        """
        Resolve flats in a DEM using the modified method of Barnes et al. (2015).
        See: https://arxiv.org/abs/1511.04433

        Parameters
        ----------
        dem : Raster
              Digital elevation dataset.
        nodata_out : int or float
                     Value to indicate nodata in output array.
        eps : float
              Step size to use when inflating flats. The inflated output digital elevation
              dataset will be equal to `dem + eps * drainage_gradient`, where the
              `drainage_gradient` is defined in Barnes et al. (2015).
        max_iter: int
                  Maximum number of iterations to use when computing the gradients from
                  higher and lower terrain, as defined in Barnes et al. (2015).


        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        inflated_dem : Raster
                       Raster representing digital elevation data with flats removed.
        """
        input_overrides = {'dtype' : np.float64}
        kwargs.update(input_overrides)
        dem = self._input_handler(dem, **kwargs)
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
        labels = labels.astype(np.int64)
        # Get high-edge cells
        hec = _self._par_get_high_edge_cells_numba(inside, fdirs_defined, higher_cells, labels)
        # Get low-edge cells
        lec = _self._par_get_low_edge_cells_numba(inside, dem, fdirs_defined, labels, numlabels)
        # Construct gradient from higher terrain
        grad_from_higher = _self._grad_from_higher_numba(hec, flats, labels, numlabels, max_iter)
        # Construct gradient towards lower terrain
        grad_towards_lower = _self._grad_towards_lower_numba(lec, flats, dem, max_iter)
        # Construct a gradient that is guaranteed to drain
        drainage_gradient = (2 * grad_towards_lower + grad_from_higher)
        # Create a flat-removed DEM by applying drainage gradient
        inflated_dem = np.asarray(dem + eps * drainage_gradient)
        inflated_dem = self._output_handler(data=inflated_dem,
                                            viewfinder=dem.viewfinder,
                                            metadata=dem.metadata)
        return inflated_dem

    def polygonize(self, data=None, mask=None, connectivity=4, transform=None, **kwargs):
        """
        Yield (polygon, value) for each set of adjacent pixels of the same value.
        Wrapper around rasterio.features.shapes

        From rasterio documentation:

        Parameters
        ----------
        data : Raster
               Data to polygonize. Defaults to `self.mask`.
        mask : Raster or np.ndarray
               Values of False or 0 will be excluded from feature generation.
        connectivity : 4 or 8 (int)
                       Use 4 or 8 pixel connectivity.
        transform : affine.Affine
                    Transformation from pixel coordinates of `image` to the
                    coordinate system of the input `shapes`.

        Additional keyword arguments (**kwargs) are passed to `self.view`.

        Returns
        -------
        shapes : generator
                 Iterable generator of polygons (see documentation for
                 rasterio.features.shapes)
        """
        if not _HAS_RASTERIO:
            raise ImportError('Requires rasterio module')
        if data is None:
            data = Raster(self.mask.astype(np.uint8),
                          viewfinder=self.viewfinder)
        data = self.view(data, affine=transform, mask=mask, **kwargs)
        mask = data.mask
        transform = data.affine
        shapes = rasterio.features.shapes(data, mask=mask, connectivity=connectivity,
                                          transform=transform)
        return shapes

    def rasterize(self, shapes, out_shape=None, fill=0, transform=None,
                  all_touched=False, default_value=1, dtype=None, mask=None,
                  crs=None):
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
        mask : np.ndarray
               Boolean mask indicating the mask of the resulting Raster.
        crs : pyproj.Proj
              Coordinate reference system of the desired Raster.

        Additional keyword arguments (**kwargs) are passed to `self.view`.

        Returns
        -------
        raster : Raster
                 Raster representing rasterized input geometries.
        """
        if not _HAS_RASTERIO:
            raise ImportError('Requires rasterio module')
        if out_shape is None:
            out_shape = self.shape
        if transform is None:
            transform = self.affine
        if mask is None:
            mask = self.mask
        if crs is None:
            crs = self.crs
        raster = rasterio.features.rasterize(shapes, out_shape=out_shape,
                                             fill=fill, transform=transform,
                                             all_touched=all_touched,
                                             default_value=default_value,
                                             dtype=dtype)
        viewfinder = ViewFinder(affine=transform, shape=out_shape,
                                nodata=fill, mask=mask, crs=crs)
        raster = Raster(raster, viewfinder=viewfinder)
        return raster

    def snap_to_mask(self, mask, xy, return_dist=False, **kwargs):
        """
        Snap a set of coordinates (given by `xy`) to the nearest nonzero cells in a
        boolean raster (given by `mask`). (Note that the mask raster is first mapped to the
        grid's ViewFinder using self.view).

        Parameters
        ----------
        mask : Raster
               A Raster dataset with nonzero elements indicating cells to match to (e.g:
               a flow accumulation grid with ones indicating cells above a certain threshold).
        xy : np.ndarray-like with shape (N, 2)
             Points to match (example: gage location coordinates).
        return_dist : If true, return the distances from xy to the nearest matched point in mask.

        Additional keyword arguments (**kwargs) are passed to self.view.

        Returns
        -------
        xy_new : np.ndarray with shape (N, 2)
                 Coordinates of nearest points where mask is nonzero.
        dist : np.ndarray with shape (N,), (optional)
               Distances from points in xy to xy_new
        """
        try:
            assert isinstance(mask, Raster)
        except:
            raise TypeError('`mask` must be a Raster instance.')
        mask_overrides = {'dtype' : np.bool8, 'nodata' : False}
        kwargs.update(mask_overrides)
        mask = self._input_handler(mask, **kwargs)
        affine = mask.affine
        return View.snap_to_mask(mask, xy, affine=affine,
                                 return_dist=return_dist)

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
