import ast
import numpy as np
import pyproj
import rasterio
import rasterio.features
from affine import Affine
from distutils.version import LooseVersion
from pysheds.sview import Raster, ViewFinder, View

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_crs = lambda Proj: Proj.crs if not _OLD_PYPROJ else Proj
_pyproj_crs_is_geographic = 'is_latlong' if _OLD_PYPROJ else 'is_geographic'
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

def read_ascii(data, skiprows=6, mask=None, crs=pyproj.Proj(_pyproj_init),
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

def read_raster(data, band=1, window=None, window_crs=None,
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
            # If window crs not specified, assume it is in raster crs
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
            # No mask was applied if all False, out of bounds
            if not mask.any():
                # Return mask to all True and deliver warning
                warnings.warn('mask_geometry does not fall within the bounds of the raster!')
                mask = ~mask
        nodata = f.nodatavals[0]
    if nodata is not None:
        nodata = data.dtype.type(nodata)
    viewfinder = ViewFinder(affine=affine, shape=shape, mask=mask, nodata=nodata, crs=crs)
    out = Raster(data, viewfinder, metadata=metadata)
    return out

def to_ascii(data, file_name, target_view=None, delimiter=' ', fmt=None,
             interpolation='nearest', apply_input_mask=False,
             apply_output_mask=True, affine=None, shape=None, crs=None,
             mask=None, nodata=None, dtype=None, **kwargs):
    """
    Writes gridded data to ascii grid files.

    Parameters
    ----------
    data_name : str
                Attribute name of dataset to write.
    file_name : str
                Name of file to write to.
    view : bool
            If True, writes the "view" of the dataset. Otherwise, writes the
            entire dataset.
    delimiter : string (optional)
                Delimiter to use in output file (defaults to ' ')
    fmt : str
            Formatting for numeric data. Passed to np.savetxt.
    apply_mask : bool
            If True, write the "masked" view of the dataset.
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

    **kwargs are passed to np.savetxt
    """
    if target_view is None:
        target_view = data.viewfinder
    data = View.view(data, target_view, interpolation=interpolation,
                     apply_input_mask=apply_input_mask,
                     apply_output_mask=apply_output_mask, affine=affine,
                     shape=shape, crs=crs, mask=mask, nodata=nodata,
                     dtype=dtype)
    try:
        assert (abs(data.affine.a) == abs(data.affine.e))
    except:
        raise ValueError('Raster cells must be square.')
    nodata = data.nodata
    shape = data.shape
    bbox = data.bbox
    cellsize = abs(data.affine.a)
    # TODO: This breaks if cells are not square; issue with ASCII format
    header_space = 9*' '
    header = (("ncols{0}{1}\nnrows{0}{2}\nxllcorner{0}{3}\n"
                "yllcorner{0}{4}\ncellsize{0}{5}\nNODATA_value{0}{6}")
                .format(header_space,
                        shape[1],
                        shape[0],
                        bbox[0],
                        bbox[1],
                        cellsize,
                        nodata))
    if fmt is None:
        if np.issubdtype(data.dtype, np.integer):
            fmt = '%d'
        else:
            fmt = '%.18e'
    np.savetxt(file_name, data, fmt=fmt, delimiter=delimiter,
               header=header, comments='', **kwargs)

def to_raster(data, file_name, target_view=None, profile=None, view=True,
              blockxsize=256, blockysize=256, interpolation='nearest',
              apply_input_mask=False, apply_output_mask=True, affine=None,
              shape=None, crs=None, mask=None, nodata=None, dtype=None,
              **kwargs):
    """
    Writes gridded data to a raster.

    Parameters
    ----------
    data_name : str
                Attribute name of dataset to write.
    file_name : str
                Name of file to write to.
    profile : dict
                Profile of driver for writing data. See rasterio documentation.
    view : bool
            If True, writes the "view" of the dataset. Otherwise, writes the
            entire dataset.
    blockxsize : int
                    Size of blocks in horizontal direction. See rasterio documentation.
    blockysize : int
                    Size of blocks in vertical direction. See rasterio documentation.
    apply_mask : bool
            If True, write the "masked" view of the dataset.
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
    if target_view is None:
        target_view = data.viewfinder
    data = View.view(data, target_view, interpolation=interpolation,
                     apply_input_mask=apply_input_mask,
                     apply_output_mask=apply_output_mask, affine=affine,
                     shape=shape, crs=crs, mask=mask, nodata=nodata,
                     dtype=dtype)
    height, width = data.shape
    default_blockx = width
    default_profile = {
        'driver' : 'GTiff',
        'blockxsize' : blockxsize,
        'blockysize' : blockysize,
        'count': 1,
        'tiled' : True
    }
    if not profile:
        profile = default_profile
    profile_updates = {
        'crs' : data.crs.srs,
        'transform' : data.affine,
        'dtype' : data.dtype.name,
        'nodata' : data.nodata,
        'height' : height,
        'width' : width
    }
    profile.update(profile_updates)
    with rasterio.open(file_name, 'w', **profile) as dst:
        dst.write(np.asarray(data), 1)

