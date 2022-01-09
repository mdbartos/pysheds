import ast
import warnings
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
    Reads data from an ascii file and returns a Raster.

    Parameters
    ----------
    data : str
           File name or path.
    skiprows : int (optional)
                The number of rows taken up by the header (defaults to 6).
    mask : np.ndarray or Raster
            Boolean array to mask dataset.
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

def read_raster(data, band=1, window=None, window_crs=None, mask_geometry=False,
                nodata=None, metadata={}, **kwargs):
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
    nodata : int or float
             Value indicating 'no data' in raster file. If None, will attempt to read
             intended 'no data' value from raster file.
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
                warnings.warn('`mask_geometry` does not fall within the bounds of the raster.')
                mask = ~mask
        # If no `nodata` value specified, read intended nodata value from file
        if nodata is None:
            nodata = f.nodatavals[0]
            # If no `nodata` value in file, default to 0
            if nodata is None:
                warnings.warn('No `nodata` value detected. Defaulting to 0.')
                nodata = 0
            # Otherwise, set nodata to value found in file
            else:
                nodata = data.dtype.type(nodata)
    viewfinder = ViewFinder(affine=affine, shape=shape, mask=mask, nodata=nodata, crs=crs)
    out = Raster(data, viewfinder, metadata=metadata)
    return out

def to_ascii(data, file_name, target_view=None, delimiter=' ', fmt=None,
             interpolation='nearest', apply_input_mask=False,
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
                  ViewFinder to use when writing data. Defaults to data.viewfinder.
    delimiter : string (optional)
                Delimiter to use in output file (defaults to ' ')
    fmt : str
            Formatting for numeric data. Passed to np.savetxt.
    interpolation : 'nearest', 'linear'
                    Interpolation method to be used if spatial reference systems
                    are not congruent.
    apply_input_mask : bool
                        If True, mask the input Raster according to data.mask.
    apply_output_mask : bool
                        If True, mask the output Raster according to target_view.mask.
    inherit_nodata : bool
                     If True, output ascii inherits `nodata` value from `data`.
                     If False, output ascii uses `nodata` value from `target_view`.
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
        target_view = data.viewfinder
    data = View.view(data, target_view, interpolation=interpolation,
                     apply_input_mask=apply_input_mask,
                     apply_output_mask=apply_output_mask,
                     inherit_nodata=inherit_nodata, affine=affine, shape=shape,
                     crs=crs, mask=mask, nodata=nodata, dtype=dtype)
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

def to_raster(data, file_name, target_view=None, profile=None, blockxsize=256,
              blockysize=256, interpolation='nearest', apply_input_mask=False,
              apply_output_mask=True, inherit_nodata=True, affine=None,
              shape=None, crs=None, mask=None, nodata=None, dtype=None,
              **kwargs):
    """
    Writes gridded data to a raster.

    Parameters
    ----------
    data: Raster
          Raster dataset to write.
    file_name : str
                Name of file or path to write to.
    target_view : ViewFinder
                  ViewFinder to use when writing data. Defaults to data.viewfinder.
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
                        If True, mask the input Raster according to data.mask.
    apply_output_mask : bool
                        If True, mask the output Raster according to target_view.mask.
    inherit_nodata : bool
                     If True, output Raster inherits `nodata` value from `data`.
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
    """
    if target_view is None:
        target_view = data.viewfinder
    data = View.view(data, target_view, interpolation=interpolation,
                     apply_input_mask=apply_input_mask,
                     apply_output_mask=apply_output_mask,
                     inherit_nodata=inherit_nodata, affine=affine, shape=shape,
                     crs=crs, mask=mask, nodata=nodata, dtype=dtype)
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

