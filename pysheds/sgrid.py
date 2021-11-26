import sys
import ast
import copy
import warnings
import pyproj
import numpy as np
import pandas as pd
from numba import njit, prange
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
from pysheds.grid import Grid

_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_crs = lambda Proj: Proj.crs if not _OLD_PYPROJ else Proj
_pyproj_crs_is_geographic = 'is_latlong' if _OLD_PYPROJ else 'is_geographic'
_pyproj_init = '+init=epsg:4326' if _OLD_PYPROJ else 'epsg:4326'

from pysheds.view import Raster
from pysheds.view import BaseViewFinder, RegularViewFinder, IrregularViewFinder
from pysheds.view import RegularGridViewer, IrregularGridViewer

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

    def __init__(self, affine=Affine(0,0,0,0,0,0), shape=(1,1), nodata=0,
                 crs=pyproj.Proj(_pyproj_init),
                 mask=None):
        super().__init__(affine, shape, nodata, crs, mask)

    def _d8_flowdir(self, dem=None, dem_mask=None, out_name='dir', nodata_in=None, nodata_out=0,
                    pits=-1, flats=-1, dirmap=(64, 128, 1, 2, 4, 8, 16, 32), inplace=True,
                    as_crs=None, apply_mask=False, ignore_metadata=False, properties={},
                    metadata={}, **kwargs):
        try:
            # Make sure nothing flows to the nodata cells
            dem.flat[dem_mask] = dem.max() + 1
            # Optionally, project DEM before computing slopes
            if as_crs is not None:
                # TODO: Not implemented
                raise NotImplementedError()
            else:
                dx = abs(dem.affine.a)
                dy = abs(dem.affine.e)
            fdir = _d8_flowdir_par(dem, dx, dy, dirmap, flat=flats, pit=pits)
        except:
            raise
        finally:
            if nodata_in is not None:
                dem.flat[dem_mask] = nodata_in
        return self._output_handler(data=fdir, out_name=out_name, properties=properties,
                                    inplace=inplace, metadata=metadata)

    def _dinf_flowdir(self, dem=None, dem_mask=None, out_name='dir', nodata_in=None, nodata_out=0,
                      pits=-1, flats=-1, dirmap=(64, 128, 1, 2, 4, 8, 16, 32), inplace=True,
                      as_crs=None, apply_mask=False, ignore_metadata=False, properties={},
                      metadata={}, **kwargs):
        try:
            # Make sure nothing flows to the nodata cells
            dem.flat[dem_mask] = dem.max() + 1
            if as_crs is not None:
                # TODO: Not implemented
                raise NotImplementedError()
            else:
                dx = abs(dem.affine.a)
                dy = abs(dem.affine.e)
            fdir = _dinf_flowdir_par(dem, dx, dy, flat=flats, pit=pits)
            fdir = fdir % (2 * np.pi)
        except:
            raise
        finally:
            if nodata_in is not None:
                dem.flat[dem_mask] = nodata_in
        return self._output_handler(data=fdir, out_name=out_name, properties=properties,
                                    inplace=inplace, metadata=metadata)

    def _d8_catchment(self, x, y, fdir=None, pour_value=None, out_name='catch', dirmap=None,
                      nodata_in=None, nodata_out=0, xytype='index', recursionlimit=15000,
                      inplace=True, apply_mask=False, ignore_metadata=False, properties={},
                      metadata={}, snap='corner', **kwargs):

        try:
            # Pad the rim
            left, right, top, bottom = self._pop_rim(fdir, nodata=nodata_in)
            # get shape of padded flow direction array, then flatten
            # if xytype is 'label', delineate catchment based on cell nearest
            # to given geographic coordinate
            # Valid if the dataset is a view.
            if xytype == 'label':
                x, y = self.nearest_cell(x, y, fdir.affine, snap)
            # get the flattened index of the pour point
            catch = _d8_catchment_numba(fdir, (y, x), dirmap)
            if pour_value is not None:
                catch[y, x] = pour_value
        except:
            raise
        finally:
            # reset recursion limit
            self._replace_rim(fdir, left, right, top, bottom)
        return self._output_handler(data=catch, out_name=out_name, properties=properties,
                                    inplace=inplace, metadata=metadata)

    def _dinf_catchment(self, x, y, fdir=None, pour_value=None, out_name='catch', dirmap=None,
                        nodata_in=None, nodata_out=0, xytype='index', recursionlimit=15000,
                        inplace=True, apply_mask=False, ignore_metadata=False, properties={},
                        metadata={}, snap='corner', **kwargs):
        try:
            # Split dinf flowdir
            fdir_0, fdir_1, prop_0, prop_1 = _angle_to_d8(fdir, dirmap)
            # Find invalid cells
            invalid_cells = ((fdir < 0) | (fdir > (np.pi * 2)))
            # Pad the rim
            left_0, right_0, top_0, bottom_0 = self._pop_rim(fdir_0, nodata=nodata_in)
            left_1, right_1, top_1, bottom_1 = self._pop_rim(fdir_1, nodata=nodata_in)
            # Ensure proportion of flow is never zero
            fdir_0[prop_0 == 0] = fdir_1[prop_0 == 0]
            fdir_1[prop_1 == 0] = fdir_0[prop_1 == 0]
            # Set nodata cells to zero
            fdir_0[invalid_cells] = 0
            fdir_1[invalid_cells] = 0
            # TODO: This relies on the bbox of the grid instance, not the dataset
            # Valid if the dataset is a view.
            if xytype == 'label':
                x, y = self.nearest_cell(x, y, fdir.affine, snap)
            catch = _dinf_catchment_numba(fdir_0, fdir_1, (y, x), dirmap)
            # if pour point needs to be a special value, set it
            if pour_value is not None:
                catch[y, x] = pour_value
        except:
            raise
        return self._output_handler(data=catch, out_name=out_name, properties=properties,
                                    inplace=inplace, metadata=metadata)

    def _d8_accumulation(self, fdir=None, weights=None, dirmap=None, nodata_in=None,
                         nodata_out=0, efficiency=None, out_name='acc', inplace=True,
                         pad=False, apply_mask=False, ignore_metadata=False, properties={},
                         metadata={}, **kwargs):
        # Pad the rim
        if pad:
            fdir = np.pad(fdir, (1,1), mode='constant', constant_values=0)
        else:
            left, right, top, bottom = self._pop_rim(fdir, nodata=0)
        mintype = np.min_scalar_type(fdir.size)
        fdir_orig_type = fdir.dtype
        # Construct flat index onto flow direction array
        domain = np.arange(fdir.size, dtype=mintype)
        try:
            if nodata_in is None:
                nodata_cells = np.zeros_like(fdir).astype(bool)
            else:
                if np.isnan(nodata_in):
                    nodata_cells = (np.isnan(fdir))
                else:
                    nodata_cells = (fdir == nodata_in)
            invalid_cells = ~np.in1d(fdir.ravel(), dirmap)
            invalid_entries = fdir.flat[invalid_cells]
            fdir.flat[invalid_cells] = 0
            # Ensure consistent types
            fdir = fdir.astype(mintype)
            # Set nodata cells to zero
            fdir[nodata_cells] = 0
            # Get matching of start and end nodes
            startnodes, endnodes = self._construct_matching(fdir, domain,
                                                            dirmap=dirmap)
            if weights is not None:
                assert(weights.size == fdir.size)
                # TODO: Why flatten? Does this prevent weights from being modified?
                acc = weights.flatten()
            else:
                acc = (~nodata_cells).ravel().astype(int)

            if efficiency is not None:
                assert(efficiency.size == fdir.size)
                eff = efficiency.flatten() # must be flattened to avoid IndexError below
                acc = acc.astype(float)
                eff_max, eff_min = np.max(eff), np.min(eff)
                assert((eff_max<=1) and (eff_min>=0))

            indegree = np.bincount(endnodes)
            indegree = indegree.reshape(acc.shape).astype(np.uint8)
            startnodes = startnodes[(indegree == 0)]
            # separate for loop to avoid performance hit when
            # efficiency is None
            if efficiency is None:
                acc = _d8_accumulation_numba(acc, fdir, indegree, startnodes)
            else:
                raise NotImplementedError()
            acc = np.reshape(acc, fdir.shape)
            if pad:
                acc = acc[1:-1, 1:-1]
        except:
            raise
        finally:
        # Clean up
            self._unflatten_fdir(fdir, domain, dirmap)
            fdir = fdir.astype(fdir_orig_type)
            fdir.flat[invalid_cells] = invalid_entries
            if nodata_in is not None:
                fdir[nodata_cells] = nodata_in
            if pad:
                fdir = fdir[1:-1, 1:-1]
            else:
                self._replace_rim(fdir, left, right, top, bottom)
        return self._output_handler(data=acc, out_name=out_name, properties=properties,
                                    inplace=inplace, metadata=metadata)

    def _dinf_accumulation(self, fdir=None, weights=None, dirmap=None, nodata_in=None,
                           nodata_out=0, efficiency=None, out_name='acc', inplace=True,
                           pad=False, apply_mask=False, ignore_metadata=False,
                           properties={}, metadata={}, cycle_size=1, **kwargs):
        # Pad the rim
        if pad:
            fdir = np.pad(fdir, (1,1), mode='constant', constant_values=nodata_in)
        else:
            left, right, top, bottom = self._pop_rim(fdir, nodata=nodata_in)
        # Construct flat index onto flow direction array
        mintype = np.min_scalar_type(fdir.size)
        domain = np.arange(fdir.size, dtype=mintype)
        try:
            invalid_cells = ((fdir < 0) | (fdir > (np.pi * 2)))
            if nodata_in is None:
                nodata_cells = np.zeros_like(fdir).astype(bool)
            else:
                if np.isnan(nodata_in):
                    nodata_cells = (np.isnan(fdir))
                else:
                    nodata_cells = (fdir == nodata_in)
            # Split d-infinity grid
            fdir_0, fdir_1, prop_0, prop_1 = _angle_to_d8(fdir, dirmap)
            # Ensure consistent types
            fdir_0 = fdir_0.astype(mintype)
            fdir_1 = fdir_1.astype(mintype)
            # Set nodata cells to zero
            fdir_0[nodata_cells | invalid_cells] = 0
            fdir_1[nodata_cells | invalid_cells] = 0
            # Get matching of start and end nodes
            startnodes, endnodes_0 = self._construct_matching(fdir_0, domain, dirmap=dirmap)
            _, endnodes_1 = self._construct_matching(fdir_1, domain, dirmap=dirmap)
            # Remove cycles
            _dinf_fix_cycles(fdir_0, fdir_1, cycle_size)
            # Initialize accumulation array
            if weights is not None:
                assert(weights.size == fdir.size)
                acc = weights.flatten().astype(float)
            else:
                acc = (~nodata_cells).ravel().astype(float)
            if efficiency is not None:
                assert(efficiency.size == fdir.size)
                eff = efficiency.flatten()
                eff_max, eff_min = np.max(eff), np.min(eff)
                assert((eff_max<=1) and (eff_min>=0))
            # Ensure no flow directions with zero proportion
            fdir_0[prop_0 == 0] = fdir_1[prop_0 == 0]
            fdir_1[prop_1 == 0] = fdir_0[prop_1 == 0]
            prop_0[prop_0 == 0] = 0.5
            prop_1[prop_0 == 0] = 0.5
            prop_0[prop_1 == 0] = 0.5
            prop_1[prop_1 == 0] = 0.5
            # Initialize indegree
            indegree_0 = np.bincount(fdir_0.ravel(), minlength=fdir.size)
            indegree_1 = np.bincount(fdir_1.ravel(), minlength=fdir.size)
            indegree = (indegree_0 + indegree_1).astype(np.uint8)
            startnodes = startnodes[(indegree == 0)]
            if efficiency is None:
                acc = _dinf_accumulation_numba(acc, fdir_0, fdir_1, indegree,
                                               startnodes, prop_0, prop_1)
            else:
                raise NotImplementedError()
            # Reshape and offset accumulation
            acc = np.reshape(acc, fdir.shape)
            if pad:
                acc = acc[1:-1, 1:-1]
        except:
            raise
        finally:
            # Clean up
            if nodata_in is not None:
                fdir[nodata_cells] = nodata_in
            if pad:
                fdir = fdir[1:-1, 1:-1]
            else:
                self._replace_rim(fdir, left, right, top, bottom)
        return self._output_handler(data=acc, out_name=out_name, properties=properties,
                                    inplace=inplace, metadata=metadata)

    def _d8_flow_distance(self, x, y, fdir, weights=None, dirmap=None, nodata_in=None,
                          nodata_out=0, out_name='dist', method='shortest', inplace=True,
                          xytype='index', apply_mask=True, ignore_metadata=False, properties={},
                          metadata={}, snap='corner', **kwargs):
        if nodata_in is None:
            nodata_cells = np.zeros_like(fdir).astype(bool)
        else:
            if np.isnan(nodata_in):
                nodata_cells = (np.isnan(fdir))
            else:
                nodata_cells = (fdir == nodata_in)
        try:
            if xytype == 'label':
                x, y = self.nearest_cell(x, y, fdir.affine, snap)
            # TODO: Currently the size of weights is hard to understand
            if weights is not None:
                weights = weights.ravel()
            else:
                weights = (~nodata_cells).ravel().astype(int)
            dist = _d8_flow_distance_numba(fdir, weights, (y, x), dirmap)
        except:
            raise
        return self._output_handler(data=dist, out_name=out_name, properties=properties,
                                    inplace=inplace, metadata=metadata)

    def _dinf_flow_distance(self, x, y, fdir, weights=None, dirmap=None, nodata_in=None,
                            nodata_out=0, out_name='dist', method='shortest', inplace=True,
                            xytype='index', apply_mask=True, ignore_metadata=False,
                            properties={}, metadata={}, snap='corner', **kwargs):
        try:
            invalid_cells = ((fdir < 0) | (fdir > (np.pi * 2)))
            if nodata_in is None:
                nodata_cells = np.zeros_like(fdir).astype(bool)
            else:
                if np.isnan(nodata_in):
                    nodata_cells = (np.isnan(fdir))
                else:
                    nodata_cells = (fdir == nodata_in)
            # Split d-infinity grid
            fdir_0, fdir_1, prop_0, prop_1 = _angle_to_d8(fdir, dirmap)
            # Set nodata cells to zero
            fdir_0[nodata_cells | invalid_cells] = 0
            fdir_1[nodata_cells | invalid_cells] = 0
            if xytype == 'label':
                x, y = self.nearest_cell(x, y, fdir.affine, snap)
            # TODO: Currently the size of weights is hard to understand
            if weights is not None:
                if isinstance(weights, list) or isinstance(weights, tuple):
                    assert(isinstance(weights[0], np.ndarray))
                    weights_0 = weights[0].ravel()
                    assert(isinstance(weights[1], np.ndarray))
                    weights_1 = weights[1].ravel()
                    assert(weights_0.size == startnodes.size)
                    assert(weights_1.size == startnodes.size)
                elif isinstance(weights, np.ndarray):
                    assert(weights.shape[0] == startnodes.size)
                    assert(weights.shape[1] == 2)
                    weights_0 = weights[:,0]
                    weights_1 = weights[:,1]
            else:
                weights_0 = (~nodata_cells).ravel().astype(int)
                weights_1 = weights_0
            if method.lower() == 'shortest':
                dist = _dinf_flow_distance_numba(fdir_0, fdir_1, weights_0,
                                                 weights_1, (y, x), dirmap)
            else:
                raise NotImplementedError("Only implemented for shortest path distance.")
        except:
            raise
        # Prepare output
        return self._output_handler(data=dist, out_name=out_name, properties=properties,
                                    inplace=inplace, metadata=metadata)

    def compute_hand(self, fdir, dem, drainage_mask, out_name='hand', dirmap=None,
                     nodata_in_fdir=None, nodata_in_dem=None, nodata_out=np.nan, routing='d8',
                     inplace=True, apply_mask=False, ignore_metadata=False, return_index=False,
                     **kwargs):
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
        # TODO: Why does this use set_dirmap but flowdir doesn't?
        dirmap = self._set_dirmap(dirmap, fdir)
        nodata_in_fdir = self._check_nodata_in(fdir, nodata_in_fdir)
        nodata_in_dem = self._check_nodata_in(dem, nodata_in_dem)
        properties = {'nodata' : nodata_out}
        # TODO: This will overwrite metadata if provided
        metadata = {'dirmap' : dirmap}
        # initialize array to collect catchment cells
        fdir = self._input_handler(fdir, apply_mask=apply_mask, nodata_view=nodata_in_fdir,
                                   properties=properties, ignore_metadata=ignore_metadata,
                                   **kwargs)
        dem = self._input_handler(dem, apply_mask=apply_mask, nodata_view=nodata_in_dem,
                                  properties=properties, ignore_metadata=ignore_metadata,
                                  **kwargs)
        mask = self._input_handler(drainage_mask, apply_mask=apply_mask, nodata_view=0,
                                   properties=properties, ignore_metadata=ignore_metadata,
                                   **kwargs)
        assert (np.asarray(dem.shape) == np.asarray(fdir.shape)).all()
        assert (np.asarray(dem.shape) == np.asarray(mask.shape)).all()
        if routing.lower() == 'dinf':
            try:
                # Split dinf flowdir
                fdir_0, fdir_1, prop_0, prop_1 = _angle_to_d8(fdir, dirmap)
                # Find invalid cells
                invalid_cells = ((fdir < 0) | (fdir > (np.pi * 2)))
                # Pad the rim
                dirleft_0, dirright_0, dirtop_0, dirbottom_0 = self._pop_rim(fdir_0,
                                                                            nodata=nodata_in_fdir)
                dirleft_1, dirright_1, dirtop_1, dirbottom_1 = self._pop_rim(fdir_1,
                                                                            nodata=nodata_in_fdir)
                maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=0)
                # Ensure proportion of flow is never zero
                fdir_0[prop_0 == 0] = fdir_1[prop_0 == 0]
                fdir_1[prop_1 == 0] = fdir_0[prop_1 == 0]
                # Set nodata cells to zero
                fdir_0[invalid_cells] = 0
                fdir_1[invalid_cells] = 0
                hand = _dinf_hand_iter(dem, mask, fdir_0, fdir_1, dirmap)
                if not return_index:
                    hand = _assign_hand_heights(hand, dem, nodata_out)
            except:
                raise
            finally:
                self._replace_rim(fdir_0, dirleft_0, dirright_0, dirtop_0, dirbottom_0)
                self._replace_rim(fdir_1, dirleft_1, dirright_1, dirtop_1, dirbottom_1)
                self._replace_rim(mask, maskleft, maskright, masktop, maskbottom)
            return self._output_handler(data=hand, out_name=out_name, properties=properties,
                                        inplace=inplace, metadata=metadata)

        elif routing.lower() == 'd8':
            try:
                dirleft, dirright, dirtop, dirbottom = self._pop_rim(fdir, nodata=nodata_in_fdir)
                maskleft, maskright, masktop, maskbottom = self._pop_rim(mask, nodata=0)
                hand = _d8_hand_iter(dem, mask, fdir, dirmap)
                if not return_index:
                    hand = _assign_hand_heights(hand, dem, nodata_out)
            except:
                raise
            finally:
                self._replace_rim(fdir, dirleft, dirright, dirtop, dirbottom)
                self._replace_rim(mask, maskleft, maskright, masktop, maskbottom)
            return self._output_handler(data=hand, out_name=out_name, properties=properties,
                                        inplace=inplace, metadata=metadata)

    def resolve_flats(self, data=None, out_name='inflated_dem', nodata_in=None, nodata_out=None,
                      inplace=True, apply_mask=False, ignore_metadata=False, eps=1e-5,
                      max_iter=1000, **kwargs):
        """
        Resolve flats in a DEM using the modified method of Garbrecht and Martz (1997).
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
        # handle nodata values in dem
        nodata_in = self._check_nodata_in(data, nodata_in)
        if nodata_out is None:
            nodata_out = nodata_in
        grid_props = {'nodata' : nodata_out}
        metadata = {}
        dem = self._input_handler(data, apply_mask=apply_mask, properties=grid_props,
                                  ignore_metadata=ignore_metadata, metadata=metadata, **kwargs)
        if nodata_in is None:
            dem_mask = np.array([]).astype(int)
        else:
            if np.isnan(nodata_in):
                dem_mask = np.where(np.isnan(dem.ravel()))[0]
            else:
                dem_mask = np.where(dem.ravel() == nodata_in)[0]
        inside = np.arange(dem.size, dtype=np.int64).reshape(dem.shape)[1:-1, 1:-1].ravel()
        fdirs_defined, flats, higher_cells = _par_get_candidates(dem, inside)
        labels, numlabels = skimage.measure.label(flats, return_num=True)
        hec = _par_get_high_edge_cells(inside, fdirs_defined, higher_cells, labels)
        # TODO: lhl no longer needed
        lec, lhl = _par_get_low_edge_cells(inside, dem, fdirs_defined, labels, numlabels)
        grad_from_higher = _grad_from_higher(hec, flats, labels, numlabels)
        grad_towards_lower = _grad_towards_lower(lec, flats, dem, max_iter)
        new_drainage_grad = (2 * grad_towards_lower + grad_from_higher)
        inflated_dem = dem + eps * new_drainage_grad
        return self._output_handler(data=inflated_dem, out_name=out_name, properties=grid_props,
                                    inplace=inplace, metadata=metadata)


# Functions for 'flowdir'

@njit(parallel=True)
def _d8_flowdir_par(dem, dx, dy, dirmap, flat=-1, pit=-2):
    fdir = np.zeros(dem.shape, dtype=np.int64)
    m, n = dem.shape
    dd = np.sqrt(dx**2 + dy**2)
    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([dy, dd, dx, dd, dy, dd, dx, dd])
    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
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

@njit
def _facet_flow(e0, e1, e2, d1=1, d2=1):
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

@njit(parallel=True)
def _dinf_flowdir_par(dem, x_dist, y_dist, flat=-1, pit=-2):
    m, n = dem.shape
    e1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    e2s = np.array([1, 1, 3, 3, 5, 5, 7, 7])
    d1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    d2s = np.array([2, 0, 4, 2, 6, 4, 0, 6])
    ac = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    af = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    angle = np.zeros(dem.shape, dtype=np.float64)
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
                angle[i, j] = (af[k_max] * r_max) + (ac[k_max] * np.pi / 2)
    return angle

@njit
def _angle_to_d8(angles, dirmap):
    n = angles.size
    mod = np.pi/4
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
    for i in range(n):
        angle = angles.flat[i]
        if np.isnan(angle):
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
        fdirs_0.flat[i] = fdir_0
        fdirs_1.flat[i] = fdir_1
        props_0.flat[i] = prop_0
        props_1.flat[i] = prop_1
    return fdirs_0, fdirs_1, props_0, props_1

# Functions for 'catchment'

@njit
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

@njit
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

@njit
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

@njit
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

# Functions for 'accumulation'

@njit
def _d8_accumulation_numba(acc, fdir, indegree, startnodes):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes[k]
        endnode = fdir.flat[startnode]
        _d8_accumulation_recursion(startnode, endnode, acc, fdir, indegree)
    return acc

@njit
def _d8_accumulation_recursion(startnode, endnode, acc, fdir, indegree):
    acc.flat[endnode] += acc.flat[startnode]
    indegree[endnode] -= 1
    if (indegree[endnode] == 0):
        new_startnode = endnode
        new_endnode = fdir.flat[endnode]
        _d8_accumulation_recursion(new_startnode, new_endnode, acc, fdir, indegree)

@njit
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

@njit
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

# Functions for 'flow_distance'

@njit
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
                                r_dirmap, 0, offsets)
    return dist

@njit
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

@njit
def _dinf_flow_distance_numba(fdir_0, fdir_1, weights_0, weights_1,
                              pour_point, dirmap):
    visits = np.zeros(fdir_0.shape, dtype=np.uint8)
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
                                  weights_0, weights_1, r_dirmap, 0, offsets)
    return dist

@njit
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

# Functions for 'resolve_flats'

@njit(parallel=True)
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
    return fdirs_defined, flats, higher_cells

@njit(parallel=False)
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

@njit(parallel=True)
def _par_get_low_edge_cells(inside, dem, fdirs_defined, labels, numlabels):
    n = inside.size
    offset = dem.shape[1]
    low_edge_cells = np.zeros(dem.shape, dtype=np.uint32)
    label_has_lec = np.zeros(numlabels, dtype=np.bool8)
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
                    label_has_lec.flat[label - 1] = True
    return low_edge_cells, label_has_lec

@njit
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

@njit
def _d8_hand_iter(dem, mask, fdir, dirmap):
    offset = dem.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])

    hand = -np.ones(dem.shape, dtype=np.int64)
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

@njit(parallel=False)
def _d8_hand_recursive(dem, parents, fdir, dirmap):
    n = parents.size
    offset = dem.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])

    hand = -np.ones(dem.shape, dtype=np.int64)

    for i in range(n):
        parent = parents[i]
        hand.flat[parent] = parent

    for i in range(n):
        parent = parents[i]
        _d8_hand_recursion(parent, parent, hand, offsets, r_dirmap)

    return hand

@njit(parallel=False)
def _d8_hand_recursion(child, parent, hand, offsets, r_dirmap):
    neighbors = offsets + child
    for k in range(8):
        neighbor = neighbors[k]
        points_to = (fdir.flat[neighbor] == r_dirmap[k])
        not_visited = (hand.flat[neighbor] == -1)
        if points_to and not_visited:
            hand.flat[neighbor] = parent
            _d8_hand_recursion(neighbor, parent, hand, offsets, r_dirmap)
    return 0

@njit
def _dinf_hand_iter(dem, mask, fdir_0, fdir_1, dirmap):
    offset = dem.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])

    hand = -np.ones(dem.shape, dtype=np.int64)
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

@njit(parallel=False)
def _dinf_hand_recursive(dem, parents, fdir_0, fdir_1, dirmap):
    n = parents.size
    offset = dem.shape[1]
    offsets = np.array([-offset, 1 - offset, 1,
                        1 + offset, offset, - 1 + offset,
                        - 1, - 1 - offset])
    r_dirmap = np.array([dirmap[4], dirmap[5], dirmap[6],
                         dirmap[7], dirmap[0], dirmap[1],
                         dirmap[2], dirmap[3]])

    hand = -np.ones(dem.shape, dtype=np.int64)

    for i in range(n):
        parent = parents[i]
        hand.flat[parent] = parent

    for i in range(n):
        parent = parents[i]
        _dinf_hand_recursion(parent, parent, hand, offsets, r_dirmap)

    return hand

@njit(parallel=False)
def _dinf_hand_recursion(child, parent, hand, offsets, r_dirmap):
    neighbors = offsets + child
    for k in range(8):
        neighbor = neighbors[k]
        points_to = ((fdir_0.flat[neighbor] == r_dirmap[k]) |
                     (fdir_1.flat[neighbor] == r_dirmap[k]))
        not_visited = (hand.flat[neighbor] == -1)
        if points_to and not_visited:
            hand.flat[neighbor] = parent
            _dinf_hand_recursion(neighbor, parent, hand, offsets, r_dirmap)
    return 0

@njit(parallel=True)
def _assign_hand_heights(hand_idx, dem, nodata_out=np.nan):
    n = hand_idx.size
    hand = np.zeros(dem.shape, dtype=np.float64)
    for i in prange(n):
        j = hand_idx.flat[i]
        if j == -1:
            hand.flat[i] = np.nan
        else:
            hand.flat[i] = dem.flat[i] - dem.flat[j]
    return hand

@njit
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

@njit
def _dinf_fix_cycles(fdir_0, fdir_1, max_cycle_size):
    n = fdir_0.size
    visited = np.zeros(fdir_0.size, dtype=np.bool8)
    depth = 0
    for node in range(n):
        _dinf_fix_cycles_recursion(node, fdir_0, fdir_1, node,
                                   depth, max_cycle_size, visited)
        visited.flat[node] = True
    return 0

@njit
def _dinf_fix_cycles_recursion(node, fdir_0, fdir_1, ancestor,
                               depth, max_cycle_size, visited):
    if visited.flat[node]:
        return 0
    if depth > max_cycle_size:
        return 0
    left = fdir_0.flat[node]
    right = fdir_1.flat[node]
    if left == ancestor:
        fdir_0.flat[node] = right
        return 1
    else:
        _dinf_fix_cycles_recursion(left, fdir_0, fdir_1, ancestor,
                                   depth + 1, max_cycle_size, visited)
    if right == ancestor:
        fdir_1.flat[node] = left
        return 1
    else:
        _dinf_fix_cycles_recursion(right, fdir_0, fdir_1, ancestor,
                                   depth + 1, max_cycle_size, visited)

