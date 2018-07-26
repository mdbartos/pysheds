import pyproj
import os
import numpy as np
from pysheds.grid import Grid

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '../data'))
data_path = os.path.join(data_dir, 'dir.asc')

# Initialize grid
grid = Grid()
crs = pyproj.Proj('+init=epsg:4326')
grid.read_ascii(data_path, 'dir', dtype=np.uint8, crs=crs)
# Initialize parameters
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
acc_in_frame = 76499
cells_in_catch = 11422
catch_shape = (159, 169)
max_distance = 209
new_crs = pyproj.Proj('+init=epsg:3083')
old_crs = pyproj.Proj('+init=epsg:4326')
x, y = -97.29416666666677, 32.73749999999989

# TODO: Need to test dtypes of different constructor methods

def test_dtype():
    assert(grid.dir.dtype == np.uint8)

def test_catchment():
    # Delineate the catchment
    grid.catchment(x, y, data='dir', dirmap=dirmap, out_name='catch',
                recursionlimit=15000, xytype='label')
    assert(np.count_nonzero(grid.catch) == cells_in_catch)

def test_clip():
    grid.clip_to('catch')
    assert(grid.shape == catch_shape)
    assert(grid.view('catch').shape == catch_shape)

def test_clip_pad():
    grid.clip_to('catch')
    no_pad = grid.view('catch')
    for p in (1, 4, 10):
        grid.clip_to('catch', pad=(p,p,p,p))
        assert((no_pad == grid.view('catch')[p:-p, p:-p]).all())
    # TODO: Should check for non-square padding

def test_accumulation():
    # TODO: This breaks if clip_to's padding of dir is nonzero
    grid.clip_to('dir')
    grid.accumulation(data='dir', dirmap=dirmap, out_name='acc')
    assert(grid.acc.max() == acc_in_frame)
    # TODO: Should eventually assert: grid.acc.dtype == np.min_scalar_type(grid.acc.max())
    grid.clip_to('catch', pad=(1,1,1,1))
    grid.accumulation(data='catch', dirmap=dirmap, out_name='acc')
    assert(grid.acc.max() == cells_in_catch)

def test_flow_distance():
    grid.clip_to('catch')
    grid.flow_distance(x, y, data='catch', dirmap=dirmap, out_name='dist', xytype='label')
    assert(grid.dist[~np.isnan(grid.dist)].max() == max_distance)

def test_set_nodata():
    grid.set_nodata('dir', 0)

def test_to_ascii():
    grid.clip_to('catch')
    grid.to_ascii('dir', 'test_dir.asc', view=False, apply_mask=False, dtype=np.float)
    grid.read_ascii('test_dir.asc', 'dir_output', dtype=np.uint8)
    assert((grid.dir_output == grid.dir).all())
    grid.to_ascii('dir', 'test_dir.asc', view=True, apply_mask=True, dtype=np.uint8)
    grid.read_ascii('test_dir.asc', 'dir_output', dtype=np.uint8)
    assert((grid.dir_output == grid.view('catch')).all())

def test_to_raster():
    grid.clip_to('catch')
    grid.to_raster('dir', 'test_dir.tif', view=False, apply_mask=False, blockxsize=16, blockysize=16)
    grid.read_raster('test_dir.tif', 'dir_output')
    assert((grid.dir_output == grid.dir).all())
    grid.to_raster('dir', 'test_dir.tif', view=True, apply_mask=True, blockxsize=16, blockysize=16)
    grid.read_raster('test_dir.tif', 'dir_output')
    assert((grid.dir_output == grid.view('catch')).all())
    # TODO: Write test for windowed reading

# def test_crs_conversion():
#     catch = grid.view('catch')
#     grid.to_crs(new_crs)
#     t_catch = grid.view('catch')
#     assert np.allclose(catch, t_catch)

