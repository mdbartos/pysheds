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
acc_in_frame = 77259
cells_in_catch = 11422
catch_shape = (159, 169)
max_distance = 208
new_crs = pyproj.Proj('+init=epsg:3083')
old_crs = pyproj.Proj('+init=epsg:4326')

# TODO: Need to test dtypes of different constructor methods

def test_dtype():
    assert(grid.dir.dtype == np.uint8)

def test_catchment():
    # Specify pour point
    x, y = -97.2937, 32.7371
    # Delineate the catchment
    grid.catchment(x, y, data='dir', dirmap=dirmap, out_name='catch',
                recursionlimit=15000, xytype='label')
    assert(np.count_nonzero(grid.catch) == cells_in_catch)

def test_clip():
    grid.clip_to('catch', precision=5)
    assert(grid.shape == catch_shape)
    assert(grid.view('catch').shape == catch_shape)

def test_accumulation():
    grid.accumulation(grid.dir, dirmap=dirmap, out_name='acc', pad=True,
                      **grid.grid_props['dir'])
    assert(grid.acc.max() == acc_in_frame)
    # TODO: Should eventually assert: grid.acc.dtype == np.min_scalar_type(grid.acc.max())
    grid.accumulation(data='catch', dirmap=dirmap, out_name='acc',
                      pad=True)
    assert(grid.acc.max() == cells_in_catch)

def test_flow_distance():
    pour_point_y, pour_point_x = np.unravel_index(np.argmax(grid.view('catch')), grid.shape)
    grid.flow_distance(pour_point_x, pour_point_y, data='catch', dirmap=dirmap, out_name='dist')
    assert(grid.dist[~np.isnan(grid.dist)].max() == max_distance)

def test_set_nodata():
    grid.set_nodata('dir', 0)

def test_to_ascii():
    grid.to_ascii('dir', 'test_dir.asc', view=False, mask=False)
    grid.read_ascii('test_dir.asc', 'dir_output', dtype=np.uint8)
    assert((grid.dir_output == grid.dir).all())
    grid.to_ascii('dir', 'test_dir.asc', view=True, mask=True)
    grid.read_ascii('test_dir.asc', 'dir_output', dtype=np.uint8)
    assert((grid.dir_output == grid.view('catch')).all())

def test_crs_conversion():
    catch = grid.view('catch')
    grid.to_crs(new_crs)
    t_catch = grid.view('catch')
    assert np.allclose(catch, t_catch)

