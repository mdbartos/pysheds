import numpy as np
from pysheds.grid import Grid

# Initialize grid
grid = Grid()
grid.read_ascii('../data/dir.asc', 'dir', dtype=np.uint8)
# Initialize parameters
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
acc_in_frame = 77259
cells_in_catch = 11422
catch_shape = (159, 169)
max_distance = 208

# TODO: Need to test dtypes of different constructor methods

def test_dtype():
    assert(grid.dir.dtype == np.uint8)

def test_catchment():
    # Specify pour point
    x, y = -97.2937, 32.7371
    # Delineate the catchment
    grid.catchment(x, y, dirmap=dirmap, out_name='catch',
                recursionlimit=15000, xytype='label')
    assert(np.count_nonzero(grid.catch) == cells_in_catch)

def test_clip():
    grid.clip_to('catch', precision=5)
    assert(grid.shape == catch_shape)
    assert(grid.view('catch').shape == catch_shape)

def test_accumulation():
    grid.accumulation(grid.dir, dirmap=dirmap, pad_inplace=False, out_name='acc')
    assert(grid.acc.max() == acc_in_frame)
    # Should eventually assert: grid.acc.dtype == np.min_scalar_type(grid.acc.max())
    grid.accumulation(direction_name='catch', dirmap=dirmap, pad_inplace=False, out_name='acc')
    assert(grid.acc.max() == cells_in_catch)

def test_flow_distance():
    pour_point_y, pour_point_x = np.unravel_index(np.argmax(grid.view('catch')), grid.shape)
    grid.flow_distance(pour_point_x, pour_point_y, dirmap=dirmap, out_name='dist')
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

