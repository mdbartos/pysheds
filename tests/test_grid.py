import pyproj
import os
import numpy as np
from pysheds.grid import Grid

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '../data'))
dir_path = os.path.join(data_dir, 'dir.asc')
dem_path = os.path.join(data_dir, 'dem.tif')

# Initialize grid
grid = Grid()
crs = pyproj.Proj('+init=epsg:4326', preserve_units=True)
grid.read_ascii(dir_path, 'dir', dtype=np.uint8, crs=crs)
grid.read_raster(dem_path, 'dem')
# Initialize parameters
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
acc_in_frame = 76499
cells_in_catch = 11422
catch_shape = (159, 169)
max_distance = 209
new_crs = pyproj.Proj('+init=epsg:3083')
old_crs = pyproj.Proj('+init=epsg:4326', preserve_units=True)
x, y = -97.29416666666677, 32.73749999999989


# TODO: Need to test dtypes of different constructor methods
def test_constructors():
    newgrid = grid.from_ascii(dir_path, 'dir', dtype=np.uint8, crs=crs)
    assert((newgrid.dir == grid.dir).all())
    del newgrid

def test_dtype():
    assert(grid.dir.dtype == np.uint8)

def test_catchment():
    # Reference routing
    grid.catchment(x, y, data='dir', dirmap=dirmap, out_name='catch',
                recursionlimit=15000, xytype='label')
    assert(np.count_nonzero(grid.catch) == cells_in_catch)

def test_clip():
    grid.clip_to('catch')
    assert(grid.shape == catch_shape)
    assert(grid.view('catch').shape == catch_shape)

def test_resolve_flats():
    flats = grid.detect_flats('dem')
    assert(flats.sum() > 100)
    grid.resolve_flats(data='dem', out_name='inflated_dem')
    flats = grid.detect_flats('inflated_dem')
    # TODO: Ideally, should show 0 flats
    assert(flats.sum() <= 30)

def test_flowdir():
    grid.clip_to('dir')
    grid.flowdir(data='inflated_dem', dirmap=dirmap, routing='d8', out_name='d8_dir')
    grid.flowdir(data='inflated_dem', dirmap=dirmap, routing='dinf', out_name='dinf_dir')
    grid.flowdir(data='inflated_dem', dirmap=dirmap, routing='d8', as_crs=new_crs,
                 out_name='proj_dir')

def test_clip_pad():
    grid.clip_to('catch')
    no_pad = grid.view('catch')
    for p in (1, 4, 10):
        grid.clip_to('catch', pad=(p,p,p,p))
        assert((no_pad == grid.view('catch')[p:-p, p:-p]).all())
    # TODO: Should check for non-square padding

def test_computed_fdir_catch():
    grid.catchment(x, y, data='d8_dir', dirmap=dirmap, out_name='d8_catch',
                   routing='d8', recursionlimit=15000, xytype='label')
    assert(np.count_nonzero(grid.catch) > 11300)
    # Reference routing
    grid.catchment(x, y, data='dinf_dir', dirmap=dirmap, out_name='dinf_catch',
                   routing='dinf', recursionlimit=15000, xytype='label')
    assert(np.count_nonzero(grid.catch) > 11300)

def test_accumulation():
    # TODO: This breaks if clip_to's padding of dir is nonzero
    grid.clip_to('dir')
    grid.accumulation(data='dir', dirmap=dirmap, out_name='acc')
    assert(grid.acc.max() == acc_in_frame)
    # TODO: Should eventually assert: grid.acc.dtype == np.min_scalar_type(grid.acc.max())
    grid.clip_to('catch', pad=(1,1,1,1))
    grid.accumulation(data='catch', dirmap=dirmap, out_name='acc')
    assert(grid.acc.max() == cells_in_catch)
    # Test accumulation on computed flowdirs
    grid.accumulation(data='d8_dir', dirmap=dirmap, out_name='d8_acc', routing='d8')
    grid.accumulation(data='dinf_dir', dirmap=dirmap, out_name='dinf_acc', routing='dinf')
    grid.accumulation(data='dinf_dir', dirmap=dirmap, out_name='dinf_acc', as_crs=new_crs,
                      routing='dinf')
    assert(grid.d8_acc.max() > 11300)
    assert(grid.dinf_acc.max() > 11400)

def test_flow_distance():
    grid.clip_to('catch')
    grid.flow_distance(x, y, data='catch', dirmap=dirmap, out_name='dist', xytype='label')
    assert(grid.dist[~np.isnan(grid.dist)].max() == max_distance)
    grid.flow_distance(x, y, data='dinf_dir', dirmap=dirmap, routing='dinf',
                       out_name='dinf_dist', xytype='label')

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
    assert((grid.view('dir_output') == grid.view('dir')).all())
    grid.to_raster('dir', 'test_dir.tif', view=True, apply_mask=True, blockxsize=16, blockysize=16)
    grid.read_raster('test_dir.tif', 'dir_output')
    assert((grid.dir_output == grid.view('catch')).all())
    # TODO: Write test for windowed reading

def test_from_raster():
    grid.clip_to('catch')
    grid.to_raster('dir', 'test_dir.tif', view=False, apply_mask=False, blockxsize=16, blockysize=16)
    newgrid = Grid.from_raster('test_dir.tif', 'dir_output')
    newgrid.clip_to('dir_output')
    assert ((newgrid.dir_output == grid.dir).all())
    grid.to_raster('dir', 'test_dir.tif', view=True, apply_mask=True, blockxsize=16, blockysize=16)
    newgrid = Grid.from_raster('test_dir.tif', 'dir_output')
    assert((newgrid.dir_output == grid.view('catch')).all())

# def test_windowed_reading():
#     newgrid = Grid.from_raster('test_dir.tif', 'dir_output', window=grid.bbox, window_crs=grid.crs)

def test_properties():
    bbox = grid.bbox
    assert(len(bbox) == 4)
    assert(isinstance(bbox, tuple))
    extent = grid.extent
    assert(len(extent) == 4)
    assert(isinstance(extent, tuple))

def test_extract_river_network():
    rivers = grid.extract_river_network('catch', 'acc', threshold=20)
    assert(isinstance(rivers, dict))
    # TODO: Need more checks here. Check if endnodes equals next startnode

def test_view_methods():
    grid.view('dem', interpolation='spline')
    grid.view('dem', interpolation='linear')
    grid.view('dem', interpolation='cubic')
    grid.view('dem', interpolation='linear', as_crs=new_crs)
    # TODO: Need checks for these

def test_pits():
    # TODO: Need dem with pits
    pits = grid.detect_pits('dem')
    assert(~pits.any())
    filled = grid.fill_pits('dem', inplace=False)

def test_other_methods():
    grid.cell_area(out_name='area', as_crs=new_crs)
    # TODO: Not a super robust test
    assert((grid.area.mean() > 7000) and (grid.area.mean() < 7500))
    # TODO: Need checks for these
    grid.cell_distances('dir', as_crs=new_crs, dirmap=dirmap)
    grid.cell_dh(fdir='dir', dem='dem', dirmap=dirmap)
    grid.cell_slopes(fdir='dir', dem='dem', dirmap=dirmap)

def test_snap_to():
    # TODO: Need checks
    grid.snap_to_mask(grid.view('acc') > 1000, [[-97.3, 32.72]])


