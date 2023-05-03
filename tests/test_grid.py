import numpy as np
import pyproj
import pytest
from pysheds.grid import Grid


# Initialize parameters
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
x, y = -97.29416666666677, 32.73749999999989


# TODO: Need to test dtypes of different constructor methods
def test_constructors(paths, grid, fdir):
    crs = pyproj.Proj("epsg:4326", preserve_units=True)

    grid.from_ascii(paths["dir_path"], dtype=np.uint8, crs=crs)
    new_fdir = grid.read_ascii(paths["dir_path"], dtype=np.uint8, crs=crs)
    assert (fdir == new_fdir).all()
    Grid(viewfinder=fdir.viewfinder)


def test_dtype(fdir):
    assert fdir.dtype == np.uint8


def test_nearest_cell(grid):
    """
    corner: snaps to nearest top/left
    center: snaps to index of cell that contains the geometry
    """
    col, row = grid.nearest_cell(x, y, snap="corner")
    assert (col, row) == (229, 101)
    col, row = grid.nearest_cell(x, y, snap="center")
    assert (col, row) == (228, 100)


def test_catchment(d, grid, fdir):
    cells_in_catch = 11422

    # Reference routing
    catch = grid.catchment(x, y, fdir, dirmap=dirmap, xytype="coordinate")
    assert np.count_nonzero(catch) == cells_in_catch
    col, row = grid.nearest_cell(x, y)
    catch_ix = grid.catchment(col, row, fdir, xytype="index")
    assert np.count_nonzero(catch_ix) == cells_in_catch


def test_clip(d, grid, dem):
    catch_shape = (159, 169)

    catch = d.catch
    grid.clip_to(catch)
    assert grid.shape == catch_shape
    assert grid.view(catch).shape == catch_shape
    # Restore viewfinder
    grid.viewfinder = dem.viewfinder


def test_input_output_mask():
    pass


def test_fill_depressions(d, grid):
    dem = d.dem
    depressions = grid.detect_depressions(dem)
    filled = grid.fill_depressions(dem)


def test_resolve_flats(d, grid):
    dem = d.dem
    flats = grid.detect_flats(dem)
    assert flats.sum() > 100
    inflated_dem = grid.resolve_flats(dem)
    flats = grid.detect_flats(inflated_dem)
    assert flats.sum() == 0


def test_flowdir(d, grid):
    fdir = d.fdir
    inflated_dem = d.inflated_dem
    grid.clip_to(fdir)
    fdir_d8 = grid.flowdir(inflated_dem, dirmap=dirmap, routing="d8")


def test_dinf_flowdir(d, grid):
    inflated_dem = d.inflated_dem
    fdir_dinf = grid.flowdir(inflated_dem, dirmap=dirmap, routing="dinf")


def test_mfd_flowdir(d, grid):
    inflated_dem = d.inflated_dem
    fdir_mfd = grid.flowdir(inflated_dem, dirmap=dirmap, routing="mfd")


def test_clip_pad(d, grid):
    catch = d.catch
    grid.clip_to(catch)
    no_pad = grid.view(catch)
    for p in (1, 4, 10):
        grid.clip_to(catch, pad=(p, p, p, p))
        assert (no_pad == grid.view(catch)[p:-p, p:-p]).all()
    # TODO: Should check for non-square padding


def test_computed_fdir_catch(d, grid):
    fdir_d8 = d.fdir_d8
    fdir_dinf = d.fdir_dinf
    fdir_mfd = d.fdir_mfd
    catch_d8 = grid.catchment(
        x, y, fdir_d8, dirmap=dirmap, routing="d8", xytype="coordinate"
    )
    assert np.count_nonzero(catch_d8) > 11300
    # Reference routing
    catch_dinf = grid.catchment(
        x, y, fdir_dinf, dirmap=dirmap, routing="dinf", xytype="coordinate"
    )
    assert np.count_nonzero(catch_dinf) > 11300
    catch_mfd = grid.catchment(
        x, y, fdir_mfd, dirmap=dirmap, routing="mfd", xytype="coordinate"
    )
    assert np.count_nonzero(catch_dinf) > 11300
    catch_d8_recur = grid.catchment(
        x,
        y,
        fdir_d8,
        dirmap=dirmap,
        routing="d8",
        xytype="coordinate",
        algorithm="recursive",
    )
    catch_dinf_recur = grid.catchment(
        x,
        y,
        fdir_dinf,
        dirmap=dirmap,
        routing="dinf",
        xytype="coordinate",
        algorithm="recursive",
    )


def test_accumulation(d, grid):
    acc_in_frame = 77261

    # D8 flow accumulation without efficiency
    # external flow direction
    fdir = d.fdir
    catch = d.catch
    fdir_d8 = d.fdir_d8
    fdir_dinf = d.fdir_dinf
    fdir_mfd = d.fdir_mfd
    # TODO: This breaks if clip_to's padding of dir is nonzero
    grid.clip_to(fdir)
    acc = grid.accumulation(fdir, dirmap=dirmap, routing="d8")
    assert acc.max() == acc_in_frame

    # catch = d.catch
    # fdir differs from fdir_d8 and fdir_dinf
    # we derive new catchment grids for assertions below
    catch = grid.catchment(x, y, d.fdir_d8, dirmap=dirmap, xytype="coordinate")
    grid.clip_to(catch)
    fdir_d8 = d.fdir_d8
    fdir_dinf = d.fdir_dinf
    # Test D8 flow accumulation on calculated flow direction
    # without efficiency
    c, r = grid.nearest_cell(x, y)
    acc_d8 = grid.accumulation(fdir_d8, dirmap=dirmap, routing="d8")
    # flow accumulation at outlet should be size of the catchment
    # CHECK:
    # acc_d8[acc_d8 > 0].size is catch[catch].size + 1
    # because two grid cells have acc_d8.max()?!
    # np.where(acc_d8 >= acc_d8.max())
    assert acc_d8[r, c] == acc_d8.max()
    assert catch[catch].size == acc_d8.max()
    # original assertion
    assert acc_d8.max() > 11300

    # ...with efficiency
    # we set the efficiency for starting cells to 0
    # this will reduce the flow accumulation by the number of starting cells
    start_cells = np.where(acc_d8 == 1)
    # default efficiency is 1 = no reduction
    eff = np.ones_like(acc_d8)
    eff[start_cells] = 0
    acc_d8_eff = grid.accumulation(fdir_d8, dirmap=dirmap, efficiency=eff, routing="d8")
    # test the outlet of the catchment
    assert acc_d8.max() - acc_d8_eff.max() - start_cells[0].size == 0

    # Test Dinf accumulation on computed flowdirs
    # without efficiency
    acc_dinf = grid.accumulation(fdir_dinf, dirmap=dirmap, routing="dinf")
    # Dinf outlet is identical to D8 outlet
    assert (acc_dinf[np.where(acc_d8 == acc_d8.max())] == acc_dinf.max()).all()
    # original assertion
    assert acc_dinf.max() > 11300
    acc_mfd = grid.accumulation(fdir_mfd, dirmap=dirmap, routing="mfd")
    assert acc_mfd.max() > 11200
    # #set nodata to 1
    # eff = grid.view(dinf_eff)
    # eff[eff==dinf_eff.nodata] = 1
    # acc_dinf_eff = grid.accumulation(fdir_dinf, dirmap=dirmap,
    #                                  routing='dinf', efficiency=eff)
    # pos = np.where(grid.dinf_acc==grid.dinf_acc.max())
    # assert(np.round(grid.dinf_acc[pos] / grid.dinf_acc_eff[pos]) == 4.)
    acc_d8_recur = grid.accumulation(
        fdir_d8, dirmap=dirmap, routing="d8", algorithm="recursive"
    )
    acc_dinf_recur = grid.accumulation(
        fdir_dinf, dirmap=dirmap, routing="dinf", algorithm="recursive"
    )
    # ...with efficiency
    # this is probably a bit hacky
    # we have two grid cells with the outlet value == max flow accumulation
    # which should actually not happen but so
    # we can set their efficiency to <1 and test the reduction
    eff = np.ones_like(acc_dinf)
    reduction = 0.25
    outlets = np.where(acc_dinf == acc_dinf.max())
    eff[outlets] = reduction
    acc_dinf_eff = grid.accumulation(
        fdir_dinf, dirmap=dirmap, routing="dinf", efficiency=eff
    )
    outlets_eff = np.sort(acc_dinf_eff[outlets])
    assert np.isclose(outlets_eff[0] / outlets_eff[1], reduction)
    # as the reduction is applied to the outflow of a grid cell
    # the higher value (which belongs to the catchment) should be
    # identical to the flow accumulation without efficiency
    assert np.isclose(outlets_eff[1], acc_dinf.max())

    # similar to Dinf:
    eff = np.ones_like(acc_d8)
    # the identity of the D8 and Dinf outlets were asserted above
    # outlets = np.where(acc_d8==acc_d8.max())
    eff[outlets] = reduction
    acc_d8_eff = grid.accumulation(fdir_d8, dirmap=dirmap, routing="d8", efficiency=eff)
    outlets_eff = np.sort(acc_d8_eff[outlets])
    assert np.isclose(outlets_eff[0] / outlets_eff[1], reduction)


def test_hand(d, grid):
    fdir = d.fdir
    dem = d.dem
    acc = d.acc
    fdir_dinf = d.fdir_dinf
    fdir_mfd = d.fdir_mfd
    hand_d8 = grid.compute_hand(fdir, dem, acc > 100, routing="d8")
    hand_dinf = grid.compute_hand(fdir_dinf, dem, acc > 100, routing="dinf")
    hand_mfd = grid.compute_hand(fdir_mfd, dem, acc > 100, routing="mfd")
    hand_d8_recur = grid.compute_hand(
        fdir, dem, acc > 100, routing="d8", algorithm="recursive"
    )
    hand_dinf_recur = grid.compute_hand(
        fdir_dinf, dem, acc > 100, routing="dinf", algorithm="recursive"
    )


# FIXME: This test is causing segfaults at line initializing `dist`
# def test_distance_to_outlet(d, grid):
#     max_distance_d8 = 209
#
#     fdir = d.fdir
#     catch = d.catch
#     fdir_dinf = d.fdir_dinf
#     fdir_mfd = d.fdir_mfd
#     grid.clip_to(catch)
#     dist = grid.distance_to_outlet(x, y, fdir, dirmap=dirmap, xytype="coordinate")
#     assert dist[np.isfinite(dist)].max() == max_distance_d8
#     col, row = grid.nearest_cell(x, y)
#     dist = grid.distance_to_outlet(col, row, fdir, dirmap=dirmap, xytype="index")
#     assert dist[np.isfinite(dist)].max() == max_distance_d8
#     weights = Raster(2 * np.ones(grid.shape), grid.viewfinder)
#     grid.distance_to_outlet(
#         x, y, fdir_dinf, dirmap=dirmap, routing="dinf", xytype="coordinate"
#     )
#     grid.distance_to_outlet(
#         x, y, fdir_mfd, dirmap=dirmap, routing="mfd", xytype="coordinate"
#     )
#     grid.distance_to_outlet(x, y, fdir, weights=weights, dirmap=dirmap, xytype="label")
#     grid.distance_to_outlet(
#         x, y, fdir_dinf, dirmap=dirmap, weights=weights, routing="dinf", xytype="label"
#     )
#     grid.distance_to_outlet(
#         x, y, fdir_mfd, dirmap=dirmap, weights=weights, routing="mfd", xytype="label"
#     )
#     # Test recursive
#     grid.distance_to_outlet(
#         x,
#         y,
#         fdir,
#         dirmap=dirmap,
#         xytype="coordinate",
#         routing="d8",
#         algorithm="recursive",
#     )
#     grid.distance_to_outlet(
#         x,
#         y,
#         fdir_dinf,
#         dirmap=dirmap,
#         xytype="coordinate",
#         routing="dinf",
#         algorithm="recursive",
#     )


def test_stream_order(d, grid):
    fdir = d.fdir
    acc = d.acc
    order = grid.stream_order(fdir, acc > 100)
    order = grid.stream_order(fdir, acc > 100, algorithm="recursive")


def test_distance_to_ridge(d, grid):
    fdir = d.fdir
    acc = d.acc
    fdir_dinf = d.fdir_dinf
    fdir_mfd = d.fdir_mfd

    grid.distance_to_ridge(fdir, acc > 100)
    grid.distance_to_ridge(fdir, acc > 100, algorithm="recursive")
    grid.distance_to_ridge(fdir_dinf, acc > 100, routing="dinf")
    grid.distance_to_ridge(fdir_mfd, acc > 100, routing="mfd")


def test_cell_dh(d, grid):
    fdir = d.fdir
    fdir_dinf = d.fdir_dinf
    fdir_mfd = d.fdir_mfd
    dem = d.dem

    grid.cell_dh(dem, fdir, routing="d8")
    grid.cell_dh(dem, fdir_dinf, routing="dinf")
    grid.cell_dh(dem, fdir_mfd, routing="mfd")


def test_cell_distances(d, grid):
    fdir = d.fdir
    fdir_dinf = d.fdir_dinf
    fdir_mfd = d.fdir_mfd
    dem = d.dem

    cdist_d8 = grid.cell_distances(fdir, routing="d8")
    cdist_dinf = grid.cell_distances(fdir_dinf, routing="dinf")
    cdist_mfd = grid.cell_distances(fdir_mfd, routing="mfd")


def test_cell_slopes(d, grid):
    fdir = d.fdir
    fdir_dinf = d.fdir_dinf
    fdir_mfd = d.fdir_mfd
    dem = d.dem
    slopes_d8 = grid.cell_slopes(dem, fdir, routing="d8")
    slopes_dinf = grid.cell_slopes(dem, fdir_dinf, routing="dinf")
    slopes_mfd = grid.cell_slopes(dem, fdir_mfd, routing="mfd")


def test_to_ascii(d, grid, tmp_path):
    catch = d.catch
    fdir = d.fdir
    grid.clip_to(catch)
    grid.to_ascii(fdir, tmp_path / "test_dir.asc", target_view=fdir.viewfinder, dtype=np.float64)
    fdir_out = grid.read_ascii(tmp_path / "test_dir.asc", dtype=np.uint8)
    assert (fdir_out == fdir).all()
    grid.to_ascii(fdir, tmp_path / "test_dir.asc", dtype=np.uint8)
    fdir_out = grid.read_ascii(tmp_path / "test_dir.asc", dtype=np.uint8)
    assert (fdir_out == grid.view(fdir)).all()


def test_read_raster(paths, grid):
    grid.read_raster(paths["multiband_path"], band=1)
    grid.read_raster(paths["multiband_path"], band=2)
    grid.read_raster(paths["multiband_path"], band=3)


def test_to_raster(d, grid, tmp_path):
    catch = d.catch
    fdir = d.fdir
    grid.clip_to(catch)
    grid.to_raster(
        fdir, tmp_path / "test_dir.tif", target_view=fdir.viewfinder, blockxsize=16, blockysize=16
    )
    fdir_out = grid.read_raster(tmp_path / "test_dir.tif")
    assert (fdir_out == fdir).all()
    assert (grid.view(fdir_out) == grid.view(fdir)).all()
    grid.to_raster(fdir, tmp_path / "test_dir.tif", blockxsize=16, blockysize=16)
    fdir_out = grid.read_raster(tmp_path / "test_dir.tif")
    assert (fdir_out == grid.view(fdir)).all()


def test_to_raster_kwargs(d, grid, fdir, tmp_path):
    """
    Test if kwargs of the "to_raster" method are passed to rasterio
    """
    import rasterio as rio

    catch = d.catch
    grid.clip_to(catch)
    grid.to_raster(
        fdir,
        tmp_path / "test_dir.tif",
        target_view=fdir.viewfinder,
        blockxsize=16,
        blockysize=16,
        compress="LZW",
    )
    with rio.open(tmp_path / "test_dir.tif") as ds:
        assert ds.profile["compress"] == "lzw"


# def test_from_raster():
#     grid.clip_to('catch')
#     grid.to_raster('dir', 'test_dir.tif', view=False, apply_mask=False, blockxsize=16, blockysize=16)
#     newgrid = Grid.from_raster('test_dir.tif', 'dir_output')
#     newgrid.clip_to('dir_output')
#     assert ((newgrid.dir_output == grid.dir).all())
#     grid.to_raster('dir', 'test_dir.tif', view=True, apply_mask=True, blockxsize=16, blockysize=16)
#     newgrid = Grid.from_raster('test_dir.tif', 'dir_output')
#     assert((newgrid.dir_output == grid.view('dir', apply_mask=True)).all())


# TODO: Write test for windowed reading
# def test_windowed_reading(d, grid, tmp_path):
#     newgrid = Grid.from_raster(tmp_path / "test_dir.tif", window=grid.bbox, window_crs=grid.crs)


# def test_mask_geometry():
#     grid = Grid.from_raster(dem_path,'dem', mask_geometry=feature_geometry)
#     rows = np.array([225, 226, 227, 228, 229, 230, 231, 232] * 7)
#     cols = np.array([np.arange(98,105)] * 8).T.reshape(1,56)
#     masked_cols, masked_rows = grid.mask.nonzero()
#     assert (masked_cols == cols).all()
#     assert (masked_rows == rows).all()
#     with warnings.catch_warnings(record=True) as warn:
#         warnings.simplefilter("always")
#         grid = Grid.from_raster(dem_path,'dem', mask_geometry=out_of_bounds)
#         assert len(warn) == 1
#         assert issubclass(warn[-1].category, UserWarning)
#         assert "does not fall within the bounds" in str(warn[-1].message)
#         assert grid.mask.all(), "mask should be returned to all True as normal"


def test_properties(grid):
    bbox = grid.bbox
    assert len(bbox) == 4
    assert isinstance(bbox, tuple)
    extent = grid.extent
    assert len(extent) == 4
    assert isinstance(extent, tuple)


def test_extract_river_network(d, grid):
    fdir = d.fdir
    catch = d.catch
    acc = d.acc
    grid.clip_to(catch)
    rivers = grid.extract_river_network(catch, acc > 20)
    assert isinstance(rivers, dict)
    grid.extract_river_network(catch, acc > 20, algorithm="recursive")
    # TODO: Need more checks here. Check if endnodes equals next startnode


def test_extract_profiles(d, grid):
    fdir = d.fdir
    catch = d.catch
    acc = d.acc
    grid.clip_to(catch)
    profiles, connections = grid.extract_profiles(catch, acc > 20)


def test_view_methods(d, grid):
    dem = d.dem
    catch = d.catch
    grid.clip_to(dem)
    grid.view(dem, interpolation="nearest")
    grid.view(dem, interpolation="linear")
    grid.clip_to(catch)
    grid.view(dem, interpolation="nearest")
    grid.view(dem, interpolation="linear")


# def test_resize():
#     new_shape = tuple(np.asarray(grid.shape) // 2)
#     grid.resize('dem', new_shape=new_shape)


def test_pits(d, grid):
    dem = d.dem
    # TODO: Need dem with pits
    pits = grid.detect_pits(dem)
    filled = grid.fill_pits(dem)
    pits = grid.detect_pits(filled)
    assert ~pits.any()


def test_to_crs(d):
    new_crs = pyproj.Proj("epsg:3083")

    dem = d.dem
    fdir = d.fdir
    dem_p = dem.to_crs(new_crs)
    fdir_p = fdir.to_crs(new_crs)


def test_snap_to(d, grid):
    acc = d.acc
    # TODO: Need checks
    grid.snap_to_mask(acc > 1000, [[-97.3, 32.72]])


# def test_set_bbox():
#     new_xmin = (grid.bbox[2] + grid.bbox[0]) / 2
#     new_ymin = (grid.bbox[3] + grid.bbox[1]) / 2
#     new_xmax = grid.bbox[2]
#     new_ymax = grid.bbox[3]
#     new_bbox = (new_xmin, new_ymin, new_xmax, new_ymax)
#     grid.set_bbox(new_bbox)
#     grid.clip_to('catch')
#     # TODO: Need to check that everything was reset properly

# def test_set_indices():
#     new_xmin = int(grid.shape[1] // 2)
#     new_ymin = int(grid.shape[0])
#     new_xmax = int(grid.shape[1])
#     new_ymax = int(grid.shape[0] // 2)
#     new_indices = (new_xmin, new_ymin, new_xmax, new_ymax)
#     grid.set_indices(new_indices)
#     grid.clip_to('catch')
#     # TODO: Need to check that everything was reset properly


@pytest.mark.xfail(reason="TypeError: `nodata` value not representable in dtype of array.")
def test_polygonize_rasterize(grid):
    shapes = grid.polygonize()
    raster = grid.rasterize(shapes)
    assert (raster == grid.mask).all()


# def test_detect_cycles():
#     cycles = grid.detect_cycles('dir')

# def test_add_gridded_data():
#     grid.add_gridded_data(grid.dem, data_name='dem_copy')

# def test_rfsm():
#     grid.clip_to('roi')
#     dem = grid.view('roi')
#     rfsm = RFSM(dem)
#     rfsm.reset_volumes()
#     area = np.abs(grid.affine.a * grid.affine.e)
#     input_vol = 0.1*area*np.ones(dem.shape)
#     waterlevel = rfsm.compute_waterlevel(input_vol)
#     end_vol = (area*np.where(waterlevel, waterlevel - dem, 0)).sum()
#     assert np.allclose(end_vol, input_vol.sum())


def test_misc(d, grid):
    dem = d.dem
    l, r, t, b = grid._pop_rim(dem, nodata=0)
    grid._replace_rim(dem, l, r, t, b)
