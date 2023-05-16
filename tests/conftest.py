import pytest
import os
from pysheds.grid import Grid
import numpy as np

dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
x, y = -97.29416666666677, 32.73749999999989


def generate_paths():
    paths = dict()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "../data"))

    paths["dir_path"] = os.path.join(data_dir, "dir.asc")
    paths["dem_path"] = os.path.join(data_dir, "dem.tif")
    paths["roi_path"] = os.path.join(data_dir, "roi.tif")
    paths["multiband_path"] = os.path.join(data_dir, "cogeo.tiff")

    return paths


def generate_grids():
    paths = generate_paths()
    grids = dict()
    grids["grid"] = Grid.from_raster(paths["dem_path"])
    grids["fdir"] = grids["grid"].read_ascii(paths["dir_path"], dtype=np.uint8, crs=grids["grid"].crs)
    grids["dem"] = grids["grid"].read_raster(paths["dem_path"])
    grids["roi"] = grids["grid"].read_raster(paths["roi_path"])

    return grids


@pytest.fixture()
def dem():
    g = generate_grids()

    return g["dem"]


@pytest.fixture()
def fdir():
    g = generate_grids()

    return g["fdir"]


@pytest.fixture()
def grid():
    g = generate_grids()

    return g["grid"]


@pytest.fixture()
def paths():
    p = generate_paths()

    return p


@pytest.fixture()
def d():
    class Datasets:
        pass

    # Initialize dataset holder
    d = Datasets()

    # Initialize grid
    grids = generate_grids()

    # Add datasets to dataset holder
    d.dem = grids["dem"]
    d.fdir = grids["fdir"]
    d.roi = grids["roi"]

    # Calculate additional grids used during tests
    d.catch = grids["grid"].catchment(x, y, d.fdir, dirmap=dirmap, xytype="coordinate")
    d.inflated_dem = grids["grid"].resolve_flats(d.dem)
    d.fdir_d8 = grids["grid"].flowdir(d.inflated_dem, dirmap=dirmap, routing="d8")
    d.fdir_dinf = grids["grid"].flowdir(d.inflated_dem, dirmap=dirmap, routing="dinf")
    d.fdir_mfd = grids["grid"].flowdir(d.inflated_dem, dirmap=dirmap, routing="mfd")
    d.acc = grids["grid"].accumulation(d.fdir, dirmap=dirmap, routing="d8")

    return d
