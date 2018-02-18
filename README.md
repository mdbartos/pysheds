# pysheds [![Build Status](https://travis-ci.org/mdbartos/pysheds.svg?branch=master)](https://travis-ci.org/mdbartos/pysheds)
Simple and fast watershed delineation in python.

## Features

- Flow Direction to Flow Accumulation
- Catchment delineation from Flow Direction
- Distance to outlet
- Fractional contributing area between differently-sized grids
- River network extraction
- Read/write raster or ASCII files

`pysheds` currently only supports a d8 routing scheme

## Example usage

See [examples/quickstart](https://github.com/mdbartos/pysheds/blob/master/examples/quickstart.ipynb) for more details.

Data available via the [USGS HydroSHEDS](https://hydrosheds.cr.usgs.gov/datadownload.php) project.

```python
    # Read elevation and flow direction rasters
    # ----------------------------
    from pysheds.grid import Grid

    grid = Grid.from_raster('n30w100_con',
                             data_name='dem', input_type='ascii')
    grid.read_raster('n30w100_dir',
                      data_name='dir', input_type='ascii')
```

![Example 1](examples/img/conditioned_dem.png)

```python
    # Delineate a catchment
    # ---------------------
    # Specify pour point
    x, y = -97.2937, 32.7371
    # Specify directional mapping
    dirmap=(64, 128, 1, 2, 4, 8, 16, 32)

    # Delineate the catchment
    grid.catchment(x, y, dirmap=dirmap, recursionlimit=15000,
                   xytype='label')

    # Crop and plot the catchment
    # ---------------------------
    # Clip the bounding box to the catchment
    grid.clip_to('catch', precision=5)
```

![Example 2](examples/img/catchment.png)

```python
    # Calculate flow accumulation
    # --------------------------
    grid.accumulation(catch, dirmap=dirmap, pad_inplace=False)
```

![Example 3](examples/img/flow_accumulation.png)

```python
    # Calculate distance to outlet from each cell
    # -------------------------------------------
    pour_point_y, pour_point_x = np.unravel_index(np.argmax(grid.view('catch')),
                                                  grid.shape)
    grid.flow_distance(pour_point_x, pour_point_y, dirmap=dirmap)
```

![Example 4](examples/img/flow_distance.png)

```python
    # Extract river network
    # ---------------------
    branches, yx = grid.extract_river_network(threshold=200)
```

![Example 5](examples/img/river_network.png)

## Installation

`pysheds` currently only supports Python3

```bash
    $ git clone https://github.com/mdbartos/pysheds.git
    $ cd pysheds
    $ python setup.py install
```


# Performance
Performance benchmarks on a 2015 MacBook Pro:

- Flow Direction --> Flow Accumulation: 36 million grid cells in 15 seconds.
- Flow Direction --> Catchment: 9.8 million grid cells in 4.55 seconds.

# To-do's:
- DEM to Flow Direction doesn't handle flats
- Float-based bbox indexing is problematic
- Add graph routines
- Allow conversion of CRS
- Raster to vector conversion and shapefile output
