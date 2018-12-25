# Flow direction

Flow directions are computed from a source DEM. The flow direction grid captures the topology of the drainage network, and is needed for delineating catchments, computing flow accumulation, and computing flow path lengths.

## D8 flow directions

By default, `pysheds` will compute flow directions using the D8 routing scheme. In this routing mode, each cell is routed to one of eight neighboring cells based on the direction of steepest descent.

### Preliminaries

Note that for most use cases, DEMs should be conditioned before computing flow directions. In other words, depressions should be filled and flats should be resolved.

```python
# Import modules
>>> from pysheds.grid import Grid

# Read raw DEM
>>> grid = Grid.from_raster('../data/roi_10m', data_name='dem')

# Fill depressions
>>> grid.fill_depressions(data='dem', out_name='flooded_dem')

# Resolve flats
>>> grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
```

### Computing D8 flow directions

After filling depressions, the flow directions can be computed using the `grid.flowdir` method:

```python
>>> grid.flowdir(data='inflated_dem', out_name='dir')
>>> grid.dir
Raster([[  0,   0,   0, ...,   0,   0,   0],
        [  0,   2,   2, ...,   4,   1,   0],
        [  0,   1,   2, ...,   4,   2,   0],
        ...,
        [  0,  64,  32, ...,   8,   1,   0],
        [  0,  64,  32, ...,  16, 128,   0],
        [  0,   0,   0, ...,   0,   0,   0]])
```

### Directional mappings

Cardinal and intercardinal directions are represented by numeric values in the output grid. By default, the ESRI scheme is used:

- **North**: 64
- **Northeast**: 128
- **East**: 1
- **Southeast**: 2
- **South**: 4
- **Southwest**: 8
- **West**: 16
- **Northwest**: 32

An alternative directional mapping can be specified using the `dirmap` keyword argument:

```python
>>> dirmap = (1, 2, 3, 4, 5, 6, 7, 8)
>>> grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
>>> grid.dir
Raster([[0, 0, 0, ..., 0, 0, 0],
        [0, 4, 4, ..., 5, 3, 0],
        [0, 3, 4, ..., 5, 4, 0],
        ...,
        [0, 1, 8, ..., 6, 3, 0],
        [0, 1, 8, ..., 7, 2, 0],
        [0, 0, 0, ..., 0, 0, 0]])
```

### Labeling pits and flats

If pits or flats are present in the originating DEM, these cells can be labeled in the output array using the `pits` and `flats` keyword arguments:

```python
>>> grid.flowdir(data='inflated_dem', out_name='dir', pits=0, flats=-1)
```

## D-infinity flow directions

While the D8 routing scheme allows each cell to be routed to only one of its nearest neighbors, the D-infinity routing scheme allows each cell to be routed to any angle between 0 and 2π. This feature allows for better resolution of flow directions on hillslopes.

D-infinity routing can be selected by using the keyword argument `routing='dinf'`.

```python
>>> grid.flowdir(data='inflated_dem', out_name='dir', routing='dinf')
>>> grid.dir
Raster([[  nan,   nan,   nan, ...,   nan,   nan,   nan],
        [  nan, 5.498, 5.3  , ..., 4.712, 0.   ,   nan],
        [  nan, 0.   , 5.498, ..., 4.712, 5.176,   nan],
        ...,
        [  nan, 1.571, 2.356, ..., 2.356, 0.   ,   nan],
        [  nan, 1.571, 2.034, ..., 3.142, 0.785,   nan],
        [  nan,   nan,   nan, ...,   nan,   nan,   nan]])
```

Note that each entry takes a value between 0 and 2π, with `np.nan` representing unknown flow directions.

Note that you must also specify `routing=dinf` when using `grid.catchment` or `grid.accumulation` with a D-infinity output grid.

## Effect of map projections on routing

The choice of map projection affects the slopes between neighboring cells. The map projection can be specified using the `as_crs` keyword argument.

```python
>>> new_crs = pyproj.Proj('+init=epsg:3083')
>>> grid.flowdir(data='inflated_dem', out_name='proj_dir', as_crs=new_crs)
```
