---
layout: default
title:  "Flow direction"
---

# Flow direction

Flow directions are computed from a source DEM. The flow direction grid captures the topology of the drainage network, and is needed for delineating catchments, computing flow accumulation, and computing flow path lengths.

## D8 flow directions

By default, `pysheds` will compute flow directions using the D8 routing scheme. In this routing mode, each cell is routed to one of eight neighboring cells based on the direction of steepest descent.

### Preliminaries

Note that for most use cases, DEMs should be conditioned before computing flow directions. In other words, depressions should be filled and flats should be resolved.

```python
# Import modules
from pysheds.grid import Grid

# Read raw DEM
grid = Grid.from_raster('./data/roi_10m')
dem = grid.read_raster('./data/roi_10m')

# Fill depressions
flooded_dem = grid.fill_depressions(dem)

# Resolve flats
inflated_dem = grid.resolve_flats(flooded_dem)
```

### Computing D8 flow directions

After filling depressions, the flow directions can be computed using the `grid.flowdir` method:

```python
fdir = grid.flowdir(inflated_dem)
```

<details>
<summary>Output...</summary>
<p>

<pre>
fdir

Raster([[  0,   0,   0, ...,   0,   0,   0],
        [  0,   2,   2, ...,   4,   1,   0],
        [  0,   1,   2, ...,   4,   2,   0],
        ...,
        [  0,  64,  32, ...,   8,   1,   0],
        [  0,  64,  32, ...,  16, 128,   0],
        [  0,   0,   0, ...,   0,   0,   0]])

</pre>

</p>
</details>

<br>


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
dirmap = (1, 2, 3, 4, 5, 6, 7, 8)
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
```

<details>
<summary>Output...</summary>
<p>

<pre>
fdir

Raster([[0, 0, 0, ..., 0, 0, 0],
        [0, 4, 4, ..., 5, 3, 0],
        [0, 3, 4, ..., 5, 4, 0],
        ...,
        [0, 1, 8, ..., 6, 3, 0],
        [0, 1, 8, ..., 7, 2, 0],
        [0, 0, 0, ..., 0, 0, 0]])
</pre>

</p>
</details>

<br>

## D-infinity flow directions

While the D8 routing scheme allows each cell to be routed to only one of its nearest neighbors, the D-infinity routing scheme allows each cell to be routed to any angle between 0 and 2π. This feature allows for better resolution of flow directions on hillslopes.

D-infinity routing can be selected by using the keyword argument `routing='dinf'`.

```python
fdir = grid.flowdir(inflated_dem, routing='dinf')
```

<details>
<summary>Output...</summary>
<p>

<pre>
fdir

Raster([[  nan,   nan,   nan, ...,   nan,   nan,   nan],
        [  nan, 5.498, 5.3  , ..., 4.712, 0.   ,   nan],
        [  nan, 0.   , 5.498, ..., 4.712, 5.176,   nan],
        ...,
        [  nan, 1.571, 2.356, ..., 2.356, 0.   ,   nan],
        [  nan, 1.571, 2.034, ..., 3.142, 0.785,   nan],
        [  nan,   nan,   nan, ...,   nan,   nan,   nan]])
</pre>

</p>
</details>

<br>

Note that each entry takes a value between 0 and 2π, with `np.nan` representing unknown flow directions.

Note that you must also specify `routing='dinf'` when using other functions that use the flow direction grid such as `grid.catchment` or `grid.accumulation`.

## Multiple flow directions (MFD)

The multiple flow direction (MFD) routing scheme partitions flow from each cell to among as many as eight of its neighbors. The proportion of the cell that flows to each of its neighbors is proportional to the height gradient between the neighboring cells.

MFD routing can be selected by using the keyword argument `routing='mfd'`.

```python
fdir = grid.flowdir(inflated_dem, routing='mfd')
```

<details>
<summary>Output...</summary>
<p>

<pre>
fdir

MultiRaster([[[0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.37428797, 0.41595555, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.35360402, 0.42297009, ..., 0.06924557,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],

             [[0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.36288963, 0.33088875, ..., 0.06863035,
               0.        , 0.        ],
              [0.        , 0.40169546, 0.36123674, ..., 0.23938736,
               0.17013502, 0.        ],
              ...,
              [0.        , 0.00850506, 0.10102002, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.04147018, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],

             [[0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.14276847, 0.06932945, ..., 0.48528137,
               0.39806072, 0.        ],
              [0.        , 0.1217316 , 0.06042334, ..., 0.4193337 ,
               0.48612365, 0.        ],
              ...,
              [0.        , 0.1683329 , 0.28176027, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.13663963, 0.24437534, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],

             ...,

             [[0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.20829874, 0.04770285, ..., 0.29010027,
               0.31952507, 0.        ],
              [0.        , 0.20128372, 0.11750307, ..., 0.23404662,
               0.28716789, 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],

             [[0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.03460397,
               0.06389793, 0.        ],
              [0.        , 0.0151827 , 0.        , ..., 0.        ,
               0.06675575, 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],

             [[0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.12005394, 0.18382625, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.12296892, 0.15536983, ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]]])
</pre>

</p>
</details>

<br>

Note that if the original DEM is of shape `(m, n)`, then the returned flow direction grid will be of shape `(8, m, n)`. The first dimension corresponds to the eight flow directions, with index 0 corresponding to North, index 1 corresponding to Northeast, index 2 corresponding to East, and so on until index 7 corresponding to Northwest. The value of a given array element (k, i, j) represents the proportion of flow that is transferred from cell (i, j) to the neighboring cell k (where k ∈ [0, 7]).

Note that you must also specify `routing='mfd'` when using other functions that use the flow direction grid such as `grid.catchment` or `grid.accumulation`.


## Effect of map projections on routing

The choice of map projection affects the slopes between neighboring cells.

```python
# Specify new map projection
import pyproj
new_crs = pyproj.Proj('epsg:3083')

# Convert CRS of dataset and grid
proj_dem = inflated_dem.to_crs(new_crs)
grid.viewfinder = proj_dem.viewfinder

# Compute flow directions on projected grid
proj_fdir = grid.flowdir(proj_dem)
```
