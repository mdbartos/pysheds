# Flow distance

## Preliminaries

The `grid.flow_distance` method operates on a flow direction grid. This flow direction grid can be computed from a DEM, as shown in [flow directions](https://mdbartos.github.io/pysheds/flow-directions.html).

```python
>>> import numpy as np
>>> from matplotlib import pyplot as plt
>>> from pysheds.grid import Grid

# Instantiate grid from raster
>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')

# Resolve flats and compute flow directions
>>> grid.resolve_flats(data='dem', out_name='inflated_dem')
>>> grid.flowdir('inflated_dem', out_name='dir')
```

## Computing flow distance

Flow distance is computed using the `grid.flow_distance` method:

```python
# Specify outlet
>>> x, y = -97.294167, 32.73750

# Delineate a catchment
>>> grid.catchment(data='dir', x=x, y=y, out_name='catch',
                   recursionlimit=15000, xytype='label')

# Clip the view to the catchment
>>> grid.clip_to('catch')

# Compute flow distance
>>> grid.flow_distance(x, y, data='catch',
                       out_name='dist', xytype='label')
```

![Flow distance](https://s3.us-east-2.amazonaws.com/pysheds/img/flow_distance.png)

Note that the `grid.flow_distance` method requires an outlet point, much like the `grid.catchment` method.

### Width function

The width function of a catchment `W(x)` represents the number of cells located at a topological distance `x` from the outlet. One can compute the width function of the catchment by counting the number of cells at a distance `x` from the outlet for each distance `x`.

```python
# Get flow distance array
>>> dists = grid.view('dist')

# Compute width function
>>> W = np.bincount(dists[dists != 0].astype(int))
```

![Width function](https://s3.us-east-2.amazonaws.com/pysheds/img/width_function.png)

## Computing weighted flow distance

Weights can be used to adjust the distance metric between cells. This can be useful if, for instance, the travel time between cells depends on characteristics such as slope, land cover, or channelization. In the following example, we will compute the weighted flow distance assuming that water in channelized cells flows 10 times faster than in hillslope cells.

```python
# Clip the bounding box to the catchment
>>> grid.clip_to('catch', pad=(1,1,1,1))

# Compute flow accumulation
>>> grid.accumulation(data='catch', out_name='acc')
>>> acc = grid.view('acc')

# Assume that water in channelized cells (>= 100 accumulation) travels 10 times faster
# than hillslope cells (< 100 accumulation)
>>> weights = (np.where(acc, 0.1, 0)
               + np.where((0 < acc) & (acc <= 100), 1, 0)).ravel()
                
# Compute weighted flow distance
>>> dists = grid.flow_distance(data='catch', x=x, y=y, weights=weights,
                               xytype='label', inplace=False)
```

![Weighted flow distance](https://s3.us-east-2.amazonaws.com/pysheds/img/weighted_flow_distance.png)

### Weighted width function

Note that because the distances are no longer integers, the weighted width function must bin the input distances.

```python
# Compute weighted width function
hist, bin_edges = np.histogram(dists[dists != 0].ravel(),
                               range=(0,dists.max()+1e-5), bins=40)
```

![Weighted width function](https://s3.us-east-2.amazonaws.com/pysheds/img/weighted_width_function.png)
