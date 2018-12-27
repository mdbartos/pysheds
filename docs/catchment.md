# Catchment delineation

## Preliminaries

The `grid.catchment` method operates on a flow direction grid. This flow direction grid can be computed from a DEM, as shown in [flow directions](https://mdbartos.github.io/pysheds/flow-directions.html).

```python
>>> from pysheds.grid import Grid

# Instantiate grid from raster
>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')

# Resolve flats and compute flow directions
>>> grid.resolve_flats(data='dem', out_name='inflated_dem')
>>> grid.flowdir('inflated_dem', out_name='dir')
```

## Delineating the catchment

To delineate a catchment, first specify a pour point (the outlet of the catchment). If the x and y components of the pour point are spatial coordinates in the grid's spatial reference system, specify `xytype='label'`.

```python
# Specify pour point
>>> x, y = -97.294167, 32.73750

# Delineate the catchment
>>> grid.catchment(data='dir', x=x, y=y, out_name='catch',
                   recursionlimit=15000, xytype='label')

# Plot the result
>>> grid.clip_to('catch')
>>> plt.imshow(grid.view('catch'))
```

![Delineated catchment](https://s3.us-east-2.amazonaws.com/pysheds/img/catchment.png)

If the x and y components of the pour point correspond to the row and column indices of the flow direction array, specify `xytype='index'`:

```python
# Reset the view
>>> grid.clip_to('dir')

# Find the row and column index corresponding to the pour point
>>> col, row = grid.nearest_cell(x, y)
>>> col, row
(229, 101)

# Delineate the catchment
>>> grid.catchment(data=grid.dir, x=col, y=row, out_name='catch',
                   recursionlimit=15000, xytype='index')

# Plot the result
>>> grid.clip_to('catch')
>>> plt.imshow(grid.view('catch'))
```

![Delineated catchment index](https://s3.us-east-2.amazonaws.com/pysheds/img/catchment.png)
