# Accumulation

## Preliminaries

The `grid.accumulation` method operates on a flow direction grid. This flow direction grid can be computed from a DEM, as shown in [flow directions](https://mdbartos.github.io/pysheds/flow-directions.html).

```python
>>> from pysheds.grid import Grid

# Instantiate grid from raster
>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')

# Resolve flats and compute flow directions
>>> grid.resolve_flats(data='dem', out_name='inflated_dem')
>>> grid.flowdir('inflated_dem', out_name='dir')
```

## Computing accumulation

Accumulation is computed using the `grid.accumulation` method.

```python
# Compute accumulation
>>> grid.accumulation(data='dir', out_name='acc')

# Plot accumulation
>>> acc = grid.view('acc')
>>> plt.imshow(acc)
```

![Full accumulation](https://s3.us-east-2.amazonaws.com/pysheds/img/full_accumulation.png)

## Computing weighted accumulation

Weights can be used to adjust the relative contribution of each cell.

```python
import pyproj

# Compute areas of each cell in new projection
new_crs = pyproj.Proj('+init=epsg:3083')
areas = grid.cell_area(as_crs=new_crs, inplace=False)

# Weight each cell by its relative area
weights = (areas / areas.max()).ravel()

# Compute accumulation with new weights
grid.accumulation(data='dir', weights=weights, out_name='acc')
```
