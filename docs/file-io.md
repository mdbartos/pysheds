# File I/O

## Reading from raster files

`pysheds` uses the `rasterio` module to read raster images. 

### Instantiating a grid from a raster

```python
from pysheds.grid import Grid
grid = Grid.from_raster('./data/dem.tif')
```

### Reading a raster file

```python
dem = grid.read_raster('./data/dem.tif')
```

## Reading from ASCII files

### Instantiating a grid from an ASCII grid

```python
grid = Grid.from_ascii('./data/dir.asc')
```

### Reading an ASCII grid

```python
fdir = grid.read_ascii('./data/dir.asc', dtype=np.uint8)
```

## Windowed reading

If the raster file is very large, you can specify a window to read data from. This window is defined by a bounding box and coordinate reference system.

```python
# Instantiate a grid with data
grid = Grid.from_raster('./data/dem.tif')

# Read windowed raster
terrain = grid.read_raster('./data/impervious_area.tiff',
                           window=grid.bbox, window_crs=grid.crs)
```

## Writing to raster files

By default, the `grid.to_raster` method will write the grid's current view of the dataset.

```python
grid = Grid.from_ascii('./data/dir.asc')
fdir = grid.read_ascii('./data/dir.asc', dtype=np.uint8)
grid.to_raster(fdir, 'test_dir.tif', blockxsize=16, blockysize=16)
```

If the full dataset is desired, set the `target_view` to the dataset's `viewfinder`:

```python
grid.to_raster(fdir, 'test_dir.tif', target_view=fdir.viewfinder,
               blockxsize=16, blockysize=16)
```

If you want the output file to be masked with the grid mask, set `apply_output_mask=True`:

```python
grid.to_raster(fdir, 'test_dir.tif', apply_output_mask=True,
               blockxsize=16, blockysize=16)
```

## Writing to ASCII files

```python
grid.to_ascii(fdir, 'test_dir.asc')
```

## Writing to shapefiles

```python
import fiona

grid = Grid.from_ascii('./data/dir.asc')

# Specify pour point
x, y = -97.294167, 32.73750

# Delineate the catchment
catch = grid.catchment(x=x, y=y, fdir=fdir,
                       xytype='coordinate')

# Clip to catchment
grid.clip_to(catch)

# Create view
catch_view = grid.view(catch, dtype=np.uint8)

# Create a vector representation of the catchment mask
shapes = grid.polygonize(catch_view)

# Specify schema
schema = {
        'geometry': 'Polygon',
        'properties': {'LABEL': 'float:16'}
}

# Write shapefile
with fiona.open('catchment.shp', 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1
```

