# File I/O

## Reading from raster files

`pysheds` uses the `rasterio` module to read raster images. 

### Instantiating a grid from a raster

```python
>>> from pysheds.grid import Grid
>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')
```

### Reading a raster file

```python
>>> grid = Grid()
>>> grid.read_raster('../data/dem.tif', data_name='dem')
```

## Reading from ASCII files

### Instantiating a grid from an ASCII grid

```python
>>> grid = Grid.from_ascii('../data/dir.asc', data_name='dir')
```

### Reading an ASCII grid

```python
>>> grid = Grid()
>>> grid.read_ascii('../data/dir.asc', data_name='dir')
```

## Windowed reading

If the raster file is very large, you can specify a window to read data from. This window is defined by a bounding box and coordinate reference system.

```python
# Instantiate a grid with data
>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')

# Read windowed raster
>>> grid.read_raster('../data/nlcd_2011_impervious_2011_edition_2014_10_10.img',
                     data_name='terrain', window=grid.bbox, window_crs=grid.crs)
```

## Adding in-memory datasets

In-memory datasets from a python session can also be added.

```python
# Instantiate a grid with data
>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')

# Add another copy of the DEM data as a Raster object
>>> grid.add_gridded_data(grid.dem, data_name='dem_copy')
```

Raw numpy arrays can also be added.

```python
>>> import numpy as np

# Generate random data
>>> data = np.random.randn(*grid.shape)

# Add data to grid
>>> grid.add_gridded_data(data=data, data_name='random',
                          affine=grid.affine,
                          crs=grid.crs,
                          nodata=0)
```

## Writing to raster files

By default, the `grid.to_raster` method will write the grid's current view of the dataset.

```python
>>> grid = Grid.from_ascii('../data/dir.asc', data_name='dir')
>>> grid.to_raster('dir', 'test_dir.tif', blockxsize=16, blockysize=16)
```

If the full dataset is desired, set `view=False`:

```python
>>> grid.to_raster('dir', 'test_dir.tif', view=False, 
                   blockxsize=16, blockysize=16)
```

If you want the output file to be masked with the grid mask, set `apply_mask=True`:

```python
>>> grid.to_raster('dir', 'test_dir.tif',
                   view=True, apply_mask=True, 
                   blockxsize=16, blockysize=16)
```

## Writing to ASCII files

```python
>>> grid.to_ascii('dir', 'test_dir.asc')
```

## Writing to shapefiles

For more detail, see the [jupyter notebook](https://github.com/mdbartos/pysheds/blob/master/recipes/write_shapefile.ipynb).

```python
>>> import fiona

>>> grid = Grid.from_ascii('../data/dir.asc', data_name='dir')

# Specify pour point
>>> x, y = -97.294167, 32.73750

# Delineate the catchment
>>> grid.catchment(data='dir', x=x, y=y, out_name='catch',
                   recursionlimit=15000, xytype='label',
                   nodata_out=0)

# Clip to catchment
>>> grid.clip_to('catch')

# Create a vector representation of the catchment mask
>>> shapes = grid.polygonize()

# Specify schema
>>> schema = {
        'geometry': 'Polygon',
        'properties': {'LABEL': 'float:16'}
    }

# Write shapefile
>>> with fiona.open('catchment.shp', 'w',
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

