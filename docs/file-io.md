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
