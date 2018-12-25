# Raster datasets

`Grid` methods operate on `Raster` objects. You can think of a `Raster` as a numpy array with additional attributes that specify the location, resolution and coordinate reference system of the data.

When a dataset is read from a file, it will automatically be saved as a `Raster` object.

```python
>>> from pysheds.grid import Grid

>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')
>>> dem = grid.dem
```

```python
>>> dem
Raster([[214, 212, 210, ..., 177, 177, 175],
        [214, 210, 207, ..., 176, 176, 174],
        [211, 209, 204, ..., 174, 174, 174],
        ...,
        [263, 262, 263, ..., 217, 217, 216],
        [266, 265, 265, ..., 217, 217, 217],
        [268, 267, 266, ..., 216, 217, 216]], dtype=int16)
```

## Calling methods on rasters

Primary `Grid` methods (such as flow direction determination and catchment delineation) can be called directly on `Raster objects`:

```python
>>> grid.resolve_flats(dem, out_name='inflated_dem')
```

Grid methods can also return `Raster` objects by specifying `inplace=False`:

```python
>>> fdir = grid.flowdir(grid.inflated_dem, inplace=False)
```

```python
>>> fdir
Raster([[  0,   0,   0, ...,   0,   0,   0],
        [  0,   2,   2, ...,   4,   1,   0],
        [  0,   1,   2, ...,   4,   2,   0],
        ...,
        [  0,  64,  32, ...,   8,   1,   0],
        [  0,  64,  32, ...,  16, 128,   0],
        [  0,   0,   0, ...,   0,   0,   0]])
```

## Raster attributes

### Affine transform

An affine transform uniquely specifies the spatial location of each cell in a gridded dataset.

```python
>>> dem.affine
Affine(0.0008333333333333, 0.0, -100.0,
       0.0, -0.0008333333333333, 34.9999999999998)
```

The elements of the affine transform `(a, b, c, d, e, f)` are:

- **a**: cell width
- **b**: row rotation (generally zero)
- **c**: x-coordinate of upper-left corner of upper-leftmost cell
- **d**: column rotation (generally zero)
- **e**: cell height
- **f**: y-coordinate of upper-left corner of upper-leftmost cell

The affine transform uses the [affine](https://pypi.org/project/affine/) module.

### Coordinate reference system

The coordinate reference system (CRS) defines a map projection for the gridded dataset. For datasets read from a raster file, the CRS will be detected and populated automaticaally.

```python
>>> dem.crs
<pyproj.Proj at 0x12363dd68>
```

A human-readable representation of the CRS can also be obtained as follows:

```python
>>> dem.crs.srs
'+init=epsg:4326 '
```

This example dataset has a geographic projection (meaning that coordinates are defined in terms of latitudes and longitudes).

The coordinate reference system uses the [pyproj](https://pypi.org/project/pyproj/) module.

### "No data" value

The `nodata` attribute specifies the value that indicates missing or invalid data.

```python
>>> dem.nodata
-32768
```

### Derived attributes

Other attributes are derived from these primary attributes:

#### Bounding box

```python
>>> dem.bbox
(-97.4849999999961, 32.52166666666537, -97.17833333332945, 32.82166666666536)
```

#### Extent

```python
>>> dem.extent
(-97.4849999999961, -97.17833333332945, 32.52166666666537, 32.82166666666536)
```

#### Coordinates

```python
>>> dem.coords
array([[ 32.82166667, -97.485     ],
       [ 32.82166667, -97.48416667],
       [ 32.82166667, -97.48333333],
       ...,
       [ 32.52333333, -97.18166667],
       [ 32.52333333, -97.18083333],
       [ 32.52333333, -97.18      ]])
```

### Numpy attributes

A `Raster` object also inherits all attributes and methods from numpy ndarrays.

```python
>>> dem.shape
(359, 367)
```
