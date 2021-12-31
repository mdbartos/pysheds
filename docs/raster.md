# Raster datasets

`Grid` methods operate on `Raster` objects. You can think of a `Raster` as a numpy array with additional attributes that specify the location, resolution and coordinate reference system of the data.

When a dataset is read from a file, it will automatically be saved as a `Raster` object.

```python
from pysheds.grid import Grid

grid = Grid.from_raster('./data/dem.tif')
dem = grid.read_raster('./data/dem.tif')
```

Here, `grid` is the `Grid` instance, and `dem` is a `Raster` object. If we call the `Raster` object, we will see that it looks much like a numpy array.

```python
dem
```
<details>
<summary>Output...</summary>
<p>

```
Raster([[214, 212, 210, ..., 177, 177, 175],
        [214, 210, 207, ..., 176, 176, 174],
        [211, 209, 204, ..., 174, 174, 174],
        ...,
        [263, 262, 263, ..., 217, 217, 216],
        [266, 265, 265, ..., 217, 217, 217],
        [268, 267, 266, ..., 216, 217, 216]], dtype=int16)
```

</p>
</details>

## Calling methods on rasters

Hydrologic functions (such as flow direction determination and catchment delineation) accept and return `Raster objects`:

```python
inflated_dem = grid.resolve_flats(dem)
fdir = grid.flowdir(inflated_dem)
```

```python
fdir
```

<details>
<summary>Output...</summary>
<p>

```
Raster([[  0,   0,   0, ...,   0,   0,   0],
        [  0,   2,   2, ...,   4,   1,   0],
        [  0,   1,   2, ...,   4,   2,   0],
        ...,
        [  0,  64,  32, ...,   8,   1,   0],
        [  0,  64,  32, ...,  16, 128,   0],
        [  0,   0,   0, ...,   0,   0,   0]])
```

</p>
</details>



## Raster attributes

### Viewfinder

The viewfinder attribute contains all the information needed to specify the Raster's spatial reference system. It can be accessed using the `viewfinder` attribute.

```python
dem.viewfinder
```

<details>
<summary>Output...</summary>
<p>

```
<pysheds.sview.ViewFinder at 0x13222f908>
```

</p>
</details>


The viewfinder contains five necessary elements that completely define the spatial reference system.

  - `affine`: An affine transformation matrix.
  - `shape`: The desired shape (rows, columns).
  - `crs` : The coordinate reference system.
  - `mask` : A boolean array indicating which cells are masked.
  - `nodata` : A sentinel value indicating 'no data'.

### Affine transformation matrix

An affine transform uniquely specifies the spatial location of each cell in a gridded dataset. In a `Raster`, the affine transform is given by the `affine` attribute.

```python
dem.affine
```
<details>
<summary>Output...</summary>
<p>

```
Affine(0.0008333333333333, 0.0, -100.0,
       0.0, -0.0008333333333333, 34.9999999999998)
```

</p>
</details>

The elements of the affine transform `(a, b, c, d, e, f)` are:

- **a**: Horizontal scaling (equal to cell width if no rotation)
- **b**: Horizontal shear
- **c**: Horizontal translation (x-coordinate of upper-left corner of upper-leftmost cell)
- **d**: Vertical shear
- **e**: Vertical scaling (equal to cell height if no rotation)
- **f**: Vertical translation (y-coordinate of upper-left corner of upper-leftmost cell)

The affine transform uses the [affine](https://pypi.org/project/affine/) module.

### Shape

The shape is equal to the shape of the underlying array (i.e. number of rows, number of columns).

```python
dem.shape
```

<details>
<summary>Output...</summary>
<p>

```
(359, 367)
```

</p>
</details>

### Coordinate reference system

The coordinate reference system (CRS) defines a map projection for the gridded
dataset. The `crs` attribute is a `pyproj.Proj` object. For datasets read from a
raster file, the CRS will be detected and populated automaticaally.

```python
dem.crs
```

<details>
<summary>Output...</summary>
<p>

```
Proj('+proj=longlat +datum=WGS84 +no_defs', preserve_units=True)
```

</p>
</details>

This example dataset has a geographic projection (meaning that coordinates are defined in terms of latitudes and longitudes).

The coordinate reference system uses the [pyproj](https://pypi.org/project/pyproj/) module.

### Mask

The mask is a boolean array indicating which cells in the dataset should be masked in the output view.

```python
dem.mask
```

<details>
<summary>Output...</summary>
<p>

```
array([[ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       ...,
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True]])
```

</p>
</details>

### "No data" value

The `nodata` attribute specifies the value that indicates missing or invalid data.

```python
dem.nodata
```

<details>
<summary>Output...</summary>
<p>

```
-32768
```

</p>
</details>

### Derived attributes

Other attributes are derived from these primary attributes:

#### Bounding box

```python
dem.bbox
```

<details>
<summary>Output...</summary>
<p>

```
(-97.4849999999961, 32.52166666666537, -97.17833333332945, 32.82166666666536)
```

</p>
</details>

#### Extent

```python
dem.extent
```

<details>
<summary>Output...</summary>
<p>

```
(-97.4849999999961, -97.17833333332945, 32.52166666666537, 32.82166666666536)
```

</p>
</details>

#### Coordinates

```python
dem.coords
```

<details>
<summary>Output...</summary>
<p>

```
array([[ 32.82166667, -97.485     ],
       [ 32.82166667, -97.48416667],
       [ 32.82166667, -97.48333333],
       ...,
       [ 32.52333333, -97.18166667],
       [ 32.52333333, -97.18083333],
       [ 32.52333333, -97.18      ]])
```

</p>
</details>

## Converting the raster coordinate reference system

The Raster can be transformed to a new coordinate reference system using the `to_crs` method:

```python
import pyproj
import numpy as np

# Initialize new CRS
new_crs = pyproj.Proj('epsg:3083')

# Convert CRS of dataset and set nodata value for better plotting
dem.nodata = np.nan
proj_dem = dem.to_crs(new_crs)
```

<details>
<summary>Plotting code...</summary>
<p>

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(12,8))
fig.patch.set_alpha(0)
ax[0].imshow(dem, cmap='terrain', zorder=1)
ax[1].imshow(proj_dem, cmap='terrain', zorder=1)
ax[0].set_title('DEM', size=14)
ax[1].set_title('Projected DEM', size=14)
plt.tight_layout()
```

</p>
</details>

Note that the projected Raster appears slightly rotated to the counterclockwise direction.

![Projection](https://s3.us-east-2.amazonaws.com/pysheds/img/rasters_projection.png)

