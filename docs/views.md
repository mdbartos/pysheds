# Views

The `grid.view` method returns a copy of a dataset cropped to the grid's current view. The grid's current view is defined by its `viewfinder` attribute, which contains five properties that fully define the spatial reference system:

  - `affine`: An affine transformation matrix.
  - `shape`: The desired shape (rows, columns).
  - `crs` : The coordinate reference system.
  - `mask` : A boolean array indicating which cells are masked.
  - `nodata` : A sentinel value indicating 'no data'.

## Initializing the grid view

The grid's view will be populated automatically upon reading the first dataset.

```python
grid = Grid.from_raster('./data/dem.tif')
```

```python
grid.affine
```

<details>
<summary>Output...</summary>
<p>

```
Affine(0.0008333333333333, 0.0, -97.4849999999961,
       0.0, -0.0008333333333333, 32.82166666666536)
```

</p>
</details>


```python
grid.crs
```

<details>
<summary>Output...</summary>
<p>

```
Proj('+proj=longlat +datum=WGS84 +no_defs', preserve_units=True)
```

</p>
</details>


```python
grid.shape
```

<details>
<summary>Output...</summary>
<p>

```
(359, 367)
```

</p>
</details>

```python
grid.mask
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

We can verify that the spatial reference system is the same as that of the originating dataset:

```python
dem = grid.read_raster('./data/dem.tif')
```

```python
grid.affine == dem.affine
```

<details>
<summary>Output...</summary>
<p>

```
True
```

</p>
</details>

```python
grid.crs == dem.crs
```

<details>
<summary>Output...</summary>
<p>

```
True
```

</p>
</details>

```python
grid.shape == dem.shape
```

<details>
<summary>Output...</summary>
<p>

```
True
```

</p>
</details>


```python
(grid.mask == dem.mask).all()
```

<details>
<summary>Output...</summary>
<p>

```
True
```

</p>
</details>


## Viewing datasets

First, let's delineate a watershed and use the `grid.view` method to get the results.

```python
# Resolve flats
inflated_dem = grid.resolve_flats(dem)

# Compute flow directions
fdir = grid.flowdir(inflated_dem)

# Specify pour point
x, y = -97.294167, 32.73750

# Delineate the catchment
catch = grid.catchment(x=x, y=y, fdir=fdir, xytype='label')

# Get the current view and plot
catch_view = grid.view(catch)
plt.imshow(catch_view, zorder=1)
```

![Catchment view](https://s3.us-east-2.amazonaws.com/pysheds/img/catchment_view.png)

Note that in this case, the original raster and its view are the same:

```python
(catch == catch_view).all()
```

<details>
<summary>Output...</summary>
<p>

```
True
```

</p>
</details>


## Clipping the view to a dataset

The `grid.clip_to` method clips the grid's current view to nonzero elements in a given dataset. This is especially useful for clipping the view to an irregular feature like a delineated watershed.

```python
# Clip the grid's view to the catchment dataset
grid.clip_to(catch)

# Get the current view and plot
catch_view = grid.view(catch)
plt.imshow(catch_view, zorder=1)
```

We can also now use the `view` method to view other datasets within the current catchment boundaries:

```python
# Get the current view of flow directions
fdir_view = grid.view(fdir)
plt.imshow(fdir_view, cmap='viridis', zorder=1)
```

![Clipped view](https://s3.us-east-2.amazonaws.com/pysheds/img/catchment_view_clipped.png)

## Tweaking the view using keyword arguments

### Setting the "no data" value

The "no data" value in the output array can be specified using the `nodata` keyword argument. This is often useful for visualization.

```python
dem_view = grid.view(dem, nodata=np.nan)
plt.imshow(dem_view, cmap='terrain', zorder=1)
```

![Setting nodata](https://s3.us-east-2.amazonaws.com/pysheds/img/dem_view_clipped_nodata.png)

### Toggling the mask

The mask can be turned off by setting `apply_output_mask=False`.

```python
dem_view = grid.view(dem, nodata=np.nan,
                     apply_output_mask=False)
plt.imshow(dem_view, cmap='terrain', zorder=1)
```

![Setting nodata](https://s3.us-east-2.amazonaws.com/pysheds/img/dem_view_nomask.png)

### Setting the interpolation method

By default, the view method uses a nearest neighbors approach for interpolation. However, this can be changed using the `interpolation` keyword argument.

```python
# Load a dataset with a different spatial reference system
terrain = grid.read_raster('./data/impervious_area.tiff', window=grid.bbox,
                           window_crs=grid.crs)
```

#### Nearest neighbor interpolation

```python
# View the new dataset with nearest neighbor interpolation
nn_interpolation = grid.view(terrain, nodata=np.nan)
plt.imshow(nn_interpolation, zorder=1, cmap='bone')
```

![Nearest neighbors](https://s3.us-east-2.amazonaws.com/pysheds/img/nn_interpolation.png)

#### Linear interpolation

```python
# View the new dataset with linear interpolation
lin_interpolation = grid.view(terrain, nodata=np.nan, interpolation='linear')
plt.imshow(lin_interpolation, zorder=1, cmap='bone')
```

![Linear interpolation](https://s3.us-east-2.amazonaws.com/pysheds/img/linear_interpolation.png)

## Setting the view manually

The `grid.viewfinder` attribute can also be set manually.

```python
# Reset the view to the dataset's original view
grid.viewfinder = dem.viewfinder

# Plot the new view
dem_view = grid.view(dem)
plt.imshow(dem_view, zorder=1, cmap='terrain')
```

![Set bbox](https://s3.us-east-2.amazonaws.com/pysheds/img/full_dem.png)
