# Views

The `grid.view` method returns a copy of a dataset cropped to the grid's current view. The grid's current view is defined by the following attributes:

- `affine`: An affine transform that defines the coordinates of the top-left cell, along with the cell resolution and rotation.
- `crs`: The coordinate reference system of the grid.
- `shape`: The shape of the grid (number of rows by number of columns)
- `mask`: A boolean array that defines which cells will be masked in the output `Raster`.

## Initializing the grid view

The grid's view will be populated automatically upon reading the first dataset.

```python
>>> grid = Grid.from_raster('../data/dem.tif',
                            data_name='dem')
>>> grid.affine
Affine(0.0008333333333333, 0.0, -97.4849999999961,
       0.0, -0.0008333333333333, 32.82166666666536)
       
>>> grid.crs
<pyproj.Proj at 0x123da4b88>

>>> grid.shape
(359, 367)

>>> grid.mask
array([[ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       ...,
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True]])
```

We can verify that the spatial reference system is the same as that of the originating dataset:

```python
>>> grid.affine == grid.dem.affine
True
>>> grid.crs == grid.dem.crs
True
>>> grid.shape == grid.dem.shape
True
>>> (grid.mask == grid.dem.mask).all()
True
```

## Viewing datasets

First, let's delineate a watershed and use the `grid.view` method to get the results.

```python
# Resolve flats
>>> grid.resolve_flats(data='dem', out_name='inflated_dem')

# Specify pour point
>>> x, y = -97.294167, 32.73750

# Delineate the catchment
>>> grid.catchment(data='dir', x=x, y=y, out_name='catch',
                   recursionlimit=15000, xytype='label')

# Get the current view and plot
>>> catch = grid.view('catch')
>>> plt.imshow(catch)
```

![Catchment view](https://s3.us-east-2.amazonaws.com/pysheds/img/catchment_view.png)

## Clipping the view to a dataset

The `grid.clip_to` method clips the grid's current view to nonzero elements in a given dataset. This is especially useful for clipping the view to an irregular feature like a delineated watershed.

```python
# Clip the grid's view to the catchment dataset
>>> grid.clip_to('catch')

# Get the current view and plot
>>> catch = grid.view('catch')
>>> plt.imshow(catch)
```

![Clipped view](https://s3.us-east-2.amazonaws.com/pysheds/img/catchment_view_clipped.png)

## Tweaking the view using keyword arguments

### Setting the "no data" value

The "no data" value in the output array can be specified using the `nodata` keyword argument. This is often useful for visualization.

```python
>>> catch = grid.view('dem', nodata=np.nan)
>>> plt.imshow(catch)
```

![Setting nodata](https://s3.us-east-2.amazonaws.com/pysheds/img/dem_view_clipped_nodata.png)

### Toggling the mask

The mask can be turned off by setting `apply_mask=False`.

```python
>>> catch = grid.view('dem', nodata=np.nan,
                      apply_mask=False)
>>> plt.imshow(catch)
```

![Setting nodata](https://s3.us-east-2.amazonaws.com/pysheds/img/dem_view_nomask.png)

### Setting the interpolation method

By default, the view method uses a nearest neighbors approach for interpolation. However, this can be changed using the `interpolation` keyword argument.

```python
>>> nn_interpolation = grid.view('terrain',
                                 nodata=np.nan)
>>> plt.imshow(nn_interpolation)
```

![Nearest neighbors](https://s3.us-east-2.amazonaws.com/pysheds/img/nn_interpolation.png)

```python
>>> linear_interpolation = grid.view('terrain',
                                     interpolation='linear',
                                     nodata=np.nan)
>>> plt.imshow(linear_interpolation)
```

![Linear interpolation](https://s3.us-east-2.amazonaws.com/pysheds/img/linear_interpolation.png)

## Clipping the view to a bounding box

The grid's view can be set to a rectangular bounding box using the `grid.set_bbox` method. 

```python
# Specify new bbox as upper-right quadrant of old bbox
>>> new_xmin = (grid.bbox[2] + grid.bbox[0]) / 2
>>> new_ymin = (grid.bbox[3] + grid.bbox[1]) / 2
>>> new_xmax = grid.bbox[2]
>>> new_ymax = grid.bbox[3]
>>> new_bbox = (new_xmin, new_ymin, new_xmax, new_ymax)

# Set new bbox
>>> grid.set_bbox(new_bbox)

# Plot the new view
>>> plt.imshow(grid.view('catch'))
```

![Set bbox](https://s3.us-east-2.amazonaws.com/pysheds/img/catch_upper_quad.png)


## Setting the view manually

The `grid.affine`, `grid.crs`, `grid.shape` and `grid.mask` attributes can also be set manually.

```python
# Reset the view to the dataset's original view
>>> grid.affine = grid.dem.affine
>>> grid.crs = grid.dem.crs
>>> grid.shape = grid.dem.shape
>>> grid.mask = grid.dem.mask

# Plot the new view
>>> plt.imshow(grid.view('catch'))
```

![Set bbox](https://s3.us-east-2.amazonaws.com/pysheds/img/full_dem.png)
