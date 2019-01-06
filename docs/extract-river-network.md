# Extract River Network

## Preliminaries

The `grid.extract_river_network` method requires both a catchment grid and an accumulation grid. The catchment grid can be obtained from a flow direction grid, as shown in [catchments](https://mdbartos.github.io/pysheds/catchment.html). The accumulation grid can also be obtained from a flow direction grid, as shown in [accumulation](https://mdbartos.github.io/pysheds/accumulation.html).

```python
>>> import numpy as np
>>> from matplotlib import pyplot as plt
>>> from pysheds.grid import Grid

# Instantiate grid from raster
>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')

# Resolve flats and compute flow directions
>>> grid.resolve_flats(data='dem', out_name='inflated_dem')
>>> grid.flowdir('inflated_dem', out_name='dir')

# Specify outlet
>>> x, y = -97.294167, 32.73750

# Delineate a catchment
>>> grid.catchment(data='dir', x=x, y=y, out_name='catch',
                   recursionlimit=15000, xytype='label')

# Clip the view to the catchment
>>> grid.clip_to('catch')

# Compute accumulation
>>> grid.accumulation(data='catch', out_name='acc')
```

## Extracting the river network

To extract the river network at a given accumulation threshold, we can call the `grid.extract_river_network` method. By default, the method will use an accumulation threshold of 100 cells:

```python
# Extract river network
>>> branches = grid.extract_river_network('catch', 'acc')
```

The `grid.extract_river_network` method returns a dictionary in the geojson format. The branches can be plotted by iterating through the features:

```python
# Plot branches
>>> for branch in branches['features']:
>>>     line = np.asarray(branch['geometry']['coordinates'])
>>>     plt.plot(line[:, 0], line[:, 1])
```

![River network](https://s3.us-east-2.amazonaws.com/pysheds/img/river_network_100.png)

## Specifying the accumulation threshold

We can change the geometry of the returned river network by specifying different accumulation thresholds:

```python
>>> branches_50 = grid.extract_river_network('catch', 'acc', threshold=50)
>>> branches_2 = grid.extract_river_network('catch', 'acc', threshold=2)
```

![River network 50](https://s3.us-east-2.amazonaws.com/pysheds/img/river_network.png)
![River network 2](https://s3.us-east-2.amazonaws.com/pysheds/img/river_network_2.png)

