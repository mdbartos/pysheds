# pysheds [![Build Status](https://travis-ci.org/mdbartos/pysheds.svg?branch=master)](https://travis-ci.org/mdbartos/pysheds) [![Coverage Status](https://coveralls.io/repos/github/mdbartos/pysheds/badge.svg?branch=master&service=github)](https://coveralls.io/github/mdbartos/pysheds?branch=master) [![Python Versions](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8%7C3.9-blue.svg)](https://www.python.org/downloads/)
🌎 Simple and fast watershed delineation in python.

## Documentation

Read the docs [here 📖](https://mdbartos.github.io/pysheds).

## Media

*Hatari Labs* - [Elevation model conditioning and stream network delineation with python and pysheds](https://www.hatarilabs.com/ih-en/elevation-model-conditioning-and-stream-network-delimitation-with-python-and-pysheds-tutorial) <sup>:uk:</sup>

*Hatari Labs* - [Watershed and stream network delineation with python and pysheds](https://www.hatarilabs.com/ih-en/watershed-and-stream-network-delimitation-with-python-and-pysheds-tutorial) <sup>:uk:</sup>

*Gidahatari* - [Delimitación de límite de cuenca y red hidrica con python y pysheds](http://gidahatari.com/ih-es/delimitacion-de-limite-de-cuenca-y-red-hidrica-con-python-y-pysheds-tutorial) <sup>:es:</sup>

*Earth Science Information Partners* - [Pysheds: a fast, open-source digital elevation model processing library](https://www.esipfed.org/student-fellow-blog/pysheds-a-fast-open-source-digital-elevation-model-processing-library) <sup>:uk:</sup>

## Example usage

Example data used in this tutorial are linked below:

  - Elevation: [elevation.tiff](https://pysheds.s3.us-east-2.amazonaws.com/data/elevation.tiff)
  - Terrain: [impervious_area.zip](https://pysheds.s3.us-east-2.amazonaws.com/data/impervious_area.zip)
  - Soil Polygons: [soils.zip](https://pysheds.s3.us-east-2.amazonaws.com/data/soils.zip)
  
Additional DEM datasets are available via the [USGS HydroSHEDS](https://www.hydrosheds.org/) project.

### Read DEM data

```python
# Read elevation raster
# ----------------------------
from pysheds.grid import Grid

grid = Grid.from_raster('elevation.tiff')
dem = grid.read_raster('elevation.tiff')
```

<details>
<summary>Plotting code...</summary>
<p>

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
plt.colorbar(label='Elevation (m)')
plt.grid(zorder=0)
plt.title('Digital elevation map', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
```

</p>
</details>

![Example 1](https://pysheds.s3.us-east-2.amazonaws.com/img/dem.png)

### Condition the elevation data

```python
# Condition DEM
# ----------------------
# Fill pits in DEM
pit_filled_dem = grid.fill_pits(dem)

# Fill depressions in DEM
flooded_dem = grid.fill_depressions(pit_filled_dem)
    
# Resolve flats in DEM
inflated_dem = grid.resolve_flats(flooded_dem)
```

### Elevation to flow direction

```python
# Determine D8 flow directions from DEM
# ----------------------
# Specify directional mapping
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    
# Compute flow directions
# -------------------------------------
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
```

<details>
<summary>Plotting code...</summary>
<p>

```python
fig = plt.figure(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
boundaries = ([0] + sorted(list(dirmap)))
plt.colorbar(boundaries= boundaries,
             values=sorted(dirmap))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow direction grid', size=14)
plt.grid(zorder=-1)
plt.tight_layout()
```

</p>
</details>

![Example 2](https://pysheds.s3.us-east-2.amazonaws.com/img/fdir.png)

### Compute accumulation from flow direction

```python
# Calculate flow accumulation
# --------------------------
acc = grid.accumulation(fdir, dirmap=dirmap)
```

<details>
<summary>Plotting code...</summary>
<p>

```python
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
```

</p>
</details>

![Example 4](https://pysheds.s3.us-east-2.amazonaws.com/img/acc.png)


### Delineate catchment from flow direction

```python
# Delineate a catchment
# ---------------------
# Specify pour point
x, y = -97.294, 32.737

# Snap pour point to high accumulation cell
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))

# Delineate the catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, 
                       xytype='coordinate')

# Crop and plot the catchment
# ---------------------------
# Clip the bounding box to the catchment
grid.clip_to(catch)
clipped_catch = grid.view(catch)
```

<details>
<summary>Plotting code...</summary>
<p>

```python
# Plot the catchment
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)
```

</p>
</details>

![Example 3](https://pysheds.s3.us-east-2.amazonaws.com/img/catch.png)

### Extract the river network

```python
# Extract river network
# ---------------------
branches = grid.extract_river_network(fdir, acc > 50, dirmap=dirmap)
```

<details>
<summary>Plotting code...</summary>
<p>

```python
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])
    
_ = plt.title('D8 channels', size=14)
```

</p>
</details>

![Example 6](https://pysheds.s3.us-east-2.amazonaws.com/img/river.png)

### Compute flow distance from flow direction

```python
# Calculate distance to outlet from each cell
# -------------------------------------------
dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,
                               xytype='coordinate')
```

<details>
<summary>Plotting code...</summary>
<p>

```python
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(dist, extent=grid.extent, zorder=2,
               cmap='cubehelix_r')
plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow Distance', size=14)
```

</p>
</details>

![Example 5](https://pysheds.s3.us-east-2.amazonaws.com/img/dist.png)

### Add land cover data

```python
# Combine with land cover data
# ---------------------
terrain = grid.read_raster('impervious_area.tiff', window=grid.bbox,
                           window_crs=grid.crs, nodata=0)
# Reproject data to grid's coordinate reference system
projected_terrain = terrain.to_crs(grid.crs)
# View data in catchment's spatial extent
catchment_terrain = grid.view(projected_terrain, nodata=np.nan)
```

<details>
<summary>Plotting code...</summary>
<p>

```python
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(catchment_terrain, extent=grid.extent, zorder=2,
               cmap='bone')
plt.colorbar(im, ax=ax, label='Percent impervious area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Percent impervious area', size=14)
```

</p>
</details>

![Example 7](https://pysheds.s3.us-east-2.amazonaws.com/img/terrain.png)

### Add vector data

```python
# Convert catchment raster to vector and combine with soils shapefile
# ---------------------
# Read soils shapefile
import pandas as pd
import geopandas as gpd
from shapely import geometry, ops
soils = gpd.read_file('soils.shp')
soil_id = 'MUKEY'
# Convert catchment raster to vector geometry and find intersection
shapes = grid.polygonize()
catchment_polygon = ops.unary_union([geometry.shape(shape)
                                     for shape, value in shapes])
soils = soils[soils.intersects(catchment_polygon)]
catchment_soils = gpd.GeoDataFrame(soils[soil_id], 
                                   geometry=soils.intersection(catchment_polygon))
# Convert soil types to simple integer values
soil_types = np.unique(catchment_soils[soil_id])
soil_types = pd.Series(np.arange(soil_types.size), index=soil_types)
catchment_soils[soil_id] = catchment_soils[soil_id].map(soil_types)
```

<details>
<summary>Plotting code...</summary>
<p>

```python
fig, ax = plt.subplots(figsize=(8, 6))
catchment_soils.plot(ax=ax, column=soil_id, categorical=True, cmap='terrain',
                     linewidth=0.5, edgecolor='k', alpha=1, aspect='equal')
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax.set_title('Soil types (vector)', size=14)
```

</p>
</details>

![Example 8](https://pysheds.s3.us-east-2.amazonaws.com/img/poly.png)

### Convert from vector to raster

```python
soil_polygons = zip(catchment_soils.geometry.values, catchment_soils[soil_id].values)
soil_raster = grid.rasterize(soil_polygons, fill=np.nan)
```

<details>
<summary>Plotting code...</summary>
<p>

```python
fig, ax = plt.subplots(figsize=(8, 6))
plt.imshow(soil_raster, cmap='terrain', extent=grid.extent, zorder=1)
boundaries = np.unique(soil_raster[~np.isnan(soil_raster)]).astype(int)
plt.colorbar(boundaries=boundaries,
             values=boundaries)
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax.set_title('Soil types (raster)', size=14)
```

</p>
</details>

![Example 9](https://pysheds.s3.us-east-2.amazonaws.com/img/rasterize.png)

## Features

- Hydrologic Functions:
  - `flowdir` : Generate a flow direction grid from a given digital elevation dataset.
  - `catchment` : Delineate the watershed for a given pour point (x, y).
  - `accumulation` : Compute the number of cells upstream of each cell; if weights are
                given, compute the sum of weighted cells upstream of each cell.
  - `distance_to_outlet` : Compute the (weighted) distance from each cell to a given
                      pour point, moving downstream.
  - `distance_to_ridge` : Compute the (weighted) distance from each cell to its originating
                    drainage divide, moving upstream.
  - `compute_hand` : Compute the height above nearest drainage (HAND).
  - `stream_order` : Compute the (strahler) stream order.
  - `extract_river_network` : Extract river segments from a catchment and return a geojson
                        object.
  - `cell_dh` : Compute the drop in elevation from each cell to its downstream neighbor.
  - `cell_distances` : Compute the distance from each cell to its downstream neighbor.
  - `cell_slopes` : Compute the slope between each cell and its downstream neighbor.
  - `fill_pits` : Fill single-celled pits in a digital elevation dataset.
  - `fill_depressions` : Fill multi-celled depressions in a digital elevation dataset.
  - `resolve_flats` : Remove flats from a digital elevation dataset.
  - `detect_pits` : Detect single-celled pits in a digital elevation dataset.
  - `detect_depressions` : Detect multi-celled depressions in a digital elevation dataset.
  - `detect_flats` : Detect flats in a digital elevation dataset.
- Viewing Functions:
  - `view` : Returns a "view" of a dataset defined by the grid's viewfinder.
  - `clip_to` : Clip the viewfinder to the smallest area containing all non-
          null gridcells for a provided dataset.
  - `nearest_cell` : Returns the index (column, row) of the cell closest
                to a given geographical coordinate (x, y).
  - `snap_to_mask` : Snaps a set of points to the nearest nonzero cell in a boolean mask;
                useful for finding pour points from an accumulation raster.
- I/O Functions:
  - `read_ascii`: Reads ascii gridded data.
  - `read_raster`: Reads raster gridded data.
  - `from_ascii` : Instantiates a grid from an ascii file.
  - `from_raster` : Instantiates a grid from a raster file or Raster object.
  - `to_ascii`: Write grids to delimited ascii files.
  - `to_raster`: Write grids to raster files (e.g. geotiff).

`pysheds` supports both D8 and D-infinity routing schemes.

## Installation

`pysheds` currently only supports Python 3.

### Using pip

You can install `pysheds` using pip:

```bash
$ pip install pysheds
```

### Using anaconda

First, add conda forge to your channels, if you have not already done so:

```bash
$ conda config --add channels conda-forge
```

Then, install pysheds:

```bash
$ conda install pysheds
```
### Installing from source

For the bleeding-edge version, you can install pysheds from this github repository.

```bash
$ git clone https://github.com/mdbartos/pysheds.git
$ cd pysheds
$ python setup.py install
```

or

```bash
$ git clone https://github.com/mdbartos/pysheds.git
$ cd pysheds
$ pip install .
```

# Performance
Performance benchmarks on a 2015 MacBook Pro (M: million, K: thousand):

| Function                | Routing | Number of cells          | Run time |
| ----------------------- | ------- | ------------------------ | -------- |
| `flowdir`               | D8      |  36M                     | 1.09 [s] |
| `flowdir`               | DINF    |  36M                     | 6.64 [s] |
| `accumulation`          | D8      |  36M                     | 3.65 [s] |
| `accumulation`          | DINF    |  36M                     | 16.2 [s] |
| `catchment`             | D8      |  9.76M                   | 3.43 [s] |
| `catchment`             | DINF    |  9.76M                   | 5.41 [s] |
| `distance_to_outlet`    | D8      |  9.76M                   | 4.74 [s] |
| `distance_to_outlet`    | DINF    |  9.76M                   | 1 [m] 13 [s] |
| `distance_to_ridge`     | D8      |  36M                     | 6.83 [s] |
| `hand`                  | D8      |  36M total, 730K channel | 12.9 [s] |
| `hand`                  | DINF    |  36M total, 770K channel | 18.7 [s] |
| `stream_order`          | D8      |  36M total, 1M channel   | 3.99 [s] |
| `extract_river_network` | D8      |  36M total, 345K channel | 4.07 [s] |
| `detect_pits`           | N/A     |  36M                     | 1.80 [s] |
| `detect_flats`          | N/A     |  36M                     | 1.84 [s] |
| `fill_pits`             | N/A     |  36M                     | 2.52 [s] |
| `fill_depressions`      | N/A     |  36M                     | 27.1 [s] |
| `resolve_flats`         | N/A     |  36M                     | 9.56 [s] |
| `cell_dh`               | D8      |  36M                     | 2.34 [s] |
| `cell_dh`               | DINF    |  36M                     | 4.92 [s] |
| `cell_distances`        | D8      |  36M                     | 1.11 [s] |
| `cell_distances`        | DINF    |  36M                     | 2.16 [s] |
| `cell_slopes`           | D8      |  36M                     | 4.01 [s] |
| `cell_slopes`           | DINF    |  36M                     | 10.2 [s] |

Speed tests were run on a conditioned DEM from the HYDROSHEDS DEM repository
(linked above as `elevation.tiff`).

# Citing

If you have used this codebase in a publication and wish to cite it, consider citing the zenodo repository:

```bibtex
@misc{bartos_2020,
    title  = {pysheds: simple and fast watershed delineation in python},
    author = {Bartos, Matt},
    url    = {https://github.com/mdbartos/pysheds},
    year   = {2020},
    doi    = {10.5281/zenodo.3822494}
}
```
