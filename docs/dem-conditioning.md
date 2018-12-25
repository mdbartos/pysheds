# DEM conditioning

Raw DEMs often contain artifacts (such as depressions and flat regions) that prevent the DEM from fully draining. In this section, methods for removing these artifacts are discussed.

## Depressions

Raw DEMs often contain depressions that must be removed before further processing. Depressions consist of regions of cells for which every surrounding cell is at a higher elevation. The following DEM contains natural depressions:

### Preliminaries

```python
# Import modules
>>> from pysheds.grid import Grid

# Read raw DEM
>>> grid = Grid.from_raster('../data/roi_10m', data_name='dem')

# Plot the raw DEM
>>> plt.imshow(grid.view('dem'))
```

### Detecting depressions
Depressions can be detected using the `grid.detect_depressions` method:

```python
# Detect depressions
depressions = grid.detect_depressions('dem')

# Plot depressions
plt.imshow(depressions)
```

### Filling depressions

Depressions can be filled using the `grid.fill_depressions` method:

```python
# Fill depressions
>>> grid.fill_depressions(data='dem', out_name='flooded_dem')

# Test result
>>> depressions = grid.detect_depressions('dem')
>>> depressions.any()
False
```

## Flats

Flats consist of cells at which every surrounding cell is at the same elevation or higher. 

### Detecting flats

Flats can be detected using the `grid.detect_flats` method:

```python
flats = grid.detect_flats('flooded_dem')
```

### Resolving flats

Flats can be resolved using the `grid.resolve_flats` method:

```python
>>> grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
```

### Finished product

After filling depressions and resolving flats, the flow direction can be determined as usual:

```python
# Compute flow direction based on corrected DEM
grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)

# Compute flow accumulation based on computed flow direction
grid.accumulation(data='dir', out_name='acc', dirmap=dirmap)
```

## Burning DEMs

Burning existing streamlines into a DEM is common practice for some applications. In `pysheds`, DEMs can be burned through a combination of boolean masking and simple addition or subtraction.
