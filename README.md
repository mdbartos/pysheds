# pysheds
Simple and fast watershed delineation in python.

# Example usage

    Read a flow direction raster
    ----------------------------
    # Import modules
    import numpy as np
    import matplotlib.pyplot as plt
    from pysheds.grid import Grid

    # Read a grid from a raster
    grid = Grid.from_raster('../data/n30w100_dir/n30w100_dir/w001001.adf',
                            data_name='dir', input_type='ascii')

![Example 1](examples/flow_direction.png)

    Delineate a catchment
    ---------------------
    # Specify pour point
    x, y = -97.2937, 32.7371

    # Delineate the catchment
    grid.catchment(x, y, dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
                   recursionlimit=15000, xytype='label')

    Crop and plot the catchment
    ---------------------------
    # Clip the bounding box to the catchment
    grid.clip_to('catch', precision=5)

    # Get a view of the catchment
    image_arr = np.where(grid.view('catch'), grid.view('catch'), np.nan)

    # Plot the catchment
    fig, ax = plt.subplots(figsize=(8,6))
    plt.grid('on', zorder=1)
    im = ax.imshow(image_arr, extent=grid.extent, zorder=2)
    plt.colorbar(im, ax=ax)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

![Example 1](examples/catchment.png)

