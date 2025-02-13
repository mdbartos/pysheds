#!/usr/bin/env python
import json
import time

import typer
import numpy as np
from typing import Tuple
from typing_extensions import Annotated


app = typer.Typer()


def lazy_import():
    global sGrid
    from pysheds.sgrid import sGrid


@app.command()
def fill_depressions(
    dem_path: str = typer.Argument(..., help="Path to the input DEM"),
    output_path: str = typer.Argument(..., help="Path to the inflated DEM"),
):
    """
    Fills depressions in the DEM and saves the result.
    """

    start = time.time()
    lazy_import()

    grid: sGrid = sGrid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    typer.echo("Filling pits ...")
    pit_filled_dem = grid.fill_pits(dem)

    typer.echo("Filling depression ...")
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    typer.echo("Resolving flats areas ...")
    inflated_dem = grid.resolve_flats(flooded_dem)

    grid.to_raster(inflated_dem, output_path)
    typer.echo(f"Writed inflated DEM as {output_path}")

    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


@app.command()
def flow_directions(
    dem_path: str = typer.Argument(..., help="The input inflated DEM"),
    output_path: str = typer.Argument(..., help="The output flow directions map"),
    dirmap: Annotated[
        Tuple[int, int, int, int, int, int, int, int],
        typer.Argument(help="8 direction map from N to NW clockwise"),
    ] = (64, 128, 1, 2, 4, 8, 16, 32),
):
    """
    Compute flow directions from inflated DEM using D8 method.
    """

    start = time.time()
    lazy_import()

    grid: sGrid = sGrid.from_raster(dem_path)
    inflated_dem = grid.read_raster(dem_path)

    typer.echo("Computing flow directions ...")
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    grid.to_raster(
        fdir.astype(np.uint8),
        output_path,
        target_view=fdir.viewfinder,
        blockxsize=16,
        blockysize=16,
    )

    typer.echo(f"Writed flow directions map as {output_path}")

    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


@app.command()
def contributing_area(
    fdir_path: str = typer.Argument(..., help="The flow directions map"),
    output_path: str = typer.Argument(
        ..., help="Path where the resulting accumulation raster will be saved."
    ),
):
    """
    Calculates the contributing area (accumulation map) from a flow direction raster
    and writes the result to an output raster file.
    """
    start = time.time()
    lazy_import()

    grid: sGrid = sGrid.from_raster(fdir_path)
    fdir = grid.read_raster(fdir_path)

    acc = grid.accumulation(fdir)

    grid.to_raster(acc, output_path, blockxsize=16, blockysize=16)

    typer.echo(f"Accumulation map writed as: {output_path}")
    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


@app.command()
def extract_network(
    acc_path: str = typer.Argument(..., help="The input accumulation raster map"),
    output_path: str = typer.Argument(..., help="The output network raster map"),
    threshold: float = typer.Option(
        1000, help="The threshold to apply on accumulation map to delineate network."
    ),
):
    """
    It extracts the drainage network from the DEM and saves it.
    """
    start = time.time()
    lazy_import()

    grid: sGrid = sGrid.from_raster(acc_path)
    acc = grid.read_raster(acc_path)

    network = acc > threshold
    grid.to_raster(network, output_path, blockxsize=16, blockysize=16)

    typer.echo(f"Network map writed as: {output_path}")
    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


@app.command()
def extract_catchment(
    x: float = typer.Argument(..., help="X-coordinate of the pour point."),
    y: float = typer.Argument(..., help="Y-coordinate of the pour point."),
    fdir_path: str = typer.Argument(..., help="Path to the flow direction raster."),
    output_path: str = typer.Argument(..., help="The output raster mask."),
    snap: str = typer.Option(
        "center",
        help="Snap tolerance for the point to the nearest flow path. "
        "Choose either 'center' or 'corner'.",
    ),
):
    """
    Calculates the catchment area for a given point (x, y) using a flow direction raster
    and saves the result as an output raster.
    """
    start = time.time()
    lazy_import()
    # Load the flow direction raster
    grid: sGrid = sGrid.from_raster(fdir_path)
    fdir = grid.read_raster(fdir_path)

    # Calculate the catchment area
    catchment_area = grid.catchment(x=x, y=y, fdir=fdir, xytype="coordinate", snap=snap)

    grid.to_raster(catchment_area.astype(np.uint8), output_path)

    typer.echo(f"Catchment area calculated and saved to: {output_path}")
    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


def parse_split_points(split_points):
    # Convert split_points from string to a list of tuples
    split_points_list = None
    if split_points:
        split_points_list = [
            tuple(map(int, point.split(","))) for point in split_points.split(";")
        ]
    return split_points_list


@app.command()
def extract_subcatchment(
    fdir_path: str = typer.Argument(..., help="Path to the flow direction raster."),
    acc_path: str = typer.Argument(..., help="Path to the accumulation raster."),
    threshold: int = typer.Option(
        1000,
        "-t",
        "--threshold",
        help="Threshold for accumulation to define the network.",
    ),
    split_points: str = typer.Option(
        None,
        help="Comma-separated list of split point coordinates (e.g., '4513192,2593168;4513258,2593127').",
    ),
    output_path: str = typer.Argument(
        ..., help="Path to save the output sub-catchment raster."
    ),
):
    """
    Extracts sub-catchments based on flow direction and accumulation data.
    Optionally, split points can be used to divide the sub-basins.
    """
    start = time.time()
    lazy_import()

    # Load the raster grid
    grid: sGrid = sGrid.from_raster(fdir_path)

    # Load flow direction and accumulation rasters
    fdir = grid.read_raster(fdir_path)
    acc = grid.read_raster(acc_path)

    split_points_list = parse_split_points(split_points)

    # Extract sub-catchments
    sub_basins = grid.extract_subcatchment(
        fdir, acc > threshold, split_points=split_points_list
    )

    grid.to_raster(
        sub_basins,
        output_path,
        apply_output_mask=True,
        blockxsize=16,
        blockysize=16,
    )

    typer.echo(f"Sub-catchment raster saved to {output_path}")
    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


@app.command()
def get_topology(
    fdir_path: str = typer.Argument(..., help="Path to the flow direction raster map."),
    acc_path: str = typer.Argument(..., help="Path to the accumulation raster map."),
    output_path: str = typer.Argument(
        ..., help="Path to json representing the catchment topology"
    ),
    split_points: str = typer.Option(
        None,
        help="Comma-separated list of split point coordinates (e.g., '4513192,2593168;4513258,2593127').",
    ),
    threshold: float = typer.Option(
        1000, "-t", "--threshold", help="Network threshold for TCA"
    ),
):
    """
    Calculates the topology of the subbasins to the outlet.
    """
    start = time.time()
    lazy_import()

    grid: sGrid = sGrid.from_raster(acc_path)
    fdir = grid.read_raster(fdir_path)
    acc = grid.read_raster(acc_path)

    split_points_list = parse_split_points(split_points)
    profiles, connections = grid.extract_profiles(
        fdir, acc > threshold, split_points=split_points_list
    )

    with open(output_path, "w") as f:
        json.dump(connections, f)

    typer.echo(f"Topology writed as: {output_path}")

    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


@app.command()
def network_vectorize(
    fdir_path: str = typer.Argument(..., help="Path to the flow direction raster."),
    acc_path: str = typer.Argument(..., help="Path to the accumulation raster map."),
    output_path: str = typer.Argument(
        ..., help="Path to json representing ths catchment topology"
    ),
    threshold: float = typer.Option(
        1000, "-t", "--threshold", help="Network threshold for TCA"
    ),
    split_points: str = typer.Option(
        None,
        help="Comma-separated list of split point coordinates (e.g., '4513192,2593168;4513258,2593127').",
    ),
    dem_path: str = typer.Option(
        None,
        "-d",
        "--dem-path",
        help="If provided some elevation statistics are added as feature attributes.",
    ),
):
    """
    Vectorizes network as LineString by adding some geometric information. If provided a DEM, it adds
    elevation statistics as well.
    """
    start = time.time()
    lazy_import()

    grid: sGrid = sGrid.from_raster(acc_path)
    fdir = grid.read_raster(fdir_path)
    acc = grid.read_raster(acc_path)

    split_points_list = parse_split_points(split_points)

    branches = grid.extract_river_network(
        fdir, acc > threshold, split_points=split_points_list, dem_path=dem_path
    )

    with open(output_path, "w") as f:
        json.dump(branches, f)

    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


@app.command()
def subcatchment_vectorize(
    subcatchments_path: str = typer.Argument(
        ..., help="Path to the input flow direction raster."
    ),
    output_path: str = typer.Argument(
        ..., help="Path to shapefile where to save vector of sub-basins"
    ),
    dem_path: str = typer.Option(
        ..., "-d", "--dem-path", help="Path to the DTM raster from which to extract further statistics."
    ),
):
    """
    Vectorizes subbasins as polygons by adding some geometric information. If provided a DEM, it adds
    elevation statistics as well.
    """
    start = time.time()
    lazy_import()

    grid: sGrid = sGrid.from_raster(subcatchments_path)
    sub_basins = grid.read_raster(subcatchments_path)

    grid.to_sb_shapefile(sub_basins, output_path, dem_path=dem_path)

    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


@app.command()
def clip_mask(
        mask_path: str = typer.Argument(..., help="File path of the raster mask with which to clip the target map"),
        target_path: str = typer.Argument(..., help="File path to the target raster to clip"),
        output_path: str = typer.Argument(..., help="Suffix for clipped maps")
):
    start = time.time()
    lazy_import()

    grid: sGrid = sGrid.from_raster(mask_path)
    mask = grid.read_raster(mask_path)
    target = grid.read_raster(target_path)

    # apply mask
    grid.clip_to(mask)
    grid.to_raster(target, output_path)

    typer.echo(f"Writed clipped map to {output_path}")
    end = time.time() - start
    typer.echo(f"Elapsed time: {end:.3f} seconds")


if __name__ == "__main__":
    app()
