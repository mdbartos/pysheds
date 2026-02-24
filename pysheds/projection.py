import pyproj
from typing import Any, Optional


def transform(
        p1: Any,
        p2: Any,
        x: Any,
        y: Any,
        z: Optional[Any] = None,
        tt: Optional[Any] = None,
        radians: bool = False,
        errcheck: bool = False,
        always_xy: bool = True,
):
    return pyproj.Transformer.from_proj(p1, p2, always_xy=always_xy).transform(
        xx=x, yy=y, zz=z, tt=tt, radians=radians, errcheck=errcheck
    )


def init():
    return pyproj.Proj('epsg:4326').crs


def to_proj(source: pyproj.Proj):
    if isinstance(source, pyproj.Proj):
        source = source.crs
    return pyproj.Proj(source, preserve_units=True).crs


def is_geographic(crs: pyproj.CRS):
    return crs.is_geographic
