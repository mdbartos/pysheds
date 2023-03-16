"""
This module handles all pyproj calls of sgrid / sview. (pgrid and pview can be changed either)

Most of the complexity comes from handling different versions. This file can be dramatically simplified
if pyproj>2.6.1 becomes a prerequisite for the next version of pysheds
"""
import pyproj
from packaging.version import Version
from typing import Any, Optional

_pyproj_version = Version(pyproj.__version__)
_OLD_PYPROJ = _pyproj_version < Version('2.2')


def __transform_before_2_2(
        p1: Any,
        p2: Any,
        x: Any,
        y: Any,
        z: Optional[Any] = None,
        tt: Optional[Any] = None,
        radians: bool = False,
        errcheck: bool = False,
        _: bool = True,
):
    """
    A pyproj version compatibility wrapper, fixes pysheds issue #210

    - For very old pyproj (<2.2), calls pyproj.transform without always_xy
    - for old pyproj (<2.6.1), calls pyproj.transform with always_xy=True
    - for recent pyproj, the deprecated call of transform is rebuild

    """

    return pyproj.transform(
        p1=p1, p2=p2, x=x, y=y, z=z, tt=tt, radians=radians, errcheck=errcheck
    )

def __transform_before_2_6(
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
    """
    A pyproj version compatibility wrapper, fixes pysheds issue #210

    - For very old pyproj (<2.2), calls pyproj.transform without always_xy
    - for old pyproj (<2.6.1), calls pyproj.transform with always_xy=True
    - for recent pyproj, the deprecated call of transform is rebuild

    """

    return pyproj.transform(
        p1=p1, p2=p2, x=x, y=y, z=z, tt=tt,
        radians=radians, errcheck=errcheck, always_xy=always_xy
    )

def __transform_after_2_6(
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
    """
    A pyproj version compatibility wrapper, fixes pysheds issue #210

    - For very old pyproj (<2.2), calls pyproj.transform without always_xy
    - for old pyproj (<2.6.1), calls pyproj.transform with always_xy=True
    - for recent pyproj, the deprecated call of transform is rebuild

    """

    # Code copied from pyproj.transform.transform
    # see: https://github.com/pyproj4/pyproj/blob/4faef1724b044ee145832591fcd7ed2639477670/pyproj/transformer.py#L1332-L1334
    return pyproj.Transformer.from_proj(p1, p2, always_xy=always_xy).transform(
        xx=x, yy=y, zz=z, tt=tt, radians=radians, errcheck=errcheck
    )


if _pyproj_version <= Version('2.2'):
    transform = __transform_before_2_6
elif _pyproj_version < Version('2.6.1'):
    transform = __transform_before_2_6
else:
    transform = __transform_after_2_6

if _OLD_PYPROJ:
    def init():
        """
        Init returns a newl
        """
        return pyproj.Proj('+init=epsg:4326')
    def to_proj(source: pyproj.Proj):
        return pyproj.Proj(source, preserve_units=True)
else:
    def init():
        return pyproj.Proj('epsg:4326').crs
    def to_proj(source: pyproj.Proj):
        if isinstance(source, pyproj.Proj):
            source = source.crs
        return pyproj.Proj(source, preserve_units=True).crs


def is_geographic(crs: pyproj.CRS):
    try:
        return crs.is_geographic
    except TypeError:
        if hasattr(crs, 'is_latlong'):
            return crs.is_latlong
        else:
            raise
