# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
pip install -e ".[dev]"        # Editable install with test deps
uv pip install -e ".[dev]"     # Or with uv
```

## Testing

```bash
pytest                                    # All tests
pytest tests/test_grid.py                 # Single file
pytest tests/test_grid.py::test_name      # Single test
pytest --cov=pysheds                      # With coverage
```

Test data lives in `/data/` (ASCII grids, GeoTIFFs). Fixtures in `conftest.py` provide pre-computed grids (`dem`, `fdir`, `grid`, `d` datasets container).

Known: `distance_to_outlet` test has segfault issues; `polygonize`/`rasterize` test is xfail.

## Linting

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics   # Syntax errors only
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics  # Full check
```

Max line length: 127. Max complexity: 10.

## Architecture

**pysheds** is a watershed delineation library. Core data flow: DEM raster → flow direction → catchment/accumulation/stream network.

Requires Python >= 3.11.

### Key modules

- **`sgrid.py`** — `sGrid` (exported as `Grid`): main API class. Reads rasters, runs hydrologic algorithms (flowdir, catchment, accumulation, distance, stream extraction, HAND, stream order). ~1500 lines.
- **`_sgrid.py`** — Numba `@njit` compiled kernels for D8, D-infinity, and MFD routing. Performance-critical inner loops with `parallel=True`.
- **`sview.py`** — `Raster` (ndarray subclass with CRS/spatial metadata), `ViewFinder` (affine transform, shape, CRS, nodata, mask), `View`.
- **`io.py`** — `read_ascii()`, `read_raster()` (via rasterio), `to_ascii()`, `to_raster()`.
- **`projection.py`** — pyproj transform/CRS helpers (requires pyproj >= 3.6).

### Design patterns

- **ViewFinder pattern**: spatial metadata (affine, CRS, nodata, mask) separated from array data.
- **Raster as ndarray subclass**: seamless numpy integration.
- **Numba JIT**: all heavy computation compiled; numba is a hard dependency (import fails without it).
- **Three routing schemes**: D8 (single steepest), D-infinity (two-cell partition), MFD (multiple proportional).

### Legacy modules (not primary API)

`grid.py`, `view.py`, `_sview.py`, `pview.py` — older implementations preserved in the codebase.

## CI

GitHub Actions on push/PR to master: Python 3.11–3.13 matrix, flake8 lint, pytest. Publish to PyPI on release via `python-publish.yml`.

## Dependencies

Core: `numba>=0.60`, `numpy>=1.26`, `scipy>=1.11`, `scikit-image>=0.21`, `pandas>=2.1`, `rasterio>=1.3`, `pyproj>=3.6`, `affine`, `geojson`, `looseversion`.
