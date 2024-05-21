from numba import njit, prange, from_dtype
from numba.typed import typedlist
from numba.types import Tuple, int64

import numpy as np
import math
from heapq import heappop, heappush, heapify
from functools import wraps


def pfwrapper(func):
    # Implemenation detail of priority-flood algorithm
    # Needed to define the types used in priority queue
    @wraps(func)
    def _wrapper(dem, mask, *args):
        # Tuple elements:
        # 0: dem data type (for elevation priority)
        # 1: int64 for insertion index (to maintain total ordering)
        # 2: int64 for row index
        # 3: int64 for col index
        tuple_type = Tuple([from_dtype(dem.dtype), int64, int64, int64])
        return func(dem, mask, tuple_type, *args)
    return _wrapper


@njit(boundscheck=True, cache=True)
def _first_true1d(arr, start=0, end=None, step=1, invert=False):
    if end is None:
        end = len(arr)

    if invert:
        for i in range(start, end, step):
            if not arr[i]:
                return i
        else:
            return -1
    else:
        for i in range(start, end, step):
            if arr[i]:
                return i
        else:
            return -1


@njit(parallel=True, cache=True)
def _top(mask):
    nc = mask.shape[1]
    rv = np.zeros(nc, dtype='int64')
    for i in prange(nc):
        rv[i] = _first_true1d(mask[:, i], invert=True)
    return rv


@njit(parallel=True, cache=True)
def _bottom(mask):
    nr, nc = mask.shape[0], mask.shape[1]
    rv = np.zeros(nc, dtype='int64')
    for i in prange(nc):
        rv[i] = _first_true1d(mask[:, i], start=nr - 1, end=-1, step=-1, invert=True)
    return rv


@njit(parallel=True, cache=True)
def _left(mask):
    nr = mask.shape[0]
    rv = np.zeros(nr, dtype='int64')
    for i in prange(nr):
        rv[i] = _first_true1d(mask[i, :], invert=True)
    return rv


@njit(parallel=True, cache=True)
def _right(mask):
    nr, nc = mask.shape[0], mask.shape[1]
    rv = np.zeros(nr, dtype='int64')
    for i in prange(nr):
        rv[i] = _first_true1d(mask[i, :], start=nc - 1, end=-1, step=-1, invert=True)
    return rv


@njit(cache=True)
def count(start=0, step=1):
    # Numba accelerated count() from itertools
    # count(10) --> 10 11 12 13 14 ...
    # count(2.5, 0.5) --> 2.5 3.0 3.5 ...
    n = start
    while True:
        yield n
        n += step


@njit(cache=True)
def heapify_border(dem, mask, open_cells, counter):
    y, x = dem.shape
    edge = _left(mask)[:-1]
    for row, col in zip(count(), edge):
        if col >= 0:
            open_cells.append((dem[row, col], next(counter), row, col))
    edge = _bottom(mask)[:-1]
    for row, col in zip(edge, count()):
        if row >= 0:
            open_cells.append((dem[row, col], next(counter), row, col))
    edge = np.flip(_right(mask))[:-1]
    for row, col in zip(count(y - 1, step=-1), edge):
        if col >= 0:
            open_cells.append((dem[row, col], next(counter), row, col))
    edge = np.flip(_top(mask))[:-1]
    for row, col in zip(edge, count(x - 1, step=-1)):
        if row >= 0:
            open_cells.append((dem[row, col], next(counter), row, col))
    heapify(open_cells)


@njit(cache=True)
def queue_empty(q, pos):
    return pos == len(q)


@pfwrapper
@njit(cache=True)
def fill_depressions(dem, dem_mask, tuple_type):
    open_cells = typedlist.List.empty_list(tuple_type)  # Priority queue
    pits = typedlist.List.empty_list(tuple_type)  # FIFO queue
    closed_cells = dem_mask.copy()

    counter = count()
    # Push the edges onto priority queue
    y, x = dem.shape
    heapify_border(dem, dem_mask, open_cells, counter)
    for _, _, i, j in open_cells:
        closed_cells[i, j] = True

    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])

    pits_pos = 0
    while open_cells or not queue_empty(pits, pits_pos):
        if not queue_empty(pits, pits_pos):
            elv, _, i, j = pits[pits_pos]
            pits_pos += 1
        else:
            elv, _, i, j = heappop(open_cells)

        for n in range(8):
            row = i + row_offsets[n]
            col = j + col_offsets[n]

            if row < 0 or row >= y or col < 0 or col >= x:
                continue

            if dem_mask[row, col] or closed_cells[row, col]:
                continue

            closed_cells[row, col] = True

            if dem[row, col] <= elv:
                dem[row, col] = elv
                pits.append((elv, 0, row, col))
            else:
                heappush(open_cells, (dem[row, col], next(counter), row, col))

        # pits book-keeping
        if queue_empty(pits, pits_pos) and len(pits) > 1024:
            # Queue is empty, lets clear it out
            pits.clear()
            pits_pos = 0

    return dem


@pfwrapper
@njit(cache=True)
def fill_depressions_epsilon(dem, dem_mask, tuple_type):
    open_cells = typedlist.List.empty_list(tuple_type)  # Priority queue
    pits = typedlist.List.empty_list(tuple_type)  # FIFO queue
    closed_cells = dem_mask.copy()

    counter = count()
    # Push the edges onto priority queue
    y, x = dem.shape
    heapify_border(dem, dem_mask, open_cells, counter)
    for _, _, i, j in open_cells:
        closed_cells[i, j] = True

    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])

    pits_pos = 0
    while open_cells or not queue_empty(pits, pits_pos):
        if not queue_empty(pits, pits_pos):
            if open_cells[0][0] == pits[pits_pos][0]:
                elv, _, i, j = heappop(open_cells)
                pit_top = None
            else:
                elv, _, i, j = pits[pits_pos]
                pits_pos += 1
                pit_top = elv
        else:
            elv, _, i, j = heappop(open_cells)
            pit_top = None

        for n in range(8):
            row = i + row_offsets[n]
            col = j + col_offsets[n]

            if row < 0 or row >= y or col < 0 or col >= x:
                continue

            if closed_cells[row, col]:
                continue

            closed_cells[row, col] =True

            # Only using numpy here because math.nextafter not supported
            if dem_mask[row, col]:
                pits.append((dem[row, col], 0, row, col))
                continue

            next_after = np.nextafter(dem[i, j], math.inf)
            if dem[row, col] <= next_after:
                if pit_top is None:
                    dem[row, col] = next_after
                    pits.append((dem[row, col], 0, row, col))
                elif pit_top < dem[row, col] and next_after >= dem[row, col]:
                    raise ValueError("DEM cell too high!")
            else:
                heappush(open_cells, (dem[row, col], next(counter), row, col))

        # pits book-keeping
        if queue_empty(pits, pits_pos) and len(pits) > 1024:
            # Queue is empty, lets clear it out
            pits.clear()
            pits_pos = 0

    return dem


@pfwrapper
@njit(cache=True)
def flow_dirs(dem, dem_mask, tuple_type, flow_values):
    # Ensure that 0 is not in flow directions
    if np.any(flow_values == 0):
        raise ValueError("Invalid flow direction: 0")
    open_cells = typedlist.List.empty_list(tuple_type)  # Priority queue
    closed_cells = dem_mask.copy()
    flows = np.zeros_like(dem, dtype=flow_values.dtype)

    counter = count()
    # Push the edges onto priority queue
    y, x = dem.shape
    heapify_border(dem, dem_mask, open_cells, counter)
    for _, _, i, j in open_cells:
        closed_cells[i, j] = True

    row_offsets = np.array([-1, 0, 1, 0, -1, -1, 1, 1])
    col_offsets = np.array([0, 1, 0, -1, -1, 1, 1, -1])
    flow_to = flow_values.take([4, 6, 0, 2, 3, 5, 7, 1])

    while open_cells:
        _, _, i, j = heappop(open_cells)

        for n in range(8):
            row = i + row_offsets[n]
            col = j + col_offsets[n]

            if row < 0 or row >= y or col < 0 or col >= x:
                continue

            if closed_cells[row, col]:
                continue

            if not dem_mask[row, col]:
                flows[row, col] = flow_to[n]

            heappush(open_cells, (dem[row, col], next(counter), row, col))
            closed_cells[row, col] = True

    return flows


@pfwrapper
@njit(cache=True)
def basins(dem, dem_mask, tuple_type):
    open_cells = typedlist.List.empty_list(tuple_type)  # Priority queue
    pits = typedlist.List.empty_list(tuple_type)  # FIFO queue
    labels = np.zeros_like(dem, dtype='int')
    label = 1
    queued = -1

    counter = count()
    # Push the edges onto priority queue
    y, x = dem.shape
    heapify_border(dem, dem_mask, open_cells, counter)
    for _, _, i, j in open_cells:
        labels[i, j] = queued

    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])

    pits_pos = 0
    while open_cells or not queue_empty(pits, pits_pos):
        if not queue_empty(pits, pits_pos):
            elv, _, i, j = pits[pits_pos]
            pits_pos += 1
        else:
            elv, _, i, j = heappop(open_cells)

        if labels[i, j] == queued and not dem_mask[i, j]:
            labels[i, j] = label
            label += 1

        for n in range(8):
            row = i + row_offsets[n]
            col = j + col_offsets[n]

            if row < 0 or row >= y or col < 0 or col >= x:
                continue

            if dem_mask[row, col] or labels[row, col] != 0:
                continue

            labels[row, col] = labels[i, j]

            if dem[row, col] <= elv:
                dem[row, col] = elv
                pits.append((elv, 0, row, col))
            else:
                heappush(open_cells, (dem[row, col], next(counter), row, col))

        # pits book-keeping
        if queue_empty(pits, pits_pos) and len(pits) > 1000:
            # Queue is empty, lets clear it out
            pits.clear()
            pits_pos = 0

    return labels
