from heapq import heappop, heappush, heapify
import math
import numpy as np
from functools import wraps
from numba import njit, prange, from_dtype
from numba.types import (
    float64,
    int64,
    uint32,
    uint16,
    uint8,
    boolean,
    UniTuple,
    Tuple,
    List,
    DictType,
    void,
)
from numba.typed import typedlist

# Functions for 'flowdir'


@njit(
    int64[:, :](
        float64[:, :],
        float64,
        float64,
        UniTuple(int64, 8),
        boolean[:, :],
        int64,
        int64,
        int64,
    ),
    parallel=True,
    cache=True,
)
def _d8_flowdir_numba(dem, dx, dy, dirmap, nodata_cells, nodata_out, flat=-1, pit=-2):
    fdir = np.full(dem.shape, nodata_out, dtype=np.int64)
    m, n = dem.shape
    dd = math.sqrt(dx**2 + dy**2)
    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    distances = np.array([dy, dd, dx, dd, dy, dd, dx, dd])
    for i in prange(m):
        for j in prange(n):
            if not nodata_cells[i, j]:
                elev = dem[i, j]
                max_slope = -np.inf
                for k in range(8):
                    row = i + row_offsets[k]
                    col = j + col_offsets[k]
                    if row < 0 or row >= m or col < 0 or col >= n:
                        # out of bounds, skip
                        continue
                    elif nodata_cells[row, col]:
                        # this neighbor is nodata, skip
                        continue
                    distance = distances[k]
                    slope = (elev - dem[row, col]) / distance
                    if slope > max_slope:
                        fdir[i, j] = dirmap[k]
                        max_slope = slope
                if max_slope == 0:
                    fdir[i, j] = flat
                elif max_slope < 0:
                    fdir[i, j] = pit
    return fdir


@njit(
    int64[:, :](
        float64[:, :],
        float64[:, :],
        float64[:, :],
        UniTuple(int64, 8),
        boolean[:, :],
        int64,
        int64,
        int64,
    ),
    parallel=True,
    cache=True,
)
def _d8_flowdir_irregular_numba(
    dem, x_arr, y_arr, dirmap, nodata_cells, nodata_out, flat=-1, pit=-2
):
    fdir = np.full(dem.shape, nodata_out, dtype=np.int64)
    m, n = dem.shape
    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    for i in prange(m):
        for j in prange(n):
            if not nodata_cells[i, j]:
                elev = dem[i, j]
                x_center = x_arr[i, j]
                y_center = y_arr[i, j]
                max_slope = -np.inf
                for k in range(8):
                    row = i + row_offsets[k]
                    col = j + col_offsets[k]
                    if row < 0 or row >= m or col < 0 or col >= n:
                        # out of bounds, skip
                        continue
                    elif nodata_cells[row, col]:
                        # this neighbor is nodata, skip
                        continue
                    dh = elev - dem[row, col]
                    dx = abs(x_center - x_arr[row, col])
                    dy = abs(y_center - y_arr[row, col])
                    distance = math.sqrt(dx**2 + dy**2)
                    slope = dh / distance
                    if slope > max_slope:
                        fdir[i, j] = dirmap[k]
                        max_slope = slope
                if max_slope == 0:
                    fdir[i, j] = flat
                elif max_slope < 0:
                    fdir[i, j] = pit
    return fdir


@njit(UniTuple(float64, 2)(float64, float64, float64, float64, float64), cache=True)
def _facet_flow(e0, e1, e2, d1=1.0, d2=1.0):
    s1 = (e0 - e1) / d1
    s2 = (e1 - e2) / d2
    r = math.atan2(s2, s1)
    s = math.hypot(s1, s2)
    diag_angle = math.atan2(d2, d1)
    diag_distance = math.hypot(d1, d2)
    b0 = r < 0
    b1 = r > diag_angle
    if b0:
        r = 0
        s = s1
    if b1:
        r = diag_angle
        s = (e0 - e2) / diag_distance
    return r, s


@njit(
    float64[:, :](float64[:, :], float64, float64, float64, float64, float64),
    parallel=True,
    cache=True,
)
def _dinf_flowdir_numba(dem, x_dist, y_dist, nodata, flat=-1.0, pit=-2.0):
    m, n = dem.shape
    e1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    e2s = np.array([1, 1, 3, 3, 5, 5, 7, 7])
    d1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    d2s = np.array([2, 0, 4, 2, 6, 4, 0, 6])
    ac = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    af = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    angle = np.full(dem.shape, nodata, dtype=np.float64)
    diag_dist = math.sqrt(x_dist**2 + y_dist**2)
    cell_dists = np.array(
        [x_dist, diag_dist, y_dist, diag_dist, x_dist, diag_dist, y_dist, diag_dist]
    )
    row_offsets = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    col_offsets = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
            e0 = dem[i, j]
            s_max = -np.inf
            k_max = 8
            r_max = 0.0
            for k in prange(8):
                edge_1 = e1s[k]
                edge_2 = e2s[k]
                row_offset_1 = row_offsets[edge_1]
                row_offset_2 = row_offsets[edge_2]
                col_offset_1 = col_offsets[edge_1]
                col_offset_2 = col_offsets[edge_2]
                e1 = dem[i + row_offset_1, j + col_offset_1]
                e2 = dem[i + row_offset_2, j + col_offset_2]
                distance_1 = d1s[k]
                distance_2 = d2s[k]
                d1 = cell_dists[distance_1]
                d2 = cell_dists[distance_2]
                r, s = _facet_flow(e0, e1, e2, d1, d2)
                if s > s_max:
                    s_max = s
                    k_max = k
                    r_max = r
            if s_max < 0:
                angle[i, j] = pit
            elif s_max == 0:
                angle[i, j] = flat
            else:
                flow_angle = (af[k_max] * r_max) + (ac[k_max] * np.pi / 2)
                flow_angle = flow_angle % (2 * np.pi)
                angle[i, j] = flow_angle
    return angle


@njit(
    float64[:, :](float64[:, :], float64[:, :], float64[:, :], float64, float64, float64),
    parallel=True,
    cache=True,
)
def _dinf_flowdir_irregular_numba(dem, x_arr, y_arr, nodata, flat=-1.0, pit=-2.0):
    m, n = dem.shape
    e1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    e2s = np.array([1, 1, 3, 3, 5, 5, 7, 7])
    d1s = np.array([0, 2, 2, 4, 4, 6, 6, 0])
    d2s = np.array([2, 0, 4, 2, 6, 4, 0, 6])
    ac = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    af = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    angle = np.full(dem.shape, nodata, dtype=np.float64)
    row_offsets = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    col_offsets = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
            e0 = dem[i, j]
            x0 = x_arr[i, j]
            y0 = y_arr[i, j]
            s_max = -np.inf
            k_max = 8
            r_max = 0.0
            for k in prange(8):
                edge_1 = e1s[k]
                edge_2 = e2s[k]
                row_offset_1 = row_offsets[edge_1]
                row_offset_2 = row_offsets[edge_2]
                col_offset_1 = col_offsets[edge_1]
                col_offset_2 = col_offsets[edge_2]
                e1 = dem[i + row_offset_1, j + col_offset_1]
                e2 = dem[i + row_offset_2, j + col_offset_2]
                x1 = x_arr[i + row_offset_1, j + col_offset_1]
                x2 = x_arr[i + row_offset_2, j + col_offset_2]
                y1 = y_arr[i + row_offset_1, j + col_offset_1]
                y2 = y_arr[i + row_offset_2, j + col_offset_2]
                d1 = math.sqrt(x1**2 + y1**2)
                d2 = math.sqrt(x2**2 + y2**2)
                r, s = _facet_flow(e0, e1, e2, d1, d2)
                if s > s_max:
                    s_max = s
                    k_max = k
                    r_max = r
            if s_max < 0:
                angle[i, j] = pit
            elif s_max == 0:
                angle[i, j] = flat
            else:
                flow_angle = (af[k_max] * r_max) + (ac[k_max] * np.pi / 2)
                flow_angle = flow_angle % (2 * np.pi)
                angle[i, j] = flow_angle
    return angle


@njit(
    Tuple((int64[:, :], int64[:, :], float64[:, :], float64[:, :]))(
        float64[:, :], UniTuple(int64, 8), boolean[:, :]
    ),
    parallel=True,
    cache=True,
)
def _angle_to_d8_numba(angles, dirmap, nodata_cells):
    n = angles.size
    min_angle = 0.0
    max_angle = 2 * np.pi
    mod = np.pi / 4
    c0_order = np.array([2, 1, 0, 7, 6, 5, 4, 3])
    c1_order = np.array([1, 0, 7, 6, 5, 4, 3, 2])
    c0 = np.zeros(8, dtype=np.uint8)
    c1 = np.zeros(8, dtype=np.uint8)
    # Need to watch typing of fdir_0 and fdir_1
    fdirs_0 = np.zeros(angles.shape, dtype=np.int64)
    fdirs_1 = np.zeros(angles.shape, dtype=np.int64)
    props_0 = np.zeros(angles.shape, dtype=np.float64)
    props_1 = np.zeros(angles.shape, dtype=np.float64)
    for i in range(8):
        c0[i] = dirmap[c0_order[i]]
        c1[i] = dirmap[c1_order[i]]
    for i in prange(n):
        angle = angles.flat[i]
        nodata = nodata_cells.flat[i]
        if np.isnan(angle) or nodata:
            zfloor = 8
            prop_0 = 0
            prop_1 = 0
            fdir_0 = 0
            fdir_1 = 0
        elif (angle < min_angle) or (angle > max_angle):
            zfloor = 8
            prop_0 = 0
            prop_1 = 0
            fdir_0 = 0
            fdir_1 = 0
        else:
            zmod = angle % mod
            zfloor = int(angle // mod)
            prop_1 = zmod / mod
            prop_0 = 1 - prop_1
            fdir_0 = c0[zfloor]
            fdir_1 = c1[zfloor]
        # Handle case where flow proportion is zero in either direction
        if prop_0 == 0:
            fdir_0 = fdir_1
            prop_0 = 0.5
            prop_1 = 0.5
        elif prop_1 == 0:
            fdir_1 = fdir_0
            prop_0 = 0.5
            prop_1 = 0.5
        fdirs_0.flat[i] = fdir_0
        fdirs_1.flat[i] = fdir_1
        props_0.flat[i] = prop_0
        props_1.flat[i] = prop_1
    return fdirs_0, fdirs_1, props_0, props_1


@njit(
    float64[:, :, :](float64[:, :], float64, float64, boolean[:, :], float64, int64),
    parallel=True,
    cache=True,
)
def _mfd_flowdir_numba(dem, dx, dy, nodata_cells, nodata_out, p=1):
    m, n = dem.shape
    fdir = np.full((8, m, n), nodata_out, dtype=np.float64)
    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dd = math.sqrt(dx**2 + dy**2)
    distances = np.array([dy, dd, dx, dd, dy, dd, dx, dd])
    for i in prange(m):
        for j in prange(n):
            if not nodata_cells[i, j]:
                elev = dem[i, j]
                den = 0.0
                for k in range(8):
                    row = i + row_offsets[k]
                    col = j + col_offsets[k]
                    if row < 0 or row >= m or col < 0 or col >= n:
                        # out of bounds, skip
                        continue
                    elif nodata_cells[row, col]:
                        # this neighbor is nodata, skip
                        continue
                    distance = distances[k]
                    num = (elev - dem[row, col]) ** p / distance
                    if num > 0:
                        fdir[k, i, j] = num
                        den += num
                if den > 0:
                    fdir[:, i, j] /= den
    return fdir


# Functions for 'catchment'


@njit(void(int64, boolean[:, :], int64[:, :], int64[:], int64[:]), cache=True)
def _d8_catchment_recursion(ix, catch, fdir, offsets, r_dirmap):
    visited = catch.flat[ix]
    if not visited:
        catch.flat[ix] = True
        neighbors = offsets + ix
        for k in range(8):
            neighbor = neighbors[k]
            points_to = fdir.flat[neighbor] == r_dirmap[k]
            if points_to:
                _d8_catchment_recursion(neighbor, catch, fdir, offsets, r_dirmap)


@njit(boolean[:, :](int64[:, :], UniTuple(int64, 2), UniTuple(int64, 8)), cache=True)
def _d8_catchment_recur_numba(fdir, pour_point, dirmap):
    catch = np.zeros(fdir.shape, dtype=np.bool_)
    offset = fdir.shape[1]
    i, j = pour_point
    ix = (i * offset) + j
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    _d8_catchment_recursion(ix, catch, fdir, offsets, r_dirmap)
    return catch


@njit(boolean[:, :](int64[:, :], UniTuple(int64, 2), UniTuple(int64, 8)))
def _d8_catchment_iter_numba(fdir, pour_point, dirmap):
    catch = np.zeros(fdir.shape, dtype=np.bool_)
    offset = fdir.shape[1]
    i, j = pour_point
    ix = (i * offset) + j
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    queue = [ix]
    while queue:
        parent = queue.pop()
        catch.flat[parent] = True
        neighbors = offsets + parent
        for k in range(8):
            neighbor = neighbors[k]
            visited = catch.flat[neighbor]
            if visited:
                continue
            else:
                points_to = fdir.flat[neighbor] == r_dirmap[k]
                if points_to:
                    queue.append(neighbor)
    return catch


@njit(void(int64, boolean[:, :], int64[:, :], int64[:, :], int64[:], int64[:]), cache=True)
def _dinf_catchment_recursion(ix, catch, fdir_0, fdir_1, offsets, r_dirmap):
    visited = catch.flat[ix]
    if not visited:
        catch.flat[ix] = True
        neighbors = offsets + ix
        for k in range(8):
            neighbor = neighbors[k]
            points_to_0 = fdir_0.flat[neighbor] == r_dirmap[k]
            points_to_1 = fdir_1.flat[neighbor] == r_dirmap[k]
            points_to = points_to_0 or points_to_1
            if points_to:
                _dinf_catchment_recursion(neighbor, catch, fdir_0, fdir_1, offsets, r_dirmap)


@njit(
    boolean[:, :](int64[:, :], int64[:, :], UniTuple(int64, 2), UniTuple(int64, 8)),
    cache=True,
)
def _dinf_catchment_recur_numba(fdir_0, fdir_1, pour_point, dirmap):
    catch = np.zeros(fdir_0.shape, dtype=np.bool_)
    dirmap = np.array(dirmap)
    offset = fdir_0.shape[1]
    i, j = pour_point
    ix = (i * offset) + j
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    _dinf_catchment_recursion(ix, catch, fdir_0, fdir_1, offsets, r_dirmap)
    return catch


@njit(
    boolean[:, :](int64[:, :], int64[:, :], UniTuple(int64, 2), UniTuple(int64, 8)),
    cache=True,
)
def _dinf_catchment_iter_numba(fdir_0, fdir_1, pour_point, dirmap):
    catch = np.zeros(fdir_0.shape, dtype=np.bool_)
    dirmap = np.array(dirmap)
    offset = fdir_0.shape[1]
    i, j = pour_point
    ix = (i * offset) + j
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    queue = [ix]
    while queue:
        parent = queue.pop()
        catch.flat[parent] = True
        neighbors = offsets + parent
        for k in range(8):
            neighbor = neighbors[k]
            visited = catch.flat[neighbor]
            if visited:
                continue
            else:
                points_to_0 = fdir_0.flat[neighbor] == r_dirmap[k]
                points_to_1 = fdir_1.flat[neighbor] == r_dirmap[k]
                points_to = points_to_0 or points_to_1
                if points_to:
                    queue.append(neighbor)
    return catch


@njit(boolean[:, :](float64[:, :, :], UniTuple(int64, 2)), cache=True)
def _mfd_catchment_iter_numba(fdir, pour_point):
    _, m, n = fdir.shape
    mn = m * n
    catch = np.zeros((m, n), dtype=np.bool_)
    i, j = pour_point
    ix = (i * n) + j
    offsets = np.array([-n, 1 - n, 1, 1 + n, n, -1 + n, -1, -1 - n])
    r_dirmap = np.array([4, 5, 6, 7, 0, 1, 2, 3])
    queue = [ix]
    while queue:
        parent = queue.pop()
        catch.flat[parent] = True
        neighbors = offsets + parent
        for k in range(8):
            neighbor = neighbors[k]
            neighbor_dir = r_dirmap[k]
            visited = catch.flat[neighbor]
            if visited:
                continue
            else:
                kix = neighbor + (neighbor_dir * mn)
                points_to = fdir.flat[kix] > 0.0
                if points_to:
                    queue.append(neighbor)
    return catch


# Functions for 'accumulation'


@njit(void(int64, int64, float64[:, :], int64[:, :], uint8[:]), cache=True)
def _d8_accumulation_recursion(startnode, endnode, acc, fdir, indegree):
    acc.flat[endnode] += acc.flat[startnode]
    indegree[endnode] -= 1
    if indegree[endnode] == 0:
        new_startnode = endnode
        new_endnode = fdir.flat[endnode]
        _d8_accumulation_recursion(new_startnode, new_endnode, acc, fdir, indegree)


@njit(float64[:, :](float64[:, :], int64[:, :], uint8[:], int64[:]), cache=True)
def _d8_accumulation_recur_numba(acc, fdir, indegree, startnodes):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes[k]
        endnode = fdir.flat[startnode]
        _d8_accumulation_recursion(startnode, endnode, acc, fdir, indegree)
    return acc


@njit(float64[:, :](float64[:, :], int64[:, :], uint8[:], int64[:]), cache=True)
def _d8_accumulation_iter_numba(acc, fdir, indegree, startnodes):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes[k]
        endnode = fdir.flat[startnode]
        while indegree[startnode] == 0:
            acc.flat[endnode] += acc.flat[startnode]
            indegree[endnode] -= 1
            startnode = endnode
            endnode = fdir.flat[startnode]
    return acc


@njit(void(int64, int64, float64[:, :], int64[:, :], uint8[:], float64[:, :]), cache=True)
def _d8_accumulation_eff_recursion(startnode, endnode, acc, fdir, indegree, eff):
    acc.flat[endnode] += acc.flat[startnode] * eff.flat[startnode]
    indegree[endnode] -= 1
    if indegree[endnode] == 0:
        new_startnode = endnode
        new_endnode = fdir.flat[endnode]
        _d8_accumulation_eff_recursion(new_startnode, new_endnode, acc, fdir, indegree, eff)


@njit(
    float64[:, :](float64[:, :], int64[:, :], uint8[:], int64[:], float64[:, :]),
    cache=True,
)
def _d8_accumulation_eff_recur_numba(acc, fdir, indegree, startnodes, eff):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes[k]
        endnode = fdir.flat[startnode]
        _d8_accumulation_eff_recursion(startnode, endnode, acc, fdir, indegree, eff)
    return acc


@njit(
    float64[:, :](float64[:, :], int64[:, :], uint8[:], int64[:], float64[:, :]),
    cache=True,
)
def _d8_accumulation_eff_iter_numba(acc, fdir, indegree, startnodes, eff):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes[k]
        endnode = fdir.flat[startnode]
        while indegree[startnode] == 0:
            acc.flat[endnode] += acc.flat[startnode] * eff.flat[startnode]
            indegree[endnode] -= 1
            startnode = endnode
            endnode = fdir.flat[startnode]
    return acc


@njit(
    void(
        int64,
        int64,
        float64[:, :],
        int64[:, :],
        int64[:, :],
        uint8[:],
        float64,
        boolean[:, :],
        float64[:, :],
        float64[:, :],
    ),
    cache=True,
)
def _dinf_accumulation_recursion(
    startnode, endnode, acc, fdir_0, fdir_1, indegree, prop, visited, props_0, props_1
):
    acc.flat[endnode] += prop * acc.flat[startnode]
    indegree.flat[endnode] -= 1
    visited.flat[startnode] = True
    if indegree.flat[endnode] == 0:
        new_startnode = endnode
        new_endnode_0 = fdir_0.flat[new_startnode]
        new_endnode_1 = fdir_1.flat[new_startnode]
        prop_0 = props_0.flat[new_startnode]
        prop_1 = props_1.flat[new_startnode]
        _dinf_accumulation_recursion(
            new_startnode,
            new_endnode_0,
            acc,
            fdir_0,
            fdir_1,
            indegree,
            prop_0,
            visited,
            props_0,
            props_1,
        )
        _dinf_accumulation_recursion(
            new_startnode,
            new_endnode_1,
            acc,
            fdir_0,
            fdir_1,
            indegree,
            prop_1,
            visited,
            props_0,
            props_1,
        )


@njit(
    float64[:, :](
        float64[:, :],
        int64[:, :],
        int64[:, :],
        uint8[:],
        int64[:],
        float64[:, :],
        float64[:, :],
    ),
    cache=True,
)
def _dinf_accumulation_recur_numba(acc, fdir_0, fdir_1, indegree, startnodes, props_0, props_1):
    n = startnodes.size
    visited = np.zeros(acc.shape, dtype=np.bool_)
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode_0 = fdir_0.flat[startnode]
        endnode_1 = fdir_1.flat[startnode]
        prop_0 = props_0.flat[startnode]
        prop_1 = props_1.flat[startnode]
        _dinf_accumulation_recursion(
            startnode,
            endnode_0,
            acc,
            fdir_0,
            fdir_1,
            indegree,
            prop_0,
            visited,
            props_0,
            props_1,
        )
        _dinf_accumulation_recursion(
            startnode,
            endnode_1,
            acc,
            fdir_0,
            fdir_1,
            indegree,
            prop_1,
            visited,
            props_0,
            props_1,
        )
        # TODO: Needed?
        visited.flat[startnode] = True
    return acc


@njit(
    float64[:, :](
        float64[:, :],
        int64[:, :],
        int64[:, :],
        uint8[:],
        int64[:],
        float64[:, :],
        float64[:, :],
    ),
    cache=True,
)
def _dinf_accumulation_iter_numba(acc, fdir_0, fdir_1, indegree, startnodes, props_0, props_1):
    n = startnodes.size
    queue = [0]
    _ = queue.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        queue.append(startnode)
        while queue:
            startnode = queue.pop()
            endnode_0 = fdir_0.flat[startnode]
            endnode_1 = fdir_1.flat[startnode]
            prop_0 = props_0.flat[startnode]
            prop_1 = props_1.flat[startnode]
            acc.flat[endnode_0] += prop_0 * acc.flat[startnode]
            acc.flat[endnode_1] += prop_1 * acc.flat[startnode]
            indegree.flat[endnode_0] -= 1
            indegree.flat[endnode_1] -= 1
            if indegree.flat[endnode_0] == 0:
                queue.append(endnode_0)
            if indegree.flat[endnode_1] == 0:
                # Account for cases where both fdirs point in same direction
                if endnode_0 != endnode_1:
                    queue.append(endnode_1)
    return acc


@njit(
    void(
        int64,
        int64,
        float64[:, :],
        int64[:, :],
        int64[:, :],
        uint8[:],
        float64,
        boolean[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
    ),
    cache=True,
)
def _dinf_accumulation_eff_recursion(
    startnode,
    endnode,
    acc,
    fdir_0,
    fdir_1,
    indegree,
    prop,
    visited,
    props_0,
    props_1,
    eff,
):
    acc.flat[endnode] += prop * acc.flat[startnode] * eff.flat[startnode]
    indegree.flat[endnode] -= 1
    visited.flat[startnode] = True
    if indegree.flat[endnode] == 0:
        new_startnode = endnode
        new_endnode_0 = fdir_0.flat[new_startnode]
        new_endnode_1 = fdir_1.flat[new_startnode]
        prop_0 = props_0.flat[new_startnode]
        prop_1 = props_1.flat[new_startnode]
        _dinf_accumulation_eff_recursion(
            new_startnode,
            new_endnode_0,
            acc,
            fdir_0,
            fdir_1,
            indegree,
            prop_0,
            visited,
            props_0,
            props_1,
            eff,
        )
        _dinf_accumulation_eff_recursion(
            new_startnode,
            new_endnode_1,
            acc,
            fdir_0,
            fdir_1,
            indegree,
            prop_1,
            visited,
            props_0,
            props_1,
            eff,
        )


@njit(
    float64[:, :](
        float64[:, :],
        int64[:, :],
        int64[:, :],
        uint8[:],
        int64[:],
        float64[:, :],
        float64[:, :],
        float64[:, :],
    ),
    cache=True,
)
def _dinf_accumulation_eff_numba(acc, fdir_0, fdir_1, indegree, startnodes, props_0, props_1, eff):
    n = startnodes.size
    visited = np.zeros(acc.shape, dtype=np.bool_)
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode_0 = fdir_0.flat[startnode]
        endnode_1 = fdir_1.flat[startnode]
        prop_0 = props_0.flat[startnode]
        prop_1 = props_1.flat[startnode]
        _dinf_accumulation_eff_recursion(
            startnode,
            endnode_0,
            acc,
            fdir_0,
            fdir_1,
            indegree,
            prop_0,
            visited,
            props_0,
            props_1,
            eff,
        )
        _dinf_accumulation_eff_recursion(
            startnode,
            endnode_1,
            acc,
            fdir_0,
            fdir_1,
            indegree,
            prop_1,
            visited,
            props_0,
            props_1,
            eff,
        )
        # TODO: Needed?
        visited.flat[startnode] = True
    return acc


@njit(
    float64[:, :](
        float64[:, :],
        int64[:, :],
        int64[:, :],
        uint8[:],
        int64[:],
        float64[:, :],
        float64[:, :],
        float64[:, :],
    ),
    cache=True,
)
def _dinf_accumulation_eff_iter_numba(
    acc, fdir_0, fdir_1, indegree, startnodes, props_0, props_1, eff
):
    n = startnodes.size
    queue = [0]
    _ = queue.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        queue.append(startnode)
        while queue:
            startnode = queue.pop()
            endnode_0 = fdir_0.flat[startnode]
            endnode_1 = fdir_1.flat[startnode]
            prop_0 = props_0.flat[startnode]
            prop_1 = props_1.flat[startnode]
            transfer = acc.flat[startnode] * eff.flat[startnode]
            acc.flat[endnode_0] += prop_0 * transfer
            acc.flat[endnode_1] += prop_1 * transfer
            indegree.flat[endnode_0] -= 1
            indegree.flat[endnode_1] -= 1
            if indegree.flat[endnode_0] == 0:
                queue.append(endnode_0)
            if indegree.flat[endnode_1] == 0:
                # Account for cases where both fdirs point in same direction
                if endnode_0 != endnode_1:
                    queue.append(endnode_1)
    return acc


@njit(uint8[:](int64[:, :, :]), parallel=True)
def _mfd_bincount(fdir):
    p, m, n = fdir.shape
    mn = m * n
    out = np.zeros(mn, dtype=np.uint8)
    for i in range(p):
        fdir_i = fdir[i]
        for j in prange(mn):
            endnode = fdir_i.flat[j]
            if endnode != j:
                out[endnode] += 1
    return out


@njit(
    float64[:, :](float64[:, :], int64[:, :, :], float64[:, :, :], uint8[:], int64[:]),
    cache=True,
)
def _mfd_accumulation_iter_numba(acc, fdir, props, indegree, startnodes):
    n = startnodes.size
    queue = [0]
    _ = queue.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        queue.append(startnode)
        while queue:
            startnode = queue.pop()
            for i in range(8):
                fdir_i = fdir[i]
                props_i = props[i]
                endnode = fdir_i.flat[startnode]
                if endnode == startnode:
                    continue
                else:
                    prop = props_i.flat[startnode]
                    acc.flat[endnode] += prop * acc.flat[startnode]
                    indegree.flat[endnode] -= 1
                    if indegree.flat[endnode] == 0:
                        queue.append(endnode)
    return acc


@njit(
    float64[:, :](
        float64[:, :],
        int64[:, :, :],
        float64[:, :, :],
        uint8[:],
        int64[:],
        float64[:, :],
    ),
    cache=True,
)
def _mfd_accumulation_eff_iter_numba(acc, fdir, props, indegree, startnodes, eff):
    n = startnodes.size
    queue = [0]
    _ = queue.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        queue.append(startnode)
        while queue:
            startnode = queue.pop()
            for i in range(8):
                fdir_i = fdir[i]
                props_i = props[i]
                endnode = fdir_i.flat[startnode]
                if endnode == startnode:
                    continue
                else:
                    prop = props_i.flat[startnode]
                    acc.flat[endnode] += prop * acc.flat[startnode] * eff.flat[startnode]
                    indegree.flat[endnode] -= 1
                    if indegree.flat[endnode] == 0:
                        queue.append(endnode)
    return acc


# Functions for 'flow_distance'


@njit(
    void(
        int64,
        int64[:, :],
        boolean[:, :],
        float64[:, :],
        float64[:, :],
        int64[:],
        float64,
        int64[:],
    ),
    cache=True,
)
def _d8_flow_distance_recursion(ix, fdir, visits, dist, weights, r_dirmap, inc, offsets):
    visited = visits.flat[ix]
    if not visited:
        visits.flat[ix] = True
        dist.flat[ix] = inc
        neighbors = offsets + ix
        for k in range(8):
            neighbor = neighbors[k]
            points_to = fdir.flat[neighbor] == r_dirmap[k]
            if points_to:
                next_inc = inc + weights.flat[neighbor]
                _d8_flow_distance_recursion(
                    neighbor, fdir, visits, dist, weights, r_dirmap, next_inc, offsets
                )


@njit(
    float64[:, :](int64[:, :], float64[:, :], UniTuple(int64, 2), UniTuple(int64, 8)),
    cache=True,
)
def _d8_flow_distance_recur_numba(fdir, weights, pour_point, dirmap):
    visits = np.zeros(fdir.shape, dtype=np.bool_)
    dist = np.full(fdir.shape, np.inf, dtype=np.float64)
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    m, n = fdir.shape
    offsets = np.array([-n, 1 - n, 1, 1 + n, n, -1 + n, -1, -1 - n])
    i, j = pour_point
    ix = (i * n) + j
    _d8_flow_distance_recursion(ix, fdir, visits, dist, weights, r_dirmap, 0.0, offsets)
    return dist


@njit(
    float64[:, :](int64[:, :], float64[:, :], UniTuple(int64, 2), UniTuple(int64, 8)),
    cache=True,
)
def _d8_flow_distance_iter_numba(fdir, weights, pour_point, dirmap):
    visits = np.zeros(fdir.shape, dtype=np.bool_)
    dist = np.full(fdir.shape, np.inf, dtype=np.float64)
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    m, n = fdir.shape
    offsets = np.array([-n, 1 - n, 1, 1 + n, n, -1 + n, -1, -1 - n])
    i, j = pour_point
    ix = (i * n) + j
    dist.flat[ix] = 0.0
    queue = [ix]
    while queue:
        parent = queue.pop()
        visits.flat[parent] = True
        neighbors = offsets + parent
        for k in range(8):
            neighbor = neighbors[k]
            visited = visits.flat[neighbor]
            if visited:
                continue
            else:
                points_to = fdir.flat[neighbor] == r_dirmap[k]
                if points_to:
                    dist.flat[neighbor] = dist.flat[parent] + weights.flat[neighbor]
                    queue.append(neighbor)
    return dist


@njit(
    void(
        int64,
        int64[:, :],
        int64[:, :],
        boolean[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        int64[:],
        float64,
        int64[:],
    ),
    cache=True,
)
def _dinf_flow_distance_recursion(
    ix, fdir_0, fdir_1, visits, dist, weights_0, weights_1, r_dirmap, inc, offsets
):
    current_dist = dist.flat[ix]
    if inc < current_dist:
        dist.flat[ix] = inc
        neighbors = offsets + ix
        for k in range(8):
            neighbor = neighbors[k]
            points_to_0 = fdir_0.flat[neighbor] == r_dirmap[k]
            points_to_1 = fdir_1.flat[neighbor] == r_dirmap[k]
            if points_to_0:
                next_inc = inc + weights_0.flat[neighbor]
                _dinf_flow_distance_recursion(
                    neighbor,
                    fdir_0,
                    fdir_1,
                    visits,
                    dist,
                    weights_0,
                    weights_1,
                    r_dirmap,
                    next_inc,
                    offsets,
                )
            elif points_to_1:
                next_inc = inc + weights_1.flat[neighbor]
                _dinf_flow_distance_recursion(
                    neighbor,
                    fdir_0,
                    fdir_1,
                    visits,
                    dist,
                    weights_0,
                    weights_1,
                    r_dirmap,
                    next_inc,
                    offsets,
                )


@njit(
    float64[:, :](
        int64[:, :],
        int64[:, :],
        float64[:, :],
        float64[:, :],
        UniTuple(int64, 2),
        UniTuple(int64, 8),
    ),
    cache=True,
)
def _dinf_flow_distance_recur_numba(fdir_0, fdir_1, weights_0, weights_1, pour_point, dirmap):
    visits = np.zeros(fdir_0.shape, dtype=np.bool_)
    dist = np.full(fdir_0.shape, np.inf, dtype=np.float64)
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    m, n = fdir_0.shape
    offsets = np.array([-n, 1 - n, 1, 1 + n, n, -1 + n, -1, -1 - n])
    i, j = pour_point
    ix = (i * n) + j
    _dinf_flow_distance_recursion(
        ix, fdir_0, fdir_1, visits, dist, weights_0, weights_1, r_dirmap, 0.0, offsets
    )
    return dist


@njit(
    float64[:, :](
        int64[:, :],
        int64[:, :],
        float64[:, :],
        float64[:, :],
        UniTuple(int64, 2),
        UniTuple(int64, 8),
    ),
    cache=True,
)
def _dinf_flow_distance_iter_numba(fdir_0, fdir_1, weights_0, weights_1, pour_point, dirmap):
    dist = np.full(fdir_0.shape, np.inf, dtype=np.float64)
    visited = np.zeros(fdir_0.shape, dtype=np.bool_)
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    m, n = fdir_0.shape
    offsets = np.array([-n, 1 - n, 1, 1 + n, n, -1 + n, -1, -1 - n])
    i, j = pour_point
    ix = (i * n) + j
    dist.flat[ix] = 0.0
    queue = [(0.0, ix)]
    while queue:
        parent_dist, parent = heappop(queue)
        visited.flat[parent] = True
        neighbors = offsets + parent
        for k in range(8):
            neighbor = neighbors[k]
            if visited.flat[neighbor]:
                continue
            else:
                current_neighbor_dist = dist.flat[neighbor]
                points_to_0 = fdir_0.flat[neighbor] == r_dirmap[k]
                points_to_1 = fdir_1.flat[neighbor] == r_dirmap[k]
                if points_to_0:
                    neighbor_dist_0 = parent_dist + weights_0.flat[neighbor]
                    if neighbor_dist_0 < current_neighbor_dist:
                        dist.flat[neighbor] = neighbor_dist_0
                        heappush(queue, (neighbor_dist_0, neighbor))
                elif points_to_1:
                    neighbor_dist_1 = parent_dist + weights_1.flat[neighbor]
                    if neighbor_dist_1 < current_neighbor_dist:
                        dist.flat[neighbor] = neighbor_dist_1
                        heappush(queue, (neighbor_dist_1, neighbor))
    return dist


# TODO: Weights should actually by (8, m, n)
# neighbor_dist = parent_dist + weights.flat[kix]
@njit(float64[:, :](float64[:, :, :], UniTuple(int64, 2), float64[:, :]), cache=True)
def _mfd_flow_distance_iter_numba(fdir, pour_point, weights):
    _, m, n = fdir.shape
    mn = m * n
    dist = np.full((m, n), np.inf, dtype=np.float64)
    visited = np.zeros((m, n), dtype=np.bool_)
    i, j = pour_point
    ix = (i * n) + j
    offsets = np.array([-n, 1 - n, 1, 1 + n, n, -1 + n, -1, -1 - n])
    r_dirmap = np.array([4, 5, 6, 7, 0, 1, 2, 3])
    dist.flat[ix] = 0.0
    queue = [(0.0, ix)]
    while queue:
        parent_dist, parent = heappop(queue)
        visited.flat[parent] = True
        neighbors = offsets + parent
        for k in range(8):
            neighbor = neighbors[k]
            if visited.flat[neighbor]:
                continue
            else:
                neighbor_dir = r_dirmap[k]
                current_neighbor_dist = dist.flat[neighbor]
                kix = neighbor + (neighbor_dir * mn)
                points_to = fdir.flat[kix] > 0.0
                if points_to:
                    neighbor_dist = parent_dist + weights.flat[neighbor]
                    if neighbor_dist < current_neighbor_dist:
                        dist.flat[neighbor] = neighbor_dist
                        heappush(queue, (neighbor_dist, neighbor))
    return dist


# Functions for 'reverse_flow_distance'


@njit(void(int64, int64, float64[:, :], int64[:, :], uint8[:], float64[:, :]), cache=True)
def _d8_reverse_distance_recursion(startnode, endnode, rdist, fdir, indegree, weights):
    rdist.flat[endnode] = max(rdist.flat[endnode], rdist.flat[startnode] + weights.flat[endnode])
    indegree.flat[endnode] -= 1
    if indegree.flat[endnode] == 0:
        new_startnode = endnode
        new_endnode = fdir.flat[new_startnode]
        _d8_reverse_distance_recursion(new_startnode, new_endnode, rdist, fdir, indegree, weights)


@njit(
    float64[:, :](float64[:, :], int64[:, :], uint8[:], int64[:], float64[:, :]),
    cache=True,
)
def _d8_reverse_distance_recur_numba(rdist, fdir, indegree, startnodes, weights):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        _d8_reverse_distance_recursion(startnode, endnode, rdist, fdir, indegree, weights)
    return rdist


@njit(
    float64[:, :](float64[:, :], int64[:, :], uint8[:], int64[:], float64[:, :]),
    cache=True,
)
def _d8_reverse_distance_iter_numba(rdist, fdir, indegree, startnodes, weights):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        while indegree.flat[startnode] == 0:
            rdist.flat[endnode] = max(
                rdist.flat[endnode], rdist.flat[startnode] + weights.flat[endnode]
            )
            indegree.flat[endnode] -= 1
            startnode = endnode
            endnode = fdir.flat[startnode]
    return rdist


# TODO: This should probably have two weights vectors
@njit(
    float64[:, :](float64[:, :], int64[:, :], int64[:, :], uint8[:], int64[:], float64[:, :]),
    cache=True,
)
def _dinf_reverse_distance_iter_numba(rdist, fdir_0, fdir_1, indegree, startnodes, weights):
    n = startnodes.size
    queue = [0]
    _ = queue.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        queue.append(startnode)
        while queue:
            startnode = queue.pop()
            endnode_0 = fdir_0.flat[startnode]
            endnode_1 = fdir_1.flat[startnode]
            rdist.flat[endnode_0] = max(
                rdist.flat[endnode_0], rdist.flat[startnode] + weights.flat[endnode_0]
            )
            rdist.flat[endnode_1] = max(
                rdist.flat[endnode_1], rdist.flat[startnode] + weights.flat[endnode_1]
            )
            indegree.flat[endnode_0] -= 1
            indegree.flat[endnode_1] -= 1
            if indegree.flat[endnode_0] == 0:
                queue.append(endnode_0)
            if indegree.flat[endnode_1] == 0:
                # Account for cases where both fdirs point in same direction
                if endnode_0 != endnode_1:
                    queue.append(endnode_1)
    return rdist


@njit(
    float64[:, :](float64[:, :], int64[:, :, :], uint8[:], int64[:], float64[:, :]),
    cache=True,
)
def _mfd_reverse_distance_iter_numba(rdist, fdir, indegree, startnodes, weights):
    n = startnodes.size
    queue = [0]
    _ = queue.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        queue.append(startnode)
        while queue:
            startnode = queue.pop()
            for i in range(8):
                fdir_i = fdir[i]
                endnode = fdir_i.flat[startnode]
                if endnode == startnode:
                    continue
                else:
                    weight = weights.flat[startnode]
                    rdist.flat[endnode] = max(
                        rdist.flat[endnode],
                        rdist.flat[startnode] + weights.flat[endnode],
                    )
                    indegree.flat[endnode] -= 1
                    if indegree.flat[endnode] == 0:
                        queue.append(endnode)
    return rdist


# Functions for 'resolve_flats'


@njit(UniTuple(boolean[:, :], 3)(float64[:, :], int64[:]), parallel=True, cache=True)
def _par_get_candidates_numba(dem, inside):
    n = inside.size
    offset = dem.shape[1]
    fdirs_defined = np.zeros(dem.shape, dtype=np.bool_)
    flats = np.zeros(dem.shape, dtype=np.bool_)
    higher_cells = np.zeros(dem.shape, dtype=np.bool_)
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    for i in prange(n):
        k = inside[i]
        inner_neighbors = k + offsets
        fdir_defined = False
        is_pit = True
        higher_cell = False
        same_elev_cell = False
        for j in prange(8):
            neighbor = inner_neighbors[j]
            diff = dem.flat[k] - dem.flat[neighbor]
            fdir_defined |= diff > 0
            is_pit &= diff < 0
            higher_cell |= diff < 0
        is_flat = ~fdir_defined & ~is_pit
        fdirs_defined.flat[k] = fdir_defined
        flats.flat[k] = is_flat
        higher_cells.flat[k] = higher_cell
    fdirs_defined[0, :] = True
    fdirs_defined[:, 0] = True
    fdirs_defined[-1, :] = True
    fdirs_defined[:, -1] = True
    return flats, fdirs_defined, higher_cells


@njit(
    uint32[:, :](int64[:], boolean[:, :], boolean[:, :], int64[:, :]),
    parallel=True,
    cache=True,
)
def _par_get_high_edge_cells_numba(inside, fdirs_defined, higher_cells, labels):
    n = inside.size
    high_edge_cells = np.zeros(fdirs_defined.shape, dtype=np.uint32)
    for i in range(n):
        k = inside[i]
        fdir_defined = fdirs_defined.flat[k]
        higher_cell = higher_cells.flat[k]
        # Find high-edge cells
        is_high_edge_cell = ~fdir_defined & higher_cell
        if is_high_edge_cell:
            high_edge_cells.flat[k] = labels.flat[k]
    return high_edge_cells


@njit(
    uint32[:, :](int64[:], float64[:, :], boolean[:, :], int64[:, :], int64),
    parallel=True,
    cache=True,
)
def _par_get_low_edge_cells_numba(inside, dem, fdirs_defined, labels, numlabels):
    n = inside.size
    offset = dem.shape[1]
    low_edge_cells = np.zeros(dem.shape, dtype=np.uint32)
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    for i in prange(n):
        k = inside[i]
        # Find low-edge cells
        inner_neighbors = k + offsets
        fdir_defined = fdirs_defined.flat[k]
        if ~fdir_defined:
            for j in range(8):
                neighbor = inner_neighbors[j]
                diff = dem.flat[k] - dem.flat[neighbor]
                is_same_elev = diff == 0
                neighbor_direction_defined = fdirs_defined.flat[neighbor]
                neighbor_is_low_edge_cell = (is_same_elev) & (neighbor_direction_defined)
                if neighbor_is_low_edge_cell:
                    label = labels.flat[k]
                    low_edge_cells.flat[neighbor] = label
    return low_edge_cells


@njit(uint16[:, :](uint32[:, :], boolean[:, :], int64[:, :], int64, int64), cache=True)
def _grad_from_higher_numba(hec, flats, labels, numlabels, max_iter=1000):
    offset = flats.shape[1]
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    z = np.zeros(flats.shape, dtype=np.uint16)
    n = z.size
    cur_queue = []
    next_queue = []
    # Increment gradient
    for i in range(n):
        if hec.flat[i]:
            z.flat[i] = 1
            cur_queue.append(i)
    for i in range(2, max_iter + 1):
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            neighbors = offsets + k
            for j in range(8):
                neighbor = neighbors[j]
                if (flats.flat[neighbor]) & (z.flat[neighbor] == 0):
                    z.flat[neighbor] = i
                    next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    # Invert gradient
    max_incs = np.zeros(numlabels + 1)
    for i in range(n):
        label = labels.flat[i]
        inc = z.flat[i]
        max_incs[label] = max(max_incs[label], inc)
    for i in range(n):
        if z.flat[i]:
            label = labels.flat[i]
            z.flat[i] = max_incs[label] - z.flat[i]
    return z


@njit(uint16[:, :](uint32[:, :], boolean[:, :], float64[:, :], int64), cache=True)
def _grad_towards_lower_numba(lec, flats, dem, max_iter=1000):
    offset = flats.shape[1]
    size = flats.size
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    z = np.zeros(flats.shape, dtype=np.uint16)
    cur_queue = []
    next_queue = []
    for i in range(size):
        label = lec.flat[i]
        if label:
            z.flat[i] = 1
            cur_queue.append(i)
    for i in range(2, max_iter + 1):
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            on_left = (k % offset) == 0
            on_right = ((k + 1) % offset) == 0
            on_top = k < offset
            on_bottom = k > (size - offset - 1)
            on_boundary = on_left | on_right | on_top | on_bottom
            neighbors = offsets + k
            for j in range(8):
                if on_boundary:
                    if (on_left) & ((j == 5) | (j == 6) | (j == 7)):
                        continue
                    if (on_right) & ((j == 1) | (j == 2) | (j == 3)):
                        continue
                    if (on_top) & ((j == 0) | (j == 1) | (j == 7)):
                        continue
                    if (on_bottom) & ((j == 3) | (j == 4) | (j == 5)):
                        continue
                neighbor = neighbors[j]
                neighbor_is_flat = flats.flat[neighbor]
                not_visited = z.flat[neighbor] == 0
                same_elev = dem.flat[neighbor] == dem.flat[k]
                if neighbor_is_flat & not_visited & same_elev:
                    z.flat[neighbor] = i
                    next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    return z


# Functions for 'compute_hand'


@njit(int64[:, :](int64[:, :], boolean[:, :], UniTuple(int64, 8)), cache=True)
def _d8_hand_iter_numba(fdir, mask, dirmap):
    offset = fdir.shape[1]
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    hand = -np.ones(fdir.shape, dtype=np.int64)
    cur_queue = []
    next_queue = []
    for i in range(hand.size):
        if mask.flat[i]:
            hand.flat[i] = i
            cur_queue.append(i)
    while True:
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            neighbors = offsets + k
            for j in range(8):
                neighbor = neighbors[j]
                points_to = fdir.flat[neighbor] == r_dirmap[j]
                not_visited = hand.flat[neighbor] < 0
                if points_to and not_visited:
                    hand.flat[neighbor] = hand.flat[k]
                    next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    return hand


@njit(void(int64, int64, int64[:, :], int64[:], int64[:], int64[:, :]), cache=True)
def _d8_hand_recursion(child, parent, hand, offsets, r_dirmap, fdir):
    neighbors = offsets + child
    for k in range(8):
        neighbor = neighbors[k]
        points_to = fdir.flat[neighbor] == r_dirmap[k]
        not_visited = hand.flat[neighbor] == -1
        if points_to and not_visited:
            hand.flat[neighbor] = parent
            _d8_hand_recursion(neighbor, parent, hand, offsets, r_dirmap, fdir)


@njit(int64[:, :](int64[:, :], boolean[:, :], UniTuple(int64, 8)), cache=True)
def _d8_hand_recur_numba(fdir, mask, dirmap):
    offset = fdir.shape[1]
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    hand = -np.ones(fdir.shape, dtype=np.int64)
    parents = np.flatnonzero(mask)
    n = parents.size
    for i in range(n):
        parent = parents[i]
        hand.flat[parent] = parent
    for i in range(n):
        parent = parents[i]
        _d8_hand_recursion(parent, parent, hand, offsets, r_dirmap, fdir)
    return hand


@njit(int64[:, :](int64[:, :], int64[:, :], boolean[:, :], UniTuple(int64, 8)), cache=True)
def _dinf_hand_iter_numba(fdir_0, fdir_1, mask, dirmap):
    offset = fdir_0.shape[1]
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    hand = -np.ones(fdir_0.shape, dtype=np.int64)
    cur_queue = []
    next_queue = []
    for i in range(hand.size):
        if mask.flat[i]:
            hand.flat[i] = i
            cur_queue.append(i)
    while True:
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            neighbors = offsets + k
            for j in range(8):
                neighbor = neighbors[j]
                points_to = (fdir_0.flat[neighbor] == r_dirmap[j]) | (
                    fdir_1.flat[neighbor] == r_dirmap[j]
                )
                not_visited = hand.flat[neighbor] < 0
                if points_to and not_visited:
                    hand.flat[neighbor] = hand.flat[k]
                    next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    return hand


@njit(
    void(int64, int64, int64[:, :], int64[:], int64[:], int64[:, :], int64[:, :]),
    cache=True,
)
def _dinf_hand_recursion(child, parent, hand, offsets, r_dirmap, fdir_0, fdir_1):
    neighbors = offsets + child
    for k in range(8):
        neighbor = neighbors[k]
        points_to = (fdir_0.flat[neighbor] == r_dirmap[k]) | (fdir_1.flat[neighbor] == r_dirmap[k])
        not_visited = hand.flat[neighbor] == -1
        if points_to and not_visited:
            hand.flat[neighbor] = parent
            _dinf_hand_recursion(neighbor, parent, hand, offsets, r_dirmap, fdir_0, fdir_1)


@njit(int64[:, :](int64[:, :], int64[:, :], boolean[:, :], UniTuple(int64, 8)), cache=True)
def _dinf_hand_recur_numba(fdir_0, fdir_1, mask, dirmap):
    offset = fdir_0.shape[1]
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    r_dirmap = np.array(
        [
            dirmap[4],
            dirmap[5],
            dirmap[6],
            dirmap[7],
            dirmap[0],
            dirmap[1],
            dirmap[2],
            dirmap[3],
        ]
    )
    hand = -np.ones(fdir_0.shape, dtype=np.int64)
    parents = np.flatnonzero(mask)
    n = parents.size
    for i in range(n):
        parent = parents[i]
        hand.flat[parent] = parent
    for i in range(n):
        parent = parents[i]
        _dinf_hand_recursion(parent, parent, hand, offsets, r_dirmap, fdir_0, fdir_1)
    return hand


@njit(int64[:, :](float64[:, :, :], boolean[:, :]), cache=True)
def _mfd_hand_iter_numba(fdir, mask):
    _, m, n = fdir.shape
    mn = m * n
    offsets = np.array([-n, 1 - n, 1, 1 + n, n, -1 + n, -1, -1 - n])
    r_dirmap = np.array([4, 5, 6, 7, 0, 1, 2, 3])
    hand = -np.ones((m, n), dtype=np.int64)
    cur_queue = []
    next_queue = []
    for i in range(hand.size):
        if mask.flat[i]:
            hand.flat[i] = i
            cur_queue.append(i)
    while True:
        if not cur_queue:
            break
        while cur_queue:
            k = cur_queue.pop()
            neighbors = offsets + k
            for j in range(8):
                neighbor = neighbors[j]
                visited = hand.flat[neighbor] >= 0
                if visited:
                    continue
                else:
                    neighbor_dir = r_dirmap[j]
                    kix = neighbor + (neighbor_dir * mn)
                    points_to = fdir.flat[kix] > 0.0
                    if points_to:
                        hand.flat[neighbor] = hand.flat[k]
                        next_queue.append(neighbor)
        while next_queue:
            next_cell = next_queue.pop()
            cur_queue.append(next_cell)
    return hand


@njit(float64[:, :](int64[:, :], float64[:, :], float64), parallel=True, cache=True)
def _assign_hand_heights_numba(hand_idx, dem, nodata_out=np.nan):
    n = hand_idx.size
    hand = np.zeros(dem.shape, dtype=np.float64)
    for i in prange(n):
        j = hand_idx.flat[i]
        if j == -1:
            hand.flat[i] = np.nan
        else:
            hand.flat[i] = dem.flat[i] - dem.flat[j]
    return hand


# Functions for 'streamorder'


@njit(
    void(
        int64,
        int64,
        int64[:, :],
        int64[:, :],
        int64[:, :],
        int64[:, :],
        uint8[:],
        uint8[:],
    ),
    cache=True,
)
def _d8_streamorder_recursion(
    startnode, endnode, min_order, max_order, order, fdir, indegree, orig_indegree
):
    min_order.flat[endnode] = min(min_order.flat[endnode], order.flat[startnode])
    max_order.flat[endnode] = max(max_order.flat[endnode], order.flat[startnode])
    indegree.flat[endnode] -= 1
    if indegree.flat[endnode] == 0:
        if (min_order.flat[endnode] == max_order.flat[endnode]) and (
            orig_indegree.flat[endnode] > 1
        ):
            order.flat[endnode] = max_order.flat[endnode] + 1
        else:
            order.flat[endnode] = max_order.flat[endnode]
        new_startnode = endnode
        new_endnode = fdir.flat[new_startnode]
        _d8_streamorder_recursion(
            new_startnode,
            new_endnode,
            min_order,
            max_order,
            order,
            fdir,
            indegree,
            orig_indegree,
        )


@njit(
    int64[:, :](int64[:, :], int64[:, :], int64[:, :], int64[:, :], uint8[:], uint8[:], int64[:]),
    cache=True,
)
def _d8_streamorder_recur_numba(
    min_order, max_order, order, fdir, indegree, orig_indegree, startnodes
):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        _d8_streamorder_recursion(
            startnode,
            endnode,
            min_order,
            max_order,
            order,
            fdir,
            indegree,
            orig_indegree,
        )
    return order


@njit(
    int64[:, :](int64[:, :], int64[:, :], int64[:, :], int64[:, :], uint8[:], uint8[:], int64[:]),
    cache=True,
)
def _d8_streamorder_iter_numba(
    min_order, max_order, order, fdir, indegree, orig_indegree, startnodes
):
    n = startnodes.size
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        while indegree.flat[startnode] == 0:
            min_order.flat[endnode] = min(min_order.flat[endnode], order.flat[startnode])
            max_order.flat[endnode] = max(max_order.flat[endnode], order.flat[startnode])
            indegree.flat[endnode] -= 1
            if (min_order.flat[endnode] == max_order.flat[endnode]) and (
                orig_indegree.flat[endnode] > 1
            ):
                order.flat[endnode] = max_order.flat[endnode] + 1
            else:
                order.flat[endnode] = max_order.flat[endnode]
            startnode = endnode
            endnode = fdir.flat[startnode]
    return order


@njit(
    void(int64, int64, int64[:, :], uint8[:], uint8[:], List(List(int64)), List(int64)),
    cache=True,
)
def _d8_stream_network_recursion(
    startnode, endnode, fdir, indegree, orig_indegree, profiles, profile
):
    profile.append(endnode)
    if orig_indegree[endnode] > 1:
        profiles.append(profile)
    indegree.flat[endnode] -= 1
    if indegree.flat[endnode] == 0:
        if orig_indegree[endnode] > 1:
            profile = [endnode]
        new_startnode = endnode
        new_endnode = fdir.flat[new_startnode]
        _d8_stream_network_recursion(
            new_startnode, new_endnode, fdir, indegree, orig_indegree, profiles, profile
        )


@njit(List(List(int64))(int64[:, :], uint8[:], uint8[:], int64[:]), cache=True)
def _d8_stream_network_recur_numba(fdir, indegree, orig_indegree, startnodes):
    n = startnodes.size
    profiles = [[0]]
    _ = profiles.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        profile = [startnode]
        _d8_stream_network_recursion(
            startnode, endnode, fdir, indegree, orig_indegree, profiles, profile
        )
    return profiles


@njit(List(List(int64))(int64[:, :], uint8[:], uint8[:], int64[:]), cache=True)
def _d8_stream_network_iter_numba(fdir, indegree, orig_indegree, startnodes):
    n = startnodes.size
    profiles = [[0]]
    _ = profiles.pop()
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        profile = [startnode]
        while indegree.flat[startnode] == 0:
            profile.append(endnode)
            indegree.flat[endnode] -= 1
            if orig_indegree[endnode] > 1:
                profiles.append(profile)
                profile = [endnode]
            startnode = endnode
            endnode = fdir.flat[startnode]
    return profiles


@njit(
    Tuple((List(List(int64)), DictType(int64, int64)))(
        int64[:, :], uint8[:], uint8[:], int64[:], boolean
    ),
    cache=True,
)
def _d8_stream_connection_iter_numba(fdir, indegree, orig_indegree, startnodes, include_endpoint):
    n = startnodes.size
    profiles = [[0]]
    connections = {0: 0}
    _ = profiles.pop()
    _ = connections.pop(0)
    for k in range(n):
        startnode = startnodes.flat[k]
        endnode = fdir.flat[startnode]
        profile = [startnode]
        while indegree.flat[startnode] == 0:
            profile.append(endnode)
            indegree.flat[endnode] -= 1
            if orig_indegree.flat[endnode] > 1:
                chain_start = profile[0]
                chain_end = profile[-1]
                connections[chain_start] = chain_end
                if not include_endpoint:
                    _ = profile.pop()
                profiles.append(profile)
                if indegree.flat[endnode] == 0:
                    profile = [endnode]
            startnode = endnode
            endnode = fdir.flat[startnode]
    return profiles, connections


@njit(float64[:, :](int64[:, :], int64[:, :], float64[:, :]), parallel=True, cache=True)
def _d8_cell_dh_numba(startnodes, endnodes, dem):
    n = startnodes.size
    dh = np.zeros_like(dem)
    for k in prange(n):
        startnode = startnodes.flat[k]
        endnode = endnodes.flat[k]
        dh.flat[k] = dem.flat[startnode] - dem.flat[endnode]
    return dh


@njit(
    float64[:, :](
        int64[:, :],
        int64[:, :],
        int64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
    ),
    parallel=True,
    cache=True,
)
def _dinf_cell_dh_numba(startnodes, endnodes_0, endnodes_1, props_0, props_1, dem):
    n = startnodes.size
    dh = np.zeros(dem.shape, dtype=np.float64)
    for k in prange(n):
        startnode = startnodes.flat[k]
        endnode_0 = endnodes_0.flat[k]
        endnode_1 = endnodes_1.flat[k]
        prop_0 = props_0.flat[k]
        prop_1 = props_1.flat[k]
        dh.flat[k] = prop_0 * (dem.flat[startnode] - dem.flat[endnode_0]) + prop_1 * (
            dem.flat[startnode] - dem.flat[endnode_1]
        )
    return dh


@njit(
    float64[:, :](int64[:, :], int64[:, :, :], float64[:, :, :], float64[:, :]),
    parallel=True,
    cache=True,
)
def _mfd_cell_dh_numba(startnodes, endnodes, props, dem):
    k, m, n = props.shape
    mn = m * n
    N = startnodes.size
    dh = np.zeros((m, n), dtype=np.float64)
    for i in prange(N):
        startnode = startnodes.flat[i]
        elev = dem.flat[startnode]
        for j in prange(k):
            kix = startnode + (j * mn)
            endnode = endnodes.flat[kix]
            prop = props.flat[kix]
            neighbor_elev = dem.flat[endnode]
            dh.flat[startnode] += prop * (elev - neighbor_elev)
    return dh


@njit(
    float64[:, :](int64[:, :], UniTuple(int64, 8), float64, float64),
    parallel=True,
    cache=True,
)
def _d8_cell_distances_numba(fdir, dirmap, dx, dy):
    n = fdir.size
    cdist = np.zeros(fdir.shape, dtype=np.float64)
    dd = math.sqrt(dx**2 + dy**2)
    distances = (dy, dd, dx, dd, dy, dd, dx, dd)
    dist_map = {0: 0.0}
    for i in range(8):
        dist_map[dirmap[i]] = distances[i]
    for k in prange(n):
        fdir_k = fdir.flat[k]
        cdist.flat[k] = dist_map[fdir_k]
    return cdist


@njit(
    float64[:, :](
        int64[:, :],
        int64[:, :],
        float64[:, :],
        float64[:, :],
        UniTuple(int64, 8),
        float64,
        float64,
    ),
    parallel=True,
    cache=True,
)
def _dinf_cell_distances_numba(fdir_0, fdir_1, prop_0, prop_1, dirmap, dx, dy):
    n = fdir_0.size
    cdist = np.zeros(fdir_0.shape, dtype=np.float64)
    dd = math.sqrt(dx**2 + dy**2)
    distances = (dy, dd, dx, dd, dy, dd, dx, dd)
    dist_map = {0: 0.0}
    for i in range(8):
        dist_map[dirmap[i]] = distances[i]
    for k in prange(n):
        fdir_k_0 = fdir_0.flat[k]
        fdir_k_1 = fdir_1.flat[k]
        dist_k_0 = dist_map[fdir_k_0]
        dist_k_1 = dist_map[fdir_k_1]
        prop_k_0 = prop_0.flat[k]
        prop_k_1 = prop_1.flat[k]
        dist_k = prop_k_0 * dist_k_0 + prop_k_1 * dist_k_1
        cdist.flat[k] = dist_k
    return cdist


@njit(
    float64[:, :](int64[:, :], int64[:, :, :], float64[:, :, :], float64, float64),
    parallel=True,
    cache=True,
)
def _mfd_cell_distances_numba(startnodes, endnodes, props, dx, dy):
    k, m, n = props.shape
    mn = m * n
    N = startnodes.size
    dd = math.sqrt(dx**2 + dy**2)
    distances = (dy, dd, dx, dd, dy, dd, dx, dd)
    cdist = np.zeros((m, n), dtype=np.float64)
    for i in prange(N):
        startnode = startnodes.flat[i]
        for j in prange(k):
            kix = startnode + (j * mn)
            endnode = endnodes.flat[kix]
            prop = props.flat[kix]
            cdist.flat[startnode] += prop * distances[j]
    return cdist


@njit(float64[:, :](float64[:, :], float64[:, :]), parallel=True, cache=True)
def _cell_slopes_numba(dh, cdist):
    n = dh.size
    slopes = np.zeros(dh.shape, dtype=np.float64)
    for k in prange(n):
        dh_k = dh.flat[k]
        cdist_k = cdist.flat[k]
        if cdist_k == 0:
            slopes.flat[k] = 0.0
        else:
            slopes.flat[k] = dh_k / cdist_k
    return slopes


@njit(
    void(int64, int64[:, :], int64[:, :], int64, int64, int64, boolean[:, :]),
    cache=True,
)
def _dinf_fix_cycles_recursion(node, fdir_0, fdir_1, ancestor, depth, max_cycle_size, visited):
    if visited.flat[node]:
        return None
    if depth > max_cycle_size:
        return None
    left = fdir_0.flat[node]
    right = fdir_1.flat[node]
    if left == ancestor:
        fdir_0.flat[node] = right
        return None
    else:
        _dinf_fix_cycles_recursion(
            left, fdir_0, fdir_1, ancestor, depth + 1, max_cycle_size, visited
        )
    if right == ancestor:
        fdir_1.flat[node] = left
        return None
    else:
        _dinf_fix_cycles_recursion(
            right, fdir_0, fdir_1, ancestor, depth + 1, max_cycle_size, visited
        )


@njit(void(int64[:, :], int64[:, :], int64), cache=True)
def _dinf_fix_cycles_numba(fdir_0, fdir_1, max_cycle_size):
    n = fdir_0.size
    visited = np.zeros(fdir_0.shape, dtype=np.bool_)
    depth = 0
    for node in range(n):
        _dinf_fix_cycles_recursion(node, fdir_0, fdir_1, node, depth, max_cycle_size, visited)
        visited.flat[node] = True


# TODO: Assumes pits and flats are removed
@njit(int64[:, :](int64[:, :], UniTuple(int64, 8)), parallel=True, cache=True)
def _flatten_fdir_numba(fdir, dirmap):
    r, c = fdir.shape
    n = fdir.size
    flat_fdir = np.zeros((r, c), dtype=np.int64)
    offsets = (0 - c, 1 - c, 1 + 0, 1 + c, 0 + c, -1 + c, -1 + 0, -1 - c)
    offset_map = {0: 0}
    left_map = {0: 0}
    right_map = {0: 0}
    top_map = {0: 0}
    bottom_map = {0: 0}
    for i in range(8):
        # Inside cells
        offset_map[dirmap[i]] = offsets[i]
        # Left boundary
        if i in {5, 6, 7}:
            left_map[dirmap[i]] = 0
        else:
            left_map[dirmap[i]] = offsets[i]
        # Right boundary
        if i in {1, 2, 3}:
            right_map[dirmap[i]] = 0
        else:
            right_map[dirmap[i]] = offsets[i]
        # Top boundary
        if i in {7, 0, 1}:
            top_map[dirmap[i]] = 0
        else:
            top_map[dirmap[i]] = offsets[i]
        # Bottom boundary
        if i in {3, 4, 5}:
            bottom_map[dirmap[i]] = 0
        else:
            bottom_map[dirmap[i]] = offsets[i]
    for k in prange(n):
        cell_dir = fdir.flat[k]
        on_left = (k % c) == 0
        on_right = ((k + 1) % c) == 0
        on_top = k < c
        on_bottom = k > (n - c - 1)
        on_boundary = on_left | on_right | on_top | on_bottom
        if on_boundary:
            # TODO: This seems like it could cause errors at corner points
            # TODO: Check if offset is already zero
            if on_left:
                offset = left_map[cell_dir]
            if on_right:
                offset = right_map[cell_dir]
            if on_top:
                offset = top_map[cell_dir]
            if on_bottom:
                offset = bottom_map[cell_dir]
        else:
            offset = offset_map[cell_dir]
        flat_fdir.flat[k] = k + offset
    return flat_fdir


@njit(int64[:, :](int64[:, :], UniTuple(int64, 8)), parallel=True, cache=True)
def _flatten_fdir_no_boundary(fdir, dirmap):
    r, c = fdir.shape
    n = fdir.size
    flat_fdir = np.zeros((r, c), dtype=np.int64)
    offsets = (0 - c, 1 - c, 1 + 0, 1 + c, 0 + c, -1 + c, -1 + 0, -1 - c)
    offset_map = {0: 0}
    for i in range(8):
        offset_map[dirmap[i]] = offsets[i]
    for k in prange(n):
        cell_dir = fdir.flat[k]
        offset = offset_map[cell_dir]
        flat_fdir.flat[k] = k + offset
    return flat_fdir


# TODO: Assumes pits and flats are removed
@njit(int64[:, :, :](float64[:, :, :]), parallel=True, cache=True)
def _flatten_mfd_fdir_numba(fdir):
    p, r, c = fdir.shape
    n = r * c
    flat_fdir = np.zeros((p, r, c), dtype=np.int64)
    offsets = np.array([0 - c, 1 - c, 1 + 0, 1 + c, 0 + c, -1 + c, -1 + 0, -1 - c], dtype=np.int64)
    left_map = np.array(
        [offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], 0, 0, 0],
        dtype=np.int64,
    )
    right_map = np.array(
        [offsets[0], 0, 0, 0, offsets[4], offsets[5], offsets[6], offsets[7]],
        dtype=np.int64,
    )
    top_map = np.array(
        [0, 0, offsets[2], offsets[3], offsets[4], offsets[5], offsets[6], 0],
        dtype=np.int64,
    )
    bottom_map = np.array(
        [offsets[0], offsets[1], offsets[2], 0, 0, 0, offsets[6], offsets[7]],
        dtype=np.int64,
    )
    for i in prange(8):
        for k in prange(n):
            kix = k + (i * n)
            cell_value = fdir.flat[kix]
            if cell_value == 0:
                offset = 0
            else:
                on_left = (k % c) == 0
                on_right = ((k + 1) % c) == 0
                on_top = k < c
                on_bottom = k > (n - c - 1)
                on_boundary = on_left | on_right | on_top | on_bottom
                if on_boundary:
                    # TODO: This seems like it could cause errors at corner points
                    # TODO: Check if offset is already zero
                    if on_left:
                        offset = left_map[i]
                    if on_right and (offset != 0):
                        offset = right_map[i]
                    if on_top and (offset != 0):
                        offset = top_map[i]
                    if on_bottom and (offset != 0):
                        offset = bottom_map[i]
                else:
                    offset = offsets[i]
            flat_fdir.flat[kix] = k + offset
    return flat_fdir


@njit
def _construct_matching(fdir, dirmap):
    n = fdir.size
    startnodes = np.arange(n, dtype=np.int64)
    endnodes = _flatten_fdir_numba(fdir, dirmap).ravel()
    return startnodes, endnodes


@njit(boolean[:, :](float64[:, :], int64[:]), parallel=True, cache=True)
def _find_pits_numba(dem, inside):
    n = inside.size
    offset = dem.shape[1]
    pits = np.zeros(dem.shape, dtype=np.bool_)
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    for i in prange(n):
        k = inside[i]
        inner_neighbors = k + offsets
        is_pit = True
        for j in prange(8):
            neighbor = inner_neighbors[j]
            diff = dem.flat[k] - dem.flat[neighbor]
            is_pit &= diff < 0
        pits.flat[k] = is_pit
    return pits


@njit(float64[:, :](float64[:, :], int64[:]), parallel=True, cache=True)
def _fill_pits_numba(dem, pit_indices):
    n = pit_indices.size
    offset = dem.shape[1]
    pits_filled = np.copy(dem).astype(np.float64)
    max_diff = dem.max() - dem.min()
    offsets = np.array([-offset, 1 - offset, 1, 1 + offset, offset, -1 + offset, -1, -1 - offset])
    for i in prange(n):
        k = pit_indices[i]
        inner_neighbors = k + offsets
        adjustment = max_diff
        for j in prange(8):
            neighbor = inner_neighbors[j]
            diff = dem.flat[neighbor] - dem.flat[k]
            adjustment = min(diff, adjustment)
        pits_filled.flat[k] += adjustment
    return pits_filled


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
    rv = np.zeros(nc, dtype="int64")
    for i in prange(nc):
        rv[i] = _first_true1d(mask[:, i], invert=True)
    return rv


@njit(parallel=True, cache=True)
def _bottom(mask):
    nr, nc = mask.shape[0], mask.shape[1]
    rv = np.zeros(nc, dtype="int64")
    for i in prange(nc):
        rv[i] = _first_true1d(mask[:, i], start=nr - 1, end=-1, step=-1, invert=True)
    return rv


@njit(parallel=True, cache=True)
def _left(mask):
    nr = mask.shape[0]
    rv = np.zeros(nr, dtype="int64")
    for i in prange(nr):
        rv[i] = _first_true1d(mask[i, :], invert=True)
    return rv


@njit(parallel=True, cache=True)
def _right(mask):
    nr, nc = mask.shape[0], mask.shape[1]
    rv = np.zeros(nr, dtype="int64")
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


@pfwrapper
@njit(cache=True)
def _priority_flood(dem, dem_mask, tuple_type):
    open_cells = typedlist.List.empty_list(tuple_type)  # Priority queue
    pits = typedlist.List.empty_list(tuple_type)  # FIFO queue
    closed_cells = dem_mask.copy()
    isertn = count()

    # Push the edges onto priority queue
    y, x = dem.shape

    edge = _left(dem_mask)[:-1]
    for row, col in zip(count(), edge):
        if col >= 0:
            open_cells.append((dem[row, col], next(isertn), row, col))
            closed_cells[row, col] = True
    edge = _bottom(dem_mask)[:-1]
    for row, col in zip(edge, count()):
        if row >= 0:
            open_cells.append((dem[row, col], next(isertn), row, col))
            closed_cells[row, col] = True
    edge = np.flip(_right(dem_mask))[:-1]
    for row, col in zip(count(y - 1, step=-1), edge):
        if col >= 0:
            open_cells.append((dem[row, col], next(isertn), row, col))
            closed_cells[row, col] = True
    edge = np.flip(_top(dem_mask))[:-1]
    for row, col in zip(edge, count(x - 1, step=-1)):
        if row >= 0:
            open_cells.append((dem[row, col], next(isertn), row, col))
            closed_cells[row, col] = True
    heapify(open_cells)

    row_offsets = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    col_offsets = np.array([0, 1, 1, 1, 0, -1, -1, -1])

    pits_pos = 0
    while open_cells or pits_pos < len(pits):
        if pits_pos < len(pits):
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

            if dem[row, col] <= elv:
                dem[row, col] = elv
                pits.append((elv, next(isertn), row, col))
            else:
                heappush(open_cells, (dem[row, col], next(isertn), row, col))
            closed_cells[row, col] = True

        # pits book-keeping
        if pits_pos == len(pits) and len(pits) > 1024:
            # Queue is empty, lets clear it out
            pits.clear()
            pits_pos = 0

    return dem
