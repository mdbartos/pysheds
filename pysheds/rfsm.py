import numpy as np
import pandas as pd
from scipy import ndimage, optimize
from pysheds.grid import Grid
from pysheds.view import Raster
import skimage.morphology
from itertools import combinations, chain

class RFSM:
    def __init__(self, dem, max_levels=100, max_spills=100, min_size=0, boundary=None):
        self.dem = dem
        if isinstance(dem, Raster):
            self.x, self.y = abs(dem.affine.a), abs(dem.affine.e)
        else:
            self.x, self.y = 1, 1
        self.max_levels = max_levels
        self.max_spills = max_spills
        self.min_size = min_size
        self.boundary = boundary
        self.shape = dem.shape
        self.size = dem.size
        self.exit_node = Node(name='exit', vol=np.inf, cumulative_vol=np.inf)
        self.construct_topology()

    def construct_topology(self):
        self.find_depressions()
        self.find_hierarchy()
        self.find_watersheds()
        self.find_connections()
        self.find_drop()
        self.create_tree()
        self.set_node_capacities()
        self.set_node_transfers()

    def compute_waterlevel(self, input_vol):
        runoff = self.compute_runoff(input_vol)
        self.initialize_volumes(runoff)
        self.spill()
        depth = self.compute_depths()
        return depth

    def find_depressions(self):
        # Identify depressions in DEM
        if self.boundary is None:
            # Rotate the barrier to ensure that all depressions touching edges are captured
            # TODO: This may not capture all depressions if they are touching all edges
            ix = np.arange(self.size).reshape(self.shape)
            top = ix[0, :]
            bottom = ix[-1, :]
            left = ix[:, 0]
            right = ix[:, -1]
            holes = np.zeros(self.shape)
            for combination in combinations([top, bottom, left, right], 3):
                mask = np.copy(self.dem)
                exterior = np.concatenate(combination)
                mask.flat[exterior] = self.dem.max()
                seed = np.copy(mask)
                seed[1:-1, 1:-1] = self.dem.max()
                rec = skimage.morphology.reconstruction(seed, mask, method='erosion')
                holes[(rec - self.dem) > 0] = self.dem[(rec - self.dem) > 0]
        else:
            mask = np.copy(self.dem)
            mask[self.boundary > 0] = self.dem.max() + 1
            mask[self.boundary < 0] = self.dem.min() - 1
            seed = np.copy(mask)
            seed[1:-1, 1:-1] = mask.max()
            rec = skimage.morphology.reconstruction(seed, mask, method='erosion')
            holes = rec - mask
        if self.min_size:
            holes_mask = skimage.morphology.remove_small_objects(holes != 0, min_size=self.min_size)
            holes = np.where(holes_mask, holes, 0)
        # Specify levels
        levels = []
        mask = np.where(holes == 0, holes.min(), holes)
        seed = np.where(holes == 0, mask, holes.max())
        rec = skimage.morphology.reconstruction(seed, mask, method='erosion')
        rec[0, :] = np.where((rec[0, :] != 0) & (rec[1, :] != 0), rec[0, :], 0)
        rec[-1, :] = np.where((rec[-1, :] != 0) & (rec[-2, :] != 0), rec[-1, :], 0)
        rec[:, 0] = np.where((rec[:, 0] != 0) & (rec[:, 1] != 0), rec[:, 0], 0)
        rec[:, -1] = np.where((rec[:, -1] != 0) & (rec[:, -2] != 0), rec[:, -1], 0)
        levels.append(rec)
        for _ in range(self.max_levels):
            diff = rec - mask
            holes = np.where(diff, holes, 0)
            if self.min_size:
                holes_mask = skimage.morphology.remove_small_objects(holes != 0,
                                                                     min_size=self.min_size)
                holes = np.where(holes_mask, holes, 0)
            mask = np.where(holes == 0, holes.min(), holes)
            seed = np.where(holes == 0, mask, holes.max())
            rec = skimage.morphology.reconstruction(seed, mask, method='erosion')
            rec[0, :] = np.where((rec[0, :] != 0) & (rec[1, :] != 0), rec[0, :], 0)
            rec[-1, :] = np.where((rec[-1, :] != 0) & (rec[-2, :] != 0), rec[-1, :], 0)
            rec[:, 0] = np.where((rec[:, 0] != 0) & (rec[:, 1] != 0), rec[:, 0], 0)
            rec[:, -1] = np.where((rec[:, -1] != 0) & (rec[:, -2] != 0), rec[:, -1], 0)
            if not rec.any():
                break
            levels.append(rec)
        levels = levels[::-1]
        levels.append(np.ones(self.shape))
        # Find number of unique depressions in each level
        struct = np.ones((3,3), dtype=bool)
        ns = []
        for index, level in enumerate(levels):
            _level, _n = ndimage.label(level, structure=struct)
            levels[index] = _level
            ns.append(_n)
        # Set instance variables
        self.levels = levels
        self.ns = ns

    def find_hierarchy(self):
        lup = {}
        has_lower = []
        for n in self.ns:
            has_lower.append(np.zeros(n + 1, dtype=bool))
        for index in range(len(self.levels) - 1):
            labelmap = (pd.Series(self.levels[index + 1][self.levels[index] != 0],
                                index=self.levels[index][self.levels[index] != 0])
                        .groupby(level=0).first())
            lup[index] = dict(labelmap)
            has_lower[index + 1][labelmap.unique()] = True
        self.lup = lup
        self.has_lower = has_lower

    def find_watersheds(self):
        ws = []
        for index in range(len(self.levels) - 1):
            mask = (self.has_lower[index + 1][self.levels[index + 1]])
            w = skimage.morphology.watershed(self.dem, self.levels[index],
                                             mask=mask, watershed_line=True)
            w = np.where(mask, w, -1)
            ws.append(w)
        ws.append(self.levels[-1])
        self.ws = ws

    def find_connections(self):
        c = {}
        b = {}
        inside = np.zeros(self.shape, dtype=bool)
        inside[1:-1, 1:-1] = True
        for index in range(len(self.levels) - 1):
            c[index] = {}
            b[index] = {}
            comm = np.flatnonzero((self.ws[index] == 0) & inside)
            # TODO: Not super elegant
            neighbors = self.ws[index].flat[Grid._select_surround_ravel(self, comm,
                                                                        self.ws[index].shape)]
            comms = dict(zip(comm, [set() for i in comm]))
            for region in self.lup[index].keys():
                for elem in comm[(neighbors == region).any(axis=1)]:
                    comms[elem].add(region)
            for elem in comms:
                comms[elem] = tuple(comms[elem])
            for comm, pair in comms.items():
                if len(pair) > 2:
                    for combination in combinations(pair, 2):
                        subpair = tuple(sorted(combination))
                        if subpair in c[index]:
                            b[index][subpair].append(comm)
                            if self.dem.flat[comm] < self.dem.flat[c[index][subpair]]:
                                c[index][subpair] = comm
                        else:
                            c[index][subpair] = comm
                            b[index][subpair] = [comm]
                elif len(pair) < 2:
                    # TODO: Not really sure what's happening in this case
                    pass
                else:
                    if pair in c[index]:
                        b[index][pair].append(comm)
                        if self.dem.flat[comm] < self.dem.flat[c[index][pair]]:
                            c[index][pair] = comm
                    else:
                        c[index][pair] = comm
                        b[index][pair] = [comm]
        self.c = c
        self.b = b

    def find_drop(self):
        level = []
        subnum = []
        full = []
        n_i = 0
        for index, n in enumerate(self.ns):
            s = np.arange(1, n + 1)
            f = s + n_i
            l = np.repeat(index, n)
            subnum.append(s)
            full.append(f)
            level.append(l)
            n_i += n
        level = np.concatenate(level)
        subnum = np.concatenate(subnum)
        full = np.concatenate(full)
        dropmap = pd.DataFrame(np.column_stack([level, subnum]), index=full)
        # Figure out where each drop will end up
        drop = skimage.morphology.watershed(self.dem,
                                            np.where(self.ws[0] > 0, self.ws[0], 0),
                                            mask=self.ws[1] > 0)
        for index in range(1, len(self.ws) - 1):
            num_lower = sum(self.ns[:index])
            base = drop + np.where((self.ws[index] > 0) & (drop == 0),
                                   num_lower + self.ws[index], 0)
            drop = skimage.morphology.watershed(self.dem, base,
                                                mask=self.ws[index + 1] > 0)
        self.dropmap = dropmap
        self.drop = drop

    def create_tree(self):
        # Make nodes
        self.nodes = []
        for index in range(len(self.levels)):
            self.nodes.append({})
            for i in range(1, self.ns[index] + 1):
                self.nodes[index][i] = Node(name=i, level=index)
        # Construct tree from nodes
        self.tmap = []
        for index in range(len(self.levels) - 1):
            g = set(range(1, self.ns[index] + 1))
            s = pd.DataFrame(pd.Series(self.c[index]))
            s[0] = s[0].astype(int)
            s[1] = self.dem.flat[s[0].values]
            s[2] = [self.lup[index][i] for i in s.index.get_level_values(0)]
            s = s.sort_values([2,1])
            num_connections = dict(s.groupby(2).size())
            self.tmap.append({})
            for i in s.index:
                l = self.nodes[index][i[0]]
                r = self.nodes[index][i[1]]
                # Kind of ugly, but need to make sure both haven't been added already
                # TODO: This could be introducing a bug, because it's eliminating a connection
                if not (self.get_root(l) is self.get_root(r)):
                    parent = Node()
                    parent.comm = s.loc[i, 0]
                    parent.elev = s.loc[i, 1]
                    parent.level = index
                    upper_label = int(s.loc[i, 2])
                    self.tmap[index][i] = parent
                    # Join watersheds at same level
                    for j, d in zip(i, ('l', 'r')):
                        g.discard(j)
                        child = self.get_root(self.nodes[index][j])
                        child.parent = parent
                        setattr(parent, d, child)
                    num_connections[upper_label] -= 1
                    # If all watersheds in the depressions have been joined, move up
                    if num_connections[upper_label] == 0:
                        parent.level = (index + 1)
                        parent.name = upper_label
                        self.nodes[index + 1][parent.name] = parent
                else:
                    num_connections[upper_label] -= 1
                    # TODO: Not totally sure if this part is correct
                    if num_connections[upper_label] == 0:
                        parent.level = (index + 1)
                        parent.name = upper_label
                        self.nodes[index + 1][parent.name] = parent
            if g:
                for j in g:
                    upper_label = self.lup[index][j]
                    child = self.nodes[index][j]
                    parent = self.nodes[index + 1][upper_label]
                    parent.elev = np.asscalar(self.dem[self.levels[index + 1]
                                                       == upper_label].min())
                    child.parent = parent
                    parent.l = child
                    self.nodes[index][j] = child
                    self.nodes[index + 1][parent.name] = parent
        self.root = self.nodes[-1][1]
        self.root.t = self.exit_node

    def set_node_capacities(self):
        self.set_cumulative_capacities(self.root)
        self.root.vol = np.inf
        self.set_marginal_capacities(self.root)

    def set_node_transfers(self):
        for index, mapping in enumerate(self.tmap):
            for pair, node in mapping.items():
                i, j = pair
                comm = int(node.comm)
                comm_elev = node.elev
                neighbors = Grid._select_surround_ravel(self, comm, self.dem.shape)
                ser = pd.DataFrame(np.column_stack([neighbors, self.dem.flat[neighbors],
                                                    self.ws[index].flat[neighbors]]))
                ser = ser[ser[2].isin(list(pair))]
                g = ser.groupby(2).idxmin()[1].apply(lambda x: ser.loc[x, 0])
                fullix = self.drop.flat[g.values.astype(int)]
                lv = self.dropmap.loc[fullix][0].values
                nm = self.dropmap.loc[fullix][1].values
                g = pd.DataFrame(np.column_stack([lv, nm]), index=g.index.values.astype(int),
                                columns=['level', 'name']).to_dict(orient='index')
                # Children will always be in numeric order from left to right
                lt, rt = g[j], g[i]
                node.l.t = self.nodes[lt['level']][lt['name']]
                node.r.t = self.nodes[rt['level']][rt['name']]
        self.set_singleton_transfer(self.root)

    def compute_runoff(self, input_vol):
        v = (pd.DataFrame(np.column_stack([self.dropmap[0].loc[self.drop.ravel()].values,
                                           self.dropmap[1].loc[self.drop.ravel()].values,
                                           input_vol.ravel()]))
             .groupby([0,1]).sum().reset_index())
        v[0] = v[0].astype(int)
        v[1] = v[1].astype(int)
        v = v.sort_values([0,1])
        return v

    def initialize_volumes(self, runoff):
        for level, name, vol in zip(runoff[0].values,
                                    runoff[1].values,
                                    runoff[2].values):
            node = self.nodes[level][name]
            node.current_vol += vol

    def compute_vol(self, z, node, target_vol):
        under_vol = node.cumulative_vol - node.vol
        if node.name:
            mask = (self.ws[node.level] == node.name)
            full = z - self.dem[mask]
        else:
            leaves = []
            self.enumerate_leaves(node, level=node.level, stack=leaves)
            mask = np.isin(self.ws[node.level], leaves)
            boundary = list(chain.from_iterable([self.b[node.level].setdefault(pair, [])
                                                    for pair in combinations(leaves, 2)]))
            mask.flat[boundary] = True
            full = z - self.dem[mask]
        vol = abs(np.asscalar(full[full > 0].sum()) * self.x * self.y)
        return vol - target_vol - under_vol

    def spill(self):
        cur_iter = 0
        for _ in range(self.max_spills):
            overflowing = np.array(False, dtype=bool)
            self.check_overflow(self.root, overflowing)
            if not overflowing:
                break
            self.spread_volumes(self.root)
            cur_iter += 1
        overflowing = np.array(False, dtype=bool)
        self.check_overflow(self.root, overflowing)
        assert not overflowing

    def compute_depths(self):
        waterlevel = np.zeros(self.dem.shape)
        self.volume_to_level(self.root, waterlevel)
        return waterlevel

    def get_root(self, node):
        if node.parent:
            return self.get_root(node.parent)
        else:
            return node

    def enumerate_leaves(self, node, level=0, stack=[]):
        if node.level >= level:
            if node.name:
                stack.append(node.name)
            if node.l:
                self.enumerate_leaves(node.l, level=level, stack=stack)
            if node.r:
                self.enumerate_leaves(node.r, level=level, stack=stack)

    def set_cumulative_capacities(self, node):
        if node.l:
            self.set_cumulative_capacities(node.l)
        if node.r:
            self.set_cumulative_capacities(node.r)
        if node.parent:
            if node.name:
                elevdiff = node.parent.elev - self.dem[self.ws[node.level] == node.name]
                vol = abs(np.asscalar(elevdiff[elevdiff > 0].sum()) * self.x * self.y)
                node.vol = vol
            else:
                leaves = []
                self.enumerate_leaves(node, level=node.level, stack=leaves)
                mask = np.isin(self.ws[node.level], leaves)
                boundary = list(chain.from_iterable([self.b[node.level].setdefault(pair, [])
                                                        for pair in combinations(leaves, 2)]))
                mask.flat[boundary] = True
                elevdiff = node.parent.elev - self.dem[mask]
                vol = abs(np.asscalar(elevdiff[elevdiff > 0].sum()) * self.x * self.y)
                node.vol = vol

    def set_marginal_capacities(self, node):
        if node.l:
            node.cumulative_vol = node.vol
            node.vol -= node.l.vol
            if node.r:
                node.vol -= node.r.vol
            node.vol = max(node.vol, 0)
            self.set_marginal_capacities(node.l)
            if node.r:
                self.set_marginal_capacities(node.r)
        else:
            node.cumulative_vol = node.vol

    def set_singleton_transfer(self, node):
        if node.l:
            self.set_singleton_transfer(node.l)
        if node.r:
            self.set_singleton_transfer(node.r)
        if node.parent:
            if node.parent.l and not node.parent.r:
                node.t = node.parent

    def map_nodes(self, node, op=(lambda x: None), *args, **kwargs):
        if node.l:
            self.map_nodes(node.l, op=op, *args, **kwargs)
        if node.r:
            self.map_nodes(node.r, op=op, *args, **kwargs)
        op(node, *args, **kwargs)

    def accumulate(self, x, accumulator):
        accumulator += (x.vol)

    def check_full(self, node, full):
        if node.vol != 0:
            full &= np.array(node.current_vol >= node.vol, dtype=bool)
        if full:
            if node.l:
                self.check_full(node.l, full)
            if node.r:
                self.check_full(node.r, full)

    def node_full(self, node):
        # TODO: This should be generalized using recursion
        if node.vol == 0:
            full = np.array(True, dtype=bool)
            self.check_full(node, full)
            full = np.asscalar(full)
            return full
        return node.current_vol >= node.vol

    def spread_volumes(self, node):
        # Push down
        if (node.current_vol > node.vol) and (node.t):
            transfer = node.current_vol - node.vol
            node.current_vol -= transfer
            node.t.current_vol += transfer
        # Recurse
        if node.l:
            self.spread_volumes(node.l)
        if node.r:
            self.spread_volumes(node.r)
        # Pull up
        if node.l:
            l_full = self.node_full(node.l)
            if node.r:
                r_full = self.node_full(node.r)
            else:
                r_full = True
            # TODO: This is taking a lot of time.
            # Should probably store full/empty state on the way up.
            if l_full and r_full:
                ltransfer = node.l.current_vol - node.l.vol
                if node.r:
                    rtransfer = node.r.current_vol - node.r.vol
                else:
                    rtransfer = 0
                node.l.current_vol -= ltransfer
                node.current_vol += ltransfer
                if node.r:
                    node.r.current_vol -= rtransfer
                    node.current_vol += rtransfer

    def check_overflow(self, node, overflowing, break_level=-1):
        if node.level > break_level:
            overflowing |= np.array(node.current_vol > node.vol, dtype=bool)
            if not overflowing:
                if node.l:
                    self.check_overflow(node.l, overflowing, break_level=break_level)
                if node.r:
                    self.check_overflow(node.r, overflowing, break_level=break_level)

    def volume_to_level(self, node, waterlevel):
        if node.current_vol > 0:
            maxelev = node.parent.elev
            if node.elev:
                minelev = node.elev
            else:
                # TODO: This bound could be a lot better
                minelev = np.nanmin(self.dem)
            target_vol = node.current_vol
            elev = optimize.bisect(self.compute_vol, minelev, maxelev,
                                   args=(node, target_vol))
            if node.name:
                mask = self.ws[node.level] == node.name
            else:
                leaves = []
                self.enumerate_leaves(node, level=node.level, stack=leaves)
                mask = np.isin(self.ws[node.level], leaves)
                boundary = list(chain.from_iterable([self.b[node.level].setdefault(pair, [])
                                                     for pair in combinations(leaves, 2)]))
                mask.flat[boundary] = True
            mask = np.flatnonzero(mask & (self.dem < elev))
            waterlevel.flat[mask] = elev
        else:
            if node.l:
                self.volume_to_level(node.l, waterlevel)
            if node.r:
                self.volume_to_level(node.r, waterlevel)

    def reset_volumes(self):
        self.remove_volume(self.root)

    def remove_volume(self, node):
        node.current_vol = 0.0
        if node.l:
            self.remove_volume(node.l)
        if node.r:
            self.remove_volume(node.r)

    def show_tree(self, tree=None):
        if tree is None:
            tree = self.root
        depth = ""
        treestr = ""

        def print_push(char):
            nonlocal depth
            branch_str = ' {}  '.format(char)
            depth += branch_str

        def print_pop():
            nonlocal depth
            depth = depth[:-4]

        def print_tree(node):
            nonlocal depth
            nonlocal treestr
            if node.l:
                if (node.name):
                    treestr += '({0}-{1})\n'.format(node.level, node.name)
                else:
                    treestr += '{0}{1}\n'.format(chr(9472), '+')
                treestr += '{0} {1}{2}{2}'.format(depth, chr(9500), chr(9472))
                print_push(chr(9474))
                print_tree(node.l)
                print_pop()
                if node.r:
                    treestr += '{0} {1}{2}{2}'.format(depth, chr(9492), chr(9472))
                    print_push(' ')
                    print_tree(node.r)
                    print_pop()
            else:
                treestr += '({0}-{1})\n'.format(node.level, node.name)

        print_tree(tree)
        return treestr

    def show_vol(self, tree=None):
        if tree is None:
            tree = self.root
        depth = ""
        treestr = ""

        def print_push(char):
            nonlocal depth
            branch_str = ' {}  '.format(char)
            depth += branch_str

        def print_pop():
            nonlocal depth
            depth = depth[:-4]

        def print_tree(node):
            nonlocal depth
            nonlocal treestr
            if node.l:
                if (node.name):
                    treestr += '({0:.2f} | {1:.2f})\n'.format(node.current_vol, node.vol)
                else:
                    treestr += '({0:.2f} | {1:.2f})\n'.format(node.current_vol, node.vol)
                treestr += '{0} {1}{2}{2}'.format(depth, chr(9500), chr(9472))
                print_push(chr(9474))
                print_tree(node.l)
                print_pop()
                if node.r:
                    treestr += '{0} {1}{2}{2}'.format(depth, chr(9492), chr(9472))
                    print_push(' ')
                    print_tree(node.r)
                    print_pop()
            else:
                treestr += '({0:.2f} | {1:.2f})\n'.format(node.current_vol, node.vol)

        print_tree(tree)
        return treestr

class Node:
    def __init__(self, name=None, parent=None, l=None, r=None, level=None, elev=None,
                 comm=None, t=None, vol=0, current_vol=0, cumulative_vol=0):
        self.name = name
        self.parent = parent
        self.l = l
        self.r = r
        self.t = t
        self.elev = elev
        self.comm = comm
        self.level = level
        self.vol = vol
        self.current_vol = current_vol
        self.cumulative_vol = cumulative_vol
