import copy
import networkx as nx
from multiprocessing import Manager, Lock
from multiprocessing_logging import install_mp_handler
import numba as nb
import numpy as np
import pymap3d as pm
from shapely.geometry import LineString

import logging

LOGGER = logging.getLogger(__name__)
install_mp_handler(LOGGER)


class GraphDict:
    """
    Dict-like class that stores all its data in the external datasource.
    """
    def __init__(self, item_id, mgr, subgraphs):
        """
        Args:
            item_id (str, int, tuple): node/edge id
            mgr (multiprocessing.Manager): Manager object
            subgraphs(dict): dict-like object that contains all sub graph details
        """
        self._id = item_id
        self._subs = subgraphs
        self._gids = mgr.list()

        if isinstance(self._id, tuple):
            self._type = 'edges'
        elif isinstance(self._id, (str, int)):
            self._type = 'nodes'
        else:
            raise TypeError('item_id should be string or integer for nodes and tuple for edges')

    def _is_node(self):
        return self._type == "nodes"

    def __setitem__(self, k, v):
        # Special case: it's necessary to re-define and update sub graph ids which this item belongs to.
        # NOTE: as we need to know two coordinates to define sub graph id,
        # sub graph id definition for the node happens in 'update' method only.
        if k == 'geometry':
            new_subgraph_ids = self.get_subgraph_ids(v.coords)
            # Sub graphs that no longer contain this edge
            orphaned_ids = set(self._gids).difference(set(new_subgraph_ids))
            for gid in orphaned_ids:
                G = self._get_subgraph(gid)
                G.remove_edge(*self._id)
                self._subs[gid] = G

            self._gids[:] = new_subgraph_ids

        for gid in self._gids:
            G = self._get_subgraph(gid)
            # Equivalent to for example G.edges[<edge_id>].update({<attrs>})
            getattr(G, self._type)[self._id].update({k: v})
            self._subs[gid] = G

    def __getitem__(self, k):
        if self._gids:
            G = self._get_subgraph(self._gids[0])
            return getattr(G, self._type)[self._id][k]
        else:
            raise KeyError(k)

    def __delitem__(self, key):
        for gid in self._gids:
            G = self._get_subgraph(gid)
            del getattr(G, self._type)[self._id][key]
            self._subs[gid] = G

    def __contains__(self, k):
        if self._gids:
            G = self._get_subgraph(self._gids[0])
            return k in getattr(G, self._type)[self._id]
        else:
            return False

    def __iter__(self):
        if self._gids:
            G = self._get_subgraph(self._gids[0])
            return iter(getattr(G, self._type)[self._id])
        else:
            return iter([])

    def __len__(self):
        if self._gids:
            G = self._get_subgraph(self._gids[0])
            return len(getattr(G, self._type)[self._id])
        else:
            return 0

    def __str__(self):
        if self._gids:
            G = self._get_subgraph(self._gids[0])
            return str(getattr(G, self._type)[self._id])
        else:
            return str(dict())

    def __repr__(self):
        if self._gids:
            G = self._get_subgraph(self._gids[0])
            return repr(getattr(G, self._type)[self._id])
        else:
            return repr(dict())

    def _get_subgraph(self, gid):
        if gid not in self._subs:
            self._subs[gid] = nx.MultiDiGraph()
        return self._subs[gid]

    def get(self, k, v=None):
        G = self._get_subgraph(self._gids[0])
        return getattr(G, self._type)[self._id].get(k, v)

    def copy(self):
        G = self._get_subgraph(self._gids[0])
        return copy.deepcopy(getattr(G, self._type)[self._id])

    def items(self):
        G = self._get_subgraph(self._gids[0])
        return getattr(G, self._type)[self._id].items()

    def keys(self):
        G = self._get_subgraph(self._gids[0])
        return getattr(G, self._type)[self._id].keys()

    def values(self):
        G = self._get_subgraph(self._gids[0])
        return getattr(G, self._type)[self._id].values()

    def update(self, dct, **kwargs):
        # Special case: it's necessary to re-define and update sub graph ids which this item belongs to.
        new_subgraph_ids = []
        if 'geometry' in dct:
            new_subgraph_ids = self.get_subgraph_ids(dct['geometry'].coords)
        elif '_geometry' in dct:
            new_subgraph_ids = self.get_subgraph_ids(dct['_geometry'].coords)
        elif 'x' in dct and 'y' in dct:
            new_subgraph_ids = self.get_subgraph_ids([(dct['x'], dct['y'])])

        # Obtaining actual item attributes dict (old and those that are in kwargs)
        if self._gids:
            G = self._get_subgraph(self._gids[0])
            item_data = getattr(G, self._type)[self._id]
            item_data.update(dct)
        else:
            item_data = dct

        if new_subgraph_ids:
            # Sub graphs that no longer contain this item
            orphaned_ids = set(self._gids).difference(set(new_subgraph_ids))

            for gid in orphaned_ids:
                G = self._get_subgraph(gid)
                if self._is_node() and G.has_node(self._id):
                    getattr(G, 'remove_node')(self._id)
                elif not self._is_node() and G.has_edge(*self._id[:2]):
                    getattr(G, 'remove_edge')(*self._id)
                self._subs[gid] = G

            # Update the sub graph ids list
            self._gids[:] = new_subgraph_ids

        # Create/update the item attributes in the sub graphs
        for gid in self._gids:
            G = self._get_subgraph(gid)
            if self._is_node():
                getattr(G, 'add_node')(self._id, **item_data)
            else:
                getattr(G, 'add_edge')(*self._id, **item_data)

            # Creates parameters used in point projection into subgraph
            lane_pts0, lane_pts1, edge_map_idxs, edge_pt_idxs = get_projections_params(G)
            G.graph['lane_pts0'] = lane_pts0
            G.graph['lane_pts1'] = lane_pts1
            G.graph['edge_map_idxs'] = edge_map_idxs
            G.graph['edge_pt_idxs'] = edge_pt_idxs

            self._subs[gid] = G

    def pop(self, k):
        for gid in self._gids:
            G = self._get_subgraph(gid)
            result = getattr(G, self._type)[self._id].pop(k)
            self._subs[gid] = G
        return result

    def clear(self):
        for gid in self._gids:
            G = self._get_subgraph(gid)
            getattr(G, self._type)[self._id].clear()
            self._subs[gid] = G

    def get_subgraph_ids(self, pts):

        """
        Finds 'n' nearest projections of points list to the graph

        Arguments:
            pts {iterable} -- points coordinates for projection

        Returns:
            results {list} -- list of geo hashes
        """

        x_res = self._subs['x_res']
        y_res = self._subs['y_res']
        pad = self._subs['pad']
        min_pt = (pts // np.array((x_res, y_res))).min(axis=0).astype(np.int)
        max_pt = (pts // np.array((x_res, y_res))).max(axis=0).astype(np.int) + 1

        bboxes = []
        hashes = []

        for x in range(min_pt[0], max_pt[0]):
            for y in range(min_pt[1], max_pt[1]):
                bbox = ((x_res * (x) - pad, y_res * (y) - pad),
                        (x_res * (x + 1) + pad, y_res * (y + 1) + pad))
                bboxes.append(bbox)
                hashes.append((x, y))

        bboxes = np.array(bboxes)
        hashes = np.array(hashes)

        results = []

        for pt in pts:
            mask = ((pt >= bboxes[:, 0]) & (pt < bboxes[:, 1])).all(axis=1)
            results.extend(hashes[mask])
        results = [f'MAP_GRAPH:{coord[0]},{coord[1]}' for coord in results]

        return results


class OSMGraph(nx.MultiDiGraph):
    def __init__(self, G, geo_convert=False, x_res=0.005, y_res=0.005, pad=0.0015):

        """
        Adapts osmnx graph structure and functions to numpy and numba arrays and operations

        Arguments:
            G {MultiDiGraph} -- road graph, which contains nodes and edges to keep in the wrapper object
            ref_lat, ref_lon {float} -- origin coordinates for the projecting (default 0, 0)
            geo_convert {bool} -- if to convert graph edges coordinates to local coordinate-system
            x_res {float} -- longitude resolution of geo hashing
            y_res {float} -- latitude resolution of geo hashing
            pad {float} -- padding used for longitude and latitude hashing
        """
        self.mgr = Manager()
        self.node_attr_dict_factory = GraphDict
        self.edge_attr_dict_factory = GraphDict
        self.node_dict_factory = self.mgr.dict
        self.edge_key_dict_factory = self.mgr.dict
        self.adjlist_outer_dict_factory = self.mgr.dict
        self.adjlist_inner_dict_factory = self.mgr.dict
        # Graph attributes are stored within the object
        self.graph_attr_dict_factory = self.mgr.dict

        self.geo_origin = (0, 0)
        nx.MultiDiGraph.__init__(self)
        self.graph['crs'] = 'epsg:4326'  # WGS84
        self._subgraphs = self.mgr.dict()
        self._subgraphs.update({'x_res': x_res,
                                'y_res': y_res,
                                'pad': pad})
        self.geo_convert = geo_convert
        self._lck = Lock()

        # Adding nodes from the source graph
        for n_id in G.nodes():
            self.add_node(n_id, **G.nodes[n_id])

        # Adding edges from the source graph
        # Need to use keys=True because __get_item__ used for e_id unpacking gives an error otherwise
        for e_id in G.edges(keys=True):
            self.add_edge(*e_id, **G.edges[e_id])

    def neighbors(self, n):
        try:
            with self.lock:
                return iter(self._adj[n].keys())
        except KeyError as e:
            raise nx.NetworkXError(f"The node {n} is not in the graph.") from e

    def adjacency(self):
        with self.lock:
            return iter(self._adj.items())

    def nbunch_iter(self, nbunch=None):
        if nbunch is None:  # include all nodes via iterator
            with self.lock:
                bunch = iter(self._adj.keys())
        elif nbunch in self:  # if nbunch is a single node
            bunch = iter([nbunch])
        else:  # if nbunch is a sequence of nodes

            def bunch_iter(nlist, adj):
                try:
                    for n in nlist:
                        if n in adj:
                            yield n
                except TypeError as e:
                    message = e.args[0]
                    # capture error for non-sequence/iterator nbunch.
                    if "iter" in message:
                        msg = "nbunch is not a node or a sequence of nodes."
                        raise nx.NetworkXError(msg) from e
                    # capture error for unhashable node.
                    elif "hashable" in message:
                        msg = f"Node {n} in sequence nbunch is not a valid node."
                        raise nx.NetworkXError(msg) from e
                    else:
                        raise

            bunch = bunch_iter(nbunch, self._adj)
        return bunch

    @property
    def lock(self):
        return self._lck

    def new_edge_key(self, u, v):
        try:
            keydict = self._adj[u][v]
        except KeyError:
            return '0'
        key = len(keydict)
        while key in keydict:
            key += 1
        return str(key)

    def update_graph_from_map(self, lane_groups):
        for group in lane_groups.values():
            gid = group['gid']
            # First, remove the existing edges with the same
            # lane group ID
            edges_to_be_removed = [(u, v)
                                   for u, v, e in self.edges(data=True)
                                   if e['lane_group'] == gid]
            self.remove_edges_from(edges_to_be_removed)

            # Add nodes and edges for each lane
            for lane in group['lanes']:
                lane_geom = np.flip(lane['path'], axis=1)
                start_id = f'{lane_geom[0][0]}_{lane_geom[0][1]}'
                end_id = f'{lane_geom[-1][0]}_{lane_geom[-1][1]}'
                self.add_node(start_id, x=lane_geom[0][0], y=lane_geom[0][1])
                self.add_node(end_id, x=lane_geom[-1][0], y=lane_geom[-1][1])
                self.add_edge(start_id, end_id, '0',
                              geometry=LineString(lane_geom),
                              lane_id=lane['lid'], lane_group=gid,
                              signal_group_ids=list(lane['signal_groups'].keys()))
        # Remove all the isolated nodes
        self.remove_nodes_from(list(nx.isolates(self)))

    def _check_coordinates(self, attr):
        assert 'x' in attr and 'y' in attr, 'Node should have "x" and "y" attributes set to be added.'

        # Projecting the coordinates if necessary
        if self.geo_convert:
            attr['x'], attr['y'], _ = pm.geodetic2enu(attr['y'], attr['x'], 0, *self.geo_origin, 0)

    def _create_node(self, n, attr):
        self._succ[n] = self.adjlist_inner_dict_factory()
        self._pred[n] = self.adjlist_inner_dict_factory()
        attr_dict = self._node[n] = self.node_attr_dict_factory(n, self.mgr, self._subgraphs)
        attr_dict.update(attr)

    def add_node(self, node_for_adding, **attr):
        """
        Adds a node to the graph. Projects its coordinate if needed.

        Arguments:
            node_for_adding {str} -- id of the node
        """
        self._check_coordinates(attr)
        with self.lock:
            if node_for_adding not in self._succ:
                self._create_node(node_for_adding, attr)
            else:  # update attr even if node already exists
                self._node[node_for_adding].update(attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for n in nodes_for_adding:
            # keep all this inside try/except because
            # CPython throws TypeError on n not in self._succ,
            # while pre-2.7.5 ironpython throws on self._succ[n]
            with self.lock:
                try:
                    if n not in self._succ:
                        self._check_coordinates(attr)
                        self._create_node(n, attr)
                    else:
                        self._node[n].update(attr)
                except TypeError:
                    nn, ndict = n
                    newdict = attr.copy()
                    newdict.update(ndict)
                    if nn not in self._succ:
                        self._check_coordinates(newdict)
                        self._create_node(nn, newdict)
                    else:
                        self._node[nn].update(newdict)

    def remove_node(self, n):
        with self.lock:
            try:
                nbrs = self._succ[n]
                del self._node[n]
            except KeyError as e:  # NetworkXError if n not in self
                raise nx.NetworkXError(f"The node {n} is not in the digraph.") from e
            for u in nbrs.keys():
                del self._pred[u][n]  # remove all edges n-u in digraph
            del self._succ[n]  # remove node from succ
            for u in self._pred[n].keys():
                del self._succ[u][n]  # remove all edges n-u in digraph
            del self._pred[n]  # remove node from pred

    def remove_nodes_from(self, nodes):
        for n in nodes:
            with self.lock:
                try:
                    succs = self._succ[n]
                    del self._node[n]
                    for u in succs.keys():
                        del self._pred[u][n]  # remove all edges n-u in digraph
                    del self._succ[n]  # now remove node
                    for u in self._pred[n].keys():
                        del self._succ[u][n]  # remove all edges n-u in digraph
                    del self._pred[n]  # now remove node
                except KeyError:
                    pass  # silent failure on remove

    def add_edge(self, u, v, key=None, **attr):
        """
        Adds an edge to the graph. Projects its geometry if needed.
        Adds the additional attributes.

        Arguments:
            u, v, key {str} -- id of the edge
        """

        # Projecting edge's geometry if needed
        if self.geo_convert and 'geometry' in attr:
            # Simplified graph
            points_geo = np.array([[y, x] for x, y in attr['geometry'].coords])
            geo = np.array(pm.geodetic2enu(points_geo[:, 0], points_geo[:, 1], 0, *self.geo_origin, 0)[:2]).T
            attr['geometry'] = LineString(geo)

        with self.lock:
            if 'geometry' in attr:
                geometry = attr['geometry'].coords
            else:
                geometry = [(self._node[u]['x'], self._node[u]['y']),
                            (self._node[v]['x'], self._node[v]['y'])]
                attr['_geometry'] = LineString(geometry)

            # add nodes
            if u not in self._succ:
                self._succ[u] = self.adjlist_inner_dict_factory()
                self._pred[u] = self.adjlist_inner_dict_factory()
                self._node[u] = self.node_attr_dict_factory(u, self.mgr, self._subgraphs)
                x, y = geometry[0]
                self._node[u].update({'x': x, 'y': y})
            if v not in self._succ:
                self._succ[v] = self.adjlist_inner_dict_factory()
                self._pred[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory(v, self.mgr, self._subgraphs)
                x, y = geometry[-1]
                self._node[v].update({'x': x, 'y': y})
            if key is None:
                key = self.new_edge_key(u, v)
            if v in self._succ[u]:
                keydict = self._adj[u][v]
                datadict = keydict.get(key, self.edge_attr_dict_factory((u, v, key), self.mgr, self._subgraphs))
                datadict.update(attr)
                keydict[key] = datadict
            else:
                # selfloops work this way without special treatment
                datadict = self.edge_attr_dict_factory((u, v, key), self.mgr, self._subgraphs)
                datadict.update(attr)
                keydict = self.edge_key_dict_factory()
                keydict[key] = datadict
                self._succ[u][v] = keydict
                self._pred[v][u] = keydict
            return key

    def has_edge(self, u, v, key=None):
        try:
            with self.lock:
                if key is None:
                    return v in self._adj[u]
                else:
                    return key in self._adj[u][v]
        except KeyError:
            return False

    def get_edge_data(self, u, v, key=None, default=None):
        try:
            with self.lock:
                if key is None:
                    return self._adj[u][v]
                else:
                    return self._adj[u][v][key]
        except KeyError:
            return default

    def remove_edge(self, u, v, key=None):
        try:
            d = self._adj[u][v]
        except KeyError as e:
            raise nx.NetworkXError(f"The edge {u}-{v} is not in the graph.") from e
        # remove the edge with specified data
        if key is None:
            with self.lock:
                d.popitem()
        else:
            try:
                with self.lock:
                    del d[key]
            except KeyError as e:
                msg = f"The edge {u}-{v} with key {key} is not in the graph."
                raise nx.NetworkXError(msg) from e
        if len(d) == 0:
            # remove the key entries if last edge
            with self.lock:
                del self._succ[u][v]
                del self._pred[v][u]

    def clear_edges(self):
        with self.lock:
            for predecessor_dict in self._pred.values():
                predecessor_dict.clear()
            for successor_dict in self._succ.values():
                successor_dict.clear()

    def clear(self):
        with self.lock:
            self._succ.clear()
            self._pred.clear()
            self._node.clear()
            self.graph.clear()

    def copy(self, as_view=False):
        if as_view is True:
            return nx.graphviews.generic_graph_view(self)
        G = self.__class__(nx.MultiDiGraph(), geo_convert=self.geo_convert)
        G.graph.update(self.graph)
        G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, key, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, keydict in nbrs.items()
            for key, datadict in keydict.items()
        )
        return G

    def has_successor(self, u, v):
        with self.lock:
            return u in self._succ and v in self._succ[u]

    def has_predecessor(self, u, v):
        with self.lock:
            return u in self._pred and v in self._pred[u]

    def _project_point(self, pt, n):

        """
        Finds 'n' nearest projections of the point to the graph

        Arguments:
            pt {np.array} -- point coordinates for projection, in [lon, lat] order
            n {int} -- number of nearest projections to return for given point
            geo {bool} -- if geo, 'pts' treated as geo coordinates and projected to local coordinate system

        Returns:
            projected_pts {np.array} -- array of n-closest graph points
            edge_idxs {list} -- list of the closest edges idxs
            distances {np.array} -- distances to projection
        """

        gid = self.get_subgraph_id(pt)
        G = self._subgraphs.get(gid, None)

        if G is None:
            # Sub graph doesn't exist
            return None, None, None

        pt = np.array(pt, dtype=np.float32)

        # finds projections and distances to all edges
        projected_pts, distances = project_numba(G.graph['lane_pts0'],
                                                 G.graph['lane_pts1'],
                                                 pt)

        # finds 'n' nearest projections
        try:
            min_n_idxs = np.argpartition(distances, n)[:n]

        except ValueError:
            n = len(distances) - 1
            min_n_idxs = np.argpartition(distances, n)[:n]
            LOGGER.warning("number of n nearest projections exceeded number of edges")

        edge_idxs = [G.graph['edge_map_idxs'][idx] for idx in min_n_idxs]
        projected_pts = projected_pts[min_n_idxs]
        distances = distances[min_n_idxs]

        return projected_pts, edge_idxs, distances

    def project_points(self, pts, n):

        """
        Finds 'n' nearest projections of points list to the graph

        Arguments:
            pts {iterable} -- points coordinates for projection
            n {int} -- number of nearest projections to return for given point
            geo {bool} -- if geo, 'pts' treated as geo coordinates and projected to local coordinate system

        Returns:
            projected_pts_list {list} -- list of arrays of n the closest graph points
            edge_idxs {list} -- list of lists of the closest edges idxs
            distances {list} -- of arrays of distances to projection
        """

        projected_pts_list = []
        edge_idxs_list = []
        distances_list = []

        # iterates over all points that need to be projected
        for pt in pts:
            projected_pts, edge_idxs, distances = self._project_point(pt, n)

            projected_pts_list.append(projected_pts)
            edge_idxs_list.append(edge_idxs)
            distances_list.append(distances)

        return projected_pts_list, edge_idxs_list, distances_list

    def get_subgraph_id(self, pt):

        """
        Gets subgraph geo hash based on pt geo coordinates

        Arguments:
            pt {np.array} -- point coordinates for projection, in [lon, lat] order

        Returns:
            idx {str} -- custom geo hash id
        """

        coord = (pt // np.array((self._subgraphs['x_res'], self._subgraphs['y_res']))).astype(np.int)
        idx = f'MAP_GRAPH:{coord[0]},{coord[1]}'
        return idx

    def get_origin(self, pt):
        """
        Returns geo origin for the subgraph by the given coordinates.
        Args:
            pt(Iterable): point coordinates for projection, in [lon, lat]
        """
        graph_id = self.get_subgraph_id(pt)
        try:
            return self._subgraphs[graph_id].graph['origin']
        except KeyError:
            return None


@nb.njit((nb.types.Array(nb.types.float32, 2, 'C'),
          nb.types.Array(nb.types.float32, 2, 'C'),
          nb.types.Array(nb.types.float32, 1, 'C')
          ))
def project_numba(p0, p1, p_proj):
    """
    Finds the closest point to graph

    Arguments:
        p0 {np.array} -- array of first points on graph edges, shape: [n, 2]
        p0 {np.array} -- array of second point on graph edges, shape: [n, 2]
        p_proj {np.array} -- point to be projected on graph edges, shape: [2,]

    Returns:
        closest_pts {np.array} -- closest points to each graph edge
        distances {np.array} -- distances between the closest points on edges and point to project
    """

    # finds vectors
    a = p_proj - p0
    b = p1 - p0
    # finds projection using non-canonical line equation and directional vectors
    p_projected = b * np.expand_dims((a * b).sum(axis=-1) / (b * b).sum(axis=-1), axis=1) + p0
    # quadratic distance between edge points
    l0 = ((b) ** 2).sum(axis=-1)
    # quadratic distance between each edge point and projected point
    l1 = ((p0 - p_projected) ** 2).sum(axis=-1)
    l2 = ((p1 - p_projected) ** 2).sum(axis=-1)
    # uses mask to find the closest point to graph: projection or one of edge point
    mask0 = np.expand_dims((l0 > l1) & (l0 > l2), axis=1)

    closest_pts = mask0 * p_projected + \
                  ~mask0 * np.expand_dims(l1 > l2, axis=1) * p1 + \
                  ~mask0 * np.expand_dims(l2 > l1, axis=1) * p0
    # distances between the closest points on edges and point to project
    distances = np.sqrt(((p_proj - closest_pts) ** 2).sum(axis=-1))

    return closest_pts, distances


def get_projections_params(graph):
    """
    Computes parameters for projection points to the graph

    Returns:
        lane_pts0 {np.array} -- array of first points in edge (n, 2)
        lane_pts1 {np.array} -- array of last points in edge (n, 2)
        edge_map_idxs {list} -- list of edges indices
        edge_pt_idxs {list} -- list of consecutive points idxs
        geo_origin {tuple} -- origin used to convert geo coordinates to local coordinate system
    """

    lane_pts0 = []
    lane_pts1 = []
    edge_map_idxs = []
    edge_pt_idxs = []

    for edgge_key in graph.edges:

        edge = graph.edges[edgge_key]

        if '_geometry' in edge:
            coords = np.array(edge['_geometry'].coords, dtype=np.float32)

        else:
            coords = np.array(edge['geometry'].coords, dtype=np.float32)

        lane_pts0.append(coords[1:])
        lane_pts1.append(coords[:-1])
        edge_map_idxs += [edgge_key for i in range(coords.shape[0] - 1)]
        edge_pt_idxs += [i for i in range(coords.shape[0] - 1)]

    if len(lane_pts0):
        lane_pts0 = np.vstack(lane_pts0)
        lane_pts1 = np.vstack(lane_pts1)
        edge_pt_idxs = np.array(edge_pt_idxs)

        return lane_pts0, lane_pts1, edge_map_idxs, edge_pt_idxs

    else:
        return None, None, None, None
