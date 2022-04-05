import networkx as nx
import numba as nb
import numpy as np
import pymap3d as pm
import warnings
from shapely.geometry import LineString


class OSMGraph(nx.MultiDiGraph):

    def __init__(self, G, n_div, patch_padding, ref_lat=0, ref_lon=0, geo_convert=False):

        """
        Adapts osmnx graph structure and functions to numpy and numba arrays and operations

        Arguments:
            G {MultiDiGraph} -- road graph, which contains nodes and edges to keep in the wrapper object
            n_div {int} -- number of divisions of bbox
            patch_padding {float} -- padding for graph patches in meters
            ref_lat, ref_lon {float} -- origin coordinates for the projecting (default 0, 0)
            geo_convert {bool} -- if to convert graph edges coordinates to local coordinate-system
        """

        nx.MultiDiGraph.__init__(self)
        self.graph['crs'] = 'epsg:2223'     # Default EPSG code as the graph will not use it anyhow
        self._n_div = n_div
        self._patch_padding = patch_padding
        self.geo_origin = ref_lat, ref_lon
        self.geo_convert = geo_convert

        # Adding nodes from the source graph
        for n_id in G.nodes():
            self.add_node(n_id, **G.nodes[n_id])

        # Adding edges from the source graph
        # Need to use keys=True because __get_item__ used for e_id unpacking gives an error otherwise
        for e_id in G.edges(keys=True):
            self.add_edge(e_id, **G.edges[e_id])

        self.nodesIDs = list(G.nodes())
        self.neighbors = dict(zip(self.nodesIDs, [list(G.successors(nodeID)) for nodeID in self.nodesIDs]))

        # Checking if our object contains any edges and needs to be split to areas
        if len(self.edges()) > 0:
            self.divide_area()

    def add_node(self, node_id, **kwargs):
        """
        Adds a node to the graph. Projects its coordinate if needed.

        Arguments:
            node_id {str} -- id of the node
        """
        super(OSMGraph, self).add_node(node_id, **kwargs)
        node = self.nodes[node_id]

        # Setting the geo_origin if it's not set
        if self.geo_origin == (0, 0) and ('x' in node and 'y' in node):
            self.geo_origin = (node['y'], node['x'])

        # Projecting the coordinates if necessary
        if self.geo_convert:
            node['x'], node['y'], _ = pm.geodetic2enu(node['y'], node['x'], 0, *self.geo_origin, 0)

    def add_edge(self, u, v, k=None, **kwargs):
        """
        Adds an edge to the graph. Projects its geometry if needed.
        Adds the additional attributes.

        Arguments:
            u, v, k {str} -- id of the edge
        """

        if k is None:
            k = (max(np.array(self.edges(keys=True))[:, 2]) + 1) if (u, v) in self.edges() else 0

        super(OSMGraph, self).add_edge(u, v, k, **kwargs)

        node1 = self.nodes[u]
        node2 = self.nodes[v]
        edge = self.edges[(u, v, k)]

        # Projecting edge's geometry if needed
        if self.geo_convert and 'geometry' in edge:
            # Simplified graph
            points_geo = np.array([[y, x] for x, y in edge['geometry'].coords])
            geo = np.array(pm.geodetic2enu(points_geo[:, 0], points_geo[:, 1], 0, *self.geo_origin, 0)[:2]).T
            edge['geometry'] = LineString(geo)

        coords = np.array([[node1['x'], node1['y']], [node2['x'], node2['y']]], dtype=np.float32)
        self.edges[(u, v, k)]['length'] = self._get_length(coords)
        self.edges[(u, v, k)]['coords'] = coords

    @staticmethod
    def _get_length(arr):
        """
        Computes length of edge

        Arguments:
            arr {np.array} -- array of edge coordinates

        Returns:
            length of edge {float}
        """

        return np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(axis=-1)).sum()

    def _get_projections_params(self):

        """
        Computes parameters for projection points to the graph

        Returns:
            lane_pts0 {np.array} -- array of first points in edge (n, 2)
            lane_pts1 {np.array} -- array of last points in edge (n, 2)
            edge_map_idxs {list} -- list of edges indices
            edge_pt_idxs {list} -- list of consecutive points idxs

        """

        lane_pts0 = []
        lane_pts1 = []
        edge_map_idxs = []
        edge_pt_idxs = []

        for edgge_key in self.edges:
            edge = self.edges[edgge_key]

            lane_pts0.append(edge['coords'][1:])
            lane_pts1.append(edge['coords'][:-1])
            edge_map_idxs += [edgge_key for i in range(edge['coords'].shape[0] - 1)]
            edge_pt_idxs += [i for i in range(edge['coords'].shape[0] - 1)]

        lane_pts0 = np.vstack(lane_pts0)
        lane_pts1 = np.vstack(lane_pts1)
        edge_pt_idxs = np.array(edge_pt_idxs)

        return lane_pts0, lane_pts1, edge_map_idxs, edge_pt_idxs

    def _project_point(self, pt, n):

        """
        Finds 'n' nearest projections of the point to the graph

        Arguments:
            pt {np.array} -- point coordinates for projection
            n {int} -- number of nearest projections to return for given point

        Returns:
            projected_pts {np.array} -- array of n closest graph points
            edge_idxs {list} -- list of closest edges idxs
            distances {np.array} -- distances to projection
        """

        # finds which bbox id corresponds to the given point
        rect_mask = in_rect(self.bboxes[:, 0], self.bboxes[:, 1], pt)
        # if pt in at list one bbox we continue projection
        if np.any(rect_mask):

            bbox_key = np.argmax(rect_mask)

            # finds projections and distances to all edges
            projected_pts, distances = project_numba(self.proj_params[bbox_key]['lane_pts0'],
                                                     self.proj_params[bbox_key]['lane_pts1'],
                                                     pt)

            # finds 'n' nearest projections
            try:
                min_n_idxs = np.argpartition(distances, n)[:n]

            except ValueError:
                n = len(distances) - 1
                min_n_idxs = np.argpartition(distances, n)[:n]
                warnings.warn("number of n nearest projections exceeded number of edges")

            edge_idxs = [self.proj_params[bbox_key]['edge_map_idxs'][idx] for idx in min_n_idxs]
            projected_pts = projected_pts[min_n_idxs]
            distances = distances[min_n_idxs]
        # if pt out of all bbox returns None
        else:
            projected_pts = None
            edge_idxs = None
            distances = None

        return projected_pts, edge_idxs, distances

    def project_points(self, pts, n, geo=False):

        """
        Finds 'n' nearest projections of points list to the graph

        Arguments:
            pts {iterable} -- points coordinates for projection
            n {int} -- number of nearest projections to return for given point
            geo {bool} -- if geo, 'pts' treated as geo coordinates and projected to local coordinate system

        Returns:
            projected_pts_list {list} -- list of arrays of n closest graph points
            edge_idxs {list} -- list of lists of the closest edges idxs
            distances {list} -- of arrays of distances to projection
        """

        projected_pts_list = []
        edge_idxs_list = []
        distances_list = []

        # iterates over all points that need to be projected
        for pt in pts:

            # converts from geo to local coordinate system
            if geo:
                x, y, _ = pm.geodetic2enu(*pt, 0, *self.geo_origin, 0)
                pt = np.array([x, y], dtype=np.float32)
            else:
                pt = np.array(list(reversed(pt)), dtype=np.float32)

            projected_pts, edge_idxs, distances = self._project_point(pt, n)

            projected_pts_list.append(projected_pts)
            edge_idxs_list.append(edge_idxs)
            distances_list.append(distances)

        return projected_pts_list, edge_idxs_list, distances_list

    def divide_area(self):

        """
        Divides drivable area to patches (sub-graphs). Each patch matches graph points within it.
        Used to speed-up matching on graph.
        """

        lane_pts0, lane_pts1, edge_map_idxs, edge_pt_idxs = self._get_projections_params()

        # bounds of patch
        min_pt = np.min([lane_pts0.min(axis=0), lane_pts1.min(axis=0)], axis=0)  # - 10
        max_pt = np.max([lane_pts0.max(axis=0), lane_pts1.max(axis=0)], axis=0)  # + 10

        self.min_pt = min_pt
        self.max_pt = max_pt

        proj_params = {}

        bboxes = np.array([[min_pt, max_pt]])
        self.outer_bbox = bboxes.copy()

        # divides graph into patches by two along each coordinate alternatively
        for i in range(self._n_div):

            new_bboxes = []
            for bbox in bboxes:
                new_bboxes.append(divide_bbox(bbox[0], bbox[1], coord=i % 2))

            bboxes = np.vstack(new_bboxes)

        bbox_mask = []
        cnt = 0
        for i in range(len(bboxes)):

            # adds padding to patches
            # finds graph point mask for particular patch
            mask0 = in_rect(bboxes[[i], 0] - self._patch_padding, bboxes[[i], 1] + self._patch_padding, lane_pts0)
            mask1 = in_rect(bboxes[[i], 0] - self._patch_padding, bboxes[[i], 1] + self._patch_padding, lane_pts1)

            mask = mask0 | mask1

            if len(edge_pt_idxs[mask]) > 0:

                proj_params[cnt] = {
                    'lane_pts0': lane_pts0[mask],
                    'lane_pts1': lane_pts1[mask],
                    'edge_map_idxs': [k for k, j in zip(edge_map_idxs, mask) if j],
                    'edge_pt_idxs': edge_pt_idxs[mask]
                }

                bbox_mask.append(True)
                cnt += 1

            else:
                bbox_mask.append(False)

        self.proj_params = proj_params
        self.bboxes = bboxes[bbox_mask]


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


# @nb.njit
def in_rect(pt_min, pt_max, pts):
    """
    Checks if pt in bbox

    Arguments:
        min_pt {np.array} -- minimum coordinates of bbox
        max_pt {np.array} -- maximum coordinates of bbox
        pts {np.array} -- points to check

    Returns:
        mask {np.array} -- boolean mask for each point in 'pts'
    """

    x, y = pts.T
    mask = (x >= pt_min[:, 0]) & (x < pt_max[:, 0]) & (y >= pt_min[:, 1]) & (y < pt_max[:, 1])

    return mask


def divide_bbox(min_pt, max_pt, coord=1):
    """
    Divides bbox by two along given coordinate.

    Arguments:
        min_pt {np.array} -- minimum coordinates of bbox
        max_pt {np.array} -- maximum coordinates of bbox
        coord {int} -- if 'coord' equals '0' then bbox would be divided along 'x' axis else -- along 'y' axis

    Returns:
        bbox {np.array} -- array of new bboxes points (min_pt and max_pt)
    """

    if coord == 0:
        middle_x = (max_pt[0] + min_pt[0]) / 2

        bbox = np.array([[min_pt, np.array((middle_x, max_pt[1]))],
                         [np.array((middle_x, min_pt[1])), max_pt]
                         ]).astype(np.float32)

    else:
        middle_y = (max_pt[1] + min_pt[1]) / 2

        bbox = np.array([[min_pt, np.array((max_pt[0], middle_y))],
                         [np.array((min_pt[0], middle_y)), max_pt]
                         ]).astype(np.float32)

    return bbox
