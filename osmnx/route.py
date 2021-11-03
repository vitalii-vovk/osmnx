from tqdm import tqdm
import copy
import itertools

from . import bearing
from . import utils_graph


def get_road_dir(lat1, lon1, lat2, lon2):
    fwd_azimuth_goal = bearing.calculate_bearing(lat1, lon1, lat2, lon2)
    if fwd_azimuth_goal < 0:
        fwd_azimuth_goal += 360
    if fwd_azimuth_goal > 315 or fwd_azimuth_goal <= 45:
        direction = 'north'
    elif 45 < fwd_azimuth_goal <= 135:
        direction = 'east'
    elif 135 < fwd_azimuth_goal <= 225:
        direction = 'south'
    else:
        direction = 'west'
    return direction


def get_way_dir(first_node, last_node, nodes):
    # Find the direction between the first and the last route points
    p1 = nodes[first_node]
    p2 = nodes[last_node]
    direction = get_road_dir(p1['y'], p1['x'], p2['y'], p2['x'])
    return direction


def get_edge_dir(edge):
    # Find the direction between the first and the last route points
    p1 = (edge['geometry'].coords.xy[0][0], edge['geometry'].coords.xy[1][0])
    p2 = (edge['geometry'].coords.xy[0][-1], edge['geometry'].coords.xy[1][-1])
    direction = get_road_dir(p1[1], p1[0], p2[1], p2[0])
    return direction


def add_route_dist(G, inplace=True):
    def process_route(G, start_node_id, route_id):
        node_id = start_node_id
        route_dist = 0
        node = G.nodes[node_id]
        node['route'][route_id]['distance'] = route_dist
        visited = set()
        while True:
            visited.add(node_id)
            next_node_id = node['route'][route_id]['next']
            # Break if route is ended or self-loop
            if next_node_id is None or next_node_id in visited:
                break
            if G.has_edge(node_id, next_node_id, 0):
                edge = G.edges[(node_id, next_node_id, 0)]
                route_dist += edge['length']
            next_node = G.nodes[next_node_id]
            next_node['route'][route_id]['distance'] = route_dist
            node = next_node
            node_id = next_node_id

    if not inplace:
        G = G.copy()

    processed_routes = set()
    for n in G.nodes:
        node = G.nodes[n]
        for rid in node.get('route', {}):
            if rid in processed_routes:
                continue
            process_route(G, node['route'][rid]['start'], rid)
            processed_routes.add(rid)

    return G


def simplify_route(G, node_id, route_id, nodes_to_be_removed, inplace=True):
    """Simplify route by updating start/end/next/prev attributes
    to nodes not in `nodes_to_be_removed`
    CAUTION: route_distance attribute will not be changed!

    Args:
        G (osmnx.Graph): graph to be updated
        node (Any): node id
        route_id (Any): route id
        nodes_to_be_removed (Container): list of all nodes that will be
            removed from the graph
        inplace (bool, optional): If true, then updates the input graph,
            otherwise makes a copy before any updates. Defaults to True.

    Returns:
        osmnx.Graph: updated graph
    """

    if not inplace:
        G = G.copy()

    if not G.has_node(node_id):
        return G

    # We are guaranteed that such a node exists
    node = G.nodes[node_id]
    start_node_id = node['route'][route_id]['start']
    while start_node_id in nodes_to_be_removed:
        start_node_id = G.nodes[start_node_id]['route'][route_id]['next']
    end_node_id = node['route'][route_id]['end']
    while end_node_id in nodes_to_be_removed:
        end_node_id = G.nodes[end_node_id]['route'][route_id]['prev']

    if not (G.has_node(start_node_id) and G.has_node(end_node_id)):
        node['route'][route_id]['start'] = node_id
        node['route'][route_id]['end'] = node_id
        node['route'][route_id]['next'] = None
        node['route'][route_id]['prev'] = None
        return G

    if start_node_id:
        node_id = start_node_id
        node = G.nodes[node_id]
        node['route'][route_id]['prev'] = None
        while node_id:
            node['route'][route_id]['start'] = start_node_id
            node['route'][route_id]['end'] = end_node_id

            next_node_id = node['route'][route_id]['next']
            while next_node_id in nodes_to_be_removed:
                next_node_id = G.nodes[next_node_id]['route'][route_id]['next']
            node['route'][route_id]['next'] = next_node_id

            if next_node_id:
                next_node = G.nodes[next_node_id]
                next_node['route'][route_id]['prev'] = node_id
                node = next_node
            node_id = next_node_id
    return G


def simplify_routes(G, nodes_to_be_removed, inplace=True):
    if not inplace:
        G = G.copy()

    processed = set()
    for n in G.nodes:
        node = G.nodes[n]
        for rid in node.get('route', {}):
            if rid in processed:
                continue
            simplify_route(G, n, rid, nodes_to_be_removed)
            processed.add(rid)

    return G


def get_full_route(G, node_id, route_id):
    route = G.nodes[node_id]['route'][route_id]
    start_id = route['start']

    full_route = {}
    node_id = start_id
    visited = set()
    while True:
        visited.add(node_id)
        route = G.nodes[node_id]['route'][route_id]
        full_route[node_id] = route

        node_id = route['next']
        if node_id is None or node_id in visited:
            break
        route = G.nodes[node_id]['route'][route_id]
    return full_route


def update_truncated_routes(G, nodes_to_be_removed, inplace=True):
    def update_start_end(route, rid, attr, update_value):
        route[rid][attr] = update_value

    gdf = utils_graph.graph_to_gdfs(G)

    for n in tqdm(nodes_to_be_removed, desc='Update truncated routes...'):
        node = G.nodes[n]
        for rid in node.get('route', {}):
            if node['route'][rid]['start'] == n or node['route'][rid]['end'] == n:
                route_gdf = gdf[0]['route'][gdf[0]['route'].apply(lambda x: isinstance(x, dict) and rid in x)]
                if node['route'][rid]['start'] == n:
                    route_gdf.update(route_gdf.apply(update_start_end, args=(rid, 'start', node['route'][rid]['next'])))

                if node['route'][rid]['end'] == n:
                    route_gdf.update(route_gdf.apply(update_start_end, args=(rid, 'end', node['route'][rid]['prev'])))

                gdf[0].update(route_gdf)

    # Updating graph with the altered nodes
    return utils_graph.graph_from_gdfs(*gdf)


def process_route_relation_id(r, relations, ways, parent_id=tuple(), parent_name=tuple()):
    first_node = last_node = None
    if r is None:
        return first_node, last_node

    parent_id_ext = tuple([r['id'], *parent_id])
    parent_name_ext = tuple([r.get('name', ''), *parent_name])

    for m in r['members']:
        if m['type'] == 'relation':
            f_, l_ = process_route_relation_id(
                relations.get(m['ref'], None), relations, ways, parent_id_ext, parent_name_ext)
            if first_node is None:
                first_node = f_
            if l_:
                last_node = l_
        elif m['type'] == 'way':
            way = ways.get(m['ref'], None)
            if way:
                # Extend the route_id
                way['route_id'] = parent_id_ext
                way['route_name'] = parent_name_ext
                if first_node is None:
                    first_node = way['nodes'][0]
                last_node = way['nodes'][-1]
    return first_node, last_node


def process_route_relation_dir(r, relations, ways, nodes, parent_dir=None):
    if r is None:
        return

    rid = r['id']
    route_dir = r['tags'].get('direction', None)
    if route_dir:
        if parent_dir is None:
            parent_dir = {}
        parent_dir[rid] = route_dir
    for m in r['members']:
        pdir = copy.deepcopy(parent_dir)
        if m['type'] == 'relation':
            process_route_relation_dir(
                relations.get(m['ref'], None), relations, ways, nodes, pdir)
        elif m['type'] == 'way':
            way = ways.get(m['ref'], None)
            if way is None:
                continue
            wid = way['osmid']
            # Compute the direction attribute value
            if 'direction' not in way:
                way['direction'] = get_way_dir(way['nodes'][0], way['nodes'][-1], nodes)
            if 'route' not in way:
                way['route'] = {}
            if wid not in way['route']:
                way['route'][wid] = {}
            way['route'][wid]['direction'] = way['direction']
            if pdir:
                for r in pdir:
                    if r not in way['route']:
                        way['route'][r] = {}
                    if 'direction' not in way['route'][r]:
                        way['route'][r]['direction'] = pdir[r]

            for n in way['nodes']:
                node = nodes[n]
                if 'route' not in node:
                    node['route'] = {}
                if wid not in node['route']:
                    node['route'][wid] = {}
                node['route'][wid]['direction'] = way['direction']

                if pdir:
                    for r in pdir:
                        if r not in node['route']:
                            node['route'][r] = {}
                        if 'direction' not in node['route'][r]:
                            node['route'][r]['direction'] = pdir[r]


def update_nodes_with_route_id(route_nodes, rid, nodes, skip_missing: bool = True):
    if skip_missing:
        route_nodes = [n for n in route_nodes if n in nodes]

    processed_nodes = set()
    start_id = route_nodes[0]
    end_id = route_nodes[-1]
    for i, n in enumerate(route_nodes):
        if n in processed_nodes:
            continue

        next_id = route_nodes[i + 1] if i < len(route_nodes) - 1 else None
        prev_id = None if i == 0 else route_nodes[i - 1]

        node = nodes[n]
        if 'route' not in node:
            node['route'] = {}
        if rid not in node['route']:
            node['route'][rid] = {}
        node['route'][rid]['start'] = start_id
        node['route'][rid]['end'] = end_id
        node['route'][rid]['next'] = next_id
        node['route'][rid]['prev'] = prev_id

        # Self-loops
        if next_id in processed_nodes:
            next_node = nodes[next_id]
            next_node['route'][rid]['prev'] = n
        processed_nodes.add(n)


def process_route_links(
    r,
    relations,
    ways,
    nodes,
    parent_route_id=tuple(),
    parent_route_name=tuple(),
    parent_route_alt_name=tuple(),
    parent_route_off_name=tuple(),
    parent_route_ref=tuple(),
    skip_missing: bool = True
):
    """Updates route links withi the given relation `r`
    and returs the list of ways in this relation

    Args:
        r (dict): the current relation item
        relations (dict): dictionary of all relations
        ways (dict): dictionary of all ways
        parent_route_id (tuple, optional): route id of the up-level route.
            Defaults to tuple()
        parent_route_name (tuple, optional): route name of the up-level route.
            Defaults to tuple()
        drop_missing (bool, optional): if True, then skip nodes that are
            missing in the `nodes`

    Returns:
        list: ways in the given relation `r`
    """

    if r is None:
        return []

    route_nodes = []
    rid = r['id']
    rname = r['tags'].get('name', '').lower()
    raltname = r['tags'].get('alt_name', '').lower()
    roffname = r['tags'].get('official_name', '').lower()
    rref = str(r['tags'].get('ref', '')).lower()
    for m in r['members']:
        route_id = (rid, *parent_route_id)
        route_name = (rname, *parent_route_name)
        route_aname = (raltname, *parent_route_alt_name)
        route_oname = (roffname, *parent_route_off_name)
        route_ref = (rref, *parent_route_ref)
        if m['type'] == 'relation':
            route_nodes.extend(
                process_route_links(
                    relations.get(m['ref'], None),
                    relations, ways, nodes,
                    route_id,
                    route_name, route_aname, route_oname,
                    route_ref
                )
            )
        elif m['type'] == 'way':
            way = ways.get(m['ref'], None)
            if not way:
                continue
            way_nodes = way['nodes']
            route_nodes.extend(way_nodes)

            wid = way['osmid']
            wname = way.get('name', '').lower()
            waname = way.get('alt_name', '').lower()
            woname = way.get('official_name', '').lower()
            wref = str(way.get('ref', '')).lower()

            way_route_id = way.get('route_id', (wid, ))
            way_route_name = way.get('route_name', (wname, ))
            way_route_aname = way.get('route_alt_name', (waname, ))
            way_route_oname = way.get('route_off_name', (woname, ))
            way_route_ref = way.get('route_ref', (wref, ))

            way['route_id'] = (*way_route_id, *route_id)
            way['route_name'] = (*way_route_name, *route_name)
            way['route_alt_name'] = (*way_route_aname, *route_aname)
            way['route_off_name'] = (*way_route_oname, *route_oname)
            way['route_ref'] = (*way_route_ref, *route_ref)

            update_nodes_with_route_id(way_nodes, wid, nodes, skip_missing)

    # Remove duplicate nodes (at ways/relations joints)
    route_nodes = list(
        itertools.compress(
            route_nodes,
            [1] + list(map(lambda x: x[0] != x[1], zip(route_nodes[:-1], route_nodes[1:])))
        )
    )
    update_nodes_with_route_id(route_nodes, rid, nodes)

    return route_nodes


def init_route_data(p, paths, nodes, skip_missing: bool = True):
    way = paths[p]
    if 'route_id' in way:
        return
    wid = way['osmid']
    way['route_id'] = (wid,)
    way['route_name'] = (way.get('name', '').lower(),)
    way['route_alt_name'] = (way.get('alt_name', '').lower(),)
    way['route_off_name'] = (way.get('official_name', '').lower(),)
    way['route_ref'] = (str(way.get('ref', '')).lower(),)

    wnodes = way.get('nodes', None)
    if nodes:
        first_node, last_node = wnodes[0], wnodes[-1]
        way['route'] = {
            wid: {
                'direction': get_way_dir(first_node, last_node, nodes)
            }
        }
    update_nodes_with_route_id(way['nodes'], wid, nodes, skip_missing)
