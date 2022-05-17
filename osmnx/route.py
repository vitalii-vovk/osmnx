import copy
import itertools

from . import bearing


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


def set_route_dist(G, route_id, start_id, start_dist=0):
    """
    Sets 'distance' attribute for every node along the route.
    Args:
        G (MultiDiGraph): road graph
        route_id (Any): route id
        start_id (Any): the first node id that is about to be processed
        start_dist (float): route distance till the node with the start_id
    """
    prev_id = None
    cur_id = start_id
    route_dist = start_dist
    node = G.nodes[cur_id]
    node['route'][route_id]['distance'] = route_dist
    next_id = node['route'][route_id]['neighbours'][prev_id]
    while True:
        # Reached the route's end
        if next_id is None:
            break

        if G.has_edge(cur_id, next_id, 0):
            edge = G.edges[(cur_id, next_id, 0)]
            route_dist += edge['length']

        node = G.nodes[next_id]
        node['route'][route_id]['distance'] = route_dist
        prev_id, cur_id, next_id = cur_id, next_id, node['route'][route_id]['neighbours'][cur_id]


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


def update_nodes_with_route_id(G, route_nodes, rid):
    start_id = route_nodes[0]
    end_id = route_nodes[-1]
    for i, n in enumerate(route_nodes):
        next_id = route_nodes[i + 1] if i < len(route_nodes) - 1 else None
        prev_id = None if i == 0 else route_nodes[i - 1]

        node = G.nodes[n]
        if 'route' not in node:
            node['route'] = {}
        if rid not in node['route']:
            node['route'][rid] = {}
        node['route'][rid]['start'] = start_id
        node['route'][rid]['end'] = end_id
        if 'neighbours' not in node['route'][rid]:
            node['route'][rid]['neighbours'] = {}
        node['route'][rid]['neighbours'][prev_id] = next_id


def process_route_links(
    r,
    relations,
    ways,
    nodes,
    rels,
    parent_route_id=tuple(),
    parent_route_name=tuple(),
    parent_route_alt_name=tuple(),
    parent_route_off_name=tuple(),
    parent_route_ref=tuple(),
    skip_missing: bool = True
):
    """Updates route links with the given relation `r`
    and returns the list of ways in this relation

    Args:
        r (dict): the current relation item
        relations (dict): dictionary of all relations
        ways (dict): dictionary of all ways
        rels (dict): <rel/path id> -> <rel/path nodes> mapping
        parent_route_id (tuple, optional): route id of the up-level route.
            Defaults to tuple()
        parent_route_name, parent_route_alt_name, parent_route_off_name, parent_route_ref (tuple, optional):
            route name of the up-level route. Defaults to tuple()
        skip_missing (bool, optional): if True, then skip nodes that are
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
                    relations, ways, nodes, rels,
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

    # Remove duplicate nodes (at ways/relations joints)
    route_nodes = list(
        itertools.compress(
            route_nodes,
            [1] + list(map(lambda x: x[0] != x[1], zip(route_nodes[:-1], route_nodes[1:])))
        )
    )
    # Will update rels with paths later during those paths processing
    rels[rid] = route_nodes

    return route_nodes


def init_route_data(p, paths, all_graph_nodes, skip_missing: bool = True):
    way = paths[p]
    if 'route_id' in way:
        return
    wid = way['osmid']
    way['route_id'] = (wid,)
    way['route_name'] = (way.get('name', '').lower(),)
    way['route_alt_name'] = (way.get('alt_name', '').lower(),)
    way['route_off_name'] = (way.get('official_name', '').lower(),)
    way['route_ref'] = (str(way.get('ref', '')).lower(),)

    way_nodes = way['nodes']
    if all_graph_nodes:
        first_node, last_node = way_nodes[0], way_nodes[-1]
        way['route'] = {
            wid: {
                'direction': get_way_dir(first_node, last_node, all_graph_nodes)
            }
        }
