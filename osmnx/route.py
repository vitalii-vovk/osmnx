from tqdm import tqdm
import itertools


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
    if not inplace:
        G = G.copy()

    for n in tqdm(nodes_to_be_removed, desc='Update truncated routes...'):
        node = G.nodes[n]
        for rid in node.get('route', {}):
            route = get_full_route(G, n, rid)
            # Update prev/next
            next_id = node['route'][rid]['next']
            prev_id = node['route'][rid]['prev']
            start_id = node['route'][rid]['start']
            end_id = node['route'][rid]['end']

            # Update start/end
            if n == start_id:
                visited = set()
                # Node is a start node
                nid = n
                while True:
                    visited.add(nid)
                    G.nodes[nid]['route'][rid]['start'] = next_id
                    nid = G.nodes[nid]['route'][rid]['next']
                    if (nid is None) or (nid in visited):
                        break
            if n == end_id:
                visited = set()
                # Node is an end node
                nid = n
                while True:
                    visited.add(nid)
                    G.nodes[nid]['route'][rid]['end'] = prev_id
                    nid = G.nodes[nid]['route'][rid]['prev']
                    if (nid is None) or (nid in visited):
                        break
            if prev_id:
                prev_node = G.nodes[prev_id]
                prev_node['route'][rid]['next'] = next_id
            if next_id:
                next_node = G.nodes[next_id]
                next_node['route'][rid]['prev'] = prev_id

    return G


def process_route_relation_id(r, relations, ways, parent_id=None):
    first_node = last_node = None
    if r is None:
        return first_node, last_node
    for m in r['members']:
        if parent_id is None:
            parent_id = tuple()
        parent_id_ext = tuple([r['id'], *parent_id])

        if m['type'] == 'relation':
            f_, l_ = process_route_relation_id(
                relations.get(m['ref'], None), relations, ways, parent_id_ext)
            if first_node is None:
                first_node = f_
            if l_:
                last_node = l_
        elif m['type'] == 'way':
            way = ways.get(m['ref'], None)
            if way:
                # Extend the route_id
                way['route_id'] = parent_id_ext
                if first_node is None:
                    first_node = way['nodes'][0]
                last_node = way['nodes'][-1]
    return first_node, last_node


def process_route_relation_dir(r, relations, ways, nodes, parent_dir=None):
    if r is None:
        return

    parent_dir = r['tags'].get('direction', parent_dir)
    rid = r['id']
    for m in r['members']:
        if m['type'] == 'relation':
            process_route_relation_dir(
                relations.get(m['ref'], None), relations, ways, nodes, parent_dir)
        elif m['type'] == 'way':
            way = ways.get(m['ref'], None)
            if way is None:
                continue
            wid = way['osmid']
            if parent_dir and ('direction' not in way):
                way['direction'] = parent_dir
            if 'route' not in way:
                way['route'] = {}
            if rid not in way['route']:
                way['route'][rid] = {}
            way['route'][rid]['direction'] = parent_dir
            for n in way['nodes']:
                node = nodes[n]
                if 'route' not in node:
                    node['route'] = {}
                if rid not in node['route']:
                    node['route'][rid] = {}
                node['route'][rid]['direction'] = parent_dir
                if wid not in node['route']:
                    node['route'][wid] = {}
                node['route'][wid]['direction'] = way['direction']


def update_nodes_with_route_id(route_nodes, rid, nodes, skip_missing: bool = True):
    if skip_missing:
        route_nodes = [n for n in route_nodes if n in nodes]

    processed_nodes = set()
    start_id = route_nodes[0]
    end_id = route_nodes[-1]
    for i, n in enumerate(route_nodes):
        if n in processed_nodes:
            continue

        next_id = route_nodes[i+1] if i < len(route_nodes)-1 else None
        prev_id = None if i == 0 else route_nodes[i-1]

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


def process_route_links(r, relations, ways, nodes, parent_route_id=tuple(), skip_missing: bool = True):
    """Updates route links withi the given relation `r`
    and returs the list of ways in this relation

    Args:
        r (dict): the current relation item
        relations (dict): dictionary of all relations
        ways (dict): dictionary of all ways
        parent_route_id (tuple, optional): route id of the up-level route.
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
    for m in r['members']:
        route_id = (rid, *parent_route_id)
        if m['type'] == 'relation':
            route_nodes.extend(
                process_route_links(
                    relations.get(m['ref'], None),
                    relations, ways, nodes, route_id
                )
            )
        elif m['type'] == 'way':
            way = ways.get(m['ref'], None)
            if not way:
                continue
            wid = way['osmid']
            way_nodes = way['nodes']
            route_nodes.extend(way_nodes)
            if 'route_id' not in way:
                way['route_id'] = (wid, )
            way['route_id'] = (*way['route_id'], *route_id)
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


def init_route_data(p, path, nodes, skip_missing: bool = True):
    way = path[p]
    if 'route_id' in way:
        return
    wid = way['osmid']
    way['route_id'] = (wid,)
    update_nodes_with_route_id(way['nodes'], wid, nodes, skip_missing)
