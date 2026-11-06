"""Kiểm tra sau khi lọc: A, B và mọi POI phải còn đi tới được nhau."""

import networkx as nx


def check_connectivity(G, required_nodes):
    """Find required nodes outside the largest strongly connected component.

    Args:
        G: Directed graph to inspect.
        required_nodes: Node IDs that must belong to the component.

    Returns:
        Required node IDs outside the component, or all of them if G has no
        edges. Nodes absent from G are included in the result.
    """
    if G.number_of_edges() == 0:
        return list(required_nodes)
    largest = max(nx.strongly_connected_components(G), key=len)
    return [n for n in required_nodes if n not in largest]
