"""Baseline để so sánh với PPO: đường ngắn nhất và tham lam theo điểm POI / thời gian."""
import networkx as nx


def shortest_path_route(G, start, end, weight="travel_time_morning"):
    # Dijkstra theo thời gian di chuyển – mốc dưới cho điểm POI
    return nx.shortest_path(G, start, end, weight=weight)


def greedy_poi_route(G, start, end, poi_nodes, budget_s, weight="travel_time_morning"):
    # Liên tục chọn POI chưa ghé có (điểm / thời gian) cao nhất mà vẫn kịp về B trong ngân sách
    route, current, used, remaining = [start], start, 0.0, set(poi_nodes)
    while remaining:
        best, best_ratio, best_path, best_t = None, 0.0, None, 0.0
        for p in remaining:
            try:
                t_go, path = nx.single_source_dijkstra(G, current, p, weight=weight)
                t_back = nx.shortest_path_length(G, p, end, weight=weight)
            except nx.NetworkXNoPath:
                continue
            if used + t_go + t_back <= budget_s:
                ratio = (G.nodes[p].get("poi_score") or 0) / max(t_go, 1)
                if ratio > best_ratio:
                    best, best_ratio, best_path, best_t = p, ratio, path, t_go
        if best is None:
            break
        route += best_path[1:]
        used += best_t
        current = best
        remaining.discard(best)
    route += nx.shortest_path(G, current, end, weight=weight)[1:]
    return route
