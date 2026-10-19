"""Chỉ số đánh giá một lộ trình."""


def route_metrics(G, route, budget_s, weight="travel_time_morning"):
    total_t, score, seen, unverified = 0.0, 0.0, set(), []
    for u, v in zip(route[:-1], route[1:]):
        data = min(G[u][v].values(), key=lambda d: d[weight])
        total_t += data[weight]
        if not data.get("verified_by_gps"):
            unverified.append((u, v))          # danh sách cần kiểm tra Street View / tài xế
        if G.nodes[v].get("is_poi") and v not in seen:
            seen.add(v)
            score += G.nodes[v].get("poi_score") or 0
    return {
        "total_min": round(total_t / 60, 1),
        "within_budget": total_t <= budget_s,
        "poi_count": len(seen),
        "poi_score": score,
        "unverified_edges": unverified,
    }
