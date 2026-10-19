"""Vẽ lộ trình và POI lên bản đồ HTML (folium)."""
import folium


def draw_route(G, route, out_html, pois=None):
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
    m = folium.Map(location=coords[0], zoom_start=13)
    folium.PolyLine(coords, weight=5).add_to(m)
    folium.Marker(coords[0], tooltip="A (start)").add_to(m)
    folium.Marker(coords[-1], tooltip="B (end)").add_to(m)
    for p in pois or []:
        folium.CircleMarker((p["lat"], p["lon"]), radius=6, tooltip=p["name"], fill=True).add_to(m)
    m.save(out_html)
    return out_html
