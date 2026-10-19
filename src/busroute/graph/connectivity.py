"""Kiểm tra sau khi lọc: A, B và mọi POI phải còn đi tới được nhau."""
import networkx as nx


def check_connectivity(G, required_nodes):
    # Trả về danh sách node bắt buộc KHÔNG nằm trong thành phần liên thông mạnh lớn nhất
    if G.number_of_edges() == 0:
        return list(required_nodes)
    largest = max(nx.strongly_connected_components(G), key=len)
    return [n for n in required_nodes if n not in largest]
