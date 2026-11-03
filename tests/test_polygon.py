import pandas as pd
from busroute.config.restrictions import TIME_BANNED_AREAS
from busroute.utils.geo import point_in_polygon

nodes = pd.read_parquet("data/stage1/nodes_base.parquet").set_index("node_idx")
edges = pd.read_parquet("data/stage1/edges_base.parquet")
poly = TIME_BANNED_AREAS[0]["polygon"]

# Trung điểm từng cạnh
mid_lat = (nodes.lat[edges.from_idx].values + nodes.lat[edges.to_idx].values) / 2
mid_lon = (nodes.lon[edges.from_idx].values + nodes.lon[edges.to_idx].values) / 2
edges["inside"] = [point_in_polygon(a, b, poly) for a, b in zip(mid_lat, mid_lon)]

print("Số cạnh trong vành đai:", int(edges.inside.sum()), "/", len(edges))

# Kiểm tra mẫu: tên đường → kỳ vọng (True = phải nằm trong)
expected = {"lê duẩn": True, "hùng vương": True,          # trong trung tâm
            "điện biên phủ": False, "bạch đằng": False,     # tuyến bao
            "võ nguyên giáp": False, "ngô quyền": False}    # bên kia sông Hàn
for name, want in expected.items():
    share = edges[edges.road_name.str.contains(name, regex=False)].inside.mean()
    status = "OK" if (share > 0.8) == want else "SAI"
    print(f"{status:3} {name:16} tỉ lệ cạnh nằm trong = {share:.0%}")
