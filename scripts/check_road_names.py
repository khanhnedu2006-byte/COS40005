import difflib
import pandas as pd
from busroute.config.roads import MAIN_ROADS
from busroute.utils.geo import normalize_road_name   
osm = pd.read_csv("reports/osm_road_names.csv")

osm["road_name"] = osm.road_name.fillna("").str.split(" \\| ")
osm = osm.explode("road_name")
osm["road_name"] = osm.road_name.map(normalize_road_name)
length = osm.groupby("road_name").total_length_km.sum()
osm_names = sorted(n for n in length.index if n)
osm_set = set(osm_names)

rows = []
for raw in MAIN_ROADS:
    norm = normalize_road_name(raw)
    idempotent = normalize_road_name(norm) == norm
    if norm in osm_set:
        rows.append((raw, norm, "OK", round(length[norm], 1), "", idempotent))
    else:
        sub = [n for n in osm_names if norm in n or n in norm][:3]
        fuzzy = difflib.get_close_matches(norm, osm_names, n=3, cutoff=0.6)
        sugg = " | ".join(dict.fromkeys(sub + fuzzy))
        rows.append((raw, norm, "MISS", 0.0, sugg, idempotent))

df = pd.DataFrame(rows, columns=["main_road", "normalized", "status",
                                 "osm_km", "suggestions", "idempotent"])
df.to_csv("reports/road_name_check.csv", index=False, encoding="utf-8-sig")

print(df.to_string(index=False))
print(f"\n{(df.status == 'OK').sum()}/{len(df)} OK, {(df.status == 'MISS').sum()} MISS")
print("Not idempotent:", df[~df.idempotent].main_road.tolist())
d = df[df.duplicated("normalized", keep=False)]
if len(d):
    print("Duplicates after normalize:\n", d[["main_road", "normalized"]])