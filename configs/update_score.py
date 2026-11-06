"""
Copy values from ranked_pois.csv back into pois.py (matched by poi_id).

Run:  python update_pois.py
A backup of the original is saved as pois.py.bak.
Note: pois.py is regenerated, so comments inside it are lost,
and it must contain only the POIS list (no other variables or functions).
"""

import csv
import shutil
from pathlib import Path

from pois import POIS

BASE_DIR = Path(__file__).resolve().parent  # files are read/written next to this script
CSV_FILE = BASE_DIR / "ranked_pois.csv"
POIS_FILE = BASE_DIR / "pois.py"
FIELDS = ["score"]  # add "rating", "review_count" if you want those stored too


def format_pois(pois):
    """One key per line, 4-space indent, single quotes."""
    lines = ["POIS = ["]
    for i, poi in enumerate(pois):
        lines.append("    {")
        items = list(poi.items())
        for j, (key, value) in enumerate(items):
            comma = "," if j < len(items) - 1 else ""
            lines.append(f"        {key!r}: {value!r}{comma}")
        lines.append("    }" + ("," if i < len(pois) - 1 else ""))
    lines.append("]")
    return "\n".join(lines) + "\n"


def convert(field, raw):
    if raw in ("", None):
        return None
    if field == "review_count":
        return int(float(raw))
    return float(raw)


def main():
    with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
        rows = {str(r["poi_id"]): r for r in csv.DictReader(f)}

    updated, missing = 0, []
    for poi in POIS:
        row = rows.get(str(poi.get("poi_id")))
        if row is None:
            missing.append(poi.get("name"))
            continue
        for field in FIELDS:
            value = convert(field, row.get(field))
            if value is not None:
                poi[field] = value
        updated += 1

    shutil.copy(POIS_FILE, f"{POIS_FILE}.bak")
    with open(POIS_FILE, "w", encoding="utf-8") as f:
        f.write(format_pois(POIS))

    print(
        f"Updated {updated}/{len(POIS)} POIs in {POIS_FILE} (backup: {POIS_FILE}.bak)"
    )
    if missing:
        print("Not found in CSV:", ", ".join(map(str, missing)))


if __name__ == "__main__":
    main()
