"""CLI huấn luyện và đánh giá MaskablePPO ở mức waypoint trên đồ thị rút gọn.

Ví dụ:
    python scripts/ppo_waypoint_train.py --epochs 80
    python scripts/ppo_waypoint_train.py --epochs 3 --episodes-per-epoch 8 --eval-count 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from busroute.routing.ppo_env import REDUCED_DIR  # noqa: E402
from busroute.routing.ppo_waypoint import (  # noqa: E402
    CHECKPOINT_WAYPOINT,
    WaypointTable,
    evaluate_waypoint,
    train_waypoint,
)
from busroute.routing.road_graph import RoadGraph, configured_points  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser dòng lệnh cho huấn luyện PPO mức waypoint."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=80, help="Số epoch huấn luyện.")
    ap.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=64,
        help="Số episode thu thập trước mỗi lần cập nhật PPO.",
    )
    ap.add_argument(
        "--update-passes",
        type=int,
        default=6,
        help="Số lượt duyệt minibatch mỗi epoch.",
    )
    ap.add_argument("--batch-size", type=int, default=256, help="Kích thước minibatch.")
    ap.add_argument("--seed", type=int, default=0, help="Seed ngẫu nhiên.")
    ap.add_argument(
        "--eval-count",
        type=int,
        default=30,
        help="Số episode đánh giá so sánh baseline sau khi train (0 = bỏ qua).",
    )
    ap.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_WAYPOINT),
        help="Đường dẫn tệp .zip lưu checkpoint Waypoint PPO.",
    )
    ap.add_argument(
        "--report",
        default=str(ROOT / "results" / "ppo_waypoint_reduced.json"),
        help="Đường dẫn tệp JSON lưu kết quả đánh giá baseline.",
    )
    ap.add_argument(
        "--nodes",
        default=str(ROOT / "data" / "stage1" / "nodes_base.parquet"),
        help="Đường dẫn Parquet bảng nút đường cơ sở.",
    )
    ap.add_argument(
        "--edges",
        default=str(ROOT / "data" / "stage1" / "edges_base.parquet"),
        help="Đường dẫn Parquet bảng cạnh đường cơ sở.",
    )
    ap.add_argument(
        "--widths",
        default=str(ROOT / "data" / "final" / "duong_pho_clean.csv"),
        help="Đường dẫn CSV bề rộng mặt đường.",
    )
    ap.add_argument(
        "--reduced-dir",
        default=str(REDUCED_DIR),
        help="Thư mục chứa đồ thị rút gọn.",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="In chi tiết từng step.")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Điểm vào CLI huấn luyện và đánh giá PPO ở mức waypoint."""
    args = build_parser().parse_args(argv)
    road_graph = RoadGraph.from_files(
        args.nodes,
        args.edges,
        args.widths,
        configured_points(),
        reduced_dir=args.reduced_dir,
    )
    table = WaypointTable.from_road_graph(road_graph)
    print(
        f"waypoint table: {len(table.point_ids)} snapped points, "
        f"{len(table.poi_ids)} POIs, {len(road_graph.nodes)} nodes",
        flush=True,
    )
    model = train_waypoint(
        table,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        update_passes=args.update_passes,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint=Path(args.checkpoint),
        verbose=args.verbose,
    )
    report = []
    if args.eval_count:
        for name in ("random", "shortest", "greedy", "ppo"):
            row = evaluate_waypoint(table, name, model, args.eval_count)
            print(row, flush=True)
            report.append(row)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
