"""CLI so sánh ma trận [poi, street-reduced, street-full] × [dijkstra, alns, vns, aco, ppo].

Ví dụ:
    python scripts/evaluate_algorithms.py --start H01 --end H04 --budget-min 90
    python scripts/evaluate_algorithms.py --mode all --workers 4 --start H01 --end H04 --budget-min 90
    python scripts/evaluate_algorithms.py --mode street-full --full-checkpoint checkpoints/ppo_full.zip --start P001 --end P010 --budget-min 45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from busroute.routing.benchmark import (  # noqa: E402
    ALL_ALGORITHMS,
    DEFAULT_EDGES,
    DEFAULT_FULL_CHECKPOINT,
    DEFAULT_NODES,
    DEFAULT_POI_CHECKPOINT,
    DEFAULT_REDUCED_CHECKPOINT,
    DEFAULT_REDUCED_DIR,
    DEFAULT_WIDTHS,
    compare_algorithms,
    format_report,
    parse_algorithms,
    parse_modes,
    save_csv,
    save_json,
)
from busroute.routing.road_graph import RoadGraph, configured_points  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser dòng lệnh cho công cụ đánh giá ma trận 3×5."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default="H01",
        help="Mã điểm xuất phát (hub hoặc POI, mặc định: H01).",
    )
    parser.add_argument(
        "--end",
        default="H04",
        help="Mã điểm đích (hub hoặc POI, mặc định: H04).",
    )
    parser.add_argument(
        "--budget-min",
        type=float,
        default=90.0,
        help="Ngân sách thời gian di chuyển tối đa tính bằng phút (mặc định: 90).",
    )
    parser.add_argument(
        "--departure",
        default="08:00",
        help="Giờ khởi hành địa phương định dạng HH:MM (mặc định: 08:00).",
    )
    parser.add_argument(
        "--mode",
        default="all",
        help=(
            "Chế độ định tuyến: poi, street-reduced, street-full, street, all "
            "(hoặc tên cũ waypoint, edge, both; mặc định: all)."
        ),
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(ALL_ALGORITHMS),
        help=(
            "Danh sách thuật toán cách nhau bởi dấu phẩy "
            f"(mặc định: {','.join(ALL_ALGORITHMS)})."
        ),
    )
    parser.add_argument(
        "--poi-checkpoint",
        "--waypoint-checkpoint",
        dest="poi_checkpoint",
        default=None,
        help=(
            "Đường dẫn tệp .zip MaskablePPO cho chế độ poi "
            f"(mặc định nếu bỏ trống: {DEFAULT_POI_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--reduced-checkpoint",
        "--checkpoint",
        dest="reduced_checkpoint",
        default=None,
        help=(
            "Đường dẫn tệp .zip MaskablePPO cho chế độ street-reduced "
            f"(mặc định nếu bỏ trống: {DEFAULT_REDUCED_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--full-checkpoint",
        default=None,
        help=(
            "Đường dẫn tệp .zip MaskablePPO cho chế độ street-full "
            f"(mặc định nếu bỏ trống: {DEFAULT_FULL_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Số vòng lặp tối ưu cho alns, vns và aco (mặc định: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed ngẫu nhiên cho alns, vns và aco (mặc định: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Số luồng/tiến trình song song cho Dijkstra đa nguồn và ALNS/VNS/ACO (mặc định: 1).",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=1.05,
        help="Hệ số dự phòng thời gian về B cho chế độ street-* (mặc định: 1.05).",
    )
    parser.add_argument(
        "--nodes",
        default=str(DEFAULT_NODES),
        help="Đường dẫn Parquet bảng nút đường cơ sở.",
    )
    parser.add_argument(
        "--edges",
        default=str(DEFAULT_EDGES),
        help="Đường dẫn Parquet bảng cạnh đường cơ sở.",
    )
    parser.add_argument(
        "--widths",
        default=str(DEFAULT_WIDTHS),
        help="Đường dẫn CSV bề rộng mặt đường.",
    )
    parser.add_argument(
        "--reduced-dir",
        default=str(DEFAULT_REDUCED_DIR),
        help="Thư mục chứa đồ thị rút gọn cho poi và street-reduced.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Đường dẫn tệp JSON tùy chọn để xuất báo cáo đầy đủ.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Đường dẫn tệp CSV tùy chọn để xuất bảng chỉ số so sánh.",
    )
    parser.add_argument(
        "--show-pois",
        action="store_true",
        help="In thêm danh sách điểm dừng và POI thu thập được của từng thuật toán.",
    )
    parser.add_argument(
        "--list-points",
        action="store_true",
        help="Liệt kê toàn bộ hub và POI cấu hình (kèm điểm số và trạng thái snap) rồi thoát.",
    )
    return parser


def _print_points(
    nodes_path: str | Path,
    edges_path: str | Path,
    widths_path: str | Path,
    reduced_dir: str | Path,
) -> None:
    """In bảng danh sách hub và POI cấu hình cùng khả năng snap vào đồ thị đường."""
    points = configured_points()
    snapped: set[str] = set()
    distances: dict[str, float] = {}
    if Path(nodes_path).is_file() and Path(edges_path).is_file() and Path(widths_path).is_file():
        graph = RoadGraph.from_files(
            nodes_path, edges_path, widths_path, points, reduced_dir=reduced_dir
        )
        snapped = set(graph.snapped)
        distances = graph.snap_distances

    print(f"{'ID':<6}  {'Kind':<4}  {'Score':>6}  {'Snap(m)':>8}  {'Avail':<5}  Name")
    print("-" * 80)
    for point in points:
        pid = point["id"]
        dist = distances.get(pid)
        dist_str = f"{dist:.1f}" if dist is not None else "-"
        avail = "yes" if (not snapped or pid in snapped) else "no"
        print(
            f"{pid:<6}  {point['kind']:<4}  {point['score']:>6.3f}  "
            f"{dist_str:>8}  {avail:<5}  {point['name']}"
        )


def main(argv: list[str] | None = None) -> None:
    """Điểm vào CLI: kiểm tra tham số, chạy so sánh và in/xuất báo cáo."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_points:
        try:
            _print_points(args.nodes, args.edges, args.widths, args.reduced_dir)
        except OSError as exc:
            parser.exit(2, f"error: {exc}\n")
        return

    try:
        algorithms = parse_algorithms(args.algorithms)
        modes = parse_modes(args.mode)
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.workers < 1:
        parser.exit(2, "error: --workers must be >= 1\n")

    explicit_poi_ckpt = args.poi_checkpoint is not None
    poi_ckpt = Path(args.poi_checkpoint) if explicit_poi_ckpt else DEFAULT_POI_CHECKPOINT
    if explicit_poi_ckpt and "poi" in modes and "ppo" in algorithms and not poi_ckpt.is_file():
        parser.exit(2, f"error: POI PPO checkpoint not found: {poi_ckpt}\n")

    explicit_red_ckpt = args.reduced_checkpoint is not None
    red_ckpt = (
        Path(args.reduced_checkpoint) if explicit_red_ckpt else DEFAULT_REDUCED_CHECKPOINT
    )
    if (
        explicit_red_ckpt
        and "street-reduced" in modes
        and "ppo" in algorithms
        and not red_ckpt.is_file()
    ):
        parser.exit(2, f"error: street-reduced PPO checkpoint not found: {red_ckpt}\n")

    explicit_full_ckpt = args.full_checkpoint is not None
    full_ckpt = Path(args.full_checkpoint) if explicit_full_ckpt else DEFAULT_FULL_CHECKPOINT
    if (
        explicit_full_ckpt
        and "street-full" in modes
        and "ppo" in algorithms
        and not full_ckpt.is_file()
    ):
        parser.exit(2, f"error: street-full PPO checkpoint not found: {full_ckpt}\n")

    try:
        report = compare_algorithms(
            args.start,
            args.end,
            args.budget_min,
            departure=args.departure,
            algorithms=algorithms,
            modes=modes,
            checkpoint=red_ckpt,
            full_checkpoint=full_ckpt,
            poi_checkpoint=poi_ckpt,
            iterations=args.iterations,
            seed=args.seed,
            workers=args.workers,
            safety_margin=args.safety_margin,
            nodes_path=args.nodes,
            edges_path=args.edges,
            widths_path=args.widths,
            reduced_dir=args.reduced_dir,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(format_report(report, show_pois=args.show_pois))
    if args.json:
        json_path = save_json(report, args.json)
        print(f"JSON saved: {json_path}")
    if args.csv:
        csv_path = save_csv(report, args.csv)
        print(f"CSV saved:  {csv_path}")


if __name__ == "__main__":
    main()
