"""CLI huấn luyện và đánh giá PPO theo lô episode trên đồ thị đã rút gọn.

Ví dụ:
    python scripts/ppo_train.py --epochs 100
    python scripts/ppo_train.py --epochs 3 --episodes-per-epoch 4 --eval-count 5
    python scripts/ppo_train.py --epochs 1 --episodes-per-epoch 1 --eval-count 0 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from busroute.routing.ppo_env import GraphData  # noqa: E402
from busroute.routing.ppo_train import CHECKPOINT, evaluate, train  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh với mô tả đầy đủ cho từng cờ.

    Returns:
        Thực thể ``argparse.ArgumentParser`` đã khai báo toàn bộ tham số huấn luyện.
    """
    ap = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá PPO theo lô episode (POI -> POI) trên đồ thị rút gọn."
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Số epoch huấn luyện (mỗi epoch thu đủ số episode rồi học một lượt).",
    )
    ap.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=16,
        help="Số episode trọn vẹn (POI A -> POI B) thu thập trong mỗi epoch.",
    )
    ap.add_argument(
        "--update-passes",
        type=int,
        default=6,
        help="Số lượt duyệt qua toàn bộ lô dữ liệu khi tối ưu PPO-clip.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Kích thước minibatch trong mỗi bước cập nhật gradient.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed ngẫu nhiên cho mô hình và bộ sinh kịch bản.",
    )
    ap.add_argument(
        "--eval-count",
        type=int,
        default=30,
        help="Số episode đánh giá trên tập kịch bản cố định sau khi train (0 = bỏ qua).",
    )
    ap.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT),
        help="Đường dẫn tệp .zip lưu trọng số mô hình PPO sau khi huấn luyện.",
    )
    ap.add_argument(
        "--report",
        default=str(ROOT / "results" / "ppo_reduced.json"),
        help="Đường dẫn tệp JSON lưu bảng kết quả đánh giá các chính sách.",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Hiển thị chi tiết từng step trong mỗi episode (tên nút/POI, tên đường, bề rộng, reward).",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    """Điểm vào CLI: nạp đồ thị, chạy huấn luyện PPO và đánh giá so sánh baseline.

    Args:
        argv: Danh sách tham số dòng lệnh tùy chọn (mặc định đọc từ ``sys.argv``).
    """
    args = build_parser().parse_args(argv)

    graph = GraphData.load()
    known_widths = int(np.isfinite(graph.width).sum()) if hasattr(graph, "width") else 0
    print(
        f"graph: {graph.n} nodes, {len(graph.dst)} edges, {len(graph.poi_node)} POIs (start/end), "
        f"max out-degree {graph.max_degree}, known-width edges {known_widths}",
        flush=True,
    )
    model = train(
        graph,
        args.epochs,
        args.episodes_per_epoch,
        args.update_passes,
        args.batch_size,
        seed=args.seed,
        checkpoint=Path(args.checkpoint),
        verbose=args.verbose,
    )
    report = []
    if args.eval_count:
        for name in ("random", "shortest", "greedy", "ppo"):
            row = evaluate(graph, name, model, args.eval_count)
            print(row, flush=True)
            report.append(row)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
