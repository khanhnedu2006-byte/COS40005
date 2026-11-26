"""CLI tạo trực quan hóa HTML tương tác và ảnh PNG cho lộ trình PPO trên đồ thị rút gọn.

Ví dụ:
    python scripts/visualize_ppo.py
    python scripts/visualize_ppo.py --start P001 --end P010 --seed 0
    python scripts/visualize_ppo.py --policies greedy,shortest --no-png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from busroute.routing.ppo_env import EDGE_FEATURES, GLOBAL_FEATURES, GraphData  # noqa: E402
from busroute.routing.ppo_train import CHECKPOINT  # noqa: E402
from busroute.utils.timeslots import to_minutes  # noqa: E402
from busroute.viz.ppo_viz import (  # noqa: E402
    DEFAULT_BENCHMARK,
    RING_FILE,
    build_html_visualization,
    load_orig_coords,
    rollout_trace,
    save_static_figure,
)


def build_parser() -> argparse.ArgumentParser:
    """Khởi tạo parser dòng lệnh với mô tả chi tiết cho mọi cờ trực quan hóa.

    Returns:
        Thực thể ``argparse.ArgumentParser`` đã cấu hình đầy đủ.
    """
    ap = argparse.ArgumentParser(
        description="Xuất bản đồ HTML tương tác và ảnh PNG cho agent PPO trên đồ thị rút gọn."
    )
    ap.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT),
        help="Đường dẫn tệp .zip mô hình MaskablePPO đã huấn luyện.",
    )
    ap.add_argument(
        "--start",
        default=None,
        help="Mã POI xuất phát (ví dụ P001); bỏ trống để lấy ngẫu nhiên theo --seed.",
    )
    ap.add_argument(
        "--end",
        default=None,
        help="Mã POI đích (ví dụ P010); bỏ trống để lấy ngẫu nhiên theo --seed.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed chọn kịch bản POI -> POI, giờ xuất phát và ngân sách.",
    )
    ap.add_argument(
        "--departure",
        default=None,
        help="Giờ xuất phát định dạng HH:MM (tùy chọn, mặc định ngẫu nhiên theo --seed).",
    )
    ap.add_argument(
        "--budget-min",
        type=float,
        default=None,
        help="Ngân sách thời gian tính bằng phút (tùy chọn).",
    )
    ap.add_argument(
        "--policies",
        default="ppo,greedy,shortest",
        help="Danh sách chính sách cách nhau bởi dấu phẩy để chạy và so sánh.",
    )
    ap.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK),
        help="Tệp JSON kết quả đánh giá tổng hợp (bỏ qua nếu tệp chưa tồn tại).",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "ppo_visualization.html"),
        help="Đường dẫn tệp HTML trực quan hóa tương tác đầu ra.",
    )
    ap.add_argument(
        "--png",
        default=str(ROOT / "results" / "ppo_visualization.png"),
        help="Đường dẫn tệp ảnh PNG tóm tắt 3 khung đầu ra.",
    )
    ap.add_argument(
        "--no-png",
        action="store_true",
        help="Chỉ xuất HTML, bỏ qua việc tạo ảnh PNG tĩnh.",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    """Điểm vào CLI: nạp đồ thị/mô hình, chạy thử các chính sách và xuất HTML/PNG.

    Args:
        argv: Danh sách tham số dòng lệnh tùy chọn (mặc định đọc từ ``sys.argv``).

    Raises:
        SystemExit: Khi thiếu checkpoint hoặc kích thước quan sát không khớp đồ thị hiện tại.
    """
    args = build_parser().parse_args(argv)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies:
        raise SystemExit("Cần ít nhất một chính sách trong --policies.")

    graph = GraphData.load()
    expected_shape = (graph.max_degree * EDGE_FEATURES + GLOBAL_FEATURES,)
    model = None
    if "ppo" in policies:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            raise SystemExit(
                f"Không tìm thấy checkpoint '{ckpt}'. Hãy huấn luyện trước bằng scripts/ppo_train.py."
            )
        model = MaskablePPO.load(ckpt, device="cpu")
        if tuple(model.observation_space.shape or ()) != expected_shape:
            raise SystemExit(
                f"Checkpoint '{ckpt}' có observation shape {model.observation_space.shape}, "
                f"không khớp môi trường hiện tại {expected_shape}. Hãy huấn luyện lại bằng scripts/ppo_train.py."
            )

    orig_coords = load_orig_coords()
    start_minute = to_minutes(args.departure) if args.departure else None
    budget_s = args.budget_min * 60.0 if args.budget_min is not None else None

    traces = {}
    for pname in policies:
        trace = rollout_trace(
            graph,
            pname,
            model=model,
            seed=args.seed,
            start=args.start,
            end=args.end,
            start_minute=start_minute,
            budget_s=budget_s,
            orig_coords=orig_coords,
        )
        traces[pname] = trace
        s = trace["summary"]
        print(
            f"[{pname}] {trace['start_name']} -> {trace['end_name']} | "
            f"steps={s['steps']} score={s['score']:.2f} ({s['pois']} POIs) "
            f"narrow={s['narrow_m']:.0f}m time={s['elapsed_min']:.1f}/{s['budget_min']:.1f}m "
            f"{'reached B' if s['success'] else 'FAILED'}",
            flush=True,
        )

    bench_path = Path(args.benchmark)
    benchmark = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.is_file() else None
    ring = json.loads(RING_FILE.read_text(encoding="utf-8")) if RING_FILE.is_file() else None

    out_html = build_html_visualization(
        traces, graph, Path(args.out), benchmark=benchmark, ring_geojson=ring
    )
    print(f"HTML saved: {out_html}", flush=True)
    if not args.no_png:
        out_png = save_static_figure(
            traces, graph, Path(args.png), benchmark=benchmark, ring_geojson=ring
        )
        print(f"PNG saved:  {out_png}", flush=True)


if __name__ == "__main__":
    main()
