"""Run solve_orienteering's arguments with a live native OSMnx map window.

ALNS/VNS show selected operator actions. Other solvers show available progress
and the final route. Road geometry is reconstructed, not the matrix provider's
original path. Close the window to stop the run.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path

from solve_orienteering import build_parser, solve

from Web.road_paths import RoadPaths, load_points

ROOT = Path(__file__).resolve().parents[1]


class Cancelled(Exception):
    """The viewer was closed while its worker was running."""



def run_window(args):
    # Import only after argument parsing, so --help works without GUI packages.
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import Button, Slider

    messages = queue.Queue(maxsize=8)
    stopped = threading.Event()
    state = {"paused": False, "exit_code": 0, "finished": False}

    def send(message):
        while not stopped.is_set():
            try:
                messages.put(message, timeout=0.1)
                return
            except queue.Full:
                pass
        raise Cancelled()

    def worker():
        try:
            import yaml

            from busroute.graph.osm_fetch import fetch_road_graph

            points = load_points(args.matrix)
            if args.start not in points or args.end not in points:
                raise ValueError("Start and end must be known points")
            cfg = yaml.safe_load((ROOT / "configs/settings.yaml").read_text(encoding="utf-8"))
            osm = cfg["osm"]
            # The fetcher logs downloads to stdout; keep result stdout as JSON.
            with redirect_stdout(sys.stderr):
                graph = fetch_road_graph(
                    place=osm.get("place"), network_type=osm["network_type"],
                    cache_dir=str(ROOT / osm["cache_dir"]), bbox=osm.get("bbox"),
                )
            paths = RoadPaths(graph, points, cfg["poi"]["max_snap_distance_m"])
            send({"event": "map_ready", "graph": graph, "points": points})

            def progress(event):
                if stopped.is_set():
                    raise Cancelled()
                event = dict(event)
                missing = set()
                for name in ("best", "current", "candidate"):
                    if name in event:
                        event[f"{name}_segments"], absent = paths.route(event[name])
                        missing.update(absent)
                event["missing"] = sorted(missing)
                send(event)

            result = solve(args, progress=progress, trace=True)
            progress({"event": "result", "best": result.to_dict(),
                      "current": result.to_dict()})
        except Cancelled:
            pass
        except Exception as exc:  # noqa: BLE001 -- forward worker failures to the GUI
            try:
                send({"event": "error", "message": str(exc)})
            except Cancelled:
                pass

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.17, top=0.79)
    fig.canvas.manager.set_window_title(f"{args.algorithm.upper()} — route search")
    title = fig.text(0.04, 0.95, "Loading OSM graph…", fontsize=14, weight="bold")
    metrics = fig.text(0.04, 0.90, f"{args.start} → {args.end}  |  {args.departure}  |  budget {args.budget_min:g} min")
    route_text = fig.text(0.04, 0.84, "", fontsize=9)
    warning = fig.text(0.04, 0.11, "", fontsize=9, color="#a23e00")
    fig.text(0.04, 0.025, "OSM-reconstructed paths; score/time from OD matrix. Map data © OpenStreetMap contributors.", fontsize=8)
    ax.set_axis_off()
    layers = {}
    button = Button(fig.add_axes((0.04, 0.05, 0.1, 0.035)), "Pause")
    speed = Slider(fig.add_axes((0.28, 0.055, 0.35, 0.02)), "Steps/s", 1, 30, valinit=8, valstep=1)

    def pause(_):
        state["paused"] = not state["paused"]
        button.label.set_text("Resume" if state["paused"] else "Pause")
        fig.canvas.draw_idle()

    button.on_clicked(pause)
    timer = fig.canvas.new_timer(interval=125)
    speed.on_changed(lambda value: setattr(timer, "interval", int(1000 / value)))

    def tick():
        if state["paused"]:
            return
        try:
            event = messages.get_nowait()
        except queue.Empty:
            return
        kind = event["event"]
        if kind == "map_ready":
            import osmnx as ox

            from busroute.config.restrictions import RING_FILE

            ox.plot_graph(event["graph"], ax=ax, show=False, close=False,
                          node_size=0, edge_color="#c3cbd1", edge_linewidth=0.5,
                          bgcolor="white")
            ring = json.loads(RING_FILE.read_text(encoding="utf-8"))
            for feature in ring["features"]:
                for boundary in feature["geometry"]["coordinates"]:
                    ax.plot(*zip(*boundary), color="#a23e00", linewidth=1, alpha=0.7)
            points = event["points"]
            ax.scatter([p[1] for p in points.values()], [p[0] for p in points.values()],
                       s=15, color="#333333", zorder=4)
            for point, (lat, lon) in points.items():
                ax.annotate(point, (lon, lat), xytext=(3, 3), textcoords="offset points", fontsize=7)
            for point, marker in ((args.start, "s"), (args.end, "*")):
                lat, lon = points[point]
                ax.scatter([lon], [lat], s=100, marker=marker, color="#111111", zorder=6)
            lats, lons = zip(*points.values())
            ax.set_xlim(min(lons) - 0.01, max(lons) + 0.01)
            ax.set_ylim(min(lats) - 0.01, max(lats) + 0.01)
            for name, color, width, style in (
                ("best", "#2563eb", 5, "solid"),
                ("current", "#159447", 3, "solid"),
                ("candidate", "#ed8b16", 2, "dashed"),
            ):
                layer = LineCollection([], colors=color, linewidths=width,
                                       linestyles=style, alpha=0.8, label=name.title(), zorder=3)
                ax.add_collection(layer)
                layers[name] = layer
            ax.legend(loc="upper right")
            title.set_text("Map ready — starting search")
        elif kind == "error":
            state["exit_code"] = 2
            state["finished"] = True
            title.set_text("Search failed")
            warning.set_text(event["message"])
            print(f"error: {event['message']}", file=sys.stderr)
            timer.stop()
        else:
            for name, layer in layers.items():
                if f"{name}_segments" in event:
                    layer.set_segments(event[f"{name}_segments"])
            if kind in ("initial_solution", "result", "search_end") and "candidate" in layers:
                layers["candidate"].set_segments([])
            action = event.get("action", kind).replace("_", " ")
            detail = ""
            if "accepted" in event:
                detail = " — accepted" if event["accepted"] else " — rejected"
            if "neighbourhood" in event:
                detail += f" | neighbourhood {event['neighbourhood']}"
            if "destroy_operator" in event:
                detail += f" | {event['destroy_operator']} removal / {event['repair_operator']} repair"
            title.set_text(f"{args.algorithm.upper()} | {action}{detail}"
                           f" | {event.get('iteration', 0)}/{args.iterations}")
            result = event.get("candidate") or event.get("current") or event.get("best")
            if result:
                metrics.set_text(f"Displayed: score {result['score']:.2f} | {result['duration_min']:.1f} min"
                                 f" / budget {args.budget_min:g} min")
                route = " → ".join(result["route"])
                route_text.set_text(route if len(route) <= 140 else route[:137] + "…")
            if "best" in event:
                best = event["best"]
                metrics.set_text(metrics.get_text() + f" | Best {best['score']:.2f}, {best['duration_min']:.1f} min")
            missing = event.get("missing", [])
            warning.set_text("Missing road paths (or snap >300 m): " + ", ".join(missing[:4]) if missing else "")
            if kind == "result":
                state["finished"] = True
                print(json.dumps(event["best"], indent=2), flush=True)
                title.set_text(f"{args.algorithm.upper()} — complete (close window to exit)")
                timer.stop()
        fig.canvas.draw_idle()

    timer.add_callback(tick)
    fig.canvas.mpl_connect("close_event", lambda _: stopped.set())
    threading.Thread(target=worker, daemon=True, name="route-search").start()
    timer.start()
    plt.show()
    stopped.set()
    return state["exit_code"] if state["finished"] else 130


def main():
    parser = build_parser()
    parser.description = __doc__
    args = parser.parse_args()
    try:
        import tkinter
    except ImportError:
        parser.exit(2, "error: Tkinter is required (Ubuntu/Debian: python3-tk).\n")
    try:
        return run_window(args)
    except ImportError as exc:
        parser.exit(2, f"error: {exc}\nInstall plotting support: uv sync --extra visualization\n"
                    "Tkinter is also required (Ubuntu/Debian: python3-tk).\n")
    except (RuntimeError, tkinter.TclError) as exc:
        parser.exit(2, f"error: {exc}\nLaunch from a graphical desktop with Tk support.\n")


if __name__ == "__main__":
    raise SystemExit(main())
