"""Run an orienteering solver or shortest-route baseline on the OD matrix."""

from __future__ import annotations

import argparse
import json
import sys

from busroute.routing.metaheuristics import ALNS, VNS, Algorithm, AntColony
from busroute.routing.od_evaluation import shortest_route
from busroute.routing.orienteering import ODMatrix
from busroute.utils.timeslots import to_minutes


def build_parser() -> argparse.ArgumentParser:
    """Shared command line for the text solver and desktop visualizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="data/final/od_matrix.csv")
    parser.add_argument(
        "--algorithm", choices=("alns", "vns", "aco", "dijkstra"), default="alns"
    )
    parser.add_argument("--start", required=True, help="Start point ID, e.g. H01")
    parser.add_argument("--end", required=True, help="Destination point ID, e.g. H02")
    parser.add_argument(
        "--departure", default="08:00", help="Local departure time HH:MM"
    )
    parser.add_argument("--budget-min", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write search progress as JSON lines to stderr",
    )
    return parser


def solve(args, *, progress=None, trace=False):
    """Run the requested solver, optionally forwarding progress to a viewer."""
    observer = progress

    def log_progress(event):
        event = {"algorithm": args.algorithm, **event}
        if args.verbose:
            print(json.dumps(event), file=sys.stderr, flush=True)
        if observer is not None:
            observer(event)

    progress = log_progress if args.verbose or observer is not None else None
    parts = args.departure.split(":")
    if len(parts) != 2 or not all(part.isdigit() and len(part) == 2 for part in parts):
        raise ValueError("Departure must be HH:MM")
    if int(parts[0]) >= 24 or int(parts[1]) >= 60:
        raise ValueError("Departure must be a valid time in HH:MM")
    matrix = ODMatrix.from_csv(args.matrix)
    if args.algorithm == "alns":
        algorithm: Algorithm = ALNS(
            iterations=args.iterations, seed=args.seed, progress=progress, trace=trace
        )
    elif args.algorithm == "vns":
        algorithm = VNS(
            iterations=args.iterations, seed=args.seed, progress=progress, trace=trace
        )
    elif args.algorithm == "aco":
        algorithm = AntColony(
            iterations=args.iterations, seed=args.seed, progress=progress, trace=trace
        )
    if args.algorithm == "dijkstra":
        # Validate inputs consistently with the other solvers.
        if args.start == args.end:
            raise ValueError("Start and end must be distinct known points")
        matrix.evaluate(
            [args.start, args.end], to_minutes(args.departure), args.budget_min * 60
        )
        result = shortest_route(
            matrix,
            args.start,
            args.end,
            to_minutes(args.departure),
            args.budget_min * 60,
            progress=progress,
        )
        if result is None:
            raise ValueError("No feasible route found within the time budget")
    else:
        result = algorithm(
            matrix,
            args.start,
            args.end,
            to_minutes(args.departure),
            args.budget_min * 60,
        )
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = solve(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
