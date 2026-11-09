"""Launch the local route-search map app with ``python -m Web``."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main(argv: Sequence[str] | None = None) -> None:
    """Start the loopback-only web server and optionally open its browser UI.

    Args:
        argv: Command-line arguments, or None to read the process arguments.

    Raises:
        SystemExit: Arguments or startup data are invalid, dependencies are
            missing, or the user interrupts the server with Ctrl+C.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="data/final/od_matrix.csv")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        import uvicorn

        from Web.app import create_app
    except ImportError as exc:
        parser.exit(2, f"error: {exc}\nInstall web support: uv sync --extra web\n")
    try:
        app = create_app(Path(args.matrix))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    url = f"http://127.0.0.1:{args.port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info")
    server = uvicorn.Server(config)

    def open_when_ready() -> None:
        """Open the browser only after this server has successfully started."""
        while not server.should_exit:
            if server.started:
                try:
                    with urlopen(url + "/api/status", timeout=2):
                        webbrowser.open(url)
                    return
                except (OSError, URLError):
                    return
            threading.Event().wait(0.1)

    if not args.no_browser:
        threading.Thread(target=open_when_ready, daemon=True).start()
    print(f"Route Lab: {url} (Ctrl+C to stop)", flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
