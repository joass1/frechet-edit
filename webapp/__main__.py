"""``python -m webapp`` - serve the course audit on localhost."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the course audit web app.")
    parser.add_argument("--host", default="127.0.0.1", help="interface (default: localhost only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Course audit running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run("webapp.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
