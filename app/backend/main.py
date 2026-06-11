# -*- coding: utf-8 -*-
"""
Entry point for the research API (the app backend)

Runs the FastAPI app (app/backend/app) with uvicorn

The backend reuses the pipeline library in src/ (config, the rag pipeline) but is
itself part of the self-contained application under app/

Run from the project root:
  python -m app.backend.main
  python -m app.backend.main --host 0.0.0.0 --port 8000 --reload
"""

import os
import argparse
import uvicorn

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Serve the RADAR research API")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev)")
    args = parser.parse_args()

    uvicorn.run("app.backend.app:create_app", host=args.host, port=args.port,
                reload=args.reload, factory=True)


if __name__ == "__main__":
    main()
