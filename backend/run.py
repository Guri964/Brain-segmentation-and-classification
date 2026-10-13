#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import create_app
from app.config import get_config
app = create_app()
if __name__ == "__main__":
    config = get_config(os.environ.get("BRAIN_ENV"))
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host=host,
        port=port,
        debug=config.DEBUG,
        use_reloader=config.DEBUG and os.environ.get("BRAIN_RELOADER", "0") == "1",
        threaded=True,
    )
