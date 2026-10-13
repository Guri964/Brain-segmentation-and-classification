from __future__ import annotations
import logging
import logging.handlers
import sys
from pathlib import Path
_CONFIGURED = False
def configure_logging(level: str = "INFO", log_dir: Path | None = None, *, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                Path(log_dir) / "backend.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            root.warning("File logging disabled (cannot write to %s)", log_dir)
    for noisy in ("PIL", "matplotlib", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
