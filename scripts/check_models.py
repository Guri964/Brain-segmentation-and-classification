from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BACKEND_ROOT, configure_logging, describe_environment, resolve_data_dir, utc_now
sys.path.insert(0, str(BACKEND_ROOT))
def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that the registered checkpoints load and respond")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--json", type=Path, default=None, help="Write the report to this JSON file as well")
    parser.add_argument("--force", action="store_true", help="Ignore any cached verdict and re-run every check")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = configure_logging(args.verbose)
    from app import create_app
    app = create_app(env="testing")
    services = app.extensions["brain_services"]
    if args.force:
        for model_id in services["registry"].descriptors:
            services["health"].check(model_id, force=True)
    models = services["health"].summary()["models"]
    rows = []
    for entry in models:
        rows.append(
            {
                "model_id": entry.get("model_id"),
                "verdict": entry.get("verdict"),
                "responsive": bool(entry.get("responsive")),
                "device": entry.get("device"),
                "output_semantics": entry.get("output_semantics"),
                "message": entry.get("message"),
                "recommendation": entry.get("recommendation"),
                "checks": entry.get("checks", []),
            }
        )
    width = max((len(str(row["model_id"])) for row in rows), default=10)
    logger.info("")
    logger.info("%-*s | %-14s | %s", width, "model", "verdict", "reading")
    logger.info("%s", "-" * (width + 60))
    for row in rows:
        logger.info("%-*s | %-14s | %s", width, row["model_id"], row["verdict"], row["message"])
    responsive = [row["model_id"] for row in rows if row["responsive"]]
    logger.info("")
    logger.info("Responsive checkpoints: %s", ", ".join(responsive) if responsive else "none")
    if not responsive:
        logger.warning(
            "No checkpoint passed the reference check. Segmentation output will be reported as "
            "unavailable rather than shown as a mask."
        )
    report = {
        "generated_at": utc_now(),
        "environment": describe_environment(),
        "data_dir": str(resolve_data_dir(args.data_dir)),
        "models": rows,
        "responsive_models": responsive,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Report written to %s", args.json)
    else:
        default_path = resolve_data_dir(args.data_dir) / "models" / "reference_checks.json"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        default_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Report written to %s", default_path)
    return 0 if responsive else 1
if __name__ == "__main__":
    raise SystemExit(main())
