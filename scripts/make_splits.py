from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import configure_logging, utc_now
def main() -> int:
    parser = argparse.ArgumentParser(description="Assign patient-level splits inside a dataset manifest")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--output", type=Path, default=None, help="Write to this file instead of updating the manifest in place")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = configure_logging(args.verbose)
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"--train/--val/--test must sum to 1.0 (currently {total:.3f})")
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    if not cases:
        raise SystemExit(f"No cases in {args.data}.")
    patients: dict[str, list[dict]] = defaultdict(list)
    for entry in cases:
        case_id = str(entry.get("case_id") or entry.get("id") or "")
        patient_id = str(entry.get("patient_id") or entry.get("patient") or "").strip()
        if not patient_id:
            raise SystemExit(
                f"Case {case_id} has no patient_id. Patient-level splitting is impossible; add "
                "the identifiers first (see docs/DATASET.md)."
            )
        patients[patient_id].append(entry)
    import random
    rng = random.Random(args.seed)
    strata: dict[str, list[str]] = defaultdict(list)
    for patient_id, entries in patients.items():
        labels = sorted({entry.get("label") for entry in entries if entry.get("label") is not None})
        strata[str(labels[0]) if labels else "unlabelled"].append(patient_id)
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for _, patient_ids in sorted(strata.items()):
        rng.shuffle(patient_ids)
        count = len(patient_ids)
        n_train = max(1, int(round(count * args.train))) if count >= 3 else count
        n_val = max(1, int(round(count * args.val))) if count - n_train >= 2 else max(0, count - n_train - 1)
        if count < 3:
            logger.warning(
                "Stratum '%s' has only %d patient(s); it is placed in the training split. Metrics "
                "from a test split this small would not mean anything.",
                _,
                count,
            )
            n_train, n_val = count, 0
        for patient_id in patient_ids[:n_train]:
            splits["train"].extend(str(entry.get("case_id") or entry.get("id")) for entry in patients[patient_id])
        for patient_id in patient_ids[n_train : n_train + n_val]:
            splits["validation"].extend(str(entry.get("case_id") or entry.get("id")) for entry in patients[patient_id])
        for patient_id in patient_ids[n_train + n_val :]:
            splits["test"].extend(str(entry.get("case_id") or entry.get("id")) for entry in patients[patient_id])
    payload["splits"] = splits
    payload["split_metadata"] = {
        "strategy": "patient_level_stratified",
        "seed": args.seed,
        "ratios": {"train": args.train, "validation": args.val, "test": args.test},
        "created_at": utc_now(),
        "note": (
            "Splits are assigned per patient. The test split must stay untouched until "
            "scripts/evaluate_models.py runs; any tuning that reads it invalidates the reported "
            "numbers."
        ),
    }
    target = args.output or args.data
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Patients: %d · cases: %d", len(patients), len(cases))
    for name, ids in splits.items():
        logger.info("  %-10s %4d case(s)", name, len(ids))
    logger.info("Written to %s", target)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
