from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "initial_schema",
        """
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id        TEXT PRIMARY KEY,
            scan_identifier    TEXT NOT NULL,
            analysis_type      TEXT NOT NULL,
            status             TEXT NOT NULL,
            model_ids          TEXT NOT NULL DEFAULT '[]',
            input_manifest     TEXT NOT NULL DEFAULT '{}',
            preprocessing      TEXT NOT NULL DEFAULT '{}',
            device             TEXT,
            notes              TEXT,
            error_message      TEXT,
            created_at         TEXT NOT NULL,
            completed_at       TEXT,
            duration_seconds   REAL
        );
        CREATE TABLE IF NOT EXISTS segmentation_results (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id        TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
            model_id           TEXT NOT NULL,
            display_name       TEXT,
            threshold          REAL NOT NULL,
            foreground_pixels  INTEGER NOT NULL,
            image_pixels       INTEGER NOT NULL,
            foreground_fraction REAL NOT NULL,
            probability_mean   REAL,
            probability_max    REAL,
            inference_seconds  REAL,
            tta_used           INTEGER NOT NULL DEFAULT 0,
            mc_dropout_used    INTEGER NOT NULL DEFAULT 0,
            metrics            TEXT NOT NULL DEFAULT '{}',
            artifact_paths     TEXT NOT NULL DEFAULT '{}',
            warnings           TEXT NOT NULL DEFAULT '[]',
            created_at         TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_segmentation_analysis ON segmentation_results(analysis_id);
        CREATE TABLE IF NOT EXISTS radiomics_features (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id        TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
            mask_source        TEXT NOT NULL,
            backend            TEXT NOT NULL,
            dimensionality      TEXT NOT NULL,
            feature_count      INTEGER NOT NULL,
            spacing_available  INTEGER NOT NULL DEFAULT 0,
            features           TEXT NOT NULL,
            categories         TEXT NOT NULL DEFAULT '{}',
            quality            TEXT NOT NULL DEFAULT '{}',
            warnings           TEXT NOT NULL DEFAULT '[]',
            artifact_path     TEXT,
            created_at         TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_radiomics_analysis ON radiomics_features(analysis_id);
        CREATE TABLE IF NOT EXISTS classification_results (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id        TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
            available          INTEGER NOT NULL DEFAULT 0,
            model_id           TEXT,
            model_version      TEXT,
            predicted_label    TEXT,
            scores             TEXT NOT NULL DEFAULT '[]',
            calibrated         INTEGER NOT NULL DEFAULT 0,
            metrics            TEXT NOT NULL DEFAULT '{}',
            contributions      TEXT NOT NULL DEFAULT '[]',
            reason             TEXT,
            created_at         TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS explainability_artifacts (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id        TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
            model_id           TEXT NOT NULL,
            method             TEXT NOT NULL,
            target_layer       TEXT,
            target_score       TEXT,
            heatmap_path       TEXT,
            overlay_path       TEXT,
            metadata           TEXT NOT NULL DEFAULT '{}',
            warnings           TEXT NOT NULL DEFAULT '[]',
            created_at         TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS uncertainty_results (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id        TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
            method             TEXT NOT NULL,
            n_passes           INTEGER NOT NULL,
            mean_dispersion    REAL,
            max_dispersion     REAL,
            high_uncertainty_fraction REAL,
            payload            TEXT NOT NULL DEFAULT '{}',
            created_at         TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reports (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id        TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
            report_id          TEXT NOT NULL UNIQUE,
            path               TEXT NOT NULL,
            size_bytes         INTEGER,
            generator_version  TEXT,
            created_at         TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assistant_messages (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id        TEXT REFERENCES analyses(analysis_id) ON DELETE CASCADE,
            role               TEXT NOT NULL,
            content            TEXT NOT NULL,
            sources            TEXT NOT NULL DEFAULT '[]',
            created_at         TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_versions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id           TEXT NOT NULL,
            status             TEXT NOT NULL,
            path               TEXT,
            checksum           TEXT,
            parameter_count    INTEGER,
            input_spec         TEXT NOT NULL DEFAULT '{}',
            health_verdict     TEXT,
            recorded_at        TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_model_versions_model ON model_versions(model_id, recorded_at);
        """,
    ),
    (
        2,
        "analysis_review_fields",
        """
        ALTER TABLE analyses ADD COLUMN reviewed_by TEXT;
        ALTER TABLE analyses ADD COLUMN reviewed_at TEXT;
        """,
    ),
]
@dataclass
class MigrationResult:
    applied: list[int]
    already_present: list[int]
    schema_version: int
def connect(database_path: Path) -> sqlite3.Connection:
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection
def migrate(connection: sqlite3.Connection) -> MigrationResult:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    existing = {
        row["version"] for row in connection.execute("SELECT version FROM schema_migrations")
    }
    applied: list[int] = []
    already: list[int] = []
    from datetime import datetime, timezone
    for version, name, script in MIGRATIONS:
        if version in existing:
            already.append(version)
            continue
        try:
            connection.executescript(script)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        connection.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (version, name, datetime.now(timezone.utc).replace(microsecond=0).isoformat()),
        )
        connection.commit()
        applied.append(version)
    version = connection.execute(
        "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations"
    ).fetchone()["v"]
    return MigrationResult(applied=applied, already_present=already, schema_version=int(version))
def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    payload = dict(row)
    for key, value in list(payload.items()):
        if isinstance(value, str) and value.startswith(("[", "{")):
            try:
                payload[key] = json.loads(value)
            except json.JSONDecodeError:
                continue
    return payload
def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [row_to_dict(row) for row in rows if row is not None]
