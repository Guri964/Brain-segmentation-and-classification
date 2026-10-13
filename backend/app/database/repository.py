from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from ..utils.io import utc_now
from .schema import connect, migrate, row_to_dict, rows_to_dicts
ANALYSIS_STATUSES = ("pending", "running", "completed", "failed", "partial")
def _dump(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str)
class AnalysisRepository:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = connect(self.database_path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
    def initialise(self) -> dict:
        with self.connection() as connection:
            result = migrate(connection)
        return {
            "applied": result.applied,
            "already_present": result.already_present,
            "schema_version": result.schema_version,
            "database_path": str(self.database_path),
        }
    def schema_version(self) -> int:
        with self.connection() as connection:
            migrate(connection)
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations"
            ).fetchone()
            return int(row["v"])
    def create_analysis(
        self,
        *,
        analysis_id: str,
        scan_identifier: str,
        analysis_type: str,
        model_ids: list[str],
        input_manifest: dict,
        preprocessing: dict,
        device: str,
        notes: str | None = None,
        status: str = "pending",
    ) -> dict:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO analyses (
                    analysis_id, scan_identifier, analysis_type, status, model_ids,
                    input_manifest, preprocessing, device, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    scan_identifier,
                    analysis_type,
                    status,
                    _dump(model_ids),
                    _dump(input_manifest),
                    _dump(preprocessing),
                    device,
                    notes,
                    utc_now(),
                ),
            )
        return self.get_analysis(analysis_id)
    def update_status(
        self,
        analysis_id: str,
        status: str,
        *,
        error_message: str | None = None,
        completed: bool = False,
    ) -> None:
        with self.connection() as connection:
            if completed:
                row = connection.execute(
                    "SELECT created_at FROM analyses WHERE analysis_id = ?", (analysis_id,)
                ).fetchone()
                duration = None
                if row is not None:
                    try:
                        started = datetime.fromisoformat(row["created_at"])
                        duration = (
                            datetime.now(timezone.utc) - started
                        ).total_seconds()
                    except ValueError:
                        duration = None
                connection.execute(
                    """
                    UPDATE analyses
                       SET status = ?, error_message = COALESCE(?, error_message),
                           completed_at = ?, duration_seconds = ?
                     WHERE analysis_id = ?
                    """,
                    (status, error_message, utc_now(), duration, analysis_id),
                )
            else:
                connection.execute(
                    "UPDATE analyses SET status = ?, error_message = COALESCE(?, error_message) WHERE analysis_id = ?",
                    (status, error_message, analysis_id),
                )
    def mark_reviewed(self, analysis_id: str, reviewer: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE analyses SET reviewed_by = ?, reviewed_at = ? WHERE analysis_id = ?",
                (reviewer, utc_now(), analysis_id),
            )
    def get_analysis(self, analysis_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
        return row_to_dict(row)
    def list_analyses(
        self,
        *,
        search: str | None = None,
        status: str | None = None,
        analysis_type: str | None = None,
        model_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "created_at",
        direction: str = "desc",
    ) -> dict:
        clauses = []
        params: list[Any] = []
        if search:
            clauses.append("(analysis_id LIKE ? OR scan_identifier LIKE ? OR notes LIKE ?)")
            pattern = f"%{search}%"
            params.extend([pattern, pattern, pattern])
        if status:
            clauses.append("status = ?")
            params.append(status)
        if analysis_type:
            clauses.append("analysis_type = ?")
            params.append(analysis_type)
        if model_id:
            clauses.append("model_ids LIKE ?")
            params.append(f'%"{model_id}"%')
        if date_from:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("created_at <= ?")
            params.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = order if order in {"created_at", "duration_seconds", "scan_identifier"} else "created_at"
        direction = "ASC" if str(direction).lower() == "asc" else "DESC"
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        with self.connection() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) AS n FROM analyses {where}", params
            ).fetchone()["n"]
            rows = connection.execute(
                f"""
                SELECT analyses.*,
                       (SELECT COUNT(*) FROM segmentation_results s
                         WHERE s.analysis_id = analyses.analysis_id) AS segmentation_count,
                       (SELECT COUNT(*) FROM radiomics_features r
                         WHERE r.analysis_id = analyses.analysis_id) AS radiomics_count,
                       (SELECT COUNT(*) FROM reports p
                         WHERE p.analysis_id = analyses.analysis_id) AS report_count
                  FROM analyses {where}
                 ORDER BY {order} {direction}
                 LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return {
            "items": rows_to_dicts(rows),
            "total": int(total),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < int(total),
        }
    def delete_analysis(self, analysis_id: str) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,)
            )
            return cursor.rowcount > 0
    def counts_by_status(self) -> dict:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS n FROM analyses GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}
    def add_segmentation_result(self, analysis_id: str, payload: dict) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO segmentation_results (
                    analysis_id, model_id, display_name, threshold, foreground_pixels,
                    image_pixels, foreground_fraction, probability_mean, probability_max,
                    inference_seconds, tta_used, mc_dropout_used, metrics, artifact_paths,
                    warnings, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    payload["model_id"],
                    payload.get("display_name"),
                    float(payload["threshold"]),
                    int(payload["foreground_pixels"]),
                    int(payload["image_pixels"]),
                    float(payload["foreground_fraction"]),
                    payload.get("probability_mean"),
                    payload.get("probability_max"),
                    payload.get("inference_seconds"),
                    1 if payload.get("tta_used") else 0,
                    1 if payload.get("mc_dropout_used") else 0,
                    _dump(payload.get("metrics", {})),
                    _dump(payload.get("artifact_paths", {})),
                    json.dumps(payload.get("warnings", [])),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)
    def add_radiomics(self, analysis_id: str, payload: dict) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO radiomics_features (
                    analysis_id, mask_source, backend, dimensionality, feature_count,
                    spacing_available, features, categories, quality, warnings,
                    artifact_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    payload["mask_source"],
                    payload["backend"],
                    payload["dimensionality"],
                    int(payload["feature_count"]),
                    1 if payload.get("spacing_available") else 0,
                    _dump(payload["features"]),
                    _dump(payload.get("categories", {})),
                    _dump(payload.get("quality", {})),
                    json.dumps(payload.get("warnings", [])),
                    payload.get("artifact_path"),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)
    def add_classification(self, analysis_id: str, payload: dict) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO classification_results (
                    analysis_id, available, model_id, model_version, predicted_label,
                    scores, calibrated, metrics, contributions, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    1 if payload.get("available") else 0,
                    payload.get("model_id"),
                    payload.get("model_version"),
                    payload.get("predicted_label"),
                    json.dumps(payload.get("scores", [])),
                    1 if payload.get("calibrated") else 0,
                    _dump(payload.get("metrics", {})),
                    json.dumps(payload.get("contributions", [])[:50]),
                    payload.get("reason"),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)
    def add_explainability(self, analysis_id: str, payload: dict) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO explainability_artifacts (
                    analysis_id, model_id, method, target_layer, target_score,
                    heatmap_path, overlay_path, metadata, warnings, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    payload["model_id"],
                    payload["method"],
                    payload.get("target_layer"),
                    payload.get("target_score"),
                    payload.get("heatmap_path"),
                    payload.get("overlay_path"),
                    _dump(payload.get("metadata", {})),
                    json.dumps(payload.get("warnings", [])),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)
    def add_uncertainty(self, analysis_id: str, payload: dict) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO uncertainty_results (
                    analysis_id, method, n_passes, mean_dispersion, max_dispersion,
                    high_uncertainty_fraction, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    payload["method"],
                    int(payload.get("n_passes", 0)),
                    payload.get("mean_dispersion"),
                    payload.get("max_dispersion"),
                    payload.get("high_uncertainty_fraction"),
                    _dump(payload.get("payload", {})),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)
    def add_report(self, analysis_id: str, payload: dict) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO reports (
                    analysis_id, report_id, path, size_bytes, generator_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    payload["report_id"],
                    payload["path"],
                    payload.get("size_bytes"),
                    payload.get("generator_version"),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)
    def add_assistant_message(
        self, analysis_id: str | None, role: str, content: str, sources: list | None = None
    ) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO assistant_messages (analysis_id, role, content, sources, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (analysis_id, role, content, json.dumps(sources or []), utc_now()),
            )
            return int(cursor.lastrowid)
    def get_assistant_messages(self, analysis_id: str, limit: int = 100) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, sources, created_at FROM assistant_messages
                 WHERE analysis_id = ? ORDER BY id ASC LIMIT ?
                """,
                (analysis_id, max(1, min(int(limit), 500))),
            ).fetchall()
        return rows_to_dicts(rows)
    def get_full_analysis(self, analysis_id: str) -> dict | None:
        analysis = self.get_analysis(analysis_id)
        if analysis is None:
            return None
        with self.connection() as connection:
            segmentation = rows_to_dicts(
                connection.execute(
                    "SELECT * FROM segmentation_results WHERE analysis_id = ? ORDER BY id",
                    (analysis_id,),
                ).fetchall()
            )
            radiomics = rows_to_dicts(
                connection.execute(
                    "SELECT * FROM radiomics_features WHERE analysis_id = ? ORDER BY id",
                    (analysis_id,),
                ).fetchall()
            )
            classification = rows_to_dicts(
                connection.execute(
                    "SELECT * FROM classification_results WHERE analysis_id = ? ORDER BY id DESC LIMIT 1",
                    (analysis_id,),
                ).fetchall()
            )
            explainability = rows_to_dicts(
                connection.execute(
                    "SELECT * FROM explainability_artifacts WHERE analysis_id = ? ORDER BY id",
                    (analysis_id,),
                ).fetchall()
            )
            uncertainty = rows_to_dicts(
                connection.execute(
                    "SELECT * FROM uncertainty_results WHERE analysis_id = ? ORDER BY id DESC LIMIT 1",
                    (analysis_id,),
                ).fetchall()
            )
            reports = rows_to_dicts(
                connection.execute(
                    "SELECT * FROM reports WHERE analysis_id = ? ORDER BY id DESC",
                    (analysis_id,),
                ).fetchall()
            )
        analysis.update(
            {
                "segmentation": segmentation,
                "radiomics": radiomics,
                "classification": classification[0] if classification else None,
                "explainability": explainability,
                "uncertainty": uncertainty[0] if uncertainty else None,
                "reports": reports,
            }
        )
        return analysis
    def get_report(self, report_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM reports WHERE report_id = ?", (report_id,)
            ).fetchone()
        return row_to_dict(row)
    def list_reports(self, *, limit: int = 100) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT reports.*, analyses.scan_identifier, analyses.analysis_type
                  FROM reports LEFT JOIN analyses USING (analysis_id)
                 ORDER BY reports.id DESC LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return rows_to_dicts(rows)
    def record_model_versions(self, entries: list[dict]) -> int:
        written = 0
        with self.connection() as connection:
            for entry in entries:
                connection.execute(
                    """
                    INSERT INTO model_versions (
                        model_id, status, path, checksum, parameter_count,
                        input_spec, health_verdict, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry["model_id"],
                        entry.get("status", "unknown"),
                        entry.get("path"),
                        entry.get("checksum"),
                        entry.get("parameter_count"),
                        _dump(entry.get("input_spec", {})),
                        entry.get("health_verdict"),
                        utc_now(),
                    ),
                )
                written += 1
        return written
    def list_model_versions(self, model_id: str | None = None, limit: int = 50) -> list[dict]:
        query = "SELECT * FROM model_versions"
        params: list[Any] = []
        if model_id:
            query += " WHERE model_id = ?"
            params.append(model_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self.connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return rows_to_dicts(rows)
    def statistics(self) -> dict:
        with self.connection() as connection:
            totals = connection.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM analyses) AS analyses,
                  (SELECT COUNT(*) FROM segmentation_results) AS segmentations,
                  (SELECT COUNT(*) FROM radiomics_features) AS radiomics,
                  (SELECT COUNT(*) FROM reports) AS reports,
                  (SELECT COUNT(*) FROM classification_results WHERE available = 1) AS classifications
                """
            ).fetchone()
            recent = rows_to_dicts(
                connection.execute(
                    "SELECT * FROM analyses ORDER BY created_at DESC LIMIT 5"
                ).fetchall()
            )
        return {"totals": dict(totals), "recent": recent}
