from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
logger = logging.getLogger(__name__)
GENERATOR_VERSION = "reportgen-1.0"
FEATURE_FAMILY_LABELS = {
    "shape": "Shape and size",
    "firstorder": "First-order intensity",
    "glcm": "GLCM texture",
    "glrlm": "Run-length texture",
    "diagnostics": "Diagnostics and histogram",
}
class ReportService:
    def __init__(self, config, repository=None):
        self.config = config
        self.repository = repository
        self.styles = self._build_styles()
    @property
    def report_dir(self) -> Path:
        return Path(self.config.DATA_DIR) / self.config.REPORT_DIR_NAME
    def _build_styles(self):
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=17, spaceAfter=4))
        styles.add(ParagraphStyle("SectionHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4))
        styles.add(ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=12))
        styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontSize=7.5, leading=10, textColor=colors.HexColor("#444444")))
        styles.add(ParagraphStyle("Notice", parent=styles["BodyText"], fontSize=9, leading=12, backColor=colors.HexColor("#fdf2f2"), borderPadding=6))
        return styles
    def generate(self, payload: dict, artifact_root: Path, display_reference: np.ndarray) -> dict:
        from ..utils.io import utc_now, write_bytes_atomic
        analysis_id = payload["analysis_id"]
        report_id = f"{analysis_id}-report"
        directory = self.report_dir / analysis_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{report_id}.pdf"
        story: list[Any] = []
        story += self._header(payload)
        story += self._inputs(payload)
        story += self._models(payload)
        story += self._segmentation(payload, artifact_root, display_reference)
        story += self._ensemble(payload)
        story += self._radiomics(payload)
        story += self._classification(payload)
        story += self._explainability(payload, artifact_root)
        story += self._uncertainty(payload)
        story += self._evaluation_state(payload)
        story += self._limitations()
        document = SimpleDocTemplate(
            str(target),
            pagesize=A4,
            leftMargin=16 * mm,
            rightMargin=16 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
            title=f"Brain abnormality analysis {analysis_id}",
            author="Brain Abnormality Research Platform",
            subject="AI-generated research analysis - not a clinical report",
        )
        document.build(story, onFirstPage=self._footer, onLaterPages=self._footer)
        size = target.stat().st_size
        return {
            "available": True,
            "report_id": report_id,
            "path": str(target),
            "relative_path": str(target.relative_to(Path(self.config.DATA_DIR))),
            "size_bytes": size,
            "generator_version": GENERATOR_VERSION,
            "generated_at": utc_now(),
        }
    def _footer(self, canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(16 * mm, 8 * mm, self.config.DISCLAIMER)
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()
    def _header(self, payload: dict) -> list:
        blocks = [
            Paragraph("Brain Abnormality Analysis Report", self.styles["ReportTitle"]),
            Paragraph(
                "AI-generated research output. Not a medical diagnosis and not a substitute "
                "for review by a qualified radiologist.",
                self.styles["Small"],
            ),
            Spacer(1, 6),
        ]
        rows = [
            ["Analysis ID", payload.get("analysis_id", "-")],
            ["Scan identifier (pseudonymous)", payload.get("scan_identifier", "-")],
            ["Analysis type", payload.get("analysis_type", "-")],
            ["Created (UTC)", payload.get("created_at", "-")],
            ["Compute device", payload.get("device", "-")],
            ["Decision threshold", str(payload.get("threshold", "-"))],
            ["Report generator", GENERATOR_VERSION],
        ]
        blocks.append(self._table(rows, col_widths=[60 * mm, 100 * mm]))
        return blocks
    def _inputs(self, payload: dict) -> list:
        preprocessing = payload.get("preprocessing", {})
        blocks = [Paragraph("1. Input and preprocessing", self.styles["SectionHeading"])]
        rows = [
            ["Pipeline", str(preprocessing.get("pipeline", "-"))],
            ["Tensor shape", str(preprocessing.get("tensor_shape", "-"))],
            ["Channels", str(preprocessing.get("channels", "-"))],
            ["Spatial size", str(preprocessing.get("spatial_size", "-"))],
            ["Normalisation", str(preprocessing.get("normalisation", "-"))],
        ]
        metadata = preprocessing.get("metadata") or {}
        if metadata.get("modalities"):
            rows.append(["Modalities", ", ".join(str(m) for m in metadata["modalities"])])
        if metadata.get("source_format") or metadata.get("original_size"):
            rows.append(
                [
                    "Source image",
                    f"{metadata.get('source_format', '-')} {metadata.get('original_size', '')}".strip(),
                ]
            )
        if metadata.get("global_mean") is not None:
            rows.append(["Global mean / std", f"{metadata['global_mean']:.4f} / {metadata.get('global_std', float('nan')):.4f}"])
        blocks.append(self._table(rows))
        warnings = list(metadata.get("warnings") or [])
        for warning in warnings:
            blocks.append(Paragraph(f"Warning: {warning}", self.styles["Small"]))
        return blocks
    def _models(self, payload: dict) -> list:
        blocks = [Paragraph("2. Model versions used", self.styles["SectionHeading"])]
        rows = [["Model", "Status", "Architecture", "Parameters", "Input specification"]]
        for model_id, entry in (payload.get("models") or {}).items():
            if not isinstance(entry, dict):
                continue
            descriptor = entry.get("descriptor") or {}
            spec = descriptor.get("input_spec") or {}
            health = (entry.get("health") or {}).get("verdict", "-")
            rows.append(
                [
                    model_id,
                    f"{entry.get('status', '-')} / {health}",
                    (descriptor.get("architecture") or "-")[:60],
                    str(entry.get("parameter_count", "-")),
                    f"{spec.get('channels', '-')}ch {spec.get('height', '-')}x{spec.get('width', '-')}",
                ]
            )
        if len(rows) == 1:
            rows.append(["No model information recorded", "-", "-", "-", "-"])
        blocks.append(self._table(rows, small=True))
        for model_id, entry in (payload.get("models") or {}).items():
            health = (entry or {}).get("health") or {}
            if health.get("verdict") and health["verdict"] != "responsive":
                blocks.append(
                    Paragraph(
                        f"{model_id}: {health.get('message', 'health check did not pass')}",
                        self.styles["Small"],
                    )
                )
        return blocks
    def _segmentation(self, payload: dict, artifact_root: Path, display_reference: np.ndarray) -> list:
        from ..utils.images import to_uint8
        blocks = [Paragraph("3. Segmentation output", self.styles["SectionHeading"])]
        panels: list[tuple[str, np.ndarray]] = [("Original input", to_uint8(display_reference))]
        for model_id, entry in (payload.get("models") or {}).items():
            if not isinstance(entry, dict) or not entry.get("available"):
                continue
            artifacts = entry.get("artifacts") or {}
            overlay_name = artifacts.get("overlay")
            if overlay_name:
                path = artifact_root / overlay_name
                if path.exists():
                    panels.append((f"{model_id} overlay", _read_png(path)))
        ensemble = payload.get("ensemble") or {}
        if ensemble.get("available") and (ensemble.get("artifacts") or {}).get("overlay"):
            path = artifact_root / ensemble["artifacts"]["overlay"]
            if path.exists():
                panels.append(("Ensemble overlay", _read_png(path)))
        for index in range(0, len(panels), 3):
            row = panels[index : index + 3]
            blocks.append(self._image_row(row))
            blocks.append(Spacer(1, 4))
        rows = [["Model", "Threshold", "Foreground pixels", "Foreground %", "Mean probability (in mask)", "Time (s)"]]
        for model_id, entry in (payload.get("models") or {}).items():
            if not isinstance(entry, dict) or not entry.get("available"):
                continue
            summary = entry.get("summary") or {}
            rows.append(
                [
                    model_id,
                    str(summary.get("threshold", "-")),
                    str(summary.get("foreground_pixels", "-")),
                    _percent(summary.get("foreground_fraction")),
                    _number(summary.get("probability_mean_in_mask")),
                    _number(summary.get("inference_seconds")),
                ]
            )
        if len(rows) == 1:
            rows.append(["No segmentation output", "-", "-", "-", "-", "-"])
        blocks.append(self._table(rows, small=True))
        blocks.append(
            Paragraph(
                "No reference segmentation exists for this scan, so no Dice, IoU, precision, recall or "
                "specificity value can be computed. Accuracy figures require labelled data.",
                self.styles["Small"],
            )
        )
        return blocks
    def _ensemble(self, payload: dict) -> list:
        ensemble = payload.get("ensemble") or {}
        blocks = [Paragraph("4. Ensemble fusion", self.styles["SectionHeading"])]
        if not ensemble.get("available"):
            reasons = ensemble.get("reasons") or ["The ensemble was not available for this analysis."]
            for reason in reasons:
                blocks.append(Paragraph(f"Unavailable: {reason}", self.styles["Body"]))
            return blocks
        rows = [
            ["Strategy", str(ensemble.get("strategy", "-"))],
            ["Weight source", str(ensemble.get("weight_source", "-"))],
            ["Members", ", ".join(ensemble.get("members", []))],
            ["Weights", ", ".join(f"{k}={v:.3f}" for k, v in (ensemble.get("weights") or {}).items())],
            ["Foreground pixels", str(ensemble.get("foreground_pixels", "-"))],
        ]
        disagreement = ensemble.get("disagreement") or {}
        if disagreement.get("mean_pairwise_dice_agreement") is not None:
            rows.append(["Mean pairwise agreement (Dice)", _number(disagreement["mean_pairwise_dice_agreement"])])
        blocks.append(self._table(rows))
        if disagreement.get("note"):
            blocks.append(Paragraph(disagreement["note"], self.styles["Small"]))
        return blocks
    def _radiomics(self, payload: dict) -> list:
        radiomics = payload.get("radiomics") or {}
        blocks = [Paragraph("5. Quantitative radiomics", self.styles["SectionHeading"])]
        if not radiomics.get("available"):
            blocks.append(
                Paragraph(
                    f"Unavailable: {radiomics.get('unavailable_reason', 'no region of interest')}",
                    self.styles["Body"],
                )
            )
            return blocks
        quality = radiomics.get("quality") or {}
        rows = [
            ["Backend", str(radiomics.get("backend", "-"))],
            ["Dimensionality", str(radiomics.get("dimensionality", "-"))],
            ["Features computed", str(radiomics.get("feature_count", "-"))],
            ["Mask source", str(quality.get("mask_source", "-"))],
            ["Region size", f"{quality.get('roi_voxel_count', '-')} pixels/voxels"],
            ["Physical spacing available", "yes" if radiomics.get("spacing_available") else "no"],
        ]
        blocks.append(self._table(rows))
        features = radiomics.get("features") or {}
        categories = radiomics.get("categories") or {}
        for family, names in categories.items():
            if not names:
                continue
            blocks.append(
                Paragraph(FEATURE_FAMILY_LABELS.get(family, family.title()), self.styles["Body"])
            )
            rows = [["Feature", "Value"]]
            for name in names[:24]:
                rows.append([name, _number(features.get(name))])
            if len(names) > 24:
                rows.append([f"... {len(names) - 24} more", ""])
            blocks.append(self._table(rows, small=True))
            blocks.append(Spacer(1, 3))
        for warning in radiomics.get("warnings") or []:
            blocks.append(Paragraph(f"Note: {warning}", self.styles["Small"]))
        if quality.get("note"):
            blocks.append(Paragraph(quality["note"], self.styles["Small"]))
        return blocks
    def _classification(self, payload: dict) -> list:
        classification = payload.get("classification") or {}
        blocks = [Paragraph("6. Classification", self.styles["SectionHeading"])]
        if not classification.get("available"):
            blocks.append(
                Paragraph(
                    f"Not available. {classification.get('reason', 'No validated classifier artefact is present.')}",
                    self.styles["Body"],
                )
            )
            missing = classification.get("missing_features") or []
            if missing:
                blocks.append(Paragraph("Missing features: " + ", ".join(missing[:20]), self.styles["Small"]))
            return blocks
        rows = [
            ["Model", str(classification.get("model_id", "-"))],
            ["Version", str(classification.get("model_version", "-"))],
            ["Predicted class", str(classification.get("predicted_label", "-"))],
            ["Scores", ", ".join(_number(s) for s in classification.get("scores", []))],
            ["Calibrated", "yes" if classification.get("calibrated") else "no"],
        ]
        metrics = classification.get("metrics") or {}
        for key in ("accuracy", "macro_f1", "weighted_f1", "roc_auc", "pr_auc"):
            if metrics.get(key) is not None:
                rows.append([key, _number(metrics[key])])
        for key in ("evaluated_model", "evaluation_split", "samples", "precision_macro", "recall_macro"):
            if metrics.get(key) is not None:
                rows.append([key, str(metrics[key])])
        blocks.append(self._table(rows))
        label_mapping = classification.get("label_mapping") or {}
        per_class = metrics.get("per_class") or {}
        if per_class:
            blocks.append(Paragraph("Per-class metrics on the held-out split", self.styles["Body"]))
            table = [["Class", "Precision", "Recall", "F1", "Support"]]
            for index, entry in sorted(per_class.items()):
                name = label_mapping.get(str(index), str(index))
                table.append(
                    [
                        f"{name} ({index})",
                        _number(entry.get("precision")),
                        _number(entry.get("recall")),
                        _number(entry.get("f1")),
                        str(entry.get("support")),
                    ]
                )
            blocks.append(self._table(table, small=True))
        matrix = metrics.get("confusion_matrix") or []
        order = metrics.get("confusion_matrix_order") or sorted(label_mapping)
        if matrix:
            blocks.append(Paragraph("Confusion matrix (rows: reference, columns: predicted)", self.styles["Body"]))
            table = [["Reference \\ Predicted"] + [label_mapping.get(str(item), str(item)) for item in order]]
            for row_index, row in enumerate(matrix):
                label = label_mapping.get(str(order[row_index]), str(order[row_index]))
                table.append([label] + [str(value) for value in row])
            blocks.append(self._table(table, small=True))
        comparison = classification.get("model_comparison") or {}
        if len(comparison) > 1:
            blocks.append(Paragraph("Baseline versus improved model", self.styles["Body"]))
            table = [["Model", "Split", "Accuracy", "Macro-F1", "ROC-AUC"]]
            for name, payload in comparison.items():
                for split in ("validation", "test"):
                    entry = (payload or {}).get(split) or {}
                    if not entry:
                        continue
                    table.append(
                        [
                            name,
                            split,
                            _number(entry.get("accuracy")),
                            _number(entry.get("macro_f1")),
                            _number(entry.get("roc_auc")),
                        ]
                    )
            blocks.append(self._table(table, small=True))
        if classification.get("statement"):
            blocks.append(Paragraph(classification["statement"], self.styles["Small"]))
        for caveat in (classification.get("caveats") or [])[:4]:
            blocks.append(Paragraph(caveat, self.styles["Small"]))
        contributions = classification.get("feature_contributions") or []
        if contributions:
            blocks.append(Paragraph("Per-case feature contributions", self.styles["Body"]))
            rows = [["Feature", "Value", "Contribution"]]
            for item in contributions[:15]:
                rows.append(
                    [item["feature"], _number(item.get("value")), _number(item.get("contribution"))]
                )
            blocks.append(self._table(rows, small=True))
        return blocks
    def _explainability(self, payload: dict, artifact_root: Path) -> list:
        blocks = [Paragraph("7. Explainability", self.styles["SectionHeading"])]
        entries = payload.get("explainability") or []
        if not entries:
            blocks.append(Paragraph("No explanation was generated for this analysis.", self.styles["Body"]))
            return blocks
        panels = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("available"):
                blocks.append(
                    Paragraph(
                        f"{entry.get('model_id', 'model')}: explanation unavailable - {entry.get('reason', 'unknown reason')}",
                        self.styles["Small"],
                    )
                )
                continue
            overlay = (entry.get("artifacts") or {}).get("overlay")
            if overlay and (artifact_root / overlay).exists():
                panels.append((f"{entry['model_id']} Grad-CAM", _read_png(artifact_root / overlay)))
            rows = [
                ["Model", str(entry.get("model_id", "-"))],
                ["Method", str(entry.get("method", "-"))],
                ["Target layer", str(entry.get("target_layer", "-"))],
                ["Target score", str(entry.get("target_score", "-"))],
                ["Activation shape", str(entry.get("activation_shape", "-"))],
            ]
            blocks.append(self._table(rows, small=True))
            if entry.get("target_layer_semantics"):
                blocks.append(Paragraph(entry["target_layer_semantics"], self.styles["Small"]))
            if entry.get("explanation"):
                blocks.append(Paragraph(entry["explanation"], self.styles["Small"]))
            for warning in entry.get("warnings") or []:
                blocks.append(Paragraph(f"Note: {warning}", self.styles["Small"]))
            blocks.append(Spacer(1, 4))
        if panels:
            blocks.append(self._image_row(panels[:3]))
        blocks.append(
            Paragraph(
                "A Grad-CAM map shows which regions influenced the model's output. It is not a "
                "radiological finding and it does not establish the presence of a tumour.",
                self.styles["Small"],
            )
        )
        return blocks
    def _uncertainty(self, payload: dict) -> list:
        uncertainty = payload.get("uncertainty")
        blocks = [Paragraph("8. Uncertainty", self.styles["SectionHeading"])]
        if not uncertainty:
            blocks.append(
                Paragraph(
                    "Not estimated for this analysis. Uncertainty is computed only when requested, "
                    "and only for methods that are valid for the selected model.",
                    self.styles["Body"],
                )
            )
            return blocks
        if not uncertainty.get("available"):
            blocks.append(Paragraph(str(uncertainty.get("unavailable_reason", "Unavailable.")), self.styles["Body"]))
            return blocks
        rows = [
            ["Method", str(uncertainty.get("method", "-"))],
            ["Passes", str(uncertainty.get("n_passes", "-"))],
            ["Mean dispersion", _number(uncertainty.get("mean_dispersion"))],
            ["95th percentile dispersion", _number(uncertainty.get("p95_dispersion"))],
            ["Maximum dispersion", _number(uncertainty.get("max_dispersion"))],
            ["Pixels with dispersion > 0.1", _percent(uncertainty.get("high_uncertainty_fraction"))],
            ["Boundary dispersion", _number(uncertainty.get("boundary_dispersion"))],
            ["Interior dispersion", _number(uncertainty.get("interior_dispersion"))],
        ]
        blocks.append(self._table(rows))
        for note in uncertainty.get("notes") or []:
            if note:
                blocks.append(Paragraph(note, self.styles["Small"]))
        blocks.append(
            Paragraph(
                "Dispersion describes how much the model's own prediction changes under "
                "perturbation or across passes. It is not a probability that an abnormality is "
                "present, and it cannot detect a mistake that every pass makes identically.",
                self.styles["Small"],
            )
        )
        return blocks
    def _evaluation_state(self, payload: dict) -> list:
        blocks = [Paragraph("9. Model comparison on this scan", self.styles["SectionHeading"])]
        comparison = payload.get("model_comparison") or {}
        blocks.append(Paragraph(comparison.get("note", ""), self.styles["Small"]))
        rows = [["Model", "Foreground pixels", "Mean probability", "Max probability", "Health", "Time (s)"]]
        for entry in comparison.get("models", []):
            rows.append(
                [
                    entry.get("model_id", "-"),
                    str(entry.get("foreground_pixels", "-")),
                    _number(entry.get("probability_mean")),
                    _number(entry.get("probability_max")),
                    str(entry.get("health_verdict", "-")),
                    _number(entry.get("inference_seconds")),
                ]
            )
        if len(rows) == 1:
            rows.append(["No model output", "-", "-", "-", "-", "-"])
        blocks.append(self._table(rows, small=True))
        agreement = comparison.get("pairwise_agreement") or {}
        if agreement:
            blocks.append(Paragraph("Agreement between models (not accuracy)", self.styles["Body"]))
            rows = [["Pair", "Dice agreement", "IoU agreement", "Only in first", "Only in second"]]
            for pair, values in agreement.items():
                rows.append(
                    [
                        pair,
                        _number(values.get("dice_agreement")),
                        _number(values.get("iou_agreement")),
                        str(values.get("only_in_first", "-")),
                        str(values.get("only_in_second", "-")),
                    ]
                )
            blocks.append(self._table(rows, small=True))
        blocks.append(
            Paragraph(
                "Evaluation against a reference standard: Not evaluated. This deployment holds no "
                "labelled test set, so Dice, IoU, precision, recall, specificity, accuracy and "
                "AUC values are not reported. Any such value must come from "
                "scripts/evaluate_models.py run on the reserved test split.",
                self.styles["Body"],
            )
        )
        return blocks
    def _limitations(self) -> list:
        blocks = [
            PageBreak(),
            Paragraph("10. Limitations and disclaimer", self.styles["SectionHeading"]),
        ]
        limitations = [
            "The analysis is produced by machine-learning models trained on research data; their behaviour on data outside that distribution is not characterised.",
            "No reference segmentation was available for this scan, so no segmentation accuracy metric can be reported.",
            "An empty predicted region is not evidence that no abnormality is present; it may also mean the input does not match the model's training distribution.",
            "Radiomics features describe the region produced by the segmentation model. If that region is inaccurate, every derived value inherits the error.",
            "Shape features describe the segmented region; for a two-dimensional slice no physical volume is computed, and areas are given in pixels unless pixel spacing was supplied.",
            "Model scores are not calibrated clinical probabilities unless a calibration artefact recorded in this report says so.",
            "Explainability maps describe how the model reached its output; they are not radiological findings.",
            "This system cannot determine malignancy, tumour grade, histological type, stage or prognosis, and it must not be used to make treatment decisions.",
            "The output must be reviewed by a qualified professional and must not be used as the sole basis for any clinical decision.",
        ]
        for item in limitations:
            blocks.append(Paragraph(f"- {item}", self.styles["Body"]))
        blocks.append(Spacer(1, 6))
        blocks.append(Paragraph(self.config.DISCLAIMER, self.styles["Notice"]))
        return blocks
    def _table(self, rows: list[list[str]], col_widths: list[float] | None = None, small: bool = False):
        style = self.styles["Small"] if small else self.styles["Body"]
        width = A4[0] - 32 * mm
        if col_widths is None:
            columns = len(rows[0])
            if columns == 2:
                col_widths = [width * 0.35, width * 0.65]
            else:
                col_widths = [width / columns] * columns
        table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5 if small else 8.5),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d2dd")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        del style
        return table
    def _image_row(self, panels: list[tuple[str, np.ndarray]]):
        from ..utils.images import encode_png
        cells = []
        for label, array in panels:
            payload = encode_png(array)
            import io
            image = Image(io.BytesIO(payload))
            max_width = (A4[0] - 40 * mm) / 3
            ratio = image.imageHeight / max(image.imageWidth, 1)
            image.drawWidth = max_width
            image.drawHeight = max_width * ratio
            image.hAlign = "CENTER"
            cells.append([image, Paragraph(label, self.styles["Small"])])
        if not cells:
            return Spacer(1, 0)
        table = Table([cells], hAlign="LEFT")
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return KeepTogether(table)
def _read_png(path: Path) -> np.ndarray:
    from PIL import Image as PILImage
    with PILImage.open(path) as img:
        return np.asarray(img.convert("RGB"))
def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) >= 1e5 or abs(number) < 1e-4:
        return f"{number:.3e}"
    return f"{round(number, digits):g}"
def _percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.3f}%"
    except (TypeError, ValueError):
        return str(value)
