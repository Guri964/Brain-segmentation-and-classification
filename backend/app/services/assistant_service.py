from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable
logger = logging.getLogger(__name__)
REFUSAL_PATTERNS = (
    "malignan",
    "cancer",
    "is it tumor",
    "is it a tumor",
    "is it a tumour",
    "survival",
    "prognos",
    "how long",
    "treatment",
    "chemo",
    "surgery",
    "stage",
    "clinical grade",
    "histolog",
    "biopsy",
    "benign",
    "tumour type",
    "tumor type",
    "grade does",
    "will the patient",
    "cure",
    "diagnose",
    "diagnosis of",
    "should we operate",
)
SAFETY_REPLY = (
    "I can only explain what this analysis produced; I cannot diagnose, predict prognosis, "
    "assign a clinical grade or advise on treatment, and the models behind this analysis were "
    "not trained for those questions. What the analysis does contain is a segmented region, "
    "measurements of that region, radiomics features, an explanation map, a dataset-label "
    "prediction when a validated classifier is installed, and an honest statement of what "
    "could not be computed. Ask me about any of those and I will quote the actual values."
)
DISCLAIMER = (
    "This is a research prototype. Everything described here is AI-generated model output and "
    "must be reviewed by a qualified professional; it is not a medical diagnosis."
)
@dataclass
class AssistantAnswer:
    answer: str
    topic: str
    sources: list[str] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    backend: str = "local"
    suggestions: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)
    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "topic": self.topic,
            "sources": self.sources,
            "grounding": self.grounding,
            "backend": self.backend,
            "suggestions": self.suggestions,
            "disclaimers": self.disclaimers,
            "scope": (
                "Answers describe this analysis only. Educational statements are marked as such."
            ),
        }
SUGGESTED_QUESTIONS = [
    "What does the segmented region mean?",
    "How large is the segmented region?",
    "What do the radiomics features measure?",
    "What does the Grad-CAM map show?",
    "Why is there no accuracy value?",
    "Why is classification unavailable?",
    "What does the uncertainty value mean?",
    "What are the limitations of this analysis?",
]
class AssistantService:
    def __init__(self, config, repository=None):
        self.config = config
        self.repository = repository
    def status(self) -> dict:
        enabled = bool(getattr(self.config, "ASSISTANT_LLM_ENABLED", False))
        key_present = bool(os.environ.get("BRAIN_ASSISTANT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        return {
            "backend": "llm" if (enabled and key_present) else "local",
            "local_backend_available": True,
            "llm_requested": enabled,
            "llm_key_present": key_present,
            "llm_model": getattr(self.config, "ASSISTANT_LLM_MODEL", None) if enabled else None,
            "note": (
                "The local backend answers from the stored analysis payload without any network "
                "call. The optional LLM backend reads its API key from the environment; no key is "
                "stored by the application."
                if enabled
                else "The assistant runs locally and offline. Set BRAIN_ASSISTANT_LLM_ENABLED=1 and "
                "provide an API key in the environment to enable the optional language-model "
                "backend; the grounded, offline answers remain the default."
            ),
            "suggested_questions": SUGGESTED_QUESTIONS,
            "disclaimer": DISCLAIMER,
        }
    def answer(self, question: str, analysis: dict | None) -> AssistantAnswer:
        question = (question or "").strip()
        if not question:
            return AssistantAnswer(
                answer="Ask a question about the analysis currently open, for example: "
                + SUGGESTED_QUESTIONS[0],
                topic="empty",
                suggestions=SUGGESTED_QUESTIONS,
            )
        if _is_unsafe(question):
            return AssistantAnswer(
                answer=SAFETY_REPLY,
                topic="out_of_scope",
                sources=["safety_policy"],
                disclaimers=[DISCLAIMER],
                suggestions=SUGGESTED_QUESTIONS,
            )
        if analysis is None:
            return AssistantAnswer(
                answer=(
                    "No analysis is open, so I have no results to describe. Load an analysis from "
                    "the history first; I only answer from the analysis that is actually stored, "
                    "never from assumptions."
                ),
                topic="no_context",
                disclaimers=[DISCLAIMER],
                suggestions=SUGGESTED_QUESTIONS,
            )
        local = self._local_answer(question, analysis)
        if not (bool(getattr(self.config, "ASSISTANT_LLM_ENABLED", False)) and _api_key()):
            return local
        enriched = self._llm_answer(question, analysis, local)
        return enriched
    def _local_answer(self, question: str, analysis: dict) -> AssistantAnswer:
        text = question.lower()
        intent = _classify(text)
        handler: Callable[[dict], AssistantAnswer] = getattr(self, f"_topic_{intent}")
        answer = handler(analysis)
        if not answer.disclaimers:
            answer.disclaimers = [DISCLAIMER]
        answer.suggestions = answer.suggestions or SUGGESTED_QUESTIONS
        return answer
    def _topic_segmentation(self, analysis: dict) -> AssistantAnswer:
        models = analysis.get("models") or {}
        completed = {
            model_id: entry
            for model_id, entry in models.items()
            if isinstance(entry, dict) and entry.get("available")
        }
        if not completed:
            return AssistantAnswer(
                answer="This analysis has no segmentation output. "
                + _why_no_segmentation(models),
                topic="segmentation",
                sources=["models"],
            )
        lines = []
        for model_id, entry in completed.items():
            summary = entry.get("summary") or {}
            pixels = summary.get("foreground_pixels")
            fraction = summary.get("foreground_fraction")
            lines.append(
                f"{model_id}: marked {pixels} pixels as the region of interest "
                f"({fraction * 100:.3f}% of the image)" if isinstance(fraction, (int, float)) else f"{model_id}: output available"
            )
            health = (entry.get("health") or {}).get("verdict")
            if health and health != "responsive":
                lines.append(
                    f"  Its reference health check returned '{health}': "
                    f"{(entry.get('health') or {}).get('message', '')}"
                )
        reference = analysis.get("reference_mask") or {}
        answer = (
            "Segmentation assigns a tumour/abnormality probability to every pixel and then "
            "thresholds it. For this analysis: "
            + " ".join(lines)
            + f" The region used for the measurements came from {reference.get('source', 'the primary model')}."
        )
        note = (
            " A segmented region is the model's own estimate of where abnormal tissue might be. "
            "Without a reference segmentation it is not possible to say whether that estimate is "
            "correct, which is why no accuracy figure appears anywhere in this analysis."
        )
        return AssistantAnswer(
            answer=answer + note,
            topic="segmentation",
            sources=["models", "reference_mask"],
            grounding={"reference_mask": reference},
        )
    def _topic_measurements(self, analysis: dict) -> AssistantAnswer:
        radiomics = analysis.get("radiomics") or {}
        if not radiomics.get("available"):
            return AssistantAnswer(
                answer=(
                    "No region measurements exist for this analysis. "
                    + str(radiomics.get("unavailable_reason", ""))
                    + " When the region is empty, that is recorded as an unavailable measurement, "
                    "not as a measurement of zero."
                ),
                topic="measurements",
                sources=["radiomics"],
            )
        features = radiomics.get("features") or {}
        quality = radiomics.get("quality") or {}
        measurements = analysis.get("region_measurements") or {}
        region_line = ""
        if measurements.get("detected"):
            bbox = measurements.get("bounding_box") or {}
            centroid = measurements.get("centroid") or {}
            region_line = (
                f" The region covers {measurements.get('pixel_area')} pixels "
                f"({measurements.get('image_coverage_percent')} % of the image); its bounding box is "
                f"{bbox.get('width')} x {bbox.get('height')} pixels starting at "
                f"(x={bbox.get('x')}, y={bbox.get('y')}) and its centroid is at "
                f"(x={centroid.get('x')}, y={centroid.get('y')})."
            )
        interesting = [
            ("shape_pixel_area", "pixels covered by the region"),
            ("shape_perimeter_px", "boundary length in pixels"),
            ("shape_circularity", "how round the region is (an ideal disc on this grid scores about 0.88)"),
            ("shape_elongation", "how stretched the region is"),
            ("firstorder_mean", "mean intensity inside the region"),
            ("firstorder_median", "median intensity inside the region"),
            ("firstorder_standard_deviation", "intensity spread inside the region"),
            ("glcm_contrast", "how quickly intensity changes between neighbouring pixels"),
            ("glcm_homogeneity", "how uniform the local texture is"),
            ("glrlm_short_run_emphasis", "how fine-grained the texture is"),
            ("diagnostics_entropy", "entropy of the intensity histogram inside the region"),
        ]
        lines = []
        for name, description in interesting:
            if name in features and features[name] is not None:
                value = features[name]
                lines.append(f"{name} = {value:.4g} ({description})")
        spacing = (
            "Pixel spacing was supplied, so physical units are available."
            if radiomics.get("spacing_available")
            else "No pixel spacing was supplied, so areas are reported in pixels and no physical "
            "volume is computed for this 2-D slice."
        )
        return AssistantAnswer(
            answer=(
                "The measurements describe the segmented region. "
                f"There are {radiomics.get('feature_count')} features in total, computed with the "
                f"{radiomics.get('backend')} extractor on a {radiomics.get('dimensionality')} region. "
                + "; ".join(lines[:8])
                + f". Region size: {quality.get('roi_voxel_count', 'unknown')} pixels. {spacing}"
                + region_line
            ),
            topic="measurements",
            sources=["radiomics.features", "radiomics.quality", "region_measurements"],
            grounding={"feature_count": radiomics.get("feature_count"), "region_size": quality.get("roi_voxel_count")},
        )
    def _topic_radiomics(self, analysis: dict) -> AssistantAnswer:
        radiomics = analysis.get("radiomics") or {}
        if not radiomics.get("available"):
            return AssistantAnswer(
                answer="No radiomics features were computed for this analysis. "
                + str(radiomics.get("unavailable_reason", "")),
                topic="radiomics",
                sources=["radiomics"],
            )
        categories = radiomics.get("categories") or {}
        counts = {family: len(names) for family, names in categories.items()}
        return AssistantAnswer(
            answer=(
                "Radiomics converts the segmented region into numbers that describe its shape, its "
                "intensity statistics and the texture of the pixels inside it. This analysis used "
                f"the {radiomics.get('backend')} extractor and produced {radiomics.get('feature_count')} "
                f"features: shape {counts.get('shape', 0)}, first-order intensity "
                f"{counts.get('firstorder', 0)}, GLCM texture {counts.get('glcm', 0)}, run-length "
                f"texture {counts.get('glrlm', 0)}, diagnostics {counts.get('diagnostics', 0)}. "
                "These numbers describe the region the model selected; if that region is wrong, the "
                "numbers inherit the error."
            ),
            topic="radiomics",
            sources=["radiomics.categories"],
            grounding={"counts": counts, "backend": radiomics.get("backend")},
        )
    def _topic_explainability(self, analysis: dict) -> AssistantAnswer:
        entries = [e for e in (analysis.get("explainability") or []) if isinstance(e, dict)]
        available = [e for e in entries if e.get("available")]
        if not available:
            reason = entries[0].get("reason") if entries else "no explanation was generated"
            return AssistantAnswer(
                answer=f"No Grad-CAM map is available for this analysis ({reason}).",
                topic="explainability",
                sources=["explainability"],
            )
        parts = []
        for entry in available:
            parts.append(
                f"{entry.get('model_id')}: heatmap computed from layer '{entry.get('target_layer')}' "
                f"with target score '{entry.get('target_score')}'"
            )
            if entry.get("warnings"):
                parts.append("notes: " + "; ".join(entry["warnings"]))
        return AssistantAnswer(
            answer=(
                "Grad-CAM back-propagates the model's own output to a chosen layer and shows which "
                "image regions pushed that output up. "
                + " | ".join(parts)
                + ". Warm colours are the regions that influenced the model's decision most. The map "
                "explains the model, not the patient: high activation is not a diagnosis, and a "
                "region the model ignored is not proof that it is healthy."
            ),
            topic="explainability",
            sources=["explainability"],
            grounding={"models": [e.get("model_id") for e in available]},
        )
    def _topic_uncertainty(self, analysis: dict) -> AssistantAnswer:
        uncertainty = analysis.get("uncertainty")
        if not uncertainty:
            return AssistantAnswer(
                answer=(
                    "No uncertainty estimate was requested for this analysis. It is computed only "
                    "when enabled, because it costs extra forward passes."
                ),
                topic="uncertainty",
                sources=["uncertainty"],
            )
        if not uncertainty.get("available"):
            return AssistantAnswer(
                answer="Uncertainty could not be estimated: "
                + str(uncertainty.get("unavailable_reason", "reason not recorded")),
                topic="uncertainty",
                sources=["uncertainty"],
            )
        return AssistantAnswer(
            answer=(
                f"Uncertainty was estimated with {uncertainty.get('method')} over "
                f"{uncertainty.get('n_passes')} passes. The mean pixel-wise dispersion is "
                f"{_fmt(uncertainty.get('mean_dispersion'))}, the 95th percentile is "
                f"{_fmt(uncertainty.get('p95_dispersion'))}, and "
                f"{_pct(uncertainty.get('high_uncertainty_fraction'))} of pixels have a dispersion "
                "above 0.1. Dispersion measures how much the model's own prediction moves when the "
                "input is perturbed or when dropout is active. It is not the probability that an "
                "abnormality exists, and it cannot reveal an error that every pass repeats."
            ),
            topic="uncertainty",
            sources=["uncertainty"],
        )
    def _topic_classification(self, analysis: dict) -> AssistantAnswer:
        classification = analysis.get("classification") or {}
        if not classification.get("available"):
            missing = classification.get("missing_features") or []
            detail = f" Missing features: {', '.join(missing[:10])}." if missing else ""
            return AssistantAnswer(
                answer=(
                    "Classification is disabled for this deployment, so no class was predicted. "
                    + str(classification.get("reason", ""))
                    + detail
                    + " Presenting a class label without a validated model, a recorded feature "
                    "schema and held-out metrics would mean inventing a result."
                ),
                topic="classification",
                sources=["classification.reason"],
            )
        metrics = classification.get("metrics") or {}
        wanted = (
            ("accuracy", "test accuracy"),
            ("precision_macro", "test macro precision"),
            ("recall_macro", "test macro recall"),
            ("macro_f1", "test macro F1"),
            ("roc_auc", "test ROC-AUC"),
            ("samples", "test samples"),
        )
        metric_text = ", ".join(
            f"{label} {_fmt(metrics[key])}" for key, label in wanted if isinstance(metrics.get(key), (int, float))
        )
        comparison = classification.get("model_comparison") or {}
        selected = (classification.get("metrics") or {}).get("evaluated_model")
        comparison_text = ""
        if comparison:
            entries = []
            for name, payload in comparison.items():
                test = (payload or {}).get("test") or {}
                if test.get("macro_f1") is None:
                    continue
                entries.append(f"{name} (test macro-F1 {_fmt(test['macro_f1'])})")
            if entries:
                comparison_text = (
                    " Both models were scored on the same held-out split: "
                    + "; ".join(entries)
                    + f". The served one is {selected}."
                )
        evaluated_on = classification.get("evaluated_on")
        return AssistantAnswer(
            answer=(
                f"The classifier ({classification.get('model_id')}) predicted class "
                f"'{classification.get('predicted_label')}' from {len(classification.get('scores', []))} "
                f"class scores. Calibrated: {'yes' if classification.get('calibrated') else 'no'}. "
                + (f"Recorded held-out metrics: {metric_text}. " if metric_text else "")
                + (f"They were measured on the {evaluated_on}. " if evaluated_on else "")
                + comparison_text
                + " This is a dataset-defined label, not a clinical diagnosis, and it is not derived "
                "from the segmentation mask."
            ),
            topic="classification",
            sources=["classification"],
        )
    def _topic_metrics(self, analysis: dict) -> AssistantAnswer:
        return AssistantAnswer(
            answer=(
                "This analysis reports no accuracy metric, and that is intentional. Dice, IoU, "
                "precision, recall, sensitivity, specificity and AUC all need a reference standard - "
                "a mask drawn by an expert - to compare against. The uploaded scan has no such "
                "reference in this deployment, so any number of that kind would be fabricated. What "
                "you can legitimately see here is the model's own output statistics (how many pixels "
                "were marked, with what probability) and the agreement between the models, which is a "
                "consistency measure rather than a correctness measure. Accuracy is reported only by "
                "scripts/evaluate_models.py, on a reserved labelled test set, and is displayed as "
                "'Not evaluated' until that has been run."
            ),
            topic="metrics",
            sources=["policy"],
        )
    def _topic_limitations(self, analysis: dict) -> AssistantAnswer:
        health = {
            model_id: (entry.get("health") or {}).get("verdict")
            for model_id, entry in (analysis.get("models") or {}).items()
            if isinstance(entry, dict)
        }
        non_responsive = [model_id for model_id, verdict in health.items() if verdict and verdict != "responsive"]
        text = (
            "The limitations that apply to this analysis: no reference segmentation is available, so "
            "segmentation accuracy cannot be reported; the segmentations come from research models "
            "whose behaviour outside their training distribution is not characterised; radiomics "
            "values describe the predicted region and inherit its errors; the models cannot determine "
            "malignancy, grade, histology or prognosis; and the explanation maps describe the model "
            "rather than the patient."
        )
        if non_responsive:
            text += (
                " In addition, these checkpoints failed the reference health check and their masks "
                "should be treated as invalid rather than as evidence of a healthy scan: "
                + ", ".join(non_responsive)
                + "."
            )
        if not analysis.get("radiomics", {}).get("available"):
            text += " No radiomics measurement is available for this case."
        return AssistantAnswer(answer=text, topic="limitations", sources=["analysis", "models"])
    def _topic_models(self, analysis: dict) -> AssistantAnswer:
        models = analysis.get("models") or {}
        lines = []
        for model_id, entry in models.items():
            if not isinstance(entry, dict):
                continue
            descriptor = entry.get("descriptor") or {}
            spec = descriptor.get("input_spec") or {}
            lines.append(
                f"{model_id}: {descriptor.get('architecture', 'architecture not recorded')[:90]} "
                f"(input {spec.get('channels')}x{spec.get('height')}x{spec.get('width')}, "
                f"output {descriptor.get('output_semantics')}, status {entry.get('status')})"
            )
        ensemble = analysis.get("ensemble") or {}
        ensemble_text = (
            f"The ensemble ran with strategy '{ensemble.get('strategy')}' and weights "
            f"{ensemble.get('weights')} taken from {ensemble.get('weight_source')}."
            if ensemble.get("available")
            else "The ensemble did not run: " + "; ".join(ensemble.get("reasons") or ["not requested"])
        )
        return AssistantAnswer(
            answer="Models used in this analysis: " + " | ".join(lines) + ". " + ensemble_text,
            topic="models",
            sources=["models", "ensemble"],
        )
    def _topic_overview(self, analysis: dict) -> AssistantAnswer:
        models = analysis.get("models") or {}
        completed = [m for m, e in models.items() if isinstance(e, dict) and e.get("available")]
        radiomics = analysis.get("radiomics") or {}
        return AssistantAnswer(
            answer=(
                f"Analysis {analysis.get('analysis_id')} ({analysis.get('analysis_type')}) ran on "
                f"{analysis.get('device')} with threshold {analysis.get('threshold')}. "
                f"Models that produced an output: {', '.join(completed) or 'none'}. "
                f"Radiomics: {'available with ' + str(radiomics.get('feature_count')) + ' features' if radiomics.get('available') else 'unavailable'}. "
                f"Classification: {'available' if (analysis.get('classification') or {}).get('available') else 'disabled'}. "
                f"Uncertainty: {'estimated' if analysis.get('uncertainty') else 'not requested'}. "
                "Ask about any of these and I will quote the stored values."
            ),
            topic="overview",
            sources=["analysis"],
        )
    def _llm_answer(self, question: str, analysis: dict, local: AssistantAnswer) -> AssistantAnswer:
        context = self._llm_context(analysis)
        system = (
            "You explain the results of a brain MRI research analysis to the person who ran it. "
            "Use only the JSON context provided; never invent values, never diagnose, never discuss "
            "prognosis or treatment. If a value is absent from the context, say that it is not "
            "available in this analysis. Always separate 'what this analysis shows' from 'general "
            "background'. End with the sentence: " + DISCLAIMER
        )
        try:
            text = _call_chat_completion(
                base_url=getattr(self.config, "ASSISTANT_LLM_BASE_URL", "") or "https://api.openai.com/v1",
                model=getattr(self.config, "ASSISTANT_LLM_MODEL", "gpt-4o-mini"),
                api_key=_api_key(),
                system=system,
                user=json.dumps({"question": question, "analysis": context}, default=str)[:24000],
            )
        except Exception as exc:
            logger.warning("LLM backend failed (%s); using the local grounded answer", type(exc).__name__)
            local.answer += (
                " (The optional language-model backend could not be reached, so this answer was "
                "produced by the offline grounded backend.)"
            )
            return local
        return AssistantAnswer(
            answer=text,
            topic=local.topic,
            sources=local.sources + ["llm_backend"],
            grounding=local.grounding,
            backend="llm",
            suggestions=SUGGESTED_QUESTIONS,
            disclaimers=[DISCLAIMER],
        )
    @staticmethod
    def _llm_context(analysis: dict) -> dict:
        return {
            "analysis_id": analysis.get("analysis_id"),
            "analysis_type": analysis.get("analysis_type"),
            "threshold": analysis.get("threshold"),
            "device": analysis.get("device"),
            "models": {
                model_id: {
                    "status": entry.get("status"),
                    "summary": (entry.get("summary") or {}),
                    "health": (entry.get("health") or {}).get("verdict"),
                }
                for model_id, entry in (analysis.get("models") or {}).items()
                if isinstance(entry, dict)
            },
            "ensemble": {
                key: analysis.get("ensemble", {}).get(key)
                for key in ("available", "strategy", "weights", "weight_source", "reasons", "disagreement")
            },
            "radiomics": {
                "available": (analysis.get("radiomics") or {}).get("available"),
                "feature_count": (analysis.get("radiomics") or {}).get("feature_count"),
                "dimensionality": (analysis.get("radiomics") or {}).get("dimensionality"),
                "top_features": _top_features(analysis.get("radiomics") or {}),
                "quality": (analysis.get("radiomics") or {}).get("quality"),
            },
            "explainability": [
                {k: e.get(k) for k in ("model_id", "target_layer", "target_score", "available")}
                for e in (analysis.get("explainability") or [])
                if isinstance(e, dict)
            ],
            "uncertainty": {
                k: (analysis.get("uncertainty") or {}).get(k)
                for k in ("available", "method", "mean_dispersion", "p95_dispersion", "n_passes")
            },
            "classification": {
                k: (analysis.get("classification") or {}).get(k)
                for k in ("available", "predicted_label", "reason", "calibrated", "metrics")
            },
            "policy": (
                "No accuracy metric exists for this analysis because no reference segmentation is "
                "available. Do not produce or imply one."
            ),
        }
def _why_no_segmentation(models: dict) -> str:
    if not models:
        return "No segmentation model was recorded for it."
    parts = []
    for model_id, entry in models.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("available"):
            continue
        status = entry.get("status", "unknown")
        if status == "missing":
            parts.append(
                f"{model_id} has no checkpoint in this deployment; it is produced by "
                f"{entry.get('trainer') or 'the training scripts'}."
            )
        elif status == "incompatible":
            parts.append(f"{model_id} is present but incompatible with its architecture: {entry.get('message', '')}")
        else:
            parts.append(f"{model_id} reported status '{status}': {entry.get('message', 'no further detail')}")
    return " ".join(parts) if parts else "No model produced an output."
def _top_features(radiomics: dict, limit: int = 20) -> dict:
    features = radiomics.get("features") or {}
    keys = [k for k in features if not k.startswith("diagnostics_spacing")]
    return {k: features[k] for k in keys[:limit]}
def _is_unsafe(question: str) -> bool:
    lowered = question.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)
def _classify(text: str) -> str:
    rules = [
        # Size questions are measurements, even when they are phrased around the mask or the
        # segmented region ("how large is the segmented region?"), so this rule comes first.
        ("measurements", ("how large", "size", "area", "volume", "measurement", "how big", "how many pixels")),
        ("segmentation", ("segment", "mask", "region of interest", "roi", "highlight")),
        ("radiomics", ("radiomic", "feature", "texture", "shape feature", "glcm")),
        ("explainability", ("grad-cam", "gradcam", "heatmap", "explain", "saliency", "why did the model")),
        ("uncertainty", ("uncertaint", "confidence", "reliab", "dispersion")),
        ("classification", ("classif", "class", "predict", "label", "grade")),
        ("metrics", ("accuracy", "dice", "iou", "auc", "f1", "sensitivity", "specificity", "metric", "evaluat")),
        ("models", ("which model", "ensemble", "architecture", "checkpoint", "u-net", "unet", "segnet")),
        ("limitations", ("limitation", "risk", "wrong", "trust", "safe", "caveat")),
    ]
    for topic, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return topic
    return "overview"
def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1e5 or (number != 0 and abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.4g}"
def _pct(value: Any) -> str:
    if value is None:
        return "an unreported share"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)
def _api_key() -> str | None:
    return os.environ.get("BRAIN_ASSISTANT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
def _call_chat_completion(*, base_url: str, model: str, api_key: str | None, system: str, user: str) -> str:
    import urllib.error
    import urllib.request
    if not api_key:
        raise RuntimeError("no API key in the environment")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}") from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("LLM endpoint returned no choices")
    return str(choices[0]["message"]["content"]).strip()
