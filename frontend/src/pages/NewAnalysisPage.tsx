import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, FileImage, Play, Trash2, Upload } from "lucide-react";
import {
  Card,
  CardBody,
  CardHeader,
  DisclaimerBanner,
  ErrorState,
  InfoNote,
  LoadingBlock,
  Pill,
  StatusBadge,
  Tooltip,
} from "../components/ui";
import { useToast } from "../components/AppProviders";
import { useAnalysis, useCapabilities } from "../hooks/useAnalysis";
import { api, ApiError } from "../services/api";
import { formatBytes } from "../services/format";
import type { PlanResponse, PlanStep } from "../types/api";
const MODALITY_SLOTS = [
  { key: "t1", label: "T1", hint: "Native T1-weighted anatomy" },
  { key: "t1ce", label: "T1ce", hint: "Contrast-enhanced T1" },
  { key: "t2", label: "T2", hint: "T2-weighted" },
  { key: "flair", label: "FLAIR", hint: "Fluid-attenuated inversion recovery" },
];
export default function NewAnalysisPage() {
  const navigate = useNavigate();
  const { push } = useToast();
  const { select, refresh } = useAnalysis();
  const { capabilities } = useCapabilities();
  const [analysisType, setAnalysisType] = useState("single_channel");
  const [files, setFiles] = useState<Record<string, File | null>>({});
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [scanIdentifier, setScanIdentifier] = useState("");
  const [notes, setNotes] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [enableEnsemble, setEnableEnsemble] = useState(true);
  const [ensembleStrategy, setEnsembleStrategy] = useState("equal");
  const [enableRadiomics, setEnableRadiomics] = useState(true);
  const [enableExplainability, setEnableExplainability] = useState(true);
  const [enableUncertainty, setEnableUncertainty] = useState(false);
  const [enableStackedEnsemble, setEnableStackedEnsemble] = useState(false);
  const [radiomicsMask, setRadiomicsMask] = useState("best");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [running, setRunning] = useState(false);
  const [stage, setStage] = useState("");
  const [error, setError] = useState<ApiError | null>(null);
  const abortRef = useRef(false);
  // Whether the notebook's stacked ensemble has every artefact it needs. The toggle stays
  // disabled until it does, so the run options never promise a result the backend cannot produce.
  const stackedReady = Boolean(capabilities?.stacked_ensemble?.available);
  const stackedMissing =
    capabilities?.stacked_ensemble?.missing_artifacts ?? [];
  const channelModels = useMemo(
    () =>
      capabilities?.models.filter(
        (model) => model.task === "segmentation" && model.available,
      ) ?? [],
    [capabilities],
  );
  useEffect(() => {
    const wanted = analysisType === "four_channel" ? 4 : 1;
    const matching = channelModels
      .filter((model) => model.input_spec.channels === wanted)
      .map((model) => model.model_id);
    setSelectedModels(matching);
  }, [analysisType, channelModels]);
  useEffect(() => {
    let cancelled = false;
    setPlanLoading(true);
    api
      .plan({
        analysis_type: analysisType,
        model_ids: selectedModels.length ? selectedModels : undefined,
        enable_ensemble: enableEnsemble,
        enable_radiomics: enableRadiomics,
        enable_uncertainty: enableUncertainty,
        enable_explainability: enableExplainability,
        ensemble_strategy: ensembleStrategy,
        radiomics_mask: radiomicsMask,
        enable_stacked_ensemble: enableStackedEnsemble,
      })
      .then((payload) => {
        if (!cancelled) setPlan(payload);
      })
      .catch(() => {
        if (!cancelled) setPlan(null);
      })
      .finally(() => {
        if (!cancelled) setPlanLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [
    analysisType,
    selectedModels,
    enableEnsemble,
    enableRadiomics,
    enableUncertainty,
    enableExplainability,
    ensembleStrategy,
    radiomicsMask,
    enableStackedEnsemble,
  ]);
  const filesReady =
    analysisType === "single_channel"
      ? Boolean(singleFile)
      : MODALITY_SLOTS.every((slot) => Boolean(files[slot.key]));
  const run = async () => {
    const form = new FormData();
    form.append("analysis_type", analysisType);
    if (analysisType === "single_channel" && singleFile) {
      form.append("files", singleFile, singleFile.name);
    } else {
      MODALITY_SLOTS.forEach((slot) => {
        const file = files[slot.key];
        if (file) form.append("files", file, file.name);
      });
      form.append("modalities_confirmed", "true");
    }
    selectedModels.forEach((model) => form.append("model_ids", model));
    if (scanIdentifier.trim())
      form.append("scan_identifier", scanIdentifier.trim());
    if (notes.trim()) form.append("notes", notes.trim());
    form.append("threshold", String(threshold));
    form.append("enable_ensemble", String(enableEnsemble));
    form.append("ensemble_strategy", ensembleStrategy);
    form.append("enable_radiomics", String(enableRadiomics));
    form.append("enable_explainability", String(enableExplainability));
    form.append("enable_uncertainty", String(enableUncertainty));
    form.append("radiomics_mask", radiomicsMask);
    form.append("enable_stacked_ensemble", String(enableStackedEnsemble));
    setRunning(true);
    setError(null);
    setProgress(0);
    setStage("uploading");
    abortRef.current = false;
    try {
      const result = await api.createAnalysis(form, (percent) => {
        setProgress(percent);
        if (percent >= 100) setStage("processing");
      });
      select(result.analysis_id);
      await refresh();
      push("success", "Analysis complete", `Stored as ${result.analysis_id}.`);
      navigate("/analysis/segmentation");
    } catch (caught) {
      const apiError =
        caught instanceof ApiError
          ? caught
          : new ApiError(String(caught), "unknown", 0);
      setError(apiError);
      push("error", "Analysis failed", apiError.message);
    } finally {
      setRunning(false);
      setProgress(0);
      setStage("");
    }
  };
  return (
    <div className="space-y-6">
      <DisclaimerBanner
        text={
          capabilities?.disclaimer ??
          "Research prototype. Outputs are AI-generated and are not intended to replace professional medical diagnosis."
        }
      />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Card>
            <CardHeader
              title="1. Input"
              subtitle="Choose the representation that matches the data you actually have"
              tooltip="The two pipelines feed different models. A single grayscale image is not the same thing as four co-registered MRI sequences, and the system will not convert one into the other."
            />
            <CardBody className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <TypeOption
                  active={analysisType === "single_channel"}
                  title="Single grayscale image"
                  description="One PNG or JPEG export. Runs the 512\u00D7512 single-channel U-Net."
                  onClick={() => setAnalysisType("single_channel")}
                />
                <TypeOption
                  active={analysisType === "four_channel"}
                  title="Four-channel MRI"
                  description="Four modality images or one .npz with four channels. Runs the four-channel models."
                  onClick={() => setAnalysisType("four_channel")}
                />
              </div>
              {analysisType === "single_channel" ? (
                <FileDrop
                  label="MRI image (PNG or JPEG)"
                  accept=".png,.jpg,.jpeg"
                  file={singleFile}
                  onFile={setSingleFile}
                  hint="Grayscale or RGB are both accepted; the image is converted to a single channel and resized to 512\u00D7512."
                />
              ) : (
                <div className="space-y-3">
                  <div className="grid gap-3 sm:grid-cols-2">
                    {MODALITY_SLOTS.map((slot, index) => (
                      <FileDrop
                        label={`${index + 1}. ${slot.label}`}
                        accept=".png,.jpg,.jpeg,.npz"
                        file={files[slot.key] ?? null}
                        onFile={(file) =>
                          setFiles((current) => ({
                            ...current,
                            [slot.key]: file,
                          }))
                        }
                        hint={slot.hint}
                        key={slot.key}
                      />
                    ))}
                  </div>
                  <InfoNote tone="warn">
                    All four slots must contain a <strong>different</strong>{" "}
                    sequence. The backend rejects uploads whose channels are
                    identical, because duplicating one image four times would
                    make the model read modalities that were never acquired.
                  </InfoNote>
                </div>
              )}
            </CardBody>
          </Card>
          <Card>
            <CardHeader
              title="2. Options"
              subtitle="Everything here is recorded in the analysis manifest"
            />
            <CardBody className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="label" htmlFor="scan">
                    Scan identifier (pseudonymous)
                  </label>
                  <input
                    id="scan"
                    className="input"
                    value={scanIdentifier}
                    onChange={(event) => setScanIdentifier(event.target.value)}
                    placeholder="e.g. STUDY-014"
                  />
                  <p className="prose-note mt-1">
                    No patient name, date of birth or record number is stored;
                    use a pseudonymous code.
                  </p>
                </div>
                <div>
                  <label className="label" htmlFor="threshold">
                    Mask threshold: {threshold.toFixed(2)}
                  </label>
                  <input
                    id="threshold"
                    type="range"
                    min="0.05"
                    max="0.95"
                    step="0.05"
                    value={threshold}
                    onChange={(event) =>
                      setThreshold(Number(event.target.value))
                    }
                    className="w-full accent-clinical-600"
                  />
                  <p className="prose-note mt-1">
                    Probability above this value becomes part of the predicted
                    region. The original application used 0.50.
                  </p>
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <Toggle
                  label="Radiomics extraction"
                  checked={enableRadiomics}
                  onChange={setEnableRadiomics}
                  tooltip="Computes shape, first-order and texture features from the predicted region."
                />
                <Toggle
                  label="Grad-CAM explanation"
                  checked={enableExplainability}
                  onChange={setEnableExplainability}
                  tooltip="Saliency map from the model that produced the mask."
                />
                <Toggle
                  label="Uncertainty estimation"
                  checked={enableUncertainty}
                  onChange={setEnableUncertainty}
                  tooltip="Adds test-time augmentation (4 passes) or Monte-Carlo dropout where the architecture supports it. Costs extra inference time."
                />
                <Toggle
                  label="Probability ensemble"
                  checked={enableEnsemble}
                  onChange={setEnableEnsemble}
                  tooltip="Fuses two models only when they share an identical input specification."
                  disabled={
                    !capabilities?.ensemble.groups.some(
                      (group) => group.ensemblable,
                    )
                  }
                />
                <Toggle
                  label="Stacked ensemble (notebook)"
                  checked={enableStackedEnsemble}
                  onChange={setEnableStackedEnsemble}
                  tooltip="FINAL ENSEMBLE NB.ipynb: three four-channel segmentation members + a 46-feature radiomics CNN feed a gradient-boosted head. It needs the four-channel input and predicts a mask-size proxy, not a grade."
                  disabled={
                    analysisType !== "four_channel" ||
                    !stackedReady
                  }
                />
              </div>
              {!stackedReady ? (
                <InfoNote tone="warn">
                  <strong>The notebook's stacked ensemble is not installed.</strong>{" "}
                  {(capabilities?.stacked_ensemble?.reason ?? "").trim()}{" "}
                  {stackedMissing.length ? (
                    <>
                      Missing files:{" "}
                      <code>{stackedMissing.join(", ")}</code>.{" "}
                    </>
                  ) : null}
                  Copy them into <code>models/weights/</code> (or point{" "}
                  <code>BRAIN_WEIGHTS_DIR</code> at the folder that holds them) and reload this
                  page; weights are never downloaded automatically.
                </InfoNote>
              ) : null}
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="label" htmlFor="strategy">
                    Fusion strategy{" "}
                    <Tooltip text="Equal weights reproduce the documented baseline. Calibrated weights come from a validation-only calibration file. Confidence-aware weights additionally modulate each model per pixel by its predictive spread." />
                  </label>
                  <select
                    id="strategy"
                    className="input"
                    value={ensembleStrategy}
                    onChange={(event) =>
                      setEnsembleStrategy(event.target.value)
                    }
                  >
                    <option value="equal">
                      Equal weights (documented baseline)
                    </option>
                    <option value="calibrated">
                      Validation-calibrated weights
                    </option>
                    <option value="confidence_aware">
                      Confidence-aware fusion
                    </option>
                  </select>
                </div>
                <div>
                  <label className="label" htmlFor="mask">
                    Region of interest for radiomics
                  </label>
                  <select
                    id="mask"
                    className="input"
                    value={radiomicsMask}
                    onChange={(event) => setRadiomicsMask(event.target.value)}
                  >
                    <option value="best">
                      First model that produced a non-empty mask
                    </option>
                    <option value="primary">Primary selected model</option>
                    <option value="ensemble">Ensemble mask</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="label">Models</label>
                <div className="flex flex-wrap gap-2">
                  {channelModels.length ? (
                    channelModels.map((model) => (
                      <label
                        className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-xs ${
                          selectedModels.includes(model.model_id)
                            ? "border-clinical-400 bg-clinical-50 dark:border-clinical-500 dark:bg-clinical-800"
                            : "border-clinical-200 dark:border-clinical-700"
                        }`}
                        key={model.model_id}
                      >
                        <input
                          type="checkbox"
                          className="accent-clinical-600"
                          checked={selectedModels.includes(model.model_id)}
                          onChange={(event) =>
                            setSelectedModels((current) =>
                              event.target.checked
                                ? [...current, model.model_id]
                                : current.filter((id) => id !== model.model_id),
                            )
                          }
                        />
                        <span>{model.model_id}</span>
                        <StatusBadge
                          value={
                            capabilities?.model_health.models.find(
                              (entry) => entry.model_id === model.model_id,
                            )?.verdict ?? "ready"
                          }
                        />
                      </label>
                    ))
                  ) : (
                    <p className="prose-note">
                      No segmentation model is available in this deployment.
                    </p>
                  )}
                </div>
              </div>
              <div>
                <label className="label" htmlFor="notes">
                  Notes
                </label>
                <textarea
                  id="notes"
                  className="input min-h-[70px]"
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                  placeholder="Optional context for this study (no identifying information)."
                />
              </div>
            </CardBody>
          </Card>
        </div>
        <div className="space-y-6">
          <Card>
            <CardHeader
              title="3. Execution plan"
              subtitle="What the backend will actually run"
              actions={
                planLoading ? (
                  <span className="badge-neutral">checking…</span>
                ) : null
              }
            />
            <CardBody className="space-y-3">
              {planLoading && !plan ? (
                <LoadingBlock label="Building the plan" />
              ) : null}
              {plan ? (
                <>
                  <div className="space-y-2">
                    {plan.steps.map((step: PlanStep) => (
                      <div
                        className="rounded-lg border border-clinical-200 p-2.5 text-xs dark:border-clinical-800"
                        key={step.model_id}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-medium text-clinical-800 dark:text-clinical-100">
                            {step.model_id}
                          </span>
                          <StatusBadge
                            value={step.runnable ? step.health : step.status}
                          />
                        </div>
                        {!step.runnable ? (
                          <p className="mt-1 text-clinical-500 dark:text-clinical-400">
                            {step.message}
                          </p>
                        ) : null}
                      </div>
                    ))}
                  </div>
                  <PlanLine
                    label="Ensemble"
                    ok={plan.ensemble?.available ?? false}
                    detail={
                      plan.ensemble?.available
                        ? "two compatible models"
                        : (plan.ensemble?.reasons ?? []).join(" ") ||
                          "not requested"
                    }
                  />
                  <PlanLine
                    label="Radiomics"
                    ok={plan.radiomics.runnable}
                    detail={String(
                      plan.radiomics.backend.selected_backend ?? "",
                    )}
                  />
                  <PlanLine
                    label="Explainability"
                    ok={plan.explainability.runnable}
                    detail={plan.explainability.targets
                      .map(
                        (target) =>
                          `${target.model_id} → ${target.target_layer}`,
                      )
                      .join(", ")}
                  />
                  <PlanLine
                    label="Uncertainty"
                    ok={plan.uncertainty.runnable}
                    detail={
                      plan.uncertainty.runnable ? "enabled" : "not requested"
                    }
                  />
                  <PlanLine
                    label="Classification (radiomics grade model)"
                    ok={plan.classification.available}
                    detail={
                      plan.classification.available
                        ? (plan.classification.model_id ?? "")
                        : "disabled — no validated radiomics artefact"
                    }
                  />
                  <PlanLine
                    label="Stacked ensemble"
                    ok={plan.stacked_ensemble.runnable}
                    detail={
                      plan.stacked_ensemble.requested
                        ? plan.stacked_ensemble.runnable
                          ? "three members + radiomics CNN + meta-learner"
                          : plan.stacked_ensemble.reason
                        : "not requested"
                    }
                  />
                </>
              ) : null}
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Run" />
            <CardBody className="space-y-3">
              {error ? (
                <ErrorState
                  message={error.message}
                  title={`Request failed (${error.code})`}
                />
              ) : null}
              {running ? (
                <div className="space-y-2">
                  <div className="h-2 w-full overflow-hidden rounded-full bg-clinical-200 dark:bg-clinical-800">
                    <div
                      className="h-full bg-clinical-600 transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-xs text-clinical-600 dark:text-clinical-300">
                    {stage === "uploading"
                      ? `Uploading ${progress}%`
                      : "Running the models on CPU/GPU — this can take up to a minute per model."}
                  </p>
                </div>
              ) : null}
              <button
                type="button"
                className="btn-primary w-full"
                disabled={!filesReady || running || !selectedModels.length}
                onClick={run}
              >
                <Play className="h-4 w-4" />
                {running ? "Running analysis…" : "Run analysis"}
              </button>
              {!filesReady ? (
                <p className="prose-note">
                  Select the required input file(s) to enable the run button.
                </p>
              ) : null}
              {!selectedModels.length ? (
                <p className="prose-note">Select at least one model.</p>
              ) : null}
              <button
                type="button"
                className="btn-ghost w-full text-xs"
                onClick={() => {
                  setSingleFile(null);
                  setFiles({});
                  setNotes("");
                  setScanIdentifier("");
                }}
              >
                <Trash2 className="h-3.5 w-3.5" />
                Clear the form
              </button>
            </CardBody>
          </Card>
          <InfoNote tone="neutral">
            <span className="inline-flex items-start gap-1.5">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5" />
              Analyses run synchronously and are stored locally. Nothing is
              uploaded anywhere except to this application's own backend.
            </span>
          </InfoNote>
        </div>
      </div>
    </div>
  );
}
function TypeOption({
  active,
  title,
  description,
  onClick,
}: {
  active: boolean;
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border p-3 text-left transition ${active ? "border-clinical-500 bg-clinical-50 dark:border-clinical-400 dark:bg-clinical-800/60" : "border-clinical-200 hover:border-clinical-300 dark:border-clinical-700"}`}
    >
      <div className="flex items-center gap-2">
        <span
          className={`h-3 w-3 rounded-full border-2 ${active ? "border-clinical-600 bg-clinical-600" : "border-clinical-300"}`}
        />
        <span className="text-sm font-medium text-clinical-900 dark:text-clinical-50">
          {title}
        </span>
      </div>
      <p className="mt-1 text-xs text-clinical-500 dark:text-clinical-400">
        {description}
      </p>
    </button>
  );
}
function FileDrop({
  label,
  accept,
  file,
  onFile,
  hint,
}: {
  label: string;
  accept: string;
  file: File | null;
  onFile: (file: File | null) => void;
  hint?: string;
}) {
  const [dragging, setDragging] = useState(false);
  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const dropped = event.dataTransfer.files?.[0];
        if (dropped) onFile(dropped);
      }}
      className={`rounded-lg border border-dashed p-3 transition ${dragging ? "border-clinical-500 bg-clinical-50 dark:bg-clinical-800/50" : "border-clinical-300 dark:border-clinical-700"}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs font-medium text-clinical-800 dark:text-clinical-100">
            {label}
          </p>
          {file ? (
            <p className="mt-0.5 truncate text-[11px] text-clinical-500 dark:text-clinical-400">
              {file.name} · {formatBytes(file.size)}
            </p>
          ) : (
            <p className="mt-0.5 text-[11px] text-clinical-500 dark:text-clinical-400">
              {hint ?? "Drop a file here or browse"}
            </p>
          )}
        </div>
        <label className="btn-secondary cursor-pointer px-2 py-1 text-[11px]">
          <Upload className="h-3 w-3" />
          Browse
          <input
            type="file"
            accept={accept}
            className="sr-only"
            onChange={(event) => onFile(event.target.files?.[0] ?? null)}
          />
        </label>
      </div>
      {file ? (
        <div className="mt-2 flex items-center gap-1.5 text-[11px] text-clinical-500 dark:text-clinical-400">
          <FileImage className="h-3 w-3" />
          ready
          <button
            type="button"
            className="ml-auto text-rose-600 hover:underline dark:text-rose-400"
            onClick={() => onFile(null)}
          >
            remove
          </button>
        </div>
      ) : null}
    </div>
  );
}
function Toggle({
  label,
  checked,
  onChange,
  tooltip,
  disabled = false,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  tooltip?: string;
  disabled?: boolean;
}) {
  return (
    <label
      className={`flex items-center gap-2 rounded-lg border border-clinical-200 px-3 py-2 text-xs dark:border-clinical-800 ${disabled ? "opacity-60" : "cursor-pointer"}`}
    >
      <input
        type="checkbox"
        className="accent-clinical-600"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span className="text-clinical-800 dark:text-clinical-100">{label}</span>
      {tooltip ? <Tooltip text={tooltip} /> : null}
      {disabled ? <Pill tone="amber">unavailable</Pill> : null}
    </label>
  );
}
function PlanLine({
  label,
  ok,
  detail,
}: {
  label: string;
  ok: boolean;
  detail: string;
}) {
  return (
    <div className="flex items-start justify-between gap-2 border-t border-clinical-100 pt-2 text-xs dark:border-clinical-800">
      <span className="text-clinical-600 dark:text-clinical-300">{label}</span>
      <span
        className="max-w-[60%] text-right text-clinical-800 dark:text-clinical-100"
        title={detail}
      >
        {ok ? detail || "enabled" : detail || "disabled"}
      </span>
    </div>
  );
}
