import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, FilePlus2 } from "lucide-react";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  EmptyState,
  ErrorState,
  InfoNote,
  LoadingBlock,
  MetricCard,
  Pill,
  StatusBadge,
  Tabs,
  Tooltip,
  ValueOrUnavailable,
} from "../components/ui";
import { ChannelStrip, ImageGrid } from "../components/ImageViewer";
import { useAnalysis } from "../hooks/useAnalysis";
import { api } from "../services/api";
import {
  formatDuration,
  formatInt,
  formatNumber,
  formatPercent,
  NOT_AVAILABLE,
} from "../services/format";
import type { ImagePanel } from "../components/ImageViewer";
export default function SegmentationPage() {
  const { analysisId, detail, result, loading, error } = useAnalysis();
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [view, setView] = useState("masks");
  const measurements = result?.region_measurements;
  const completedModels = useMemo(
    () =>
      Object.entries(result?.models ?? {}).filter(
        ([, entry]) => entry.available && entry.summary,
      ),
    [result],
  );
  useEffect(() => {
    if (
      completedModels.length &&
      !completedModels.some(([modelId]) => modelId === activeModel)
    ) {
      setActiveModel(completedModels[0][0]);
    }
  }, [completedModels, activeModel]);
  if (!analysisId) {
    return (
      <EmptyState
        title="No analysis selected"
        message="Run a new MRI analysis or pick one from the history to see its segmentation output."
        action={
          <div className="flex gap-2">
            <Link to="/analysis/new" className="btn-primary">
              New analysis
            </Link>
            <Link to="/history" className="btn-secondary">
              History
            </Link>
          </div>
        }
        icon={<FilePlus2 className="h-6 w-6" />}
      />
    );
  }
  if (loading && !result) return <LoadingBlock label="Loading the analysis" />;
  if (error)
    return (
      <ErrorState
        title={`Could not load ${analysisId}`}
        message={error.message}
      />
    );
  if (!result) {
    return (
      <EmptyState
        title="The stored result could not be read"
        message={
          detail
            ? `The database holds this analysis but its result payload is ${detail.payload_source}. Re-run the analysis to regenerate the artefacts.`
            : "No payload was stored for this analysis."
        }
      />
    );
  }
  const selected = activeModel ? result.models[activeModel] : undefined;
  const ensemble = result.ensemble;
  const referenceModel = result.reference_mask.model_id;
  const panels: ImagePanel[] = [];
  if (result.inputs && Object.keys(result.inputs).length) {
    const firstKey = Object.keys(result.inputs)[0];
    panels.push({
      caption: "Model input (normalised)",
      src: api.artifactUrl(result.analysis_id, result.inputs[firstKey]),
      tone: "input",
      note: `${result.preprocessing.pipeline} · ${result.preprocessing.normalisation}`,
      downloadName: `${result.analysis_id}_input.png`,
    });
  }
  completedModels.forEach(([modelId, entry]) => {
    const artifacts = entry.artifacts ?? {};
    if (view === "masks" && artifacts.mask) {
      panels.push({
        caption: `${modelId} — predicted mask`,
        src: api.artifactUrl(result.analysis_id, artifacts.mask),
        tone: "mask",
        note: `Threshold ${entry.summary?.threshold} · ${formatInt(entry.summary?.foreground_pixels ?? 0)} pixels`,
        downloadName: `${result.analysis_id}_${modelId}_mask.png`,
      });
    }
    if (view === "overlays" && artifacts.overlay) {
      panels.push({
        caption: `${modelId} — overlay`,
        src: api.artifactUrl(result.analysis_id, artifacts.overlay),
        tone: "overlay",
        note: "Predicted region blended over the input image",
        downloadName: `${result.analysis_id}_${modelId}_overlay.png`,
      });
    }
  });
  if (ensemble.available && ensemble.artifacts) {
    if (view === "masks" && ensemble.artifacts.mask) {
      panels.push({
        caption: "Ensemble — fused mask",
        src: api.artifactUrl(result.analysis_id, ensemble.artifacts.mask),
        tone: "mask",
        note: `${ensemble.strategy} fusion · weights ${Object.entries(
          ensemble.weights,
        )
          .map(([key, value]) => `${key}=${value.toFixed(3)}`)
          .join(", ")}`,
      });
    }
    if (view === "overlays" && ensemble.artifacts.overlay) {
      panels.push({
        caption: "Ensemble — overlay",
        src: api.artifactUrl(result.analysis_id, ensemble.artifacts.overlay),
        tone: "overlay",
        note: `Weight source: ${ensemble.weight_source}`,
      });
    }
  }
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
            Segmentation results
          </h2>
          <p className="mt-1 text-sm text-clinical-600 dark:text-clinical-300">
            {result.scan_identifier} · {result.analysis_id} · {result.device} ·
            threshold {result.threshold}
          </p>
        </div>
        <div className="flex gap-2">
          <Link to="/analysis/radiomics" className="btn-secondary">
            Radiomics
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
          <Link to="/analysis/explainability" className="btn-secondary">
            Explainability
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </header>
      <InfoNote tone="warn">
        No reference segmentation exists for an uploaded scan, so{" "}
        <strong>no accuracy metric is reported here</strong>. The values below
        describe the models' own output and how far the models agree with each
        other. Dice, IoU, precision, recall and specificity require labelled
        data and appear only on the Model Evaluation page, after
        <code className="mx-1 rounded bg-black/5 px-1 dark:bg-white/10">
          scripts/evaluate_models.py
        </code>{" "}
        has run on the reserved test split.
      </InfoNote>
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Models producing output"
          value={`${completedModels.length}/${Object.keys(result.models).length}`}
          hint={completedModels.map(([id]) => id).join(", ") || "none"}
        />
        <MetricCard
          label="Region size (primary)"
          value={
            selected?.summary
              ? formatInt(selected.summary.foreground_pixels)
              : NOT_AVAILABLE
          }
          unit="px"
          tooltip="Pixels whose predicted probability exceeds the decision threshold, in the model's own resolution."
          hint={
            selected?.summary
              ? formatPercent(selected.summary.foreground_fraction) +
                " of the image"
              : undefined
          }
        />
        <MetricCard
          label="Mean probability in region"
          value={
            selected?.summary
              ? formatNumber(selected.summary.probability_mean_in_mask)
              : NOT_AVAILABLE
          }
          tooltip="Average predicted probability over the pixels inside the predicted region. This is a model score, not a calibrated probability that an abnormality exists."
        />
        <MetricCard
          label="Inference time"
          value={
            selected?.summary
              ? formatDuration(selected.summary.inference_seconds)
              : NOT_AVAILABLE
          }
          hint={
            selected?.summary?.n_passes
              ? `${selected.summary.n_passes} pass(es)`
              : undefined
          }
        />
      </section>
      <Card>
        <CardHeader
          title="Visual comparison"
          subtitle="Original input, per-model masks and the fused result"
          actions={
            <Tabs
              tabs={[
                { id: "masks", label: "Masks" },
                { id: "overlays", label: "Overlays" },
                { id: "channels", label: "Input channels" },
              ]}
              active={view}
              onChange={(id) => setView(id)}
            />
          }
        />
        <CardBody>
          {view === "channels" ? (
            result.preprocessing.channels === 4 ? (
              <ChannelStrip
                channels={
                  (
                    result.preprocessing.metadata?.modalities as
                      | string[]
                      | undefined
                  )?.map((label: string, index: number) => ({
                    label,
                    src: (() => {
                      const match =
                        Object.entries(result.inputs ?? {}).find(
                          ([key]) => key.toUpperCase() === label.toUpperCase(),
                        ) ?? Object.entries(result.inputs ?? {})[index];
                      return match
                        ? api.artifactUrl(result.analysis_id, match[1])
                        : null;
                    })(),
                  })) ?? []
                }
              />
            ) : (
              <InfoNote tone="neutral">
                This analysis used the single-channel pipeline, so there is only
                one input channel.
              </InfoNote>
            )
          ) : panels.length ? (
            <ImageGrid panels={panels} columns={3} />
          ) : (
            <EmptyState
              title="No segmentation image"
              message="The selected models did not produce a mask for this analysis."
            />
          )}
        </CardBody>
      </Card>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Per-model measurements"
            subtitle="Output statistics only \u2014 not accuracy"
          />
          <CardBody>
            <DataTable
              columns={[
                "Model",
                "Reference check",
                "Region pixels",
                "Region share",
                "Mean p in region",
                "Max p",
                "Threshold",
                "Time",
              ]}
              rows={Object.entries(result.models).map(([modelId, entry]) => [
                <div key={modelId}>
                  <p className="font-medium text-clinical-900 dark:text-clinical-50">
                    {modelId}
                  </p>
                  <p className="text-[11px] text-clinical-500 dark:text-clinical-400">
                    {entry.descriptor?.architecture?.slice(0, 70) ?? ""}
                  </p>
                </div>,
                <StatusBadge
                  value={entry.health?.verdict ?? entry.status}
                  key={"h"}
                />,
                entry.summary
                  ? formatInt(entry.summary.foreground_pixels)
                  : NOT_AVAILABLE,
                entry.summary
                  ? formatPercent(entry.summary.foreground_fraction)
                  : NOT_AVAILABLE,
                entry.summary ? (
                  <ValueOrUnavailable
                    value={formatNumber(entry.summary.probability_mean_in_mask)}
                    reason="The predicted region is empty, so there are no in-region pixels to average."
                    key={"m"}
                  />
                ) : (
                  NOT_AVAILABLE
                ),
                entry.summary
                  ? formatNumber(entry.summary.probability_max)
                  : NOT_AVAILABLE,
                entry.summary ? String(entry.summary.threshold) : NOT_AVAILABLE,
                entry.summary
                  ? formatDuration(entry.summary.inference_seconds)
                  : NOT_AVAILABLE,
              ])}
            />
            {selected?.summary?.warnings?.length ? (
              <div className="mt-3">
                <InfoNote tone="warn">
                  <p className="font-semibold">Warnings for {activeModel}</p>
                  <ul className="mt-1 list-disc space-y-1 pl-4">
                    {selected.summary.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </InfoNote>
              </div>
            ) : null}
          </CardBody>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader
              title="Ensemble"
              actions={
                <Pill tone={ensemble.available ? "teal" : "amber"}>
                  {ensemble.available ? ensemble.strategy : "unavailable"}
                </Pill>
              }
              tooltip="Probability-level fusion P = w1\u00B7P1 + w2\u00B7P2. It runs only between models that take exactly the same input tensor."
            />
            <CardBody className="space-y-3 text-sm">
              {ensemble.available ? (
                <>
                  <Row
                    label="Members"
                    value={(ensemble.members ?? []).join(", ")}
                  />
                  <Row
                    label="Weights"
                    value={Object.entries(ensemble.weights)
                      .map(([key, value]) => `${key}=${value.toFixed(3)}`)
                      .join(", ")}
                  />
                  <Row
                    label="Weight source"
                    value={ensemble.weight_source ?? NOT_AVAILABLE}
                  />
                  <Row
                    label="Fused region"
                    value={`${formatInt(ensemble.foreground_pixels ?? 0)} px`}
                  />
                  {ensemble.disagreement ? (
                    <>
                      <Row
                        label="Mean pairwise agreement"
                        value={
                          <span className="inline-flex items-center gap-1">
                            {formatNumber(
                              ensemble.disagreement
                                .mean_pairwise_dice_agreement,
                            )}
                            <Tooltip text="Dice agreement between the members' masks. Agreement is not accuracy." />
                          </span>
                        }
                      />
                      <Row
                        label="Probability spread (mean)"
                        value={formatNumber(
                          ensemble.disagreement.probability_std_mean,
                        )}
                      />
                    </>
                  ) : null}
                </>
              ) : (
                <ul className="list-disc space-y-1 pl-4 text-xs text-clinical-600 dark:text-clinical-300">
                  {ensemble.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              )}
            </CardBody>
          </Card>
          <Card>
            <CardHeader
              title="Abnormal region analysis"
              subtitle="Measured from the mask itself, in pixel units"
            />
            <CardBody className="space-y-3 text-sm">
              <Row
                label="Region detected"
                value={
                  <Pill tone={measurements?.detected ? "amber" : "neutral"}>
                    {measurements?.detected ? "Yes" : "No"}
                  </Pill>
                }
              />
              <Row
                label="Region area"
                value={
                  measurements?.detected
                    ? `${formatInt(measurements.pixel_area)} px`
                    : NOT_AVAILABLE
                }
              />
              <Row
                label="Image coverage"
                value={
                  measurements?.detected
                    ? `${formatNumber(measurements.image_coverage_percent, 2)} %`
                    : NOT_AVAILABLE
                }
              />
              <Row
                label="Bounding box"
                value={
                  measurements?.bounding_box
                    ? `x=${measurements.bounding_box.x}, y=${measurements.bounding_box.y}, ` +
                      `${measurements.bounding_box.width} x ${measurements.bounding_box.height} px`
                    : NOT_AVAILABLE
                }
              />
              <Row
                label="Centroid (x, y)"
                value={
                  measurements?.centroid
                    ? `x=${formatNumber(measurements.centroid.x, 1)}, y=${formatNumber(measurements.centroid.y, 1)}`
                    : NOT_AVAILABLE
                }
              />
              <Row
                label="Boundary length"
                value={
                  measurements?.perimeter_px !== null &&
                  measurements?.perimeter_px !== undefined
                    ? `${formatNumber(measurements.perimeter_px, 1)} px`
                    : NOT_AVAILABLE
                }
              />
              <Row
                label="Equivalent diameter"
                value={
                  measurements?.equivalent_diameter_px !== null &&
                  measurements?.equivalent_diameter_px !== undefined
                    ? `${formatNumber(measurements.equivalent_diameter_px, 1)} px`
                    : NOT_AVAILABLE
                }
              />
              <Row
                label="Mask source"
                value={measurements?.mask_source ?? result.reference_mask.source}
              />
              <Row
                label="Model"
                value={result.reference_mask.model_id ?? NOT_AVAILABLE}
              />
              <p className="prose-note">
                {measurements?.note ??
                  "Measurements describe the segmented region, not the patient. No anatomical, malignancy or staging statement is derived from them."}
              </p>
              {referenceModel ? (
                <p className="prose-note">
                  Radiomics features were computed inside this region.
                </p>
              ) : null}
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
}
function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-clinical-500 dark:text-clinical-400">{label}</span>
      <span className="max-w-[65%] break-words text-right font-medium text-clinical-800 dark:text-clinical-100">
        {value}
      </span>
    </div>
  );
}
