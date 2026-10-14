import { useState } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, Brain, Sparkles } from "lucide-react";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  EmptyState,
  InfoNote,
  LoadingBlock,
  MetricCard,
  Pill,
  StatusBadge,
  Tabs,
  Tooltip,
  ValueOrUnavailable,
} from "../components/ui";
import { ImageGrid } from "../components/ImageViewer";
import { useAnalysis, useCapabilities } from "../hooks/useAnalysis";
import { api } from "../services/api";
import {
  formatNumber,
  humaniseFeature,
  NOT_AVAILABLE,
} from "../services/format";
import type { AnalysisPayload } from "../types/api";
import type { ImagePanel } from "../components/ImageViewer";
export default function ExplainabilityPage() {
  const { result, analysisId } = useAnalysis();
  const { capabilities } = useCapabilities();
  const [view, setView] = useState("overlay");
  if (!analysisId) {
    return (
      <EmptyState
        title="No analysis selected"
        message="Explanations are generated from a specific model and a specific layer, so they are always tied to one analysis."
        action={
          <Link className="btn-primary" to="/analysis/new">
            New analysis
          </Link>
        }
        icon={<Sparkles className="h-6 w-6" />}
      />
    );
  }
  if (!result) return <LoadingBlock label="Loading the explanation record" />;
  const entries = result.explainability ?? [];
  const available = entries.filter((entry) => entry.available);
  const classification = result.classification;
  const contributions = classification.feature_contributions ?? [];
  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
          Explainable AI
        </h2>
        <p className="mt-1 max-w-3xl text-sm text-clinical-600 dark:text-clinical-300">
          Two complementary views: a Grad-CAM saliency map that shows which
          parts of the image drove a model's output, and a feature attribution
          that shows how the radiomics values contributed to the classifier
          score.
        </p>
      </header>
      <InfoNote tone="warn">
        <span className="inline-flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            A heatmap is not a diagnosis and is not a lesion outline. Grad-CAM
            highlights regions that influenced the network's output; it can be
            noisy, it can highlight irrelevant anatomy, and it does not
            establish that a region is abnormal. Always read it together with
            the mask and the quantitative measurements.
          </span>
        </span>
      </InfoNote>
      {!available.length ? (
        <EmptyState
          title="No saliency map for this analysis"
          message={
            entries.length
              ? entries
                  .map(
                    (entry) =>
                      `${entry.model_id}: ${entry.reason ?? "unavailable"}`,
                  )
                  .join(" · ")
              : "The model producing output has no supported Grad-CAM path (no convolutional layer could be reached), so no explanation was generated."
          }
        />
      ) : (
        <>
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Models explained"
              value={String(available.length)}
              hint={available.map((entry) => entry.model_id).join(", ")}
            />
            <MetricCard
              label="Method"
              value={String(available[0].method ?? NOT_AVAILABLE)}
              tooltip="Grad-CAM uses gradients of the target output with respect to the feature maps of a chosen layer. It is the same class of method that the original Streamlit application used \u2014 here it is anchored on the model that actually produced the mask."
            />
            <MetricCard
              label="Target layer"
              value={String(available[0].target_layer ?? NOT_AVAILABLE)}
              hint="The last convolutional layer reached by the hook"
            />
            <MetricCard
              label="Mean activation"
              value={formatNumber(available[0].heatmap_mean ?? null)}
              tooltip="Average normalised activation over the heatmap. A near-zero value across the whole map means the layer produced almost no spatial signal."
            />
          </section>
          <Card>
            <CardHeader
              title="Saliency maps"
              subtitle="One explanation per available model \u2014 never a shared map"
              actions={
                <Tabs
                  tabs={[
                    { id: "overlay", label: "On the image" },
                    { id: "heatmap", label: "Heatmap only" },
                  ]}
                  active={view}
                  onChange={(id) => setView(id)}
                />
              }
            />
            <CardBody>
              <ImageGrid panels={buildPanels(result, view)} columns={3} />
            </CardBody>
          </Card>
          <Card>
            <CardHeader
              title="Explanation provenance"
              subtitle="Which model, which layer, which score was explained"
            />
            <CardBody>
              <DataTable
                columns={[
                  "Model",
                  "Method",
                  "Layer",
                  "Target",
                  "Activation shape",
                  "Mean",
                  "Max",
                  "Notes",
                ]}
                rows={entries.map((entry) => [
                  entry.model_id,
                  entry.available ? String(entry.method) : "unavailable",
                  entry.available ? String(entry.target_layer) : NOT_AVAILABLE,
                  String(entry.target_score ?? NOT_AVAILABLE),
                  entry.activation_shape
                    ? entry.activation_shape.join(" × ")
                    : NOT_AVAILABLE,
                  formatNumber(entry.heatmap_mean ?? null),
                  formatNumber(entry.heatmap_max ?? null),
                  <span
                    className="text-[11px] text-clinical-500 dark:text-clinical-400"
                    key={"r"}
                  >
                    {entry.available
                      ? (entry.warnings?.join(" ") ?? entry.explanation ?? "")
                      : entry.reason}
                  </span>,
                ])}
              />
            </CardBody>
          </Card>
        </>
      )}
      <Card>
        <CardHeader
          title="Radiomics feature attribution"
          subtitle="Which features pushed the classifier score, and in which direction"
          tooltip="Attribution is computed from the classifier's own weights or a linear surrogate fitted on the stored features. A positive contribution pushes the score towards the predicted class."
        />
        <CardBody>
          {!classification.available ? (
            <InfoNote tone="warn">
              <p className="font-medium">
                Attribution unavailable because the classifier is disabled.
              </p>
              <p className="mt-1">{classification.reason}</p>
              {capabilities?.classification.requirements?.length ? (
                <div className="mt-2">
                  <p className="font-medium">What is required to enable it</p>
                  <ul className="mt-1 list-disc space-y-1 pl-4">
                    {capabilities.classification.requirements.map(
                      (requirement) => (
                        <li key={requirement}>{requirement}</li>
                      ),
                    )}
                  </ul>
                </div>
              ) : null}
            </InfoNote>
          ) : contributions.length ? (
            <DataTable
              columns={["Feature", "Value", "Weight", "Contribution", "Method"]}
              rows={contributions.map((entry) => [
                <span key={"f"}>
                  {humaniseFeature(entry.feature)}
                  <span className="ml-1 text-[11px] text-clinical-400">
                    {entry.feature}
                  </span>
                </span>,
                <ValueOrUnavailable
                  value={formatNumber(entry.value ?? null)}
                  key={"v"}
                />,
                formatNumber(entry.weight ?? null),
                <span
                  className={
                    entry.contribution >= 0
                      ? "text-emerald-600 dark:text-emerald-400"
                      : "text-rose-600 dark:text-rose-400"
                  }
                  key={"c"}
                >
                  {entry.contribution >= 0 ? "+" : ""}
                  {formatNumber(entry.contribution)}
                </span>,
                String(entry.method ?? NOT_AVAILABLE),
              ])}
            />
          ) : (
            <InfoNote tone="neutral">
              The classifier produced a score but no per-feature attribution was
              recorded for this analysis.
            </InfoNote>
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader
          title="Model-specific explanation paths"
          subtitle="Segmentation and classification are explained separately"
          tooltip="A segmentation explanation answers 'which pixels made this mask'; a classification explanation answers 'which measurements moved this score'. They are not interchangeable."
        />
        <CardBody className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-clinical-500" />
              <h4 className="text-sm font-medium text-clinical-800 dark:text-clinical-100">
                Segmentation
              </h4>
            </div>
            <ul className="mt-2 space-y-1.5 text-xs">
              {(capabilities?.explainability.segmentation ?? []).map(
                (entry) => (
                  <li
                    className="flex items-center justify-between gap-2"
                    key={entry.model_id}
                  >
                    <span className="text-clinical-600 dark:text-clinical-300">
                      {entry.model_id}
                    </span>
                    {entry.available ? (
                      <span className="flex items-center gap-1.5">
                        <Pill tone="teal">{entry.target_layer}</Pill>
                        <Tooltip
                          text={`Method: ${entry.method}. Layer semantics: the final convolutional block, whose receptive field covers the predicted region.`}
                        />
                      </span>
                    ) : (
                      <StatusBadge
                        value="unavailable"
                        label={entry.reason ?? "unavailable"}
                      />
                    )}
                  </li>
                ),
              )}
            </ul>
          </div>
          <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-clinical-500" />
              <h4 className="text-sm font-medium text-clinical-800 dark:text-clinical-100">
                Classification
              </h4>
            </div>
            <p className="mt-2 text-xs text-clinical-600 dark:text-clinical-300">
              {capabilities?.explainability.classification.available
                ? `${capabilities.explainability.classification.method} — per-feature contributions are listed above.`
                : (capabilities?.explainability.classification.reason ??
                  "Unavailable.")}
            </p>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
function buildPanels(result: AnalysisPayload, view: string): ImagePanel[] {
  const panels: ImagePanel[] = [];
  if (result.inputs && Object.keys(result.inputs).length) {
    const reference = Object.values(result.inputs)[0];
    panels.push({
      caption: "Input (reference)",
      src: api.artifactUrl(result.analysis_id, reference),
      tone: "input",
      note: "The image the network actually saw",
    });
  }
  (result.explainability ?? []).forEach((entry) => {
    if (!entry.available || !entry.artifacts) return;
    const artifact =
      view === "overlay" ? entry.artifacts.overlay : entry.artifacts.heatmap;
    if (!artifact) return;
    panels.push({
      caption: `${entry.model_id} — ${view === "overlay" ? "Grad-CAM overlay" : "Grad-CAM heatmap"}`,
      src: api.artifactUrl(result.analysis_id, artifact),
      tone: "heatmap",
      note: `Layer ${entry.target_layer} · mean activation ${formatNumber(entry.heatmap_mean ?? null)}`,
      downloadName: `${result.analysis_id}_${entry.model_id}_gradcam.png`,
    });
  });
  return panels;
}
