import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { GitCompareArrows, Info } from "lucide-react";
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
} from "../components/ui";
import { ImageGrid } from "../components/ImageViewer";
import { MetricComparisonChart } from "../components/charts";
import { useAnalysis, useCapabilities } from "../hooks/useAnalysis";
import { api } from "../services/api";
import {
  formatDuration,
  formatInt,
  formatNumber,
  formatPercent,
  NOT_AVAILABLE,
} from "../services/format";
import type { ImagePanel } from "../components/ImageViewer";
import type {
  AnalysisPayload,
  EvaluationResponse,
  RuntimeEntry,
} from "../types/api";
export default function ModelComparisonPage() {
  const { result, analysisId } = useAnalysis();
  const { capabilities } = useCapabilities();
  const [version, setVersion] = useState("masks");
  const [runtime, setRuntime] = useState<EvaluationResponse | null>(null);
  useEffect(() => {
    api
      .evaluation()
      .then(setRuntime)
      .catch(() => setRuntime(null));
  }, []);
  const comparison = result?.model_comparison;
  const completed = useMemo(
    () =>
      Object.entries(result?.models ?? {}).filter(
        ([, entry]) => entry.available,
      ),
    [result],
  );
  const runtimeRows = useMemo(() => runtime?.runtime?.models ?? [], [runtime]);
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
            Model comparison
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-clinical-600 dark:text-clinical-300">
            Agreement between the segmentation models on this scan, plus the
            measured inference cost of each checkpoint. Agreement and
            correctness are different things: without a reference mask the
            platform can only say how similar two predictions are.
          </p>
        </div>
        <Pill tone="amber">
          <Info className="h-3 w-3" />
          {comparison
            ? comparison.state === "unlabelled"
              ? "Unlabelled comparison"
              : comparison.state
            : "no analysis"}
        </Pill>
      </header>
      {!analysisId ? (
        <EmptyState
          title="No analysis selected"
          message="Run an analysis first; the comparison is computed from the models' outputs on the same scan."
          action={
            <Link className="btn-primary" to="/analysis/new">
              New analysis
            </Link>
          }
          icon={<GitCompareArrows className="h-6 w-6" />}
        />
      ) : null}
      {comparison ? <InfoNote tone="warn">{comparison.note}</InfoNote> : null}
      {result ? (
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Models compared"
            value={String(completed.length)}
            hint={completed.map(([id]) => id).join(", ") || "none"}
          />
          <MetricCard
            label="Mean pairwise agreement"
            value={
              comparison?.pairwise_agreement &&
              Object.keys(comparison.pairwise_agreement).length
                ? formatNumber(meanDice(comparison.pairwise_agreement))
                : NOT_AVAILABLE
            }
            tooltip="Average Dice coefficient between the masks predicted by different models on this scan. It measures similarity between models, not similarity to the true lesion."
          />
          <MetricCard
            label="Ensemble available"
            value={result.ensemble.available ? "yes" : "no"}
            hint={
              result.ensemble.available
                ? `${result.ensemble.strategy} fusion`
                : (result.ensemble.reasons[0] ?? "").slice(0, 80)
            }
            tone={result.ensemble.available ? "good" : "warn"}
          />
          <MetricCard
            label="Fastest model"
            value={fastest(result)?.modelId ?? NOT_AVAILABLE}
            hint={
              fastest(result)?.seconds !== undefined
                ? formatDuration(fastest(result)?.seconds ?? null)
                : undefined
            }
          />
        </section>
      ) : null}
      {result ? (
        <>
          <Card>
            <CardHeader
              title="Side-by-side output"
              subtitle="Each panel is produced by a different checkpoint on the identical preprocessed input"
              actions={
                <Tabs
                  tabs={[
                    { id: "masks", label: "Binary masks" },
                    { id: "probabilities", label: "Probability maps" },
                  ]}
                  active={version}
                  onChange={(id) => setVersion(id)}
                />
              }
            />
            <CardBody>
              <ImageGrid panels={buildPanels(result, version)} columns={3} />
            </CardBody>
          </Card>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader
                title="Agreement metrics"
                tooltip="Dice and IoU here compare two predictions with each other. They are explicitly labelled as agreement, never as accuracy."
              />
              <CardBody>
                {comparison?.pairwise_agreement &&
                Object.keys(comparison.pairwise_agreement).length ? (
                  <DataTable
                    columns={[
                      "Model pair",
                      "Dice (agreement)",
                      "IoU (agreement)",
                      "Only in first",
                      "Only in second",
                    ]}
                    rows={Object.entries(comparison.pairwise_agreement).map(
                      ([pair, value]) => [
                        pair.replace("__", " vs "),
                        formatNumber(value.dice_agreement),
                        formatNumber(value.iou_agreement),
                        formatInt(value.only_in_first),
                        formatInt(value.only_in_second),
                      ],
                    )}
                  />
                ) : (
                  <InfoNote tone="neutral">
                    At least two models must produce a mask before pairwise
                    agreement can be computed.
                  </InfoNote>
                )}
                {comparison?.mask_agreement_with_ensemble ? (
                  <div className="mt-4">
                    <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                      Agreement with the fused mask
                    </h4>
                    <DataTable
                      columns={[
                        "Model",
                        "Dice vs ensemble",
                        "IoU vs ensemble",
                        "Model pixels",
                        "Ensemble pixels",
                      ]}
                      rows={Object.entries(
                        comparison.mask_agreement_with_ensemble,
                      ).map(([modelId, value]) => [
                        modelId,
                        formatNumber(value.dice_agreement),
                        formatNumber(value.iou_agreement),
                        formatInt(value.pixels_a),
                        formatInt(value.pixels_b),
                      ])}
                    />
                  </div>
                ) : null}
              </CardBody>
            </Card>
            <Card>
              <CardHeader
                title="Agreement chart"
                subtitle="Higher is more similar \u2014 not more correct"
              />
              <CardBody>
                {comparison?.pairwise_agreement &&
                Object.keys(comparison.pairwise_agreement).length ? (
                  <MetricComparisonChart
                    data={Object.entries(comparison.pairwise_agreement).map(
                      ([pair, value]) => ({
                        label: pair.replace("__", " vs "),
                        Dice: round(value.dice_agreement),
                        IoU: round(value.iou_agreement),
                      }),
                    )}
                    keys={["Dice", "IoU"]}
                  />
                ) : (
                  <InfoNote tone="neutral">
                    Nothing to chart without two models.
                  </InfoNote>
                )}
              </CardBody>
            </Card>
          </div>
          <Card>
            <CardHeader
              title="Per-model measurements"
              subtitle="Output statistics on this scan"
              tooltip="Mean probability is the average predicted probability inside each model's own predicted region; it is a model score, not a clinical probability."
            />
            <CardBody>
              <DataTable
                columns={[
                  "Model",
                  "Status",
                  "Pixels",
                  "Share",
                  "Mean p in region",
                  "Max p",
                  "Time",
                  "Ensemble member",
                ]}
                rows={Object.entries(result.models).map(([modelId, entry]) => [
                  modelId,
                  <StatusBadge
                    value={entry.health?.verdict ?? entry.status}
                    key={"s"}
                  />,
                  entry.summary
                    ? formatInt(entry.summary.foreground_pixels)
                    : NOT_AVAILABLE,
                  entry.summary
                    ? formatPercent(entry.summary.foreground_fraction)
                    : NOT_AVAILABLE,
                  entry.summary
                    ? formatNumber(entry.summary.probability_mean_in_mask)
                    : NOT_AVAILABLE,
                  entry.summary
                    ? formatNumber(entry.summary.probability_max)
                    : NOT_AVAILABLE,
                  entry.summary
                    ? formatDuration(entry.summary.inference_seconds)
                    : NOT_AVAILABLE,
                  result.ensemble.available
                    ? result.ensemble.members?.includes(modelId)
                      ? "yes"
                      : "no"
                    : "not used",
                ])}
              />
            </CardBody>
          </Card>
        </>
      ) : null}
      <Card>
        <CardHeader
          title="Checkpoint compatibility"
          subtitle="Why some model pairs cannot be fused"
          tooltip="Probability fusion requires identical input geometry and identical physical meaning. A 4-channel 256\u00D7256 model and a 1-channel 512\u00D7512 model cannot be averaged."
        />
        <CardBody className="space-y-4">
          {capabilities ? (
            <>
              <DataTable
                columns={[
                  "Model",
                  "Channels",
                  "Resolution",
                  "Normalisation",
                  "Status",
                  "Reference check",
                ]}
                rows={capabilities.models.map((model) => {
                  const health = capabilities.model_health.models.find(
                    (entry) => entry.model_id === model.model_id,
                  );
                  return [
                    model.model_id,
                    String(model.input_spec.channels),
                    `${model.input_spec.height}×${model.input_spec.width}`,
                    model.input_spec.normalisation,
                    <StatusBadge value={model.status} key={"st"} />,
                    <StatusBadge
                      value={health?.verdict ?? "not_loaded"}
                      key={"h"}
                    />,
                  ];
                })}
              />
              <div className="space-y-2">
                {capabilities.ensemble.groups.map((group) => (
                  <div
                    className="rounded-lg border border-clinical-200 p-3 text-xs dark:border-clinical-800"
                    key={`${group.channels}-${group.height}-${group.width}`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-clinical-800 dark:text-clinical-100">
                        {group.channels}×{group.height}×{group.width} ·{" "}
                        {group.normalisation}
                      </span>
                      <Pill tone={group.ensemblable ? "teal" : "amber"}>
                        {group.ensemblable ? "fusable" : "not fusable"}
                      </Pill>
                    </div>
                    <p className="mt-1 text-clinical-600 dark:text-clinical-300">
                      Members: {group.members.join(", ") || "none"}
                    </p>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <LoadingBlock label="Reading model descriptors" />
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader
          title="Inference cost"
          subtitle="Measured while running this deployment on its current device"
          tooltip="Runtime and memory are inference cost, not training cost. They come from real runs on this machine."
        />
        <CardBody>
          {runtimeRows.length ? (
            <DataTable
              columns={[
                "Model",
                "Device",
                "Seconds per image",
                "Images per second",
                "Parameters",
                "Input shape",
                "State",
              ]}
              rows={runtimeRows.map((row: RuntimeEntry) => [
                row.model_id,
                String(row.device ?? NOT_AVAILABLE),
                row.seconds_per_image !== undefined
                  ? formatNumber(row.seconds_per_image, 4)
                  : NOT_AVAILABLE,
                row.images_per_second !== undefined &&
                row.images_per_second !== null
                  ? formatNumber(row.images_per_second, 3)
                  : NOT_AVAILABLE,
                row.parameter_count !== undefined &&
                row.parameter_count !== null
                  ? formatInt(row.parameter_count)
                  : NOT_AVAILABLE,
                row.input_shape ? row.input_shape.join(" × ") : NOT_AVAILABLE,
                <StatusBadge
                  value={row.state}
                  label={
                    row.reason ? `${row.state} — ${row.reason}` : row.state
                  }
                  key={"s"}
                />,
              ])}
            />
          ) : (
            <InfoNote tone="neutral">
              No runtime measurement was returned. The evaluation endpoint
              reports the reason explicitly — see the Model Evaluation page.
            </InfoNote>
          )}
        </CardBody>
      </Card>
      {runtime && runtime.segmentation?.state !== "evaluated" ? (
        <InfoNote tone="neutral">{runtime.segmentation?.reason}</InfoNote>
      ) : null}
    </div>
  );
}
function meanDice(pairs: Record<string, { dice_agreement: number | null }>) {
  const values = Object.values(pairs)
    .map((entry) => entry.dice_agreement)
    .filter((value): value is number => typeof value === "number");
  if (!values.length) return null;
  return (
    values.reduce((sum: number, value: number) => sum + value, 0) /
    values.length
  );
}
function round(value: number | null | undefined) {
  return value === null || value === undefined
    ? null
    : Number(value.toFixed(4));
}
function fastest(result: AnalysisPayload) {
  const entries = Object.entries(result.models)
    .filter(([, entry]) => entry.summary)
    .map(([modelId, entry]) => ({
      modelId,
      seconds: entry.summary?.inference_seconds ?? 0,
    }));
  if (!entries.length) return null;
  return entries.reduce((best, current) =>
    current.seconds < best.seconds ? current : best,
  );
}
function buildPanels(result: AnalysisPayload, version: string): ImagePanel[] {
  const panels: ImagePanel[] = [];
  Object.entries(result.models).forEach(([modelId, entry]) => {
    if (!entry.available || !entry.artifacts) return;
    const artifact =
      version === "masks" ? entry.artifacts.mask : entry.artifacts.probability;
    if (!artifact) return;
    panels.push({
      caption: `${modelId} — ${version === "masks" ? "mask" : "probability map"}`,
      src: api.artifactUrl(result.analysis_id, artifact),
      tone: version === "masks" ? "mask" : "input",
      note:
        version === "masks"
          ? `threshold ${entry.summary?.threshold}`
          : "Raw sigmoid output before thresholding",
    });
  });
  if (result.ensemble.available && result.ensemble.artifacts) {
    const artifact =
      version === "masks"
        ? result.ensemble.artifacts.mask
        : result.ensemble.artifacts.probability;
    if (artifact) {
      panels.push({
        caption: `Ensemble — ${version === "masks" ? "fused mask" : "fused probability"}`,
        src: api.artifactUrl(result.analysis_id, artifact),
        tone: version === "masks" ? "mask" : "overlay",
        note: `${result.ensemble.strategy} fusion`,
      });
    }
  }
  return panels;
}
