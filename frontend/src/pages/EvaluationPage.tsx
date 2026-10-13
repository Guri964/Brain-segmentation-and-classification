import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { BarChart3, Download, FlaskConical } from "lucide-react";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
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
import { MetricComparisonChart, TrainingCurve } from "../components/charts";
import { api } from "../services/api";
import {
  formatDate,
  formatDuration,
  formatInt,
  formatMetric,
  NOT_AVAILABLE,
  NOT_EVALUATED,
} from "../services/format";
import type { EvaluationResponse, SegmentationMetricEntry } from "../types/api";
const METRIC_ORDER = [
  "dice",
  "iou",
  "precision",
  "recall",
  "sensitivity",
  "specificity",
  "hausdorff95",
  "average_surface_distance",
  "boundary_f1",
];
const METRIC_LABELS: Record<string, string> = {
  dice: "Dice",
  iou: "IoU",
  precision: "Precision",
  recall: "Recall",
  sensitivity: "Sensitivity",
  specificity: "Specificity",
  hausdorff95: "HD95 (px)",
  average_surface_distance: "ASD",
  boundary_f1: "Boundary F1",
};
export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState("segmentation");
  useEffect(() => {
    api
      .evaluation()
      .then((payload) => {
        setData(payload);
        setError(null);
      })
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : String(caught)),
      );
  }, []);
  if (error)
    return (
      <ErrorState
        title="Could not read the evaluation results"
        message={error}
      />
    );
  if (!data) return <LoadingBlock label="Reading evaluation artefacts" />;
  const segmentationMetrics = data.segmentation.models.filter(
    (model: SegmentationMetricEntry) => model.metrics,
  );
  const ablationRows = data.segmentation.ablation ?? [];
  const protocol = data.protocol;
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
            Model evaluation
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-clinical-600 dark:text-clinical-300">
            Metrics for the reserved test split, together with the
            reference-verification checks and the measured inference cost of
            this deployment. Where an evaluation artefact does not exist, the
            platform reports <em>Not evaluated</em> and says which script
            produces it.
          </p>
        </div>
        <Pill tone={data.evaluation.state === "evaluated" ? "teal" : "amber"}>
          <BarChart3 className="h-3 w-3" />
          {data.evaluation.state === "evaluated"
            ? "evaluated"
            : "Not evaluated"}
        </Pill>
      </header>
      {data.evaluation.state !== "evaluated" ? (
        <InfoNote tone="warn">
          <p className="font-medium">
            No evaluation artefact has been produced on this deployment.
          </p>
          <p className="mt-1">{data.evaluation.reason}</p>
          {data.evaluation.produced_by ? (
            <p className="mt-1">
              Produced by:{" "}
              <code className="rounded bg-black/5 px-1 dark:bg-white/10">
                {data.evaluation.produced_by}
              </code>
              {data.evaluation.expected_artifact ? (
                <>
                  {" "}
                  →{" "}
                  <code className="rounded bg-black/5 px-1 dark:bg-white/10">
                    {data.evaluation.expected_artifact}
                  </code>
                </>
              ) : null}
            </p>
          ) : null}
          <p className="mt-2">
            Until it runs, no Dice, IoU, precision, recall, specificity,
            sensitivity, curve or ablation number can be shown — inventing them
            would be worse than showing nothing.
          </p>
        </InfoNote>
      ) : null}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Segmentation models evaluated"
          value={
            segmentationMetrics.length
              ? String(segmentationMetrics.length)
              : NOT_EVALUATED
          }
          hint={
            segmentationMetrics.length
              ? segmentationMetrics.map((model) => model.model_id).join(", ")
              : "requires the reserved test split"
          }
          tone={segmentationMetrics.length ? "good" : "warn"}
        />
        <MetricCard
          label="Mean Dice (across models)"
          value={
            meanDice(segmentationMetrics) === null
              ? NOT_EVALUATED
              : formatMetric(meanDice(segmentationMetrics))
          }
          tooltip="Mean of the per-model Dice coefficients on the reserved test split. Reported only when the artefact exists."
        />
        <MetricCard
          label="Classification evaluation"
          value={String(data.classification.state ?? "not_evaluated").replace(
            /_/g,
            " ",
          )}
          hint={
            typeof data.classification.reason === "string"
              ? data.classification.reason.slice(0, 90)
              : undefined
          }
          tone={data.classification.state === "evaluated" ? "good" : "warn"}
        />
        <MetricCard
          label="Runtime measurements"
          value={
            data.runtime.state === "measured"
              ? `${data.runtime.models?.length ?? 0} model(s)`
              : String(data.runtime.state).replace(/_/g, " ")
          }
          tone={data.runtime.state === "measured" ? "good" : "warn"}
        />
      </section>
      <Card>
        <CardHeader
          title="Protocol"
          subtitle="How the numbers below were produced"
          actions={
            <a
              className="btn-secondary text-xs"
              href="/api/evaluation"
              target="_blank"
              rel="noreferrer"
            >
              <Download className="h-3.5 w-3.5" />
              JSON
            </a>
          }
        />
        <CardBody>
          <Tabs
            tabs={[
              { id: "segmentation", label: "Segmentation" },
              { id: "classification", label: "Classification" },
              { id: "ablations", label: "Ablations" },
              { id: "runtime", label: "Runtime & resources" },
              { id: "protocol", label: "Method" },
            ]}
            active={tab}
            onChange={(id) => setTab(id)}
          />
          <div className="mt-4">
            {tab === "segmentation" ? (
              <div className="space-y-4">
                <DataTable
                  columns={[
                    "Model",
                    "Status",
                    ...METRIC_ORDER.map(
                      (metric) => METRIC_LABELS[metric] ?? metric,
                    ),
                  ]}
                  rows={data.segmentation.models.map((model) => [
                    model.model_id,
                    <StatusBadge value={model.status} key={"s"} />,
                    ...METRIC_ORDER.map((metric) =>
                      model.metrics
                        ? formatMetric(model.metrics[metric])
                        : NOT_EVALUATED,
                    ),
                  ])}
                  empty="No segmentation model is registered."
                />
                {segmentationMetrics.length ? (
                  <div>
                    <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                      Published metrics
                    </h4>
                    <MetricComparisonChart
                      data={segmentationMetrics.map((model) => ({
                        label: model.model_id,
                        ...Object.fromEntries(
                          ["dice", "iou", "precision", "recall"]
                            .filter(
                              (metric) =>
                                typeof model.metrics?.[metric] === "number",
                            )
                            .map((metric) => [
                              METRIC_LABELS[metric],
                              (model.metrics ?? {})[metric],
                            ]),
                        ),
                      }))}
                      keys={["Dice", "IoU", "Precision", "Recall"]}
                    />
                  </div>
                ) : null}
                {data.segmentation.ensemble ? (
                  <div>
                    <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                      Ensemble
                    </h4>
                    <DataTable
                      columns={["Metric", "Value"]}
                      rows={Object.entries(data.segmentation.ensemble)
                        .filter(([, value]) => typeof value === "number")
                        .map(([key, value]) => [
                          METRIC_LABELS[key] ?? key.replace(/_/g, " "),
                          formatMetric(Number(value)),
                        ])}
                    />
                  </div>
                ) : null}
              </div>
            ) : null}
            {tab === "classification" ? (
              <div className="space-y-4">
                {data.classification.state === "evaluated" ? (
                  <DataTable
                    columns={["Metric", "Value"]}
                    rows={Object.entries(data.classification)
                      .filter(
                        ([key, value]) =>
                          key !== "state" && typeof value === "number",
                      )
                      .map(([key, value]) => [
                        key.replace(/_/g, " "),
                        formatMetric(Number(value)),
                      ])}
                  />
                ) : (
                  <InfoNote tone="warn">
                    <p className="font-medium">
                      Classification has not been evaluated.
                    </p>
                    <p className="mt-1">
                      {String(
                        data.classification.reason ??
                          "No classifier artefact exists.",
                      )}
                    </p>
                    <p className="mt-2">
                      Training a classifier requires a labelled radiomics table:
                      one row per case with the extracted features, the target
                      label and a patient-level group identifier so that cases
                      from the same patient stay in the same split.{" "}
                      <code className="rounded bg-black/5 px-1 dark:bg-white/10">
                        scripts/train_classifier.py
                      </code>{" "}
                      consumes exactly that file, records the label mapping, the
                      split and the calibration, and only then does the
                      classification branch activate.
                    </p>
                  </InfoNote>
                )}
              </div>
            ) : null}
            {tab === "ablations" ? (
              <div>
                {ablationRows.length ? (
                  <DataTable
                    columns={[
                      "Configuration",
                      "Change",
                      ...METRIC_ORDER.slice(0, 5).map(
                        (metric) => METRIC_LABELS[metric],
                      ),
                    ]}
                    rows={ablationRows.map((row) => [
                      String(row.configuration ?? row.name ?? "configuration"),
                      String(row.change ?? row.description ?? ""),
                      ...METRIC_ORDER.slice(0, 5).map((metric) =>
                        formatMetric(Number(row[metric] ?? NaN)),
                      ),
                    ])}
                  />
                ) : (
                  <InfoNote tone="neutral">
                    No ablation has been run. Ablations differ by exactly one
                    component — for example TTA on/off, attention gates on/off,
                    ensemble weights equal versus calibrated — and each row
                    requires a labelled test split. Run{" "}
                    <code className="rounded bg-black/5 px-1 dark:bg-white/10">
                      scripts/run_ablations.py
                    </code>{" "}
                    once the split exists.
                  </InfoNote>
                )}
              </div>
            ) : null}
            {tab === "runtime" ? (
              <div className="space-y-3">
                <DataTable
                  columns={[
                    "Model",
                    "State",
                    "Device",
                    "Seconds / image",
                    "Images / second",
                    "Parameters",
                    "Input shape",
                  ]}
                  rows={(data.runtime.models ?? []).map((row) => [
                    row.model_id,
                    <StatusBadge value={row.state} key={"s"} />,
                    String(row.device ?? NOT_AVAILABLE),
                    row.seconds_per_image !== undefined
                      ? formatDuration(row.seconds_per_image)
                      : NOT_AVAILABLE,
                    row.images_per_second !== undefined &&
                    row.images_per_second !== null
                      ? formatInt(row.images_per_second)
                      : NOT_AVAILABLE,
                    row.parameter_count !== undefined &&
                    row.parameter_count !== null
                      ? formatInt(row.parameter_count)
                      : NOT_AVAILABLE,
                    row.input_shape
                      ? row.input_shape.join(" × ")
                      : NOT_AVAILABLE,
                  ])}
                  empty="No runtime measurement is available."
                />
                <p className="prose-note">
                  These are inference measurements taken on this machine with a
                  reference input, generated{" "}
                  {data.runtime.generated_at
                    ? formatDate(data.runtime.generated_at)
                    : "at request time"}
                  . They are not training cost, and they do not transfer to
                  other hardware.
                </p>
              </div>
            ) : null}
            {tab === "protocol" ? (
              <div className="grid gap-4 lg:grid-cols-2">
                <div>
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    Splits
                  </h4>
                  <ul className="list-disc space-y-1 pl-5 text-xs text-clinical-600 dark:text-clinical-300">
                    {protocol.splits.map((split) => (
                      <li key={split}>{split}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    Principles
                  </h4>
                  <ul className="list-disc space-y-1 pl-5 text-xs text-clinical-600 dark:text-clinical-300">
                    {protocol.principles.map((principle) => (
                      <li key={principle}>{principle}</li>
                    ))}
                  </ul>
                </div>
                <div className="lg:col-span-2">
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    Reproduction scripts
                  </h4>
                  <DataTable
                    columns={["Script", "Purpose"]}
                    rows={Object.entries(protocol.scripts).map(
                      ([name, purpose]) => [
                        <code
                          className="text-xs"
                          key={"n"}
                        >{`scripts/${name}.py`}</code>,
                        purpose,
                      ],
                    )}
                  />
                </div>
              </div>
            ) : null}
          </div>
        </CardBody>
      </Card>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Reference verification"
            subtitle="Deterministic phantoms, not held-out data"
            tooltip="These checks confirm that a checkpoint loads, is wired to the right input geometry and produces a spatial response. They are a necessary condition for use, not a measure of accuracy."
          />
          <CardBody className="space-y-3">
            <div className="flex flex-wrap items-center gap-2 text-xs text-clinical-500 dark:text-clinical-400">
              <Pill tone="violet">
                phantom {data.model_health.phantom_version}
              </Pill>
              <span>
                generated {formatDate(data.model_health.generated_at)}
              </span>
              <span>
                responsive:{" "}
                {data.model_health.responsive_models.length
                  ? data.model_health.responsive_models.join(", ")
                  : "none"}
              </span>
            </div>
            <DataTable
              columns={[
                "Model",
                "Verdict",
                "Output semantics",
                "Recommendation",
              ]}
              rows={data.model_health.models.map((model) => [
                model.model_id,
                <StatusBadge value={model.verdict} key={"v"} />,
                <span className="text-xs" key={"o"}>
                  {model.output_semantics}
                </span>,
                <span
                  className="text-[11px] text-clinical-600 dark:text-clinical-300"
                  key={"r"}
                >
                  {model.recommendation}
                </span>,
              ])}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader
            title="Training runs"
            subtitle="Recorded by the training scripts"
            tooltip="A run record is written by scripts/train_*.py when training completes: dataset version, split, seed, schedule, best epoch and checkpoint hash."
          />
          <CardBody>
            {data.training_runs.length ? (
              <div className="space-y-3">
                {data.training_runs.map((run, index) => (
                  <div
                    className="rounded-lg border border-clinical-200 p-3 text-xs dark:border-clinical-800"
                    key={index}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-clinical-800 dark:text-clinical-100">
                        {String(
                          run.model_id ?? run.run_id ?? `run ${index + 1}`,
                        )}
                      </span>
                      <Pill tone="teal">
                        {String(run.status ?? "completed")}
                      </Pill>
                    </div>
                    <dl className="mt-2 grid grid-cols-2 gap-1.5">
                      {Object.entries(run)
                        .filter(
                          ([, value]) =>
                            value !== null && typeof value !== "object",
                        )
                        .slice(0, 10)
                        .map(([key, value]) => (
                          <div className="min-w-0" key={key}>
                            <dt className="text-[10px] uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                              {key.replace(/_/g, " ")}
                            </dt>
                            <dd
                              className="truncate text-clinical-800 dark:text-clinical-100"
                              title={String(value)}
                            >
                              {String(value)}
                            </dd>
                          </div>
                        ))}
                    </dl>
                    {run.history && Array.isArray(run.history) ? (
                      <div className="mt-3">
                        <TrainingCurve
                          points={run.history}
                          lines={["train_loss", "val_loss"]}
                        />
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
              <InfoNote tone="neutral">
                <span className="inline-flex items-start gap-2">
                  <FlaskConical className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  No training run has been executed in this deployment. Training
                  requires the original dataset: patient-level splits can only
                  be built from identified studies, and the scripts refuse to
                  train without them.
                </span>
              </InfoNote>
            )}
          </CardBody>
        </Card>
      </div>
      <Card>
        <CardHeader
          title="Model selection record"
          subtitle="Recorded by the evaluation script"
        />
        <CardBody>
          {data.model_selection.state === "available" ||
          data.model_selection.state === "evaluated" ? (
            <DataTable
              columns={["Field", "Value"]}
              rows={Object.entries(data.model_selection).map(([key, value]) => [
                key.replace(/_/g, " "),
                typeof value === "object"
                  ? JSON.stringify(value).slice(0, 200)
                  : String(value),
              ])}
            />
          ) : (
            <InfoNote tone="neutral">
              {data.model_selection.reason ??
                "No model selection record exists."}
              {data.model_selection.produced_by ? (
                <span className="ml-1">
                  Produced by{" "}
                  <code className="rounded bg-black/5 px-1 dark:bg-white/10">
                    {data.model_selection.produced_by}
                  </code>
                  .
                </span>
              ) : null}
            </InfoNote>
          )}
        </CardBody>
      </Card>
      <InfoNote tone="neutral">
        <span className="inline-flex items-start gap-2">
          <Tooltip text="Any number on this page originates from a stored artefact. The page never computes a metric from unlabelled predictions.">
            <ValueOrUnavailable value="Metric provenance" />
          </Tooltip>
          <span>
            Curves, ablations and confidence intervals appear here only after
            the training and evaluation scripts have run on real data. See{" "}
            <Link className="underline underline-offset-2" to="/settings">
              Settings
            </Link>{" "}
            for the exact commands.
          </span>
        </span>
      </InfoNote>
    </div>
  );
}
function meanDice(models: SegmentationMetricEntry[]) {
  const values = models
    .map((model) => model.metrics?.dice)
    .filter((value) => typeof value === "number");
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
