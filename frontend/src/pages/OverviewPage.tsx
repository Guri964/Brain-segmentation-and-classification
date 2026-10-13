import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  Upload,
  XCircle,
} from "lucide-react";
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
  SkeletonCard,
  StatusBadge,
} from "../components/ui";
import { api } from "../services/api";
import { formatDate, formatInt, NOT_AVAILABLE } from "../services/format";
import { useAnalysis, useCapabilities } from "../hooks/useAnalysis";
import type { Statistics } from "../types/api";
export default function OverviewPage() {
  const { capabilities } = useCapabilities();
  const { result, loading: analysisLoading } = useAnalysis();
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .statistics()
      .then(setStatistics)
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : String(caught)),
      );
  }, []);
  const responsive = capabilities?.model_health.responsive_models ?? [];
  const problems =
    capabilities?.model_health.models.filter(
      (model) => model.status !== "missing" && !model.responsive,
    ) ?? [];
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
            Research dashboard
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-clinical-600 dark:text-clinical-300">
            Segmentation, quantitative radiomics, explainability and uncertainty
            for brain MRI studies. Every value shown is either measured by a
            model or reported as unavailable; nothing is filled in for display
            purposes.
          </p>
        </div>
        <Link to="/analysis/new" className="btn-primary">
          <Upload className="h-4 w-4" />
          New MRI analysis
        </Link>
      </header>
      {error ? <ErrorState message={error} /> : null}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {statistics ? (
          <>
            <MetricCard
              label="Analyses stored"
              value={formatInt(statistics.totals.analyses)}
              hint={`${formatInt(statistics.totals.reports)} reports generated`}
              tooltip="Total analyses persisted in the local database."
            />
            <MetricCard
              label="Segmentations run"
              value={formatInt(statistics.totals.segmentations)}
              hint="One row per model per analysis"
            />
            <MetricCard
              label="Radiomics extractions"
              value={formatInt(statistics.totals.radiomics)}
              hint="Feature tables computed from real image-mask pairs"
            />
            <MetricCard
              label="Classification results"
              value={formatInt(statistics.totals.classifications)}
              hint="Served only from a validated classifier artefact"
              tone={statistics.totals.classifications ? "good" : "default"}
            />
          </>
        ) : (
          [0, 1, 2, 3].map((index) => <SkeletonCard key={index} />)
        )}
      </section>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Model status"
            subtitle="A checkpoint that loads is not automatically a checkpoint that works \u2014 each one is verified against a deterministic reference phantom"
            actions={
              <Link to="/evaluation" className="btn-ghost text-xs">
                Model evaluation
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            }
          />
          <CardBody>
            {!capabilities ? (
              <LoadingBlock label="Checking models" />
            ) : (
              <DataTable
                columns={[
                  "Model",
                  "Status",
                  "Reference check",
                  "Input",
                  "Notes",
                ]}
                rows={capabilities.models.map((model) => {
                  const health = capabilities.model_health.models.find(
                    (entry) => entry.model_id === model.model_id,
                  );
                  return [
                    <div key={model.model_id}>
                      <p className="font-medium text-clinical-900 dark:text-clinical-50">
                        {model.display_name}
                      </p>
                      <p className="text-[11px] text-clinical-500 dark:text-clinical-400">
                        {model.architecture}
                      </p>
                    </div>,
                    <StatusBadge value={model.status} key={"status"} />,
                    <StatusBadge
                      value={health?.verdict ?? "not_loaded"}
                      key={"health"}
                    />,
                    <span className="text-xs" key={"input"}>
                      {model.input_spec.channels}×{model.input_spec.height}×
                      {model.input_spec.width}
                    </span>,
                    <span
                      className="text-xs text-clinical-600 dark:text-clinical-300"
                      key={"note"}
                    >
                      {model.available
                        ? (health?.message ?? model.description)
                        : model.message}
                    </span>,
                  ];
                })}
              />
            )}
          </CardBody>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader
              title="Component availability"
              tooltip="Features stay disabled until the artefact they need actually exists, so the interface never offers a control that cannot work."
            />
            <CardBody className="space-y-3 text-sm">
              <AvailabilityRow
                label="Ready segmentation models"
                value={`${responsive.length} / ${capabilities?.models.length ?? 0}`}
                ok={responsive.length > 0}
              />
              <AvailabilityRow
                label="Probability ensemble"
                value={
                  capabilities?.ensemble.available
                    ? "available"
                    : "not available"
                }
                ok={Boolean(capabilities?.ensemble.available)}
              />
              <AvailabilityRow
                label="Radiomics extraction"
                value={
                  capabilities?.radiomics.selected_backend ?? NOT_AVAILABLE
                }
                ok={Boolean(capabilities)}
              />
              <AvailabilityRow
                label="Classification"
                value={
                  capabilities?.classification.available
                    ? "validated artefact present"
                    : "disabled — no validated artefact"
                }
                ok={Boolean(capabilities?.classification.available)}
              />
              <AvailabilityRow
                label="Grad-CAM"
                value={`${capabilities?.explainability.segmentation.filter((entry) => entry.available).length ?? 0} model(s)`}
                ok={Boolean(
                  capabilities?.explainability.segmentation.some(
                    (entry) => entry.available,
                  ),
                )}
              />
              <AvailabilityRow
                label="Assistant backend"
                value={
                  capabilities?.assistant.backend === "llm"
                    ? "language model (grounded)"
                    : "local, offline"
                }
                ok={true}
              />
              <div className="pt-1">
                <Pill tone="amber">
                  Evaluation:{" "}
                  {capabilities?.evaluation === "evaluated"
                    ? "evaluated"
                    : "Not evaluated"}
                </Pill>
              </div>
            </CardBody>
          </Card>
          {problems.length ? (
            <InfoNote tone="warn">
              <p className="font-semibold">Checkpoints that need attention</p>
              <ul className="mt-1 list-disc space-y-1 pl-4">
                {problems.map((model) => (
                  <li key={model.model_id}>
                    <span className="font-medium">{model.model_id}</span> —{" "}
                    {model.verdict.replace(/_/g, " ")}. {model.recommendation}
                    {model.training ? (
                      <span className="mt-0.5 block text-xs opacity-80">
                        Training data: {model.training.label}
                        {model.training.note ? ` — ${model.training.note}` : ""}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ul>
            </InfoNote>
          ) : null}
        </div>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Current analysis"
            subtitle="The analysis the results pages are showing"
            actions={
              <Link to="/history" className="btn-ghost text-xs">
                History
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            }
          />
          <CardBody>
            {analysisLoading ? (
              <LoadingBlock />
            ) : result ? (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <Field label="Analysis ID" value={result.analysis_id} />
                <Field label="Scan identifier" value={result.scan_identifier} />
                <Field
                  label="Type"
                  value={result.analysis_type.replace(/_/g, " ")}
                />
                <Field label="Created" value={formatDate(result.created_at)} />
                <Field label="Device" value={result.device} />
                <Field label="Threshold" value={String(result.threshold)} />
                <Field
                  label="Models run"
                  value={Object.keys(result.models).join(", ")}
                />
                <Field
                  label="Radiomics features"
                  value={
                    result.radiomics.available
                      ? String(result.radiomics.feature_count)
                      : NOT_AVAILABLE
                  }
                />
              </dl>
            ) : (
              <InfoNote tone="neutral">
                No analysis is selected. Run{" "}
                <span className="font-medium">New MRI analysis</span>, or pick
                one from the history.
              </InfoNote>
            )}
          </CardBody>
        </Card>
        <Card>
          <CardHeader
            title="Recent activity"
            tooltip="Persisted analyses, newest first."
          />
          <CardBody>
            {statistics?.recent?.length ? (
              <ul className="divide-y divide-clinical-100 text-sm dark:divide-clinical-800">
                {statistics.recent.map((item) => (
                  <li
                    className="flex items-center justify-between gap-3 py-2"
                    key={item.analysis_id}
                  >
                    <div className="min-w-0">
                      <p className="truncate font-medium text-clinical-800 dark:text-clinical-100">
                        {item.scan_identifier}
                      </p>
                      <p className="text-[11px] text-clinical-500 dark:text-clinical-400">
                        {formatDate(item.created_at)} ·{" "}
                        {item.analysis_type.replace(/_/g, " ")} ·{" "}
                        {item.model_ids.join(", ")}
                      </p>
                    </div>
                    <StatusBadge value={item.status} />
                  </li>
                ))}
              </ul>
            ) : (
              <InfoNote tone="neutral">
                <span className="inline-flex items-center gap-1.5">
                  <Database className="h-3.5 w-3.5" /> No analyses stored yet.
                </span>
              </InfoNote>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
function AvailabilityRow({
  label,
  value,
  ok,
}: {
  label: string;
  value: ReactNode;
  ok: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-clinical-600 dark:text-clinical-300">{label}</span>
      <span className="flex items-center gap-1.5 text-right font-medium text-clinical-800 dark:text-clinical-100">
        {ok ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
        ) : (
          <XCircle className="h-3.5 w-3.5 text-amber-500" />
        )}
        {value}
      </span>
    </div>
  );
}
function Field({ label, value }: { label: string; value: ReactNode }) {
  const title = typeof value === "string" ? value : undefined;
  return (
    <div className="min-w-0">
      <dt className="text-[11px] uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
        {label}
      </dt>
      <dd
        className="truncate text-clinical-800 dark:text-clinical-100"
        title={title}
      >
        {value}
      </dd>
    </div>
  );
}
export { AlertTriangle };
