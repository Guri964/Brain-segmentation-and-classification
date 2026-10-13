import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  Download,
  Eye,
  History as HistoryIcon,
  Search,
  Trash2,
} from "lucide-react";
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
  StatusBadge,
} from "../components/ui";
import { useToast } from "../components/AppProviders";
import { useAnalysis } from "../hooks/useAnalysis";
import { api, ApiError } from "../services/api";
import {
  formatBytes,
  formatDate,
  formatDuration,
  formatInt,
  NOT_AVAILABLE,
} from "../services/format";
import type { AnalysisDetail, HistoryResponse, Statistics } from "../types/api";
export default function HistoryPage() {
  const params = useParams();
  const navigate = useNavigate();
  const { push } = useToast();
  const { select, refresh } = useAnalysis();
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [analysisType, setAnalysisType] = useState("");
  const [page, setPage] = useState(0);
  const limit = 25;
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [history, stats] = await Promise.all([
        api.history({
          search,
          status,
          analysis_type: analysisType,
          limit,
          offset: page * limit,
        }),
        api.statistics(),
      ]);
      setData(history);
      setStatistics(stats);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  }, [search, status, analysisType, page]);
  useEffect(() => {
    void load();
  }, [load]);
  const remove = async (analysisId: string) => {
    if (
      !window.confirm(
        `Delete ${analysisId}? The stored measurements, artefacts and report are removed. This cannot be undone.`,
      )
    )
      return;
    try {
      const outcome = await api.deleteAnalysis(analysisId);
      push(
        "success",
        "Analysis deleted",
        `${outcome.files_removed} artefact(s) removed.`,
      );
      if (params.analysisId === analysisId) navigate("/history");
      await load();
      await refresh();
    } catch (caught) {
      const message =
        caught instanceof ApiError ? caught.message : String(caught);
      push("error", "Deletion failed", message);
    }
  };
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
            Analysis history
          </h2>
          <p className="mt-1 text-sm text-clinical-600 dark:text-clinical-300">
            Locally stored analyses with their measured values. Record
            identifiers are pseudonymous; no patient-identifying field exists in
            the schema.
          </p>
        </div>
        <Link to="/analysis/new" className="btn-primary">
          New analysis
        </Link>
      </header>
      {statistics ? (
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Analyses"
            value={formatInt(statistics.totals.analyses)}
            hint={`${formatInt(statistics.totals.segmentations)} segmentation records`}
          />
          <MetricCard
            label="Reports"
            value={formatInt(statistics.totals.reports)}
            hint={`${formatInt(statistics.totals.radiomics)} radiomics tables`}
          />
          <MetricCard
            label="Models ready"
            value={`${statistics.models.ready}/${statistics.models.total}`}
            hint={`${formatInt(statistics.totals.classifications)} classification result(s) stored`}
            tone={
              statistics.models.ready === statistics.models.total
                ? "good"
                : "warn"
            }
          />
          <MetricCard
            label="Evaluation state"
            value={statistics.evaluation_state.replace(/_/g, " ")}
            tone={statistics.evaluation_state === "evaluated" ? "good" : "warn"}
          />
        </section>
      ) : null}
      <Card>
        <CardHeader
          title="Filter"
          subtitle={data ? `${data.total} matching record(s)` : undefined}
          actions={
            <button
              type="button"
              className="btn-ghost text-xs"
              onClick={() => void load()}
            >
              Refresh
            </button>
          }
        />
        <CardBody className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-clinical-400" />
              <input
                className="input pl-8"
                placeholder="Search identifier or model\u2026"
                value={search}
                onChange={(event) => {
                  setSearch(event.target.value);
                  setPage(0);
                }}
              />
            </div>
            <select
              className="input"
              value={status}
              onChange={(event) => {
                setStatus(event.target.value);
                setPage(0);
              }}
            >
              <option value="">All statuses</option>
              {(data?.filters.statuses ?? []).map((value: string) => (
                <option value={value} key={value}>
                  {value}
                </option>
              ))}
            </select>
            <select
              className="input"
              value={analysisType}
              onChange={(event) => {
                setAnalysisType(event.target.value);
                setPage(0);
              }}
            >
              <option value="">All pipeline types</option>
              {(data?.filters.analysis_types ?? []).map((value: string) => (
                <option value={value} key={value}>
                  {value.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>
          {error ? <ErrorState message={error} /> : null}
          {loading ? <LoadingBlock label="Reading the database" /> : null}
          {data && !data.items.length && !loading ? (
            <EmptyState
              title="No analysis recorded"
              message="Nothing matches the current filter. Run an analysis, or clear the filter."
              icon={<HistoryIcon className="h-6 w-6" />}
            />
          ) : null}
          {data?.items.length ? (
            <DataTable
              columns={[
                "Analysis",
                "Type",
                "Models",
                "Status",
                "Created",
                "Duration",
                "Report",
                "",
              ]}
              rows={data.items.map((item) => [
                <div key={item.analysis_id}>
                  <p className="font-medium text-clinical-900 dark:text-clinical-50">
                    {item.scan_identifier}
                  </p>
                  <p className="text-[11px] text-clinical-500 dark:text-clinical-400">
                    {item.analysis_id}
                  </p>
                </div>,
                item.analysis_type.replace(/_/g, " "),
                <span className="text-xs" key={"m"}>
                  {item.model_ids.join(", ") || NOT_AVAILABLE}
                </span>,
                <StatusBadge value={item.status} key={"s"} />,
                formatDate(item.created_at),
                item.duration_seconds !== null &&
                item.duration_seconds !== undefined
                  ? formatDuration(item.duration_seconds)
                  : NOT_AVAILABLE,
                item.report_count ? String(item.report_count) : "not generated",
                <div className="flex items-center gap-1" key={"a"}>
                  <button
                    type="button"
                    className="btn-ghost p-1"
                    title="Open in the results pages"
                    onClick={() => {
                      select(item.analysis_id);
                      navigate("/analysis/segmentation");
                    }}
                  >
                    <Eye className="h-3.5 w-3.5" />
                  </button>
                  <a
                    className="btn-ghost p-1"
                    title="Download the JSON export"
                    href={api.exportAnalysisUrl(item.analysis_id)}
                    download={true}
                  >
                    <Download className="h-3.5 w-3.5" />
                  </a>
                  <button
                    type="button"
                    className="btn-ghost p-1 text-rose-600 dark:text-rose-400"
                    title="Delete"
                    onClick={() => void remove(item.analysis_id)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>,
              ])}
            />
          ) : null}
          {data && data.total > limit ? (
            <div className="flex items-center justify-between text-xs">
              <span className="text-clinical-500 dark:text-clinical-400">
                Page {page + 1} of {Math.max(1, Math.ceil(data.total / limit))}
              </span>
              <div className="flex gap-2">
                <button
                  type="button"
                  className="btn-secondary text-xs"
                  disabled={page === 0}
                  onClick={() => setPage((value) => Math.max(0, value - 1))}
                >
                  Previous
                </button>
                <button
                  type="button"
                  className="btn-secondary text-xs"
                  disabled={(page + 1) * limit >= data.total}
                  onClick={() => setPage((value) => value + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          ) : null}
        </CardBody>
      </Card>
      {params.analysisId ? (
        <DetailPanel
          analysisId={params.analysisId}
          onDeleted={() => void load()}
        />
      ) : null}
      <InfoNote tone="neutral">
        Deleting an analysis removes its database rows together with the
        uploaded input, the derived images and the generated report. Model
        checkpoints and evaluation results are never touched.
      </InfoNote>
    </div>
  );
}
function DetailPanel({
  analysisId,
  onDeleted,
}: {
  analysisId: string;
  onDeleted: () => void;
}) {
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .analysis(analysisId)
      .then(setDetail)
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : String(caught)),
      );
  }, [analysisId]);
  return (
    <Card>
      <CardHeader
        title={`Stored record — ${analysisId}`}
        subtitle="Exactly what the database holds for this analysis"
      />
      <CardBody className="space-y-4">
        {error ? <ErrorState message={error} /> : null}
        {!detail && !error ? <LoadingBlock /> : null}
        {detail ? (
          <>
            <DataTable
              columns={["Field", "Value"]}
              rows={Object.entries(detail.metadata)
                .filter(
                  ([, value]) =>
                    value !== null &&
                    value !== undefined &&
                    typeof value !== "object",
                )
                .map(([key, value]) => [key, String(value)])}
            />
            <div className="grid gap-4 lg:grid-cols-2">
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                  Segmentation records
                </h4>
                <DataTable
                  dense={true}
                  columns={[
                    "Model",
                    "Threshold",
                    "Pixels",
                    "Fraction",
                    "Mean p",
                    "Seconds",
                  ]}
                  rows={detail.stored_records.segmentation.map((row) => [
                    String(row.model_id),
                    String(row.threshold),
                    formatInt(Number(row.foreground_pixels)),
                    String(row.foreground_fraction),
                    String(row.probability_mean),
                    String(row.inference_seconds),
                  ])}
                />
              </div>
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                  Reports
                </h4>
                <DataTable
                  dense={true}
                  columns={["Report", "Size", "Created", ""]}
                  rows={detail.stored_records.reports.map((row) => [
                    row.report_id,
                    formatBytes(row.size_bytes),
                    formatDate(row.created_at),
                    <a
                      className="btn-ghost p-1"
                      href={api.reportUrl(row.report_id)}
                      download={true}
                      key={"d"}
                    >
                      <Download className="h-3.5 w-3.5" />
                    </a>,
                  ])}
                  empty="No report was generated for this analysis."
                />
                <button
                  type="button"
                  className="btn-secondary mt-3 text-xs"
                  onClick={onDeleted}
                >
                  Collapse
                </button>
              </div>
            </div>
            <InfoNote tone="neutral">
              Payload source:{" "}
              <span className="font-medium">{detail.payload_source}</span>. The
              canonical result shape is the stored analysis.json, which the
              result pages, the assistant and report regeneration all read.
            </InfoNote>
          </>
        ) : null}
      </CardBody>
    </Card>
  );
}
