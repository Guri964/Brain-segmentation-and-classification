import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Download, FileText, RefreshCw } from "lucide-react";
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
} from "../components/ui";
import { useToast } from "../components/AppProviders";
import { useAnalysis } from "../hooks/useAnalysis";
import { api } from "../services/api";
import {
  formatBytes,
  formatDate,
  formatInt,
  NOT_AVAILABLE,
} from "../services/format";
import type { ReportListResponse } from "../types/api";
const SECTIONS = [
  "Study and analysis identifiers, timestamps and software versions",
  "Input description: pipeline, channel count, tensor geometry and normalisation",
  "Preprocessing record: resize, intensity handling and duplicate-channel check",
  "Models executed, their checkpoints, architecture and reference-verification verdict",
  "Per-model segmentation statistics and the decision threshold",
  "Ensemble composition, weights and the provenance of those weights",
  "Regions of interest used for measurement, with pixel and physical extents",
  "Radiomics feature table grouped by category, with the extraction backend",
  "Quality-control flags recorded with the feature table",
  "Explainability record: model, target layer and method for each saliency map",
  "Uncertainty record: method, number of passes and dispersion statistics",
  "Classification status, including an explicit statement when it is disabled",
  "Model comparison: agreement between models, labelled as agreement rather than accuracy",
  "Limitations, disclaimer and the list of data required for a clinical-grade evaluation",
];
export default function ReportsPage() {
  const { push } = useToast();
  const { result, analysisId, refresh } = useAnalysis();
  const [reports, setReports] = useState<ReportListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const load = async () => {
    setLoading(true);
    try {
      setReports(await api.reports());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    void load();
  }, []);
  const regenerate = async () => {
    if (!analysisId) return;
    setBusy(true);
    try {
      const info = await api.regenerateReport(analysisId);
      push(
        "success",
        "Report regenerated",
        `${info.report_id} · ${info.relative_path}`,
      );
      await load();
      await refresh();
    } catch (caught) {
      push(
        "error",
        "Report generation failed",
        caught instanceof Error ? caught.message : String(caught),
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
            Reports
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-clinical-600 dark:text-clinical-300">
            PDF reports are rendered from the stored analysis payload. A value
            the pipeline could not measure is printed as unavailable and never
            as zero, and the limitations section is attached to every report.
          </p>
        </div>
        <div className="flex gap-2">
          {analysisId ? (
            <button
              type="button"
              className="btn-secondary"
              onClick={() => void regenerate()}
              disabled={busy}
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`}
              />
              Regenerate for {analysisId}
            </button>
          ) : null}
          <Link to="/history" className="btn-secondary">
            History
          </Link>
        </div>
      </header>
      {result ? (
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Report for this analysis"
            value={result.report?.available ? "generated" : "not generated"}
            tone={result.report?.available ? "good" : "warn"}
            hint={result.report?.reason ?? result.report?.report_id}
          />
          <MetricCard
            label="Size"
            value={
              result.report?.size_bytes
                ? formatBytes(result.report.size_bytes)
                : NOT_AVAILABLE
            }
          />
          <MetricCard
            label="Generator version"
            value={String(result.report?.generator_version ?? NOT_AVAILABLE)}
          />
          <MetricCard
            label="Stored reports"
            value={reports ? formatInt(reports.items.length) : NOT_AVAILABLE}
            hint="Across all analyses"
          />
        </section>
      ) : null}
      {error ? <ErrorState message={error} /> : null}
      <Card>
        <CardHeader
          title="Generated reports"
          subtitle="Each report belongs to exactly one analysis"
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
        <CardBody>
          {loading ? <LoadingBlock label="Reading stored reports" /> : null}
          {reports?.items.length ? (
            <DataTable
              columns={["Report", "Analysis", "Scan", "Size", "Created", ""]}
              rows={reports.items.map((item) => [
                <span className="font-mono text-xs" key={"r"}>
                  {item.report_id}
                </span>,
                <Link
                  className="text-clinical-600 underline-offset-2 hover:underline dark:text-clinical-300"
                  to={`/history/${item.analysis_id}`}
                  key={"a"}
                >
                  {item.analysis_id}
                </Link>,
                item.scan_identifier,
                formatBytes(item.size_bytes),
                formatDate(item.created_at),
                <a
                  className="btn-secondary px-2 py-1 text-xs"
                  href={api.reportUrl(item.report_id)}
                  download={true}
                  key={"d"}
                >
                  <Download className="h-3 w-3" />
                  Download
                </a>,
              ])}
            />
          ) : null}
          {!loading && !reports?.items.length ? (
            <EmptyState
              title="No report has been generated yet"
              message="A report is written automatically at the end of every analysis that completes successfully. If report rendering failed, the analysis payload records the reason."
              icon={<FileText className="h-6 w-6" />}
            />
          ) : null}
        </CardBody>
      </Card>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Report contents"
            subtitle="Fourteen required sections, each rendered from stored values"
          />
          <CardBody>
            <ol className="grid gap-2 text-xs sm:grid-cols-2">
              {SECTIONS.map((section, index) => (
                <li
                  className="flex items-start gap-2 rounded-lg border border-clinical-100 p-2 dark:border-clinical-800"
                  key={section}
                >
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-clinical-100 text-[11px] font-semibold text-clinical-700 dark:bg-clinical-800 dark:text-clinical-200">
                    {index + 1}
                  </span>
                  <span className="text-clinical-600 dark:text-clinical-300">
                    {section}
                  </span>
                </li>
              ))}
            </ol>
          </CardBody>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader title="Report policy" />
            <CardBody className="space-y-2 text-xs text-clinical-600 dark:text-clinical-300">
              <p>
                <Pill tone="teal">real values only</Pill>
              </p>
              <p>
                Numbers come from the payload stored at analysis time. Nothing
                is recomputed at render time, so a downloaded report always
                matches the analysis it belongs to.
              </p>
              <p>
                Missing sections are rendered with an explicit reason (for
                example, “no reference mask exists, so no Dice coefficient is
                reported”) instead of an empty cell or a zero.
              </p>
              <p>
                The limitations page is fixed prose and is appended to every
                report, including reports for analyses that succeeded.
              </p>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Analysis report state" />
            <CardBody className="space-y-2 text-sm">
              <Row label="Analysis" value={analysisId ?? "none selected"} />
              <Row
                label="State"
                value={
                  result?.report?.available ? (
                    <StatusBadge value="completed" label="generated" />
                  ) : (
                    <StatusBadge value="missing" label="not generated" />
                  )
                }
              />
              <Row
                label="Relative path"
                value={String(result?.report?.relative_path ?? NOT_AVAILABLE)}
              />
            </CardBody>
          </Card>
        </div>
      </div>
      <InfoNote tone="warn">
        A report is a research record, not a clinical document. It reproduces
        model output, it does not confirm that the outlined region corresponds
        to any pathology, and it makes no statement about diagnosis, grade or
        prognosis.
      </InfoNote>
    </div>
  );
}
function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 text-xs">
      <span className="text-clinical-500 dark:text-clinical-400">{label}</span>
      <span className="max-w-[60%] break-words text-right font-medium text-clinical-800 dark:text-clinical-100">
        {value}
      </span>
    </div>
  );
}
