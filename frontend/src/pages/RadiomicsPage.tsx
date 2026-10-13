import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Download, FlaskConical, Search } from "lucide-react";
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
  Tabs,
  Tooltip,
} from "../components/ui";
import {
  CategoryRadar,
  FeatureBarChart,
  FeatureScatter,
} from "../components/charts";
import { useAnalysis } from "../hooks/useAnalysis";
import { api } from "../services/api";
import {
  formatInt,
  formatNumber,
  humaniseFeature,
  NOT_AVAILABLE,
} from "../services/format";
import type { FeatureRow, RadiomicsCatalogue } from "../types/api";
export default function RadiomicsPage() {
  const { result, analysisId } = useAnalysis();
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [sortKey, setSortKey] = useState("name");
  const [sortAsc, setSortAsc] = useState(true);
  const [catalogue, setCatalogue] = useState<RadiomicsCatalogue | null>(null);
  useEffect(() => {
    api
      .radiomicsCatalogue()
      .then(setCatalogue)
      .catch(() => setCatalogue(null));
  }, []);
  const radiomics = result?.radiomics;
  const rows = useMemo(() => {
    if (!radiomics?.available || !radiomics.features) return [];
    const categories = radiomics.categories ?? {};
    const categoryOf = new Map();
    Object.entries(categories).forEach(([name, features]) => {
      features.forEach((feature) => categoryOf.set(feature, name));
    });
    return Object.entries(radiomics.features).map(([name, value]) => ({
      name,
      value: value === null ? null : Number(value),
      category: categoryOf.get(name) ?? name.split("_")[0],
      description: radiomics.descriptions?.[name],
    }));
  }, [radiomics]);
  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const subset = rows.filter((row) => {
      const matchesCategory = category === "all" || row.category === category;
      const matchesSearch =
        !needle ||
        row.name.toLowerCase().includes(needle) ||
        humaniseFeature(row.name).toLowerCase().includes(needle) ||
        (row.description ?? "").toLowerCase().includes(needle);
      return matchesCategory && matchesSearch;
    });
    return subset.sort((a, b) => {
      if (sortKey === "name")
        return sortAsc
          ? a.name.localeCompare(b.name)
          : b.name.localeCompare(a.name);
      const left = a.value ?? Number.NEGATIVE_INFINITY;
      const right = b.value ?? Number.NEGATIVE_INFINITY;
      return sortAsc ? left - right : right - left;
    });
  }, [rows, search, category, sortKey, sortAsc]);
  const categoryCounts = useMemo(() => {
    const counts = new Map();
    rows.forEach((row) =>
      counts.set(row.category, (counts.get(row.category) ?? 0) + 1),
    );
    return counts;
  }, [rows]);
  const chartSeries = useMemo(
    () =>
      filtered.slice(0, 18).map((row) => ({
        label: row.name.replace(
          /^(shape|firstorder|glcm|glrlm|glszm|gldm|ngtdm)_/,
          "",
        ),
        value: row.value,
      })),
    [filtered],
  );
  const radarSeries = useMemo(() => {
    const grouped = new Map();
    rows.forEach((row) => {
      if (row.value === null || !Number.isFinite(row.value)) return;
      const bucket = grouped.get(row.category) ?? [];
      bucket.push(Math.abs(row.value));
      grouped.set(row.category, bucket);
    });
    return Array.from(grouped.entries()).map(([name, values]) => ({
      category: name,
      value: Number(
        (
          values.reduce((sum: number, value: number) => sum + value, 0) /
          values.length
        ).toFixed(4),
      ),
    }));
  }, [rows]);
  const scatterSeries = useMemo(() => {
    const byName = new Map(rows.map((row) => [row.name, row.value]));
    const x =
      byName.get("shape_pixel_surface_area") ??
      byName.get("shape_mesh_surface_area");
    const y = byName.get("firstorder_mean") ?? byName.get("firstorder_median");
    if (x === undefined || x === null || y === undefined || y === null)
      return [];
    return [{ x, y, label: "region" }];
  }, [rows]);
  if (!analysisId) {
    return (
      <EmptyState
        title="No analysis selected"
        message="Quantitative radiomics are computed from the predicted region, so they need a completed analysis."
        action={
          <Link className="btn-primary" to="/analysis/new">
            New analysis
          </Link>
        }
        icon={<FlaskConical className="h-6 w-6" />}
      />
    );
  }
  if (!result) return <LoadingBlock label="Loading the radiomics record" />;
  if (!radiomics || !radiomics.available) {
    return (
      <div className="space-y-6">
        <EmptyState
          title="Radiomics unavailable for this analysis"
          message={
            radiomics?.unavailable_reason ??
            "No radiomics record was stored for this analysis."
          }
          action={
            <Link className="btn-secondary" to="/analysis/new">
              Run another analysis
            </Link>
          }
        />
        <InfoNote tone="neutral">
          Features are extracted from the <strong>predicted</strong> mask. If
          every model produced an empty region, there is no region to describe
          and the platform reports that instead of returning zeros.
        </InfoNote>
      </div>
    );
  }
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
            Quantitative radiomics
          </h2>
          <p className="mt-1 text-sm text-clinical-600 dark:text-clinical-300">
            {radiomics.feature_count} features · {radiomics.dimensionality}{" "}
            extraction · backend {radiomics.backend} · mask from{" "}
            {radiomics.quality?.mask_source ?? NOT_AVAILABLE}
          </p>
        </div>
        <div className="flex gap-2">
          <a
            className="btn-secondary"
            href={api.radiomicsCsvUrl(result.analysis_id)}
            download={true}
          >
            <Download className="h-3.5 w-3.5" />
            CSV
          </a>
          <a
            className="btn-secondary"
            href={api.exportAnalysisUrl(result.analysis_id)}
            download={true}
          >
            <Download className="h-3.5 w-3.5" />
            Full JSON
          </a>
        </div>
      </header>
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Region voxels"
          value={
            radiomics.quality?.roi_voxel_count !== undefined
              ? formatInt(radiomics.quality.roi_voxel_count)
              : NOT_AVAILABLE
          }
          tooltip="Number of pixels (2-D pipeline) or voxels (3-D pipeline) inside the region of interest used for the extraction."
        />
        <MetricCard
          label="Pixel area"
          value={
            pixelArea(rows) === null
              ? NOT_AVAILABLE
              : formatInt(pixelArea(rows))
          }
          unit="px"
          tooltip="Counted in pixels. It is a measurement of the image, not of anatomy."
        />
        <MetricCard
          label="Physical area"
          value={
            physicalArea(rows) === null
              ? NOT_AVAILABLE
              : formatNumber(physicalArea(rows), 3)
          }
          unit="mm\u00B2"
          tooltip="Reported only when the image carries pixel spacing. Without spacing, physical size cannot be derived from a pixel count."
          tone={physicalArea(rows) === null ? "warn" : "default"}
        />
        <MetricCard
          label="Spacing known"
          value={radiomics.spacing_available ? "yes" : "no"}
          hint={
            radiomics.spacing_available
              ? "Physical units reported"
              : "Physical units are not derivable from this input"
          }
          tone={radiomics.spacing_available ? "good" : "warn"}
        />
      </section>
      {!radiomics.spacing_available ? (
        <InfoNote tone="warn">
          This input has no pixel spacing metadata, so physical measurements
          such as mm², mm³ or voxel volume cannot be computed. Pixel-domain
          features are correct and complete; the platform states which is which
          rather than converting a pixel count into millimetres.
        </InfoNote>
      ) : null}
      {radiomics.warnings?.length ? (
        <InfoNote tone="warn">
          <ul className="list-disc space-y-1 pl-4">
            {radiomics.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </InfoNote>
      ) : null}
      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Feature table"
            subtitle={`${filtered.length} of ${rows.length} features shown`}
            tooltip="Values come from the extractor's output. A missing value means the extractor could not compute that feature for this region \u2014 it is not zero."
          />
          <CardBody className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative min-w-[220px] flex-1">
                <Search className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-clinical-400" />
                <input
                  className="input pl-8"
                  placeholder="Search feature or description\u2026"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </div>
              <select
                className="input max-w-[190px]"
                value={category}
                onChange={(event) => setCategory(event.target.value)}
              >
                <option value="all">All categories ({rows.length})</option>
                {Array.from(categoryCounts.entries()).map(([name, count]) => (
                  <option value={name} key={name}>
                    {name} ({count})
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="btn-secondary text-xs"
                onClick={() => {
                  setSortKey(sortKey === "name" ? "value" : "name");
                  setSortAsc(true);
                }}
              >
                Sort: {sortKey === "name" ? "name" : "value"} ↑
              </button>
            </div>
            <div className="max-h-[520px] overflow-y-auto">
              <DataTable
                dense={true}
                columns={["Feature", "Category", "Value", "Meaning"]}
                rows={filtered.map((row) => [
                  <span title={row.name} key={"n"}>
                    {humaniseFeature(row.name)}
                  </span>,
                  <Pill tone="violet" key={"c"}>
                    {row.category}
                  </Pill>,
                  row.value === null ? (
                    <span className="text-clinical-400" key={"v"}>
                      not computed
                    </span>
                  ) : (
                    formatNumber(row.value)
                  ),
                  <span
                    className="text-[11px] text-clinical-500 dark:text-clinical-400"
                    key={"d"}
                  >
                    {row.description ?? ""}
                  </span>,
                ])}
                empty="No feature matches the current filter."
              />
            </div>
          </CardBody>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader
              title="Category profile"
              tooltip="Mean absolute value per feature category. The categories live on different scales, so this chart compares the shape of the profile, not the magnitudes."
            />
            <CardBody>
              {radarSeries.length > 2 ? (
                <CategoryRadar data={radarSeries} />
              ) : (
                <InfoNote tone="neutral">
                  Too few categories for a radar view.
                </InfoNote>
              )}
            </CardBody>
          </Card>
          <Card>
            <CardHeader
              title="Quality control"
              tooltip="Extraction quality signals recorded with the feature table."
            />
            <CardBody className="space-y-2 text-sm">
              <Row
                label="ROI extent (px)"
                value={
                  radiomics.quality?.roi_extent
                    ? radiomics.quality.roi_extent
                        .map((value) => String(value))
                        .join(" × ")
                    : NOT_AVAILABLE
                }
              />
              <Row
                label="Relative ROI size"
                value={
                  radiomics.quality?.relative_roi_size !== undefined
                    ? formatNumber(radiomics.quality.relative_roi_size, 5)
                    : NOT_AVAILABLE
                }
              />
              <Row
                label="Touches field-of-view border"
                value={
                  radiomics.quality?.roi_touches_fov_border === undefined
                    ? NOT_AVAILABLE
                    : radiomics.quality.roi_touches_fov_border
                      ? "yes"
                      : "no"
                }
              />
              <Row
                label="Discretisation"
                value={String(
                  catalogue?.discretisation?.description ?? NOT_AVAILABLE,
                )}
              />
              {radiomics.quality?.note ? (
                <p className="prose-note">{radiomics.quality.note}</p>
              ) : null}
            </CardBody>
          </Card>
          <Card>
            <CardHeader
              title="Extractor"
              subtitle={
                catalogue
                  ? String(catalogue.backend.selected_backend ?? "")
                  : ""
              }
            />
            <CardBody className="space-y-2 text-xs text-clinical-600 dark:text-clinical-300">
              {catalogue?.notes?.length ? (
                <ul className="list-disc space-y-1 pl-4">
                  {catalogue.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              ) : (
                <LoadingBlock label="Reading the extractor status" />
              )}
              <p className="prose-note">
                Features follow the IBSI-style definitions used by PyRadiomics.
                Where the native extractor is used, only features it can compute
                identically are reported.
              </p>
            </CardBody>
          </Card>
        </div>
      </div>
      <Card>
        <CardHeader
          title="Feature visualisation"
          subtitle="The first 18 filtered features, and the region's area-intensity relationship"
          tooltip="Charts are a rendering of the stored table \u2014 no smoothing, interpolation or imputation is applied."
        />
        <CardBody>
          {chartSeries.length ? (
            <Tabs
              tabs={[
                { id: "bar", label: "Feature values" },
                { id: "scatter", label: "Area vs intensity" },
              ]}
              active="bar"
              onChange={() => undefined}
            />
          ) : null}
          {chartSeries.length ? (
            <FeatureBarChart data={chartSeries} />
          ) : (
            <InfoNote tone="neutral">No numeric feature to plot.</InfoNote>
          )}
          {scatterSeries.length ? (
            <div className="mt-4">
              <p className="mb-1 text-xs text-clinical-500 dark:text-clinical-400">
                Region area against mean intensity (single point: this
                analysis).
              </p>
              <FeatureScatter data={scatterSeries} xKey="x" yKey="y" />
            </div>
          ) : null}
        </CardBody>
      </Card>
      {catalogue?.descriptions ? (
        <Card>
          <CardHeader
            title="Feature glossary"
            subtitle="Definitions as recorded by the extraction catalogue"
          />
          <CardBody>
            <div className="grid gap-2 sm:grid-cols-2">
              {Object.entries(catalogue.descriptions)
                .slice(0, 60)
                .map(([name, description]) => (
                  <div
                    className="rounded border border-clinical-100 p-2 text-xs dark:border-clinical-800"
                    key={name}
                  >
                    <p className="font-medium text-clinical-800 dark:text-clinical-100">
                      {humaniseFeature(name)}
                      <Tooltip text={String(description)} />
                    </p>
                    <p className="mt-0.5 text-clinical-500 dark:text-clinical-400">
                      {String(description).slice(0, 120)}
                    </p>
                  </div>
                ))}
            </div>
          </CardBody>
        </Card>
      ) : null}
    </div>
  );
}
function pixelArea(rows: FeatureRow[]) {
  const row = rows.find(
    (entry) =>
      entry.name === "shape_pixel_area" ||
      entry.name === "shape_pixel_surface_area",
  );
  return row?.value ?? null;
}
function physicalArea(rows: FeatureRow[]) {
  const row = rows.find(
    (entry) =>
      entry.name === "shape_mesh_surface_area" ||
      entry.name === "shape_voxel_volume",
  );
  return row?.value ?? null;
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
