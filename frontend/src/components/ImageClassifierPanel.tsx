import { useCallback, useEffect, useState } from "react";
import { Loader2, Play, Upload } from "lucide-react";
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
  Tooltip,
} from "./ui";
import { FeatureBarChart } from "./charts";
import { api, ApiError } from "../services/api";
import { formatBytes, formatNumber, NOT_AVAILABLE } from "../services/format";
import { useCapabilities } from "../hooks/useAnalysis";
import type { ImageClassificationPrediction, ImageClassifierModel } from "../types/api";

const STATE_LABEL: Record<string, string> = {
  ready: "Installed and loadable",
  checkpoint_missing: "Checkpoint not installed",
  metadata_missing: "Metadata not installed",
  framework_missing: "TensorFlow not available",
  incompatible: "Not compatible with this serving path",
  error: "Could not be inspected",
};

/**
 * The four-class tumour-type model (`firstnb.ipynb`, Xception).
 *
 * This is a different task from the BraTS HGG/LGG grade classifier elsewhere on this page:
 * different dataset, different label space, different input (the image itself, 299x299x3). Its
 * availability therefore comes from `capabilities.image_classification` — the radiomics grade
 * classifier's `available` flag says nothing about this model and is never read here.
 *
 * The panel shows only what the checkpoint and its metadata can support: when either file is
 * missing it says which one and where it looked, instead of offering a button that cannot produce
 * a number. Because the two tasks are separate, this panel stays usable while the grade classifier
 * is switched off, and the metrics it displays are labelled as reported by the training notebook
 * (metadata.json), never as measured by this deployment.
 */
export default function ImageClassifierPanel() {
  const { capabilities, reload } = useCapabilities();
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ImageClassificationPrediction | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [models, setModels] = useState<ImageClassifierModel[] | null>(null);
  const [catalogueError, setCatalogueError] = useState<string | null>(null);

  const summary = capabilities?.image_classification;

  const accepted = ".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp";

  // The catalogue is what names the *files*: capabilities carries one compact summary, while
  // /api/classification/image/models reports per-file presence (checkpoint, metadata, framework)
  // and every directory that was searched.
  const loadCatalogue = useCallback(() => {
    api
      .imageClassifierModels()
      .then((payload) => {
        setModels(payload.models);
        setCatalogueError(null);
      })
      .catch((caught) =>
        setCatalogueError(caught instanceof Error ? caught.message : String(caught)),
      );
  }, []);

  useEffect(() => {
    loadCatalogue();
  }, [loadCatalogue]);

  const run = async () => {
    if (!file) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const prediction = await api.classifyImage(form);
      setResult(prediction);
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause
          : new ApiError("The image classifier request failed.", "unknown", 0),
      );
    } finally {
      setRunning(false);
    }
  };

  const checkAgain = () => {
    reload();
    loadCatalogue();
  };

  if (!capabilities) return <LoadingBlock label="Checking the image classifier" />;

  const available = Boolean(summary?.available);
  const model = models?.[0] ?? null;
  const chartData = (result?.scores ?? []).map((entry) => ({
    label: entry.label,
    value: entry.probability,
  }));

  return (
    <Card>
      <CardHeader
        title="Xception image classification (four tumour types)"
        subtitle="The image itself is the input: 299 × 299 × 3 RGB, Keras Xception trained in firstnb.ipynb. A separate task, dataset and label space from the radiomics grade classifier, and its status is read from image_classification — never from the radiomics classifier."
        actions={
          <Pill tone={available ? "teal" : "amber"}>
            {available ? "ready" : (summary?.state ?? "unknown")}
          </Pill>
        }
      />
      <CardBody className="space-y-4">
        <section className="grid gap-3 sm:grid-cols-3">
          <MetricCard
            label="Backend status"
            value={STATE_LABEL[summary?.state ?? ""] ?? summary?.state ?? NOT_AVAILABLE}
            tone={available ? "good" : "warn"}
            hint="From GET /api/capabilities → image_classification"
          />
          <MetricCard
            label="Model"
            value={summary?.display_name ?? NOT_AVAILABLE}
            hint="brain_tumor_xception"
          />
          <MetricCard
            label="Input"
            value={summary?.input_size?.join(" × ") ?? NOT_AVAILABLE}
            hint="RGB, as recorded in metadata.json"
          />
        </section>

        {available ? (
          <InfoNote tone="neutral">
            <p className="font-semibold">This classifier is ready.</p>
            <p className="mt-1">
              Classes: {summary?.classes?.length ? summary.classes.join(", ") : NOT_AVAILABLE}. The
              predicted label is the tumour-type class of the training dataset
              {summary?.model_path ? (
                <>
                  {" "}
                  (<code>{summary.model_path}</code>)
                </>
              ) : null}
              . It is not the BraTS HGG/LGG grade, not a malignancy verdict and not a stage.
            </p>
          </InfoNote>
        ) : (
          <InfoNote tone="warn">
            <p className="font-semibold">The image classifier is not usable in this deployment</p>
            <p className="mt-1">{summary?.reason}</p>
            {model ? (
              <ul className="mt-2 list-disc space-y-1 pl-4">
                <li>
                  <code>{model.model_file}</code> —{" "}
                  {model.model_file_present ? "present" : "not found"}
                  {model.model_path ? ` (looked at ${model.model_path})` : ""}
                </li>
                <li>
                  <code>{model.metadata_file}</code> —{" "}
                  {model.metadata_present ? "present" : "not found"}
                </li>
                <li>
                  TensorFlow —{" "}
                  {model.framework_available === null
                    ? "not checked (a file is missing first)"
                    : model.framework_available
                      ? `importable${model.framework_version ? ` (${model.framework_version})` : ""}`
                      : `not importable${model.framework_error ? ` (${model.framework_error})` : ""}`}
                </li>
              </ul>
            ) : null}
            {model?.searched?.length ? (
              <p className="mt-2 text-xs opacity-80">
                Directories searched: {model.searched.join(" · ")}
              </p>
            ) : null}
            <p className="mt-2">
              Weights are never downloaded automatically and nothing is predicted until both files
              are in place.
            </p>
          </InfoNote>
        )}

        {catalogueError ? (
          <InfoNote tone="neutral">
            The per-file catalogue could not be read ({catalogueError}); the status above comes from
            /api/capabilities.
          </InfoNote>
        ) : null}

        {available ? (
          <>
            <div className="rounded-lg border border-dashed border-clinical-300 p-3 dark:border-clinical-700">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-xs font-medium text-clinical-800 dark:text-clinical-100">
                    MRI image (PNG, JPEG, BMP or TIFF)
                  </p>
                  <p className="mt-0.5 truncate text-[11px] text-clinical-500 dark:text-clinical-400">
                    {file
                      ? `${file.name} · ${formatBytes(file.size)}`
                      : "Resized bilinearly to the size recorded in metadata.json before scaling."}
                  </p>
                </div>
                <label className="btn-secondary cursor-pointer px-2 py-1 text-[11px]">
                  <Upload className="h-3 w-3" />
                  <span>Choose file</span>
                  <input
                    type="file"
                    accept={accepted}
                    className="hidden"
                    onChange={(event) => {
                      setFile(event.target.files?.[0] ?? null);
                      setResult(null);
                      setError(null);
                    }}
                  />
                </label>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="btn-primary"
                onClick={run}
                disabled={!file || running}
              >
                {running ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {running ? "Running the classifier" : "Classify this image"}
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setFile(null);
                  setResult(null);
                  setError(null);
                }}
                disabled={!file && !result}
              >
                Clear
              </button>
            </div>

            {error ? (
              <ErrorState
                title="The classifier refused this request"
                message={`${error.message}${error.code ? ` (${error.code})` : ""}`}
              />
            ) : null}

            {result ? (
              <div className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-3">
                  <MetricCard
                    label="Predicted type"
                    value={result.predicted_label}
                    tone="good"
                    hint={`Softmax value ${formatNumber(result.probability, 4)}`}
                  />
                  <MetricCard
                    label="Output"
                    value={`${result.classes.length} classes`}
                    hint={result.output_transform}
                  />
                  <MetricCard
                    label="Inference"
                    value={`${formatNumber(result.inference_seconds, 3)} s`}
                    hint={`${result.input.original_shape.join(" × ")} → ${result.input.model_input_size.join(" × ")}`}
                  />
                </div>

                <DataTable
                  columns={["Class", "Probability"]}
                  rows={(result.scores ?? []).map((entry) => [
                    entry.label,
                    formatNumber(entry.probability, 6),
                  ])}
                  empty="The model returned no class probabilities."
                />

                <div>
                  <p className="mb-2 text-xs font-medium uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    Class probabilities from the model
                  </p>
                  <FeatureBarChart data={chartData} height={220} />
                </div>

                <div className="flex flex-wrap items-center gap-2 text-xs text-clinical-500 dark:text-clinical-400">
                  <Tooltip text={result.calibration_note} />
                  <span>
                    Not calibrated · class order taken from{" "}
                    <code className="rounded bg-clinical-100 px-1 py-0.5 dark:bg-clinical-800">
                      {result.classes_source}
                    </code>
                  </span>
                </div>

                <InfoNote tone="warn">
                  <strong>Tumour type, not grade and not malignancy.</strong>{" "}
                  {result.label_semantics}
                </InfoNote>

                <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
                  <p className="text-xs font-medium text-clinical-800 dark:text-clinical-100">
                    Metrics reported by the training notebook
                  </p>
                  {result.reported_metrics ? (
                    <div className="mt-2 grid gap-2 sm:grid-cols-4">
                      {Object.entries(result.reported_metrics).map(([key, value]) => (
                        <div key={key} className="text-xs">
                          <div className="text-clinical-500 dark:text-clinical-400">
                            {key.replace(/_/g, " ")}
                          </div>
                          <div className="font-medium tabular-nums text-clinical-800 dark:text-clinical-100">
                            {value <= 1 ? `${formatNumber(value * 100, 2)} %` : formatNumber(value, 4)}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-1 text-xs text-clinical-500 dark:text-clinical-400">
                      metadata.json records no metrics for this checkpoint.
                    </p>
                  )}
                  <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-400">
                    {result.metrics_provenance}
                  </p>
                </div>
              </div>
            ) : null}
          </>
        ) : (
          <EmptyState
            title="Upload is disabled until the two files are in place"
            message={
              <span className="space-y-2">
                <span className="block">
                  This panel is the Xception image classifier only. The radiomics grade classifier
                  above has its own status and does not affect this one.
                </span>
                {summary?.model_path ? (
                  <span className="block">
                    Expected in <code>{summary.model_path.replace(/\/[^/]+$/, "")}</code>, or point{" "}
                    <code>BRAIN_WEIGHTS_DIR</code> at another directory.
                  </span>
                ) : null}
              </span>
            }
            action={
              <button type="button" className="btn-secondary" onClick={checkAgain}>
                Check again
              </button>
            }
          />
        )}
      </CardBody>
    </Card>
  );
}
