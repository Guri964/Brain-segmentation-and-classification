import { useState } from "react";
import { Loader2, Play, Upload } from "lucide-react";
import {
  Card,
  CardBody,
  CardHeader,
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
import type { ImageClassificationPrediction } from "../types/api";

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
 * This is a different task from the BraTS HGG/LGG grade classifier on the rest of this page:
 * different dataset, different label space, different input (the image itself, 299x299x3).
 * The panel therefore shows only what the checkpoint and its metadata can support: when either
 * file is missing it says so instead of offering a button that cannot produce a number, and the
 * metrics it displays are labelled as reported by the training notebook, not as verified here.
 */
export default function ImageClassifierPanel() {
  const { capabilities, reload } = useCapabilities();
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ImageClassificationPrediction | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const summary = capabilities?.image_classification;

  const accepted = ".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp";

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

  if (!capabilities) return <LoadingBlock label="Checking the image classifier" />;

  const available = Boolean(summary?.available);
  const chartData = (result?.scores ?? []).map((entry) => ({
    label: entry.label,
    value: entry.probability,
  }));

  return (
    <Card>
      <CardHeader
        title="Tumour type from the image (four classes)"
        subtitle="Keras Xception trained in firstnb.ipynb. Separate task, dataset and label space from the grade model above."
      />
      <CardBody className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone={available ? "teal" : "amber"}>
            {STATE_LABEL[summary?.state ?? ""] ?? summary?.state ?? "unknown"}
          </Pill>
          <span className="text-xs text-clinical-500 dark:text-clinical-400">
            {summary?.display_name ?? NOT_AVAILABLE}
          </span>
        </div>

        {available ? (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCard
                label="Input"
                value={summary?.input_size?.join(" × ") ?? NOT_AVAILABLE}
                hint="RGB, as recorded in metadata.json"
              />
              <MetricCard
                label="Classes"
                value={summary?.classes?.length ?? NOT_AVAILABLE}
                hint={summary?.classes?.join(", ") ?? ""}
              />
              <MetricCard
                label="Preprocessing"
                value="Xception"
                hint="x / 127.5 - 1, identical to keras.applications.xception.preprocess_input"
              />
            </div>

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
            title="The image classifier is not installed"
            message={
              <span className="space-y-2">
                <span className="block">{summary?.reason}</span>
                <span className="block">
                  Expected files: <code>brain_tumor_xception_final.keras</code> and{" "}
                  <code>metadata.json</code> in the weights directory
                  {summary?.model_path ? (
                    <>
                      {" "}
                      (<code>{summary.model_path.replace(/\/[^/]+$/, "")}</code>)
                    </>
                  ) : null}
                  , or point <code>BRAIN_WEIGHTS_DIR</code> at another directory. Weights are
                  never downloaded automatically and nothing is predicted until they are there.
                </span>
              </span>
            }
            action={
              <button type="button" className="btn-secondary" onClick={reload}>
                Check again
              </button>
            }
          />
        )}
      </CardBody>
    </Card>
  );
}
