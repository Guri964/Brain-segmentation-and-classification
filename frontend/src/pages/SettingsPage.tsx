import { useEffect, useState } from "react";
import {
  Cpu,
  Database,
  HardDrive,
  RefreshCw,
  ShieldCheck,
  Sliders,
} from "lucide-react";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  InfoNote,
  LoadingBlock,
  MetricCard,
  Pill,
  StatusBadge,
  Tabs,
  Tooltip,
} from "../components/ui";
import { useToast } from "../components/AppProviders";
import { useCapabilities } from "../hooks/useAnalysis";
import { api } from "../services/api";
import {
  formatBytes,
  formatDate,
  formatDuration,
  NOT_AVAILABLE,
} from "../services/format";
import type { HealthStatus, SystemInfo } from "../types/api";
export default function SettingsPage() {
  const { capabilities, reload } = useCapabilities();
  const { push } = useToast();
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [tab, setTab] = useState("deployment");
  const [busy, setBusy] = useState(false);
  const load = () => {
    api
      .system()
      .then(setSystem)
      .catch(() => setSystem(null));
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth(null));
  };
  useEffect(load, []);
  const runChecks = async () => {
    setBusy(true);
    try {
      const outcome = await api.runHealthChecks(true);
      push(
        "success",
        "Reference checks finished",
        outcome.responsive_models.length
          ? `Responsive: ${outcome.responsive_models.join(", ")}`
          : "No model passed the reference check.",
      );
      reload();
    } catch (caught) {
      push(
        "error",
        "Reference checks failed",
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
            Settings and deployment
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-clinical-600 dark:text-clinical-300">
            Read-only view of how this deployment is configured, which artefacts
            it found and what each subsystem needs in order to become active.
          </p>
        </div>
        <div className="flex gap-2">
          <button type="button" className="btn-secondary" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={() => void runChecks()}
            disabled={busy}
          >
            <ShieldCheck
              className={`h-3.5 w-3.5 ${busy ? "animate-pulse" : ""}`}
            />
            Re-run reference checks
          </button>
        </div>
      </header>
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Compute device"
          value={system?.selected_device ?? NOT_AVAILABLE}
          hint={
            system
              ? `torch ${system.torch_version} · ${system.torch_threads} threads`
              : undefined
          }
          tooltip="Selected automatically: CUDA when a working GPU is present, otherwise CPU with a capped thread count."
        />
        <MetricCard
          label="CUDA available"
          value={
            system ? (system.cuda_available ? "yes" : "no") : NOT_AVAILABLE
          }
          hint={system?.cuda_device_name ?? "running on CPU"}
          tone={system?.cuda_available ? "good" : "warn"}
        />
        <MetricCard
          label="Radiomics backend"
          value={capabilities?.radiomics.selected_backend ?? NOT_AVAILABLE}
          hint={
            capabilities?.radiomics.pyradiomics_available
              ? "PyRadiomics importable"
              : "native extractor in use"
          }
        />
        <MetricCard
          label="Database"
          value={
            health ? `schema v${health.database.schema_version}` : NOT_AVAILABLE
          }
          hint={health?.database.path ?? "path unknown"}
        />
      </section>
      <Card>
        <CardHeader
          title="Configuration"
          subtitle="Values are read from the environment; nothing is editable from the browser"
        />
        <CardBody>
          <Tabs
            tabs={[
              { id: "deployment", label: "Deployment" },
              { id: "models", label: "Models" },
              { id: "data", label: "Data & storage" },
              { id: "features", label: "Feature flags" },
            ]}
            active={tab}
            onChange={(id) => setTab(id)}
          />
          <div className="mt-4">
            {tab === "deployment" ? (
              system ? (
                <DataTable
                  columns={["Setting", "Value"]}
                  rows={Object.entries(system.config).map(([key, value]) => [
                    <span className="font-mono text-xs" key={"k"}>
                      {key}
                    </span>,
                    <span className="break-all text-xs" key={"v"}>
                      {typeof value === "object"
                        ? JSON.stringify(value)
                        : String(value)}
                    </span>,
                  ])}
                />
              ) : (
                <LoadingBlock label="Reading the runtime" />
              )
            ) : null}
            {tab === "models" ? (
              capabilities ? (
                <div className="space-y-4">
                  <DataTable
                    columns={[
                      "Model",
                      "Task",
                      "Architecture",
                      "Input",
                      "Threshold",
                      "Output",
                      "File size",
                      "Loaded in",
                      "Status",
                    ]}
                    rows={capabilities.models.map((model) => [
                      <div key={"m"}>
                        <p className="font-medium text-clinical-900 dark:text-clinical-50">
                          {model.model_id}
                        </p>
                        <p className="font-mono text-[10px] text-clinical-500 dark:text-clinical-400">
                          {model.path}
                        </p>
                      </div>,
                      model.task,
                      <span className="text-xs" key={"a"}>
                        {model.architecture}
                      </span>,
                      <span className="text-xs" key={"i"}>
                        {model.input_spec.channels}×{model.input_spec.height}×
                        {model.input_spec.width}
                        <br />
                        <span className="text-clinical-500 dark:text-clinical-400">
                          {model.input_spec.normalisation}
                        </span>
                      </span>,
                      model.fixed_threshold !== undefined &&
                      model.fixed_threshold !== null
                        ? String(model.fixed_threshold)
                        : NOT_AVAILABLE,
                      <span className="text-xs" key={"o"}>
                        {model.output_semantics}
                      </span>,
                      model.file_size_bytes !== null
                        ? formatBytes(model.file_size_bytes)
                        : NOT_AVAILABLE,
                      model.load_seconds !== null
                        ? formatDuration(model.load_seconds)
                        : NOT_AVAILABLE,
                      <StatusBadge value={model.status} key={"s"} />,
                    ])}
                  />
                  <InfoNote tone="neutral">
                    <span className="inline-flex items-start gap-2">
                      <Cpu className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      Checkpoints are cached after the first load, so repeated
                      analyses do not pay the load cost again. A model that
                      fails to load is reported with its error rather than being
                      silently skipped.
                    </span>
                  </InfoNote>
                </div>
              ) : (
                <LoadingBlock />
              )
            ) : null}
            {tab === "data" ? (
              <div className="grid gap-4 lg:grid-cols-2">
                <div>
                  <h4 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    <Database className="h-3.5 w-3.5" /> Database
                  </h4>
                  {health ? (
                    <DataTable
                      columns={["Field", "Value"]}
                      rows={[
                        [
                          "Path",
                          <span
                            className="break-all font-mono text-xs"
                            key={"p"}
                          >
                            {health.database.path}
                          </span>,
                        ],
                        [
                          "Schema version",
                          String(health.database.schema_version),
                        ],
                        ["Environment", health.environment],
                        ["Checked at", formatDate(health.time)],
                      ]}
                    />
                  ) : (
                    <LoadingBlock />
                  )}
                </div>
                <div>
                  <h4 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    <HardDrive className="h-3.5 w-3.5" /> Storage layout
                  </h4>
                  <DataTable
                    columns={["Directory", "Contents"]}
                    rows={[
                      [
                        "data/uploads/<analysis_id>/",
                        "The exact bytes that were uploaded, with their original names and sizes",
                      ],
                      [
                        "data/artifacts/<analysis_id>/",
                        "Probability maps, masks, overlays, saliency maps, radiomics tables, manifest and analysis.json",
                      ],
                      [
                        "data/reports/<analysis_id>/",
                        "The generated PDF report",
                      ],
                      [
                        "data/models/",
                        "Checkpoints, classification artefacts and evaluation results",
                      ],
                      [
                        "data/analysis.db",
                        "SQLite database with records, features and report metadata",
                      ],
                    ]}
                  />
                </div>
              </div>
            ) : null}
            {tab === "features" ? (
              capabilities ? (
                <div className="space-y-4">
                  <DataTable
                    columns={["Capability", "State", "Why"]}
                    rows={[
                      [
                        "Segmentation",
                        <Pill
                          tone={
                            capabilities.model_health.responsive_models.length
                              ? "teal"
                              : "amber"
                          }
                          key={"a"}
                        >
                          {capabilities.model_health.responsive_models.length}{" "}
                          responsive model(s)
                        </Pill>,
                        capabilities.models
                          .filter((model) => !model.available)
                          .map((model) => `${model.model_id}: ${model.message}`)
                          .join(" ") || "All registered checkpoints loaded.",
                      ],
                      [
                        "Probability ensemble",
                        <Pill
                          tone={
                            capabilities.ensemble.available ? "teal" : "amber"
                          }
                          key={"b"}
                        >
                          {capabilities.ensemble.available
                            ? "available"
                            : "unavailable"}
                        </Pill>,
                        capabilities.ensemble.reasons.join(" ") ||
                          "Two or more models share an identical input specification.",
                      ],
                      [
                        "Radiomics",
                        <Pill tone="teal" key={"c"}>
                          {capabilities.radiomics.selected_backend}
                        </Pill>,
                        capabilities.radiomics.note,
                      ],
                      // Two independent classifiers: the radiomics grade classifier and the
                      // Xception image classifier each get their own row and their own reason, so
                      // neither status is presented as the status of both.
                      [
                        "Classification — radiomics grade model",
                        <Pill
                          tone={
                            capabilities.classification.available
                              ? "teal"
                              : "amber"
                          }
                          key={"d"}
                        >
                          {capabilities.classification.available
                            ? "enabled"
                            : "disabled"}
                        </Pill>,
                        capabilities.classification.reason ??
                          "A validated artefact with a recorded label mapping is present.",
                      ],
                      [
                        "Classification — Xception image model",
                        <Pill
                          tone={
                            capabilities.image_classification.available
                              ? "teal"
                              : "amber"
                          }
                          key={"d2"}
                        >
                          {capabilities.image_classification.available
                            ? "ready"
                            : (capabilities.image_classification.state ?? "unknown").replace(
                                /_/g,
                                " ",
                              )}
                        </Pill>,
                        capabilities.image_classification.available
                          ? `${capabilities.image_classification.classes?.length ?? 0} classes read from metadata.json; predictions are never combined with the grade model.`
                          : capabilities.image_classification.reason,
                      ],
                      [
                        "Grad-CAM",
                        <Pill
                          tone={
                            capabilities.explainability.segmentation.some(
                              (entry) => entry.available,
                            )
                              ? "teal"
                              : "amber"
                          }
                          key={"e"}
                        >
                          {
                            capabilities.explainability.segmentation.filter(
                              (entry) => entry.available,
                            ).length
                          }{" "}
                          model(s)
                        </Pill>,
                        capabilities.explainability.segmentation
                          .map(
                            (entry) =>
                              `${entry.model_id}: ${entry.available ? entry.target_layer : entry.reason}`,
                          )
                          .join(" · "),
                      ],
                      [
                        "Assistant",
                        <Pill tone="violet" key={"f"}>
                          {capabilities.assistant.backend}
                        </Pill>,
                        capabilities.assistant.note,
                      ],
                      [
                        "Evaluation",
                        <Pill
                          tone={
                            capabilities.evaluation === "evaluated"
                              ? "teal"
                              : "amber"
                          }
                          key={"g"}
                        >
                          {capabilities.evaluation.replace(/_/g, " ")}
                        </Pill>,
                        capabilities.evaluation === "evaluated"
                          ? "A test-split evaluation artefact exists."
                          : "Run scripts/evaluate_models.py on a labelled split to produce metrics.",
                      ],
                    ]}
                  />
                  {capabilities.classification.requirements?.length ? (
                    <InfoNote tone="warn">
                      <p className="font-medium">
                        Radiomics grade classifier requirements (the Xception image classifier is
                        separate and unaffected)
                      </p>
                      <ul className="mt-1 list-disc space-y-1 pl-4">
                        {capabilities.classification.requirements.map(
                          (requirement) => (
                            <li key={requirement}>{requirement}</li>
                          ),
                        )}
                      </ul>
                      {capabilities.classification.legacy_checkpoint_note ? (
                        <p className="mt-2">
                          {capabilities.classification.legacy_checkpoint_note}
                        </p>
                      ) : null}
                    </InfoNote>
                  ) : null}
                  <InfoNote tone="neutral">
                    <span className="inline-flex items-start gap-2">
                      <Sliders className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      Every flag above is derived from what is actually on disk.
                      A capability is never switched on to make the interface
                      look complete.
                    </span>
                  </InfoNote>
                </div>
              ) : (
                <LoadingBlock />
              )
            ) : null}
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardHeader
          title="Commands"
          subtitle="How to reproduce, retrain and evaluate outside the browser"
        />
        <CardBody>
          <DataTable
            columns={["Task", "Command"]}
            rows={[
              [
                "Start the API",
                <code className="text-xs" key={"1"}>
                  python backend/run.py
                </code>,
              ],
              [
                "Start the interface (development)",
                <code className="text-xs" key={"2"}>
                  cd frontend &amp;&amp; npm run dev
                </code>,
              ],
              [
                "Build the interface (production)",
                <code className="text-xs" key={"3"}>
                  cd frontend &amp;&amp; npm run build
                </code>,
              ],
              [
                "Reference checks only",
                <code className="text-xs" key={"4"}>
                  python scripts/check_models.py
                </code>,
              ],
              [
                "Train SegNet (four channel)",
                <code className="text-xs" key={"5"}>
                  python scripts/train_segnet.py --data &lt;manifest&gt;
                  --epochs 60
                </code>,
              ],
              [
                "Train the single-channel U-Net",
                <code className="text-xs" key={"6"}>
                  python scripts/train_unet.py --data &lt;manifest&gt; --epochs
                  60
                </code>,
              ],
              [
                "Evaluate the reserved test split",
                <code className="text-xs" key={"7"}>
                  python scripts/evaluate_models.py --data &lt;manifest&gt;
                  --split test
                </code>,
              ],
              [
                "Run the ablations",
                <code className="text-xs" key={"8"}>
                  python scripts/run_ablations.py --data &lt;manifest&gt;
                </code>,
              ],
              [
                "Calibrate the ensemble",
                <code className="text-xs" key={"9"}>
                  python scripts/calibrate_ensemble.py --data &lt;manifest&gt;
                  --split val
                </code>,
              ],
              [
                "Train the radiomics classifier",
                <code className="text-xs" key={"10"}>
                  python scripts/train_classifier.py --features
                  &lt;radiomics.csv&gt;
                </code>,
              ],
            ]}
          />
          <p className="prose-note mt-3">
            These commands are the only supported path to new metrics. The
            evaluation endpoint rejects calibration files that were produced
            from the test split, so weights tuned to the test set cannot be used
            to inflate a reported score.
          </p>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Privacy and safety" />
        <CardBody className="grid gap-3 text-xs sm:grid-cols-2">
          <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
            <p className="font-medium text-clinical-800 dark:text-clinical-100">
              Pseudonymous by design
            </p>
            <p className="mt-1 text-clinical-600 dark:text-clinical-300">
              The schema has no patient name, record number or date-of-birth
              column. Analyses are addressed by a generated identifier, and the
              optional study code is whatever pseudonym the operator chooses to
              type.
            </p>
          </div>
          <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
            <p className="font-medium text-clinical-800 dark:text-clinical-100">
              Local by default
            </p>
            <p className="mt-1 text-clinical-600 dark:text-clinical-300">
              The assistant answers from stored fields with no network call. If
              a language-model backend is enabled, its key is read from an
              environment variable, is never stored or logged, and only a
              reduced, non-identifying summary of the analysis is sent.
            </p>
          </div>
          <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
            <p className="font-medium text-clinical-800 dark:text-clinical-100">
              Uploads are validated
            </p>
            <p className="mt-1 text-clinical-600 dark:text-clinical-300">
              Extensions, decoded content, byte size, per-analysis file count
              and duplicate channel content are all checked before a model runs.
              A rejected file never reaches a model and never creates an
              analysis record.
            </p>
          </div>
          <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
            <p className="font-medium text-clinical-800 dark:text-clinical-100">
              Refusals are explicit
            </p>
            <p className="mt-1 text-clinical-600 dark:text-clinical-300">
              The assistant pattern-matches questions about malignancy, grade,
              staging, prognosis, survival and treatment before any backend is
              invoked, and declines them with the reason instead of producing
              prose.
            </p>
          </div>
        </CardBody>
      </Card>
      <InfoNote tone="neutral">
        <span className="inline-flex items-start gap-2">
          <Tooltip text="Changing a checkpoint or a configuration value requires restarting the API process; the interface reads the new state on the next request.">
            <span className="font-medium">Model cache</span>
          </Tooltip>
          <span>
            Loaded models stay resident for the life of the process. After
            replacing a checkpoint, restart the API, then re-run the reference
            checks so the new file is verified before it is used. Stored
            analyses keep the model identifiers they were produced with, so
            history stays reproducible.
          </span>
        </span>
      </InfoNote>
    </div>
  );
}
