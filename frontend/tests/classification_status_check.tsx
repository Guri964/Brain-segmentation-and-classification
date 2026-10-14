/**
 * Classification-page status check, rendered in jsdom against a stubbed API.
 *
 * Why this exists: the reported bug was a *status-display* bug — the page presented the radiomics
 * grade classifier's absence as though the Xception image classifier were switched off too. There
 * is no browser automation in this environment, so the interface is rendered here instead, with
 * every API response replaced by a fixture, and the rendered text is asserted. Three
 * configurations are covered:
 *
 *   A. image classifier ready, radiomics absent   -> image panel usable, upload present
 *   A2. an image is submitted                      -> it reaches POST /api/classification/image
 *                                                     and the label and probabilities are rendered
 *   B. both absent                                 -> two independent warnings, neither conflated
 *   C. radiomics present, image absent             -> image panel still reports its own absence
 *   D. sidebar, overview dashboard and settings    -> each names the two classifiers separately
 *
 * Scope, stated plainly: the API is stubbed because this checkout has no Xception checkpoint, no
 * metadata.json and no TensorFlow installed, so a real prediction cannot be produced here. The
 * stub's probabilities are fixture data used to check that the panel renders a response; they are
 * not a model result and are never reported as one. What this proves is the wiring and the status
 * logic. A real end-to-end prediction has to be run against the deployment that holds the weights.
 *
 * How to run it (needs esbuild, which vite already installs, and jsdom, which is not a dependency
 * of the application — the harness is deliberately not part of the production build):
 *
 *   cd frontend
 *   npm install --no-save jsdom
 *   npx esbuild tests/classification_status_check.tsx --bundle --outfile=tests/.bundle.cjs \
 *       --platform=node --format=cjs --jsx=automatic --external:jsdom --log-level=warning
 *   node tests/.bundle.cjs          # prints PASS/FAIL per check and exits non-zero on failure
 */
import { act } from "react-dom/test-utils";
import { createRoot } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { AppProviders } from "../src/components/AppProviders";
import TumorClassificationPage from "../src/pages/TumorClassificationPage";
import OverviewPage from "../src/pages/OverviewPage";
import SettingsPage from "../src/pages/SettingsPage";
import AppLayout from "../src/layouts/AppLayout";

type Mode = "image_ready" | "both_absent" | "radiomics_only";

const failures: string[] = [];
const passes: string[] = [];

function check(name: string, condition: boolean, detail?: string) {
  if (condition) passes.push(name);
  else failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
}

const PREDICTION = {
  model_id: "brain_tumor_xception",
  display_name: "Brain tumour image classifier (Xception)",
  available: true,
  task: "image_classification",
  filename: "slice.png",
  predicted_index: 0,
  predicted_label: "Glioma",
  probability: 0.8123,
  classes: ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
  probabilities: [0.8123, 0.0931, 0.0611, 0.0335],
  scores: [
    { index: 0, label: "Glioma", probability: 0.8123 },
    { index: 1, label: "Meningioma", probability: 0.0931 },
    { index: 2, label: "Pituitary", probability: 0.0611 },
    { index: 3, label: "No Tumor", probability: 0.0335 },
  ],
  class_to_index: { Glioma: 0, Meningioma: 1, Pituitary: 2, "No Tumor": 3 },
  classes_source: "metadata.json",
  output_transform: "softmax",
  calibrated: false,
  calibration_note: "Softmax outputs, not calibrated probabilities.",
  input: {
    filename: "slice.png",
    original_shape: [256, 256, 3],
    model_input_size: [299, 299, 3],
    resize: "bilinear",
  },
  preprocessing: "x / 127.5 - 1",
  preprocessing_implementation: "keras.applications.xception.preprocess_input",
  inference_seconds: 0.42,
  architecture: "Xception",
  framework_version_recorded: "2.15",
  reported_metrics: { test_accuracy: 0.9091 },
  metrics_provenance: "Reported by the training notebook; not re-measured here.",
  training_config: null,
  label_semantics: "Tumour type in the training dataset's own label space.",
  disclaimer: "Research prototype.",
  upload: { filename: "slice.png", bytes: 1234, decoded_shape: [256, 256, 3] },
};

const IMAGE_READY_SUMMARY = {
  available: true,
  state: "ready",
  model_id: "brain_tumor_xception",
  display_name: "Brain tumour image classifier (Xception)",
  reason: "Checkpoint and metadata are present; TensorFlow loaded the model.",
  classes: ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
  input_size: [299, 299, 3],
  preprocessing: "x / 127.5 - 1",
  reported_metrics: { test_accuracy: 0.9091 },
  metrics_provenance: "Reported by the training notebook; not re-measured here.",
  model_path: "/weights/brain_tumor_xception_final.keras",
  metadata_path: "/weights/metadata.json",
};

const IMAGE_ABSENT_SUMMARY = {
  ...IMAGE_READY_SUMMARY,
  available: false,
  state: "checkpoint_missing",
  reason:
    "brain_tumor_xception_final.keras was not found in the weights directory (or $BRAIN_WEIGHTS_DIR).",
  classes: [],
  input_size: null,
};

const IMAGE_MODEL_ROW = {
  model_id: "brain_tumor_xception",
  display_name: "Brain tumour image classifier (Xception)",
  task: "image_classification",
  framework: "tensorflow",
  state: "ready",
  available: true,
  reason: "Checkpoint and metadata are present.",
  model_file: "brain_tumor_xception_final.keras",
  model_path: "/weights/brain_tumor_xception_final.keras",
  model_file_present: true,
  metadata_file: "metadata.json",
  metadata_present: true,
  framework_available: true,
  framework_version: "2.15.0",
  framework_error: null,
  loaded: true,
  parameter_count: 20861476,
  label_semantics: "Tumour type in the training dataset's own label space.",
  preprocessing: "x / 127.5 - 1",
  preprocessing_implementation: "keras.applications.xception.preprocess_input",
  input_size: [299, 299, 3],
  classes: ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
  class_to_index: { Glioma: 0, Meningioma: 1, Pituitary: 2, "No Tumor": 3 },
  reported_metrics: { test_accuracy: 0.9091 },
  metrics_provenance: "Reported by the training notebook; not re-measured here.",
  training_config: null,
  hardware: null,
  architecture: "Xception",
  framework_version_recorded: "2.15",
  trainer: "firstnb.ipynb",
  searched: ["/weights", "/repo", "/repo/data/models"],
  accepted_formats: [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"],
};

const RADIOMICS_REASON =
  "No validated classifier artefact was found in data/models/classification. Train one with " +
  "scripts/train_classifier.py, which writes the model, the ordered feature schema, the label " +
  "mapping and the held-out metrics into that directory.";

function capabilities(mode: Mode) {
  const radiomicsAvailable = mode === "radiomics_only";
  return {
    models: [],
    model_health: { phantom_version: "v1", generated_at: "2026-10-14T00:00:00Z", models: [], responsive_models: [] },
    ensemble: { available: false, reasons: [], members: [], rule: "" },
    radiomics: { preference: "native", pyradiomics_available: false, selected_backend: "native", note: "" },
    classification: radiomicsAvailable
      ? {
          available: true,
          model_id: "radiomics_grade_logreg",
          label_mapping: { "0": "LGG", "1": "HGG" },
          feature_count: 46,
          metrics: { accuracy: 0.87 },
          required_artifacts: [],
          requirements: [],
        }
      : {
          available: false,
          reason: RADIOMICS_REASON,
          required_artifacts: ["data/models/classification/<name>.joblib"],
          legacy_checkpoint_note: null,
          requirements: ["One labelled radiomics row per case."],
        },
    image_classification: mode === "image_ready" ? IMAGE_READY_SUMMARY : IMAGE_ABSENT_SUMMARY,
    stacked_ensemble: null,
    assistant: { backend: "local", note: "" },
    explainability: { segmentation: [], classification: { method: "", available: false, reason: null }, disclaimer: "" },
    evaluation: "not_evaluated",
    disclaimer: "Research prototype.",
  };
}

function modelCard(mode: Mode) {
  const radiomicsAvailable = mode === "radiomics_only";
  return {
    status: capabilities(mode).classification,
    evaluation: {},
    training: { requirements: ["One labelled radiomics row per case."], recommended: "scripts/train_classifier.py", pipeline: [] },
    xai: { method: "coefficients", shap: { available: false, reason: "ModuleNotFoundError" } },
    label_semantics: { target: "BraTS grade column", explanation: "Dataset label, not a diagnosis." },
    disclaimer: "Research prototype.",
  };
}

function installFetch(mode: Mode) {
  const calls: string[] = [];
  (globalThis as unknown as { fetch: unknown }).fetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url = typeof input === "string" ? input : String(input);
    calls.push(`${init?.method ?? "GET"} ${url}`);
    const json = (body: unknown, status = 200) =>
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      });
    if (url.endsWith("/api/capabilities")) return json(capabilities(mode));
    if (url.endsWith("/api/classification/image/models")) {
      return json({
        models: [mode === "image_ready" ? IMAGE_MODEL_ROW : { ...IMAGE_MODEL_ROW, available: false, state: "checkpoint_missing", model_file_present: false, framework_available: null }],
      });
    }
    if (url.endsWith("/api/classification/image")) {
      const body = init?.body;
      const hasFile = typeof FormData !== "undefined" && body instanceof FormData && body.get("file") !== null;
      if (!hasFile) {
        return json({ error: { message: "Attach one image as the multipart field 'file'.", code: "validation_error" } }, 422);
      }
      return json(PREDICTION);
    }
    if (url.endsWith("/api/classification")) return json(modelCard(mode));
    if (url.endsWith("/api/history/statistics")) {
      return json({
        totals: { analyses: 0, segmentations: 0, radiomics: 0, reports: 0, classifications: 0 },
        recent: [],
        status_counts: {},
        models: { ready: 0, total: 0 },
        evaluation_state: "not_evaluated",
      });
    }
    if (url.endsWith("/api/system")) {
      return json({
        python: "3.11",
        platform: "linux",
        torch_version: "2.4.1",
        cuda_available: false,
        cuda_device_count: 0,
        cuda_device_name: null,
        torch_threads: 2,
        selected_device: "cpu",
        config: { BRAIN_ENV: "testing", BRAIN_WEIGHTS_DIR: "/weights" },
      });
    }
    if (url.endsWith("/api/health")) {
      return json({
        status: "ok",
        database: { schema_version: 2, path: "/stub/app.db", size_bytes: 4096 },
        models: [],
        notes: [],
      });
    }
    return json({ error: { message: `stub: no route for ${url}`, code: "not_found" } }, 404);
  };
  return calls;
}

async function settle(rounds = 12) {
  for (let index = 0; index < rounds; index += 1) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 15));
    });
  }
}

async function render(mode: Mode) {
  const calls = installFetch(mode);
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  await act(async () => {
    root.render(
      <MemoryRouter initialEntries={["/analysis/classification"]}>
        <AppProviders>
          <TumorClassificationPage />
        </AppProviders>
      </MemoryRouter>,
    );
  });
  await settle();
  return { root, host, calls, text: () => host.textContent ?? "" };
}

async function renderSurface(mode: Mode, surface: "sidebar" | "overview" | "settings") {
  const calls = installFetch(mode);
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  const surfaceElement =
    surface === "sidebar" ? (
      <AppLayout />
    ) : surface === "overview" ? (
      <OverviewPage />
    ) : (
      <SettingsPage />
    );
  await act(async () => {
    root.render(
      <MemoryRouter>
        <AppProviders>{surfaceElement}</AppProviders>
      </MemoryRouter>,
    );
  });
  await settle();
  return { root, host, calls, text: () => host.textContent ?? "" };
}

function fileInput(host: HTMLElement): HTMLInputElement | null {
  return host.querySelector('input[type="file"]');
}

function buttonByText(host: HTMLElement, label: string): HTMLButtonElement | null {
  return (
    (Array.from(host.querySelectorAll("button")).find((button) =>
      (button.textContent ?? "").includes(label),
    ) as HTMLButtonElement | undefined) ?? null
  );
}

async function main() {
  const { JSDOM } = await import("jsdom");
  const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "http://localhost:5173/" });
  const g = globalThis as unknown as Record<string, unknown>;
  g.window = dom.window;
  g.document = dom.window.document;
  // Node 22 exposes a read-only ``navigator``; define it instead of assigning.
  Object.defineProperty(g, "navigator", {
    value: dom.window.navigator,
    configurable: true,
    writable: true,
  });
  g.HTMLElement = dom.window.HTMLElement;
  g.HTMLInputElement = dom.window.HTMLInputElement;
  // jsdom implements no fetch API, so ``Response``/``FormData``/``File`` stay Node's own globals
  // (Node 22 has all three); only the DOM comes from jsdom.
  g.localStorage = dom.window.localStorage;
  g.IS_REACT_ACT_ENVIRONMENT = true;
  // Recharts measures its container with a ResizeObserver, which jsdom does not implement; the
  // stub keeps the chart mounted without ever firing a resize.
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  g.ResizeObserver = ResizeObserverStub;
  g.window.ResizeObserver = ResizeObserverStub;
  g.window.matchMedia = g.window.matchMedia ?? (() => ({
    matches: false,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  }));

  // ---- A. the image classifier is ready while the radiomics classifier is absent -------------
  const a = await render("image_ready");
  const textA = a.text();
  check("A: page says the image classifier is ready", textA.includes("Xception image classifier: ready"), textA.slice(0, 400));
  check("A: page says the radiomics classifier is not available", textA.includes("Radiomics grade classifier: not available"));
  check("A: the radiomics warning is scoped to the radiomics model", textA.includes("Radiomics grade classification is switched off"));
  check("A: the radiomics reason is still shown verbatim (not hidden)", textA.includes(RADIOMICS_REASON));
  check("A: the page never says the whole page is 'classification disabled'", !textA.includes("classification disabled"));
  check("A: the image panel is titled as its own pipeline", textA.includes("Xception image classification (four tumour types)"));
  check("A: the upload control is rendered", fileInput(a.host) !== null);
  check("A: the classify button is rendered", buttonByText(a.host, "Classify this image") !== null);
  check("A: the two pipelines are tabulated separately", textA.includes("Radiomics grade classification") && textA.includes("Xception image classification"));
  check("A: capabilities and the image catalogue were requested", a.calls.some((c) => c.includes("/api/classification/image/models")));

  // ---- A2. submitting an image reaches the endpoint and the result is displayed ---------------
  const input = fileInput(a.host);
  if (input) {
    const image = new File([new Uint8Array([137, 80, 78, 71])], "slice.png", { type: "image/png" });
    Object.defineProperty(input, "files", { value: [image], configurable: true });
    await act(async () => {
      input.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
    });
    await settle(4);
    const classify = buttonByText(a.host, "Classify this image");
    check("A2: the classify button is enabled once a file is chosen", Boolean(classify && !classify.disabled));
    if (classify) {
      await act(async () => {
        classify.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
      });
      await settle();
      const textAfter = a.text();
      check("A2: the image was posted to /api/classification/image", a.calls.some((c) => c === "POST /api/classification/image"), a.calls.join(" | "));
      check("A2: the predicted class is displayed", textAfter.includes("Glioma"));
      check("A2: every class probability is displayed", ["0.8123", "0.0931", "0.0611", "0.0335"].every((value) => textAfter.includes(value)) || textAfter.includes("0.812300"));
      check("A2: the probabilities are labelled with their class names", textAfter.includes("Meningioma") && textAfter.includes("No Tumor"));
      check("A2: the reported metrics are labelled as reported, not measured here", textAfter.includes("Metrics reported by the training notebook"));
      check("A2: the label semantics warning is shown", textAfter.includes("Tumour type, not grade and not malignancy."));
    }
  } else {
    check("A2: the classify button is enabled once a file is chosen", false, "no file input");
  }
  await act(async () => a.root.unmount());

  // ---- B. neither classifier is available ----------------------------------------------------
  const b = await render("both_absent");
  const textB = b.text();
  check("B: radiomics status is its own badge", textB.includes("Radiomics grade classifier: not available"));
  check("B: image status is its own badge", textB.includes("Xception image classifier: not installed"));
  check("B: the image panel names the missing checkpoint and where it looked", textB.includes("brain_tumor_xception_final.keras") && textB.includes("/weights"));
  check("B: the image panel refuses to offer an upload", textB.includes("Upload is disabled until the two files are in place"));
  check("B: the image panel does not read the radiomics flag", !textB.includes("Xception image classifier: ready"));
  check("B: the radiomics reason is still visible", textB.includes(RADIOMICS_REASON));
  await act(async () => b.root.unmount());

  // ---- C. the radiomics classifier is present, the image one is not --------------------------
  const c = await render("radiomics_only");
  const textC = c.text();
  check("C: radiomics badge reports the loaded artefact", textC.includes("Radiomics grade classifier: artefact loaded"));
  check("C: image badge still reports its own absence", textC.includes("Xception image classifier: not installed"));
  check("C: the radiomics model card is labelled as pipeline 1", textC.includes("Radiomics grade classifier — model card"));
  check("C: the radiomics 'switched off' warning is not shown when its artefact exists", !textC.includes("Radiomics grade classification is switched off"));
  await act(async () => c.root.unmount());

  // ---- D. the other status surfaces: sidebar, overview, settings -----------------------------
  await act(async () => a.root.unmount());

  const sidebar = await renderSurface("image_ready", "sidebar");
  check("D: sidebar names the grade classifier separately", sidebar.text().includes("Grade classifier (radiomics): disabled"), sidebar.text().slice(0, 300));
  check("D: sidebar names the image classifier separately", sidebar.text().includes("Image classifier (Xception): ready"));
  check("D: sidebar no longer prints one conflated 'Classification: disabled'", !/Classification:\s*disabled/.test(sidebar.text()));
  await act(async () => sidebar.root.unmount());

  const overview = await renderSurface("image_ready", "overview");
  check("D: overview row for the radiomics grade classifier", overview.text().includes("Grade classifier (radiomics)"));
  check("D: overview row for the Xception image classifier", overview.text().includes("Image classifier (Xception)"));
  check("D: overview reports the image classifier as ready with its class count", overview.text().includes("ready — 4 classes"));
  check("D: overview reports the radiomics one as disabled", overview.text().includes("disabled — no validated artefact in data/models/classification"));
  await act(async () => overview.root.unmount());

  const settings = await renderSurface("image_ready", "settings");
  // the capability rows live in the "Feature flags" tab
  const flagsTab = Array.from(settings.host.querySelectorAll("button")).find((button) =>
    (button.textContent ?? "").includes("Feature flags"),
  );
  check("D: the feature-flags tab exists", Boolean(flagsTab));
  if (flagsTab) {
    await act(async () => {
      flagsTab.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
    });
    await settle(4);
  }
  check("D: settings lists the radiomics grade model as its own row", settings.text().includes("Classification — radiomics grade model"));
  check("D: settings lists the Xception image model as its own row", settings.text().includes("Classification — Xception image model"));
  check("D: settings shows the image model as ready", settings.text().includes("ready"));
  await act(async () => settings.root.unmount());

  for (const line of passes) console.log(`PASS  ${line}`);
  for (const line of failures) console.log(`FAIL  ${line}`);
  console.log(`\n${passes.length} passed, ${failures.length} failed`);
  process.exit(failures.length ? 1 : 0);
}

void main();
