import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  HelpCircle,
  Send,
  ShieldAlert,
  Stethoscope,
} from "lucide-react";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  DisclaimerBanner,
  EmptyState,
  InfoNote,
  LoadingBlock,
  MetricCard,
  Pill,
  StatusBadge,
} from "../components/ui";
import { useAnalysis, useCapabilities } from "../hooks/useAnalysis";
import type {
  AnalysisPayload,
  AssistantStatus,
  ChatMessage,
} from "../types/api";
import { api } from "../services/api";
import {
  formatInt,
  formatNumber,
  formatPercent,
  NOT_AVAILABLE,
} from "../services/format";
export default function ClinicalSupportPage() {
  const { result } = useAnalysis();
  const { capabilities } = useCapabilities();
  const [status, setStatus] = useState<AssistantStatus | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    api
      .assistantStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);
  const ask = async (text: string) => {
    if (!text.trim() || busy) return;
    setMessages((current) => [...current, { role: "user", text }]);
    setQuestion("");
    setBusy(true);
    try {
      const answer = await api.ask(result?.analysis_id ?? null, text);
      setMessages((current) => [
        ...current,
        { role: "assistant", text: answer.answer, answer },
      ]);
    } catch (caught) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: `The assistant could not answer: ${caught instanceof Error ? caught.message : String(caught)}`,
        },
      ]);
    } finally {
      setBusy(false);
    }
  };
  const findings = result ? buildFindings(result) : [];
  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
          Clinical decision support
        </h2>
        <p className="mt-1 max-w-3xl text-sm text-clinical-600 dark:text-clinical-300">
          A structured summary of what this analysis measured, together with a
          grounded assistant that answers only from the stored result. Nothing
          here is a diagnosis, a grade or a treatment recommendation, and the
          system refuses to produce one.
        </p>
      </header>
      <DisclaimerBanner
        text={
          result?.disclaimer ??
          "Research prototype. Outputs are AI-generated and are not intended to replace professional medical diagnosis."
        }
      />
      {!result ? (
        <EmptyState
          title="No analysis selected"
          message="The summary and the assistant are grounded in a specific analysis result."
          action={
            <Link to="/analysis/new" className="btn-primary">
              New analysis
            </Link>
          }
          icon={<Stethoscope className="h-6 w-6" />}
        />
      ) : (
        <>
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Region detected"
              value={result.reference_mask.foreground_pixels > 0 ? "yes" : "no"}
              hint={`${formatInt(result.reference_mask.foreground_pixels)} pixels from ${result.reference_mask.source}`}
              tone={
                result.reference_mask.foreground_pixels > 0 ? "good" : "warn"
              }
            />
            <MetricCard
              label="Largest region share"
              value={largestShare(result)}
              tooltip="Share of the image occupied by the largest predicted region. A very small share can mean a small finding or a conservative threshold; the number alone does not distinguish them."
            />
            <MetricCard
              label="Predictive dispersion"
              value={
                result.uncertainty?.mean_dispersion !== undefined &&
                result.uncertainty?.mean_dispersion !== null
                  ? formatNumber(result.uncertainty.mean_dispersion, 4)
                  : NOT_AVAILABLE
              }
              hint={
                result.uncertainty?.method?.replace(/_/g, " ") ??
                "uncertainty estimation was not requested"
              }
            />
            <MetricCard
              label="Report"
              value={result.report?.available ? "generated" : "not generated"}
              hint={
                result.report?.reason ??
                `${formatInt(result.report?.size_bytes ?? 0)} bytes`
              }
              tone={result.report?.available ? "good" : "warn"}
            />
          </section>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader
                title="Structured findings"
                subtitle="Direct restatement of the stored measurements"
                tooltip="Every line maps to a stored field. Lines that could not be measured say so; no clinical interpretation is added."
              />
              <CardBody>
                <ul className="space-y-2 text-sm">
                  {findings.map((finding) => (
                    <li className="flex items-start gap-2" key={finding.label}>
                      {finding.ok ? (
                        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
                      ) : (
                        <HelpCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
                      )}
                      <span>
                        <span className="font-medium text-clinical-800 dark:text-clinical-100">
                          {finding.label}:{" "}
                        </span>
                        <span className="text-clinical-600 dark:text-clinical-300">
                          {finding.value}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              </CardBody>
            </Card>
            <Card>
              <CardHeader
                title="Model agreement"
                subtitle="Similarity between the models, not accuracy"
              />
              <CardBody>
                {Object.keys(result.model_comparison.pairwise_agreement ?? {})
                  .length ? (
                  <DataTable
                    columns={["Pair", "Dice (agreement)", "IoU (agreement)"]}
                    rows={Object.entries(
                      result.model_comparison.pairwise_agreement ?? {},
                    ).map(([pair, value]) => [
                      pair.replace("__", " vs "),
                      formatNumber(value.dice_agreement),
                      formatNumber(value.iou_agreement),
                    ])}
                  />
                ) : (
                  <InfoNote tone="neutral">
                    {result.model_comparison.note}
                  </InfoNote>
                )}
                <div className="mt-4 space-y-2 text-xs">
                  <p className="prose-note">
                    Where the models disagree strongly, the finding is less
                    reproducible and should be treated as tentative. Agreement
                    is still not evidence of correctness: two models trained on
                    the same data can be wrong in the same way.
                  </p>
                  {result.ensemble.available ? (
                    <p className="prose-note">
                      The fused mask uses {result.ensemble.strategy} weights (
                      {result.ensemble.weight_source}) and contains{" "}
                      {formatInt(result.ensemble.foreground_pixels ?? 0)}{" "}
                      pixels.
                    </p>
                  ) : (
                    <p className="prose-note">
                      No ensemble was produced:{" "}
                      {result.ensemble.reasons.join(" ")}
                    </p>
                  )}
                </div>
              </CardBody>
            </Card>
          </div>
          <Card>
            <CardHeader
              title="What this analysis cannot tell you"
              subtitle="Limits that apply to every result of this pipeline"
            />
            <CardBody className="grid gap-3 text-xs sm:grid-cols-2">
              {LIMITATIONS.map((limitation) => (
                <div
                  className="flex items-start gap-2 rounded-lg border border-clinical-200 p-3 dark:border-clinical-800"
                  key={limitation.title}
                >
                  <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                  <div>
                    <p className="font-medium text-clinical-800 dark:text-clinical-100">
                      {limitation.title}
                    </p>
                    <p className="mt-0.5 text-clinical-600 dark:text-clinical-300">
                      {limitation.body}
                    </p>
                  </div>
                </div>
              ))}
            </CardBody>
          </Card>
          <Card>
            <CardHeader
              title="Explanation assistant"
              subtitle={
                status
                  ? `Backend: ${status.backend === "llm" ? `language model (${status.llm_model})` : "local, offline"} · grounded in this analysis`
                  : "checking the backend"
              }
              actions={
                <Pill tone={status?.backend === "llm" ? "violet" : "teal"}>
                  {status?.backend === "llm"
                    ? "LLM grounded"
                    : "local grounded"}
                </Pill>
              }
              tooltip="The assistant answers from the stored result fields. When a language model is configured it receives a reduced, non-identifying summary and is instructed to answer only from it; on any failure the local answer is used instead."
            />
            <CardBody className="space-y-4">
              <div className="max-h-[380px] space-y-3 overflow-y-auto rounded-lg border border-clinical-100 p-3 dark:border-clinical-800">
                {!messages.length ? (
                  <div className="space-y-3">
                    <p className="text-sm text-clinical-500 dark:text-clinical-400">
                      Ask about this analysis — region size, radiomics, model
                      choice, uncertainty or the explanation. The assistant will
                      say when something was not measured.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {(status?.suggested_questions ?? []).map((suggestion) => (
                        <button
                          type="button"
                          className="btn-secondary text-xs"
                          onClick={() => void ask(suggestion)}
                          key={suggestion}
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  messages.map((message, index) => (
                    <div
                      className={
                        message.role === "user"
                          ? "flex justify-end"
                          : "flex justify-start"
                      }
                      key={index}
                    >
                      <div
                        className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                          message.role === "user"
                            ? "bg-clinical-600 text-white"
                            : "border border-clinical-200 bg-clinical-50 text-clinical-800 dark:border-clinical-700 dark:bg-clinical-900 dark:text-clinical-100"
                        }`}
                      >
                        <p className="whitespace-pre-wrap leading-relaxed">
                          {message.text}
                        </p>
                        {message.answer ? (
                          <div className="mt-2 space-y-1 border-t border-black/10 pt-2 text-[11px] text-clinical-500 dark:border-white/10 dark:text-clinical-400">
                            <p>
                              Source:{" "}
                              {message.answer.backend === "llm"
                                ? "language model, grounded in the stored result"
                                : "local, computed from the stored result"}{" "}
                              · topic: {message.answer.topic.replace(/_/g, " ")}
                            </p>
                            {message.answer.sources.length ? (
                              <p>
                                Fields used: {message.answer.sources.join(", ")}
                              </p>
                            ) : (
                              <p>
                                No stored field matched this question, so no
                                measurement was quoted.
                              </p>
                            )}
                            {message.answer.disclaimers.length ? (
                              <p>{message.answer.disclaimers.join(" ")}</p>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ))
                )}
                {busy ? (
                  <LoadingBlock label="Reading the stored result" />
                ) : null}
                <div ref={endRef} />
              </div>
              <form
                className="flex gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  void ask(question);
                }}
              >
                <input
                  className="input"
                  placeholder="Ask about this analysis\u2026"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  aria-label="Question"
                />
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={busy || !question.trim()}
                >
                  <Send className="h-4 w-4" />
                  Ask
                </button>
              </form>
              {status ? (
                <InfoNote tone="neutral">
                  <span className="inline-flex items-start gap-2">
                    <Bot className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    {status.note}
                  </span>
                </InfoNote>
              ) : null}
              <InfoNote tone="warn">
                <span className="inline-flex items-start gap-2">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  Questions about malignancy, tumour grade, staging, prognosis,
                  survival or treatment are refused by design: the models here
                  segment a region and describe its texture, and none of them
                  was trained or validated to answer those questions.
                </span>
              </InfoNote>
            </CardBody>
          </Card>
        </>
      )}
      {capabilities ? (
        <Card>
          <CardHeader
            title="Decision-support checklist"
            subtitle="What a reviewer should confirm before using a result"
          />
          <CardBody>
            <DataTable
              columns={["Item", "State", "Detail"]}
              rows={[
                [
                  "Input matches the model",
                  <StatusBadge
                    value={
                      result
                        ? result.preprocessing.channels === 1
                          ? "single channel"
                          : "four channel"
                        : "unknown"
                    }
                    key={"a"}
                  />,
                  result
                    ? `${result.preprocessing.pipeline} · ${result.preprocessing.normalisation}`
                    : NOT_AVAILABLE,
                ],
                [
                  "Model verified on reference data",
                  <StatusBadge
                    value={
                      result?.models[result.reference_mask.model_id ?? ""]
                        ?.health?.verdict ?? "unknown"
                    }
                    key={"b"}
                  />,
                  result?.models[result.reference_mask.model_id ?? ""]?.health
                    ?.message ?? NOT_AVAILABLE,
                ],
                [
                  "Reference standard available",
                  <StatusBadge
                    value={
                      capabilities.evaluation === "evaluated"
                        ? "evaluated"
                        : "not_evaluated"
                    }
                    key={"c"}
                  />,
                  capabilities.evaluation === "evaluated"
                    ? "Metrics exist for the reserved test split; see Model Evaluation."
                    : "No labelled test set has been evaluated, so no accuracy claim can be made for this deployment.",
                ],
                [
                  "Classification output",
                  <StatusBadge
                    value={
                      result?.classification.available
                        ? "available"
                        : "disabled"
                    }
                    key={"d"}
                  />,
                  result?.classification.reason ??
                    "Enabled only with a validated artefact.",
                ],
                [
                  "Human review recorded",
                  <StatusBadge value="not_recorded" key={"e"} />,
                  "This research build does not implement sign-off workflow.",
                ],
              ]}
            />
          </CardBody>
        </Card>
      ) : null}
    </div>
  );
}
const LIMITATIONS = [
  {
    title: "No diagnosis or grading",
    body: "The pipeline outlines a region and describes its intensity and texture. It does not name a pathology, assign a grade or infer a stage.",
  },
  {
    title: "No prognosis or treatment guidance",
    body: "Survival, progression and therapy decisions are outside the scope of every model shipped here, and outside the scope of the assistant.",
  },
  {
    title: "No accuracy claim without labels",
    body: "Dice, IoU, precision, recall, specificity and sensitivity are only computed against a reference segmentation, which an uploaded scan does not have.",
  },
  {
    title: "Single-image inputs are a proxy",
    body: "A PNG or JPEG export of a slice is not the volumetric acquisition. Slice selection, windowing and resampling done before the export cannot be recovered by this pipeline.",
  },
  {
    title: "Radiomics depend on the mask",
    body: "Texture features describe the predicted region. If the model included peritumoral tissue or missed part of the lesion, the feature values describe that region, not the lesion.",
  },
  {
    title: "Heatmaps are not evidence",
    body: "Grad-CAM explains the network, not the biology. A highlighted area is wherever the model is sensitive, which may be an artefact of the scan.",
  },
];
function largestShare(result: AnalysisPayload) {
  const shares = Object.values(result.models)
    .map((entry) => entry.summary?.foreground_fraction)
    .filter((value): value is number => typeof value === "number");
  if (!shares.length) return NOT_AVAILABLE;
  return formatPercent(Math.max(...shares));
}
function buildFindings(
  result: AnalysisPayload,
): Array<{ label: string; value: string; ok: boolean }> {
  const findings: Array<{ label: string; value: string; ok: boolean }> = [];
  const regionModel = result.reference_mask.model_id;
  findings.push({
    label: "Predicted region",
    value:
      result.reference_mask.foreground_pixels > 0
        ? `${formatInt(result.reference_mask.foreground_pixels)} pixels (${result.reference_mask.source})`
        : "No region exceeded the decision threshold with the models that ran.",
    ok: result.reference_mask.foreground_pixels > 0,
  });
  if (regionModel && result.models[regionModel]?.summary) {
    const summary = result.models[regionModel].summary;
    findings.push({
      label: "Region extent",
      value: `${formatPercent(summary.foreground_fraction)} of the image, mean predicted probability ${formatNumber(summary.probability_mean_in_mask)}`,
      ok: true,
    });
  }
  findings.push({
    label: "Radiomics",
    value: result.radiomics.available
      ? `${result.radiomics.feature_count} features (${result.radiomics.dimensionality}), mask from ${result.radiomics.quality?.mask_source ?? "the predicted region"}`
      : (result.radiomics.unavailable_reason ?? "Not computed."),
    ok: Boolean(result.radiomics.available),
  });
  findings.push({
    label: "Uncertainty",
    value: result.uncertainty?.available
      ? `${String(result.uncertainty.method).replace(/_/g, " ")} over ${result.uncertainty.n_passes} passes, mean dispersion ${formatNumber(result.uncertainty.mean_dispersion ?? null)}`
      : (result.uncertainty?.unavailable_reason ??
        "Not requested for this analysis."),
    ok: Boolean(result.uncertainty?.available),
  });
  findings.push({
    label: "Classifier output",
    value: result.classification.available
      ? `${result.classification.predicted_label ?? "label"} (${(result.classification.scores ?? []).map((score: number) => formatNumber(score)).join(", ")})`
      : result.classification.reason,
    ok: Boolean(result.classification.available),
  });
  findings.push({
    label: "Reference comparison",
    value:
      result.model_comparison.state === "unlabelled"
        ? "No reference mask for this scan — only inter-model agreement is available."
        : `State: ${result.model_comparison.state}.`,
    ok: result.model_comparison.state !== "unlabelled",
  });
  return findings;
}
