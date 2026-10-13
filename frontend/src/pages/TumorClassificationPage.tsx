import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  BarChart3,
  Microscope,
  ShieldAlert,
  Sparkles,
  Upload,
} from "lucide-react";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  DisclaimerBanner,
  EmptyState,
  ErrorState,
  InfoNote,
  LoadingBlock,
  MetricCard,
  Pill,
  StatusBadge,
} from "../components/ui";
import { FeatureBarChart } from "../components/charts";
import { useAnalysis, useCapabilities } from "../hooks/useAnalysis";
import { api } from "../services/api";
import {
  formatInt,
  formatMetric,
  formatNumber,
  formatPercent,
  humaniseFeature,
  NOT_AVAILABLE,
  NOT_EVALUATED,
  truncate,
} from "../services/format";
import type { ClassificationModelCard } from "../types/api";

export default function TumorClassificationPage() {
  const { result, analysisId } = useAnalysis();
  const { capabilities } = useCapabilities();
  const [card, setCard] = useState<ClassificationModelCard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .classificationModelCard()
      .then(setCard)
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : String(caught)),
      );
  }, []);

  const classification = result?.classification;
  const prediction = classification?.available ? classification : null;
  const status = card?.status ?? capabilities?.classification ?? null;
  const metrics = status?.metrics ?? {};
  const available = Boolean(classification?.available);

  const probabilityRows = useMemo(() => {
    if (!classification?.scores?.length) return [];
    return classification.scores.map((score, index) => ({
      label:
        classification.label_mapping?.[String(index)] ??
        status?.label_mapping?.[String(index)] ??
        `class ${index}`,
      value: Number(score.toFixed(6)),
    }));
  }, [classification, status]);

  const contributionRows = useMemo(() => {
    const contributions = classification?.feature_contributions ?? [];
    return contributions
      .slice(0, 20)
      .map((item) => ({
        label: humaniseFeature(item.feature),
        value: Number(Math.abs(item.contribution).toFixed(6)),
        signed: Number(item.contribution.toFixed(6)),
        method: item.method ?? "",
      }));
  }, [classification]);

  const radiomicsRows = useMemo(() => {
    const rows = result?.radiomics?.available ? result.radiomics : null;
    if (!rows || !status?.feature_schema?.length) return [];
    return status.feature_schema
      .map((name) => ({
        feature: name,
        value: rows.features?.[name] ?? null,
      }))
      .filter((row) => row.value !== null && Number.isFinite(row.value));
  }, [result, status]);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-clinical-900 dark:text-clinical-50">
            Tumour classification
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-clinical-600 dark:text-clinical-300">
            A class label predicted from the radiomics features of the region the
            segmentation found. The label is the dataset's own annotation — for
            the BraTS grade column that is the WHO grade group (HGG or LGG). It
            is not a malignancy verdict, not a clinical stage and not a
            prognosis, and the class never comes from the mask itself.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge
            value={available ? "completed" : "not_loaded"}
            label={available ? "prediction available" : "classification disabled"}
          />
          <Link to="/analysis/new" className="btn-secondary">
            <Upload className="h-4 w-4" />
            New analysis
          </Link>
        </div>
      </header>

      {error ? <ErrorState message={error} title="Could not read the model card" /> : null}

      <Card>
        <CardHeader
          title="Model card"
          subtitle="What the served classifier is, what it was scored on, and what it must not be used for"
          actions={
            <Pill tone={status?.available ? "teal" : "amber"}>
              {status?.available ? "artefact loaded" : "no artefact"}
            </Pill>
          }
        />
        <CardBody className="space-y-4">
          {!card && !error ? <LoadingBlock label="Reading the classifier status" /> : null}
          {card ? (
            <>
              <DataTable
                columns={["Field", "Value"]}
                rows={[
                  ["Model", status?.model_id ?? NOT_AVAILABLE],
                  ["Version", status?.model_version ?? NOT_AVAILABLE],
                  ["Trained", status?.trained_at ?? NOT_AVAILABLE],
                  [
                    "Target",
                    status?.target_description ??
                      card.label_semantics.target ??
                      NOT_AVAILABLE,
                  ],
                  ["Classes", Object.values(status?.label_mapping ?? {}).join(", ") || NOT_AVAILABLE],
                  ["Features", status?.feature_count ? formatInt(status.feature_count) : NOT_AVAILABLE],
                  [
                    "Preprocessing",
                    status?.preprocessing?.type
                      ? `${status.preprocessing.type}, ${status.preprocessing.fit_on ?? "fit on training data"}`
                      : NOT_AVAILABLE,
                  ],
                  ["Selected by", status?.selected_model ?? NOT_AVAILABLE],
                ]}
              />
              {!status?.available ? (
                <InfoNote tone="warn">
                  <p className="font-semibold">Classification is switched off</p>
                  <p className="mt-1">{status?.reason}</p>
                  {card.training.requirements?.length ? (
                    <ul className="mt-2 list-disc space-y-1 pl-4">
                      {card.training.requirements.slice(0, 5).map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  ) : null}
                  <p className="mt-2">
                    Producing an artefact: {card.training.recommended}
                  </p>
                </InfoNote>
              ) : null}
              {status?.legacy_checkpoint_note ? (
                <InfoNote tone="neutral">{status.legacy_checkpoint_note}</InfoNote>
              ) : null}
            </>
          ) : null}
        </CardBody>
      </Card>

      {!analysisId ? (
        <EmptyState
          title="No analysis selected"
          message="Run an analysis to see the predicted class, the class probabilities, the features that drove the decision and the recorded held-out metrics."
          icon={<Microscope className="h-6 w-6" />}
          action={
            <Link to="/analysis/new" className="btn-primary">
              New MRI analysis
            </Link>
          }
        />
      ) : null}

      {analysisId ? (
        <Card>
          <CardHeader
            title="Prediction for the current analysis"
            subtitle={result?.scan_identifier ? `Scan ${result.scan_identifier}` : undefined}
          />
          <CardBody className="space-y-4">
            {!classification ? (
              <InfoNote tone="neutral">No classification record is attached to this analysis.</InfoNote>
            ) : !classification.available ? (
              <InfoNote tone="warn">
                <p className="font-semibold">No prediction was produced for this analysis</p>
                <p className="mt-1">{classification.reason}</p>
                {classification.missing_features?.length ? (
                  <p className="mt-1">
                    Missing features: {classification.missing_features.slice(0, 8).join(", ")}
                    {classification.missing_features.length > 8 ? "…" : ""}
                  </p>
                ) : null}
                <p className="mt-2">
                  The platform does not substitute, reorder or pad feature values to force a
                  prediction. See the model card above for what the classifier expects.
                </p>
              </InfoNote>
            ) : (
              <>
                <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  <MetricCard
                    label="Predicted class"
                    value={classification.predicted_label ?? NOT_AVAILABLE}
                    tooltip="The class with the highest predicted probability. It is the dataset label, not a diagnosis."
                  />
                  <MetricCard
                    label="Probability"
                    value={
                      classification.scores?.[classification.predicted_index ?? 0] !== undefined
                        ? formatPercent(
                            classification.scores[classification.predicted_index ?? 0],
                          )
                        : NOT_AVAILABLE
                    }
                    hint="Model output for the selected class, not a clinical confidence"
                  />
                  <MetricCard
                    label="Classes"
                    value={Object.values(classification.label_mapping ?? {}).length || NOT_AVAILABLE}
                    hint={Object.values(classification.label_mapping ?? {}).join(", ")}
                  />
                  <MetricCard
                    label="Calibrated"
                    value={classification.calibrated ? "yes" : "no"}
                    hint={
                      classification.calibrated
                        ? classification.calibration_method ?? ""
                        : "Probabilities are raw model outputs"
                    }
                    tone={classification.calibrated ? "good" : "warn"}
                  />
                </section>

                <DataTable
                  columns={["Class probabilities", "Value"]}
                  rows={probabilityRows.map((row) => [
                    row.label,
                    formatNumber(row.value, 6),
                  ])}
                  empty="The model returned no class scores."
                />

                {probabilityRows.length ? (
                  <div>
                    <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                      Probability distribution
                    </h4>
                    <FeatureBarChart
                      data={probabilityRows.map((row) => ({
                        label: row.label,
                        value: row.value,
                      }))}
                      color="#8b5cf6"
                      height={220}
                    />
                  </div>
                ) : null}

                <InfoNote tone="warn">
                  <span className="inline-flex items-start gap-2">
                    <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    {classification.statement}
                  </span>
                </InfoNote>
                {classification.evaluated_on ? (
                  <p className="text-xs text-clinical-500 dark:text-clinical-400">
                    Metrics recorded for this artefact were measured on:{" "}
                    {classification.evaluated_on}
                  </p>
                ) : null}
              </>
            )}
          </CardBody>
        </Card>
      ) : null}

      {prediction ? (
        <Card>
          <CardHeader
            title="What drove the decision"
            subtitle="Feature attribution computed from the served model itself"
            actions={<Pill tone="violet"><Sparkles className="h-3 w-3" /> XAI</Pill>}
          />
          <CardBody className="space-y-4">
            {contributionRows.length ? (
              <>
                <FeatureBarChart
                  data={contributionRows.map((row) => ({
                    label: truncate(row.label, 34),
                    value: row.value,
                  }))}
                  color="#0ea5e9"
                  height={300}
                />
                <DataTable
                  dense
                  columns={["Feature", "Standardised value", "Contribution", "Direction", "Method"]}
                  rows={(prediction.feature_contributions ?? []).slice(0, 20).map((item) => [
                    humaniseFeature(item.feature),
                    item.value !== undefined ? formatNumber(item.value, 4) : NOT_AVAILABLE,
                    formatNumber(item.contribution, 6),
                    item.contribution >= 0 ? "towards the class" : "away from the class",
                    item.method ?? "",
                  ])}
                />
              </>
            ) : (
              <InfoNote tone="neutral">
                The artefact does not carry coefficients or training quantiles, so no feature
                attribution could be computed for this analysis.
              </InfoNote>
            )}
            <InfoNote tone="neutral">
              {card?.xai.method ?? "Attribution method recorded in the artefact."}{" "}
              {card?.xai.shap?.available === false
                ? `SHAP is not installed in this deployment (${card.xai.shap.reason}); the recorded fallback is used instead.`
                : null}
            </InfoNote>
            {card?.label_semantics ? (
              <InfoNote tone="warn">{card.label_semantics.explanation}</InfoNote>
            ) : null}
          </CardBody>
        </Card>
      ) : null}

      {radiomicsRows.length ? (
        <Card>
          <CardHeader
            title="Radiomics features used as input"
            subtitle="Values taken from this analysis and matched against the classifier's own schema, in the order the model expects"
          />
          <CardBody>
            <DataTable
              dense
              columns={["Feature", "Value"]}
              rows={radiomicsRows.slice(0, 40).map((row) => [
                humaniseFeature(row.feature),
                formatNumber(row.value, 6),
              ])}
              empty="No feature of the classifier schema is available for this analysis."
            />
            <p className="mt-2 text-xs text-clinical-500 dark:text-clinical-400">
              {radiomicsRows.length > 40
                ? `Showing 40 of ${radiomicsRows.length} features. The full table is on the Quantitative Radiomics page and in the CSV export.`
                : "Physical units are reported only when the input image carried pixel spacing."}
            </p>
          </CardBody>
        </Card>
      ) : null}

      <Card>
        <CardHeader
          title="Recorded evaluation"
          subtitle="Held-out test metrics stored with the artefact that produced the predictions above"
          actions={<Pill tone={metrics.accuracy ? "teal" : "amber"}><BarChart3 className="h-3 w-3" />{metrics.accuracy ? "evaluated" : "not evaluated"}</Pill>}
        />
        <CardBody className="space-y-4">
          {!metrics.accuracy ? (
            <InfoNote tone="warn">
              <p className="font-semibold">No held-out metrics are recorded for this deployment</p>
              <p className="mt-1">
                {status?.available
                  ? "The artefact exists but carries no test metrics; treat any prediction as unvalidated."
                  : "Train a classifier with scripts/train_classification.py, which records accuracy, per-class precision/recall/F1, macro-F1, the confusion matrix and ROC-AUC for the reserved test split."}
              </p>
            </InfoNote>
          ) : (
            <>
              <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Accuracy"
                  value={formatMetric(metrics.accuracy)}
                  hint={metrics.evaluation_split ?? "held-out test split"}
                />
                <MetricCard
                  label="Macro F1"
                  value={formatMetric(metrics.macro_f1)}
                  tooltip="Unweighted mean of the per-class F1 scores, so a rare class counts as much as a common one."
                />
                <MetricCard
                  label="ROC-AUC"
                  value={formatMetric(metrics.roc_auc)}
                  hint={metrics.roc_auc_average ?? undefined}
                />
                <MetricCard
                  label="Test samples"
                  value={formatInt(metrics.samples ?? null)}
                  hint={`Selected model: ${metrics.evaluated_model ?? NOT_AVAILABLE}`}
                />
              </section>

              {metrics.per_class ? (
                <div>
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    Per-class metrics
                  </h4>
                  <DataTable
                    columns={["Class", "Precision", "Recall", "F1", "Support"]}
                    rows={Object.entries(metrics.per_class).map(([key, entry]) => [
                      status?.label_mapping?.[key] ?? key,
                      formatNumber(entry.precision, 4),
                      formatNumber(entry.recall, 4),
                      formatNumber(entry.f1, 4),
                      formatInt(entry.support),
                    ])}
                  />
                </div>
              ) : null}

              {metrics.confusion_matrix?.length ? (
                <div>
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    Confusion matrix (rows: reference, columns: predicted)
                  </h4>
                  <DataTable
                    columns={[
                      "Reference \\ Predicted",
                      ...(metrics.confusion_matrix_order ?? []).map(
                        (key) => status?.label_mapping?.[key] ?? key,
                      ),
                    ]}
                    rows={metrics.confusion_matrix.map((row, index) => [
                      status?.label_mapping?.[String(index)] ?? String(index),
                      ...row.map((value) => formatInt(value)),
                    ])}
                  />
                </div>
              ) : null}

              {Object.keys(card?.evaluation?.comparison ?? {}).length ? (
                <div>
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
                    Baseline versus improved model
                  </h4>
                  <DataTable
                    columns={["Model", "Split", "Accuracy", "Macro F1", "ROC-AUC"]}
                    rows={Object.entries(
                      (card?.evaluation?.comparison ?? {}) as Record<
                        string,
                        { validation?: Record<string, number | null>; test?: Record<string, number | null> }
                      >,
                    ).flatMap(([name, payload]) =>
                      (["validation", "test"] as const).flatMap((split) => {
                        const entry = payload?.[split];
                        if (!entry) return [];
                        return [
                          [
                            name,
                            split,
                            formatNumber(entry.accuracy ?? null, 4),
                            formatNumber(entry.macro_f1 ?? null, 4),
                            formatNumber(entry.roc_auc ?? null, 4),
                          ],
                        ];
                      }),
                    )}
                  />
                </div>
              ) : null}

              {metrics.class_distribution ? (
                <InfoNote tone="neutral">
                  Test class distribution:{" "}
                  {Object.entries(metrics.class_distribution)
                    .map(
                      ([key, value]) =>
                        `${status?.label_mapping?.[key] ?? key}: ${value}`,
                    )
                    .join(" · ")}
                </InfoNote>
              ) : null}
            </>
          )}

          {status?.splits && Object.keys(status.splits).length ? (
            <DataTable
              columns={["Split", "Class distribution"]}
              rows={Object.entries(status.splits).map(([name, distribution]) => [
                name,
                Object.entries(distribution)
                  .map(([key, value]) => `${status.label_mapping?.[key] ?? key}: ${value}`)
                  .join(" · "),
              ])}
            />
          ) : null}

          {status?.caveats?.length ? (
            <div className="space-y-2">
              {status.caveats.map((caveat) => (
                <InfoNote tone="warn" key={caveat}>
                  <span className="inline-flex items-start gap-2">
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    {caveat}
                  </span>
                </InfoNote>
              ))}
            </div>
          ) : null}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Reproducing this pipeline"
          subtitle="The commands that produce the artefact this page reads"
        />
        <CardBody className="space-y-3">
          {card?.training.pipeline?.length ? (
            <ol className="list-decimal space-y-1 pl-5 text-xs text-clinical-600 dark:text-clinical-300">
              {card.training.pipeline.map((step) => (
                <li key={step}>
                  <code className="rounded bg-black/5 px-1 dark:bg-white/10">{step}</code>
                </li>
              ))}
            </ol>
          ) : null}
          <InfoNote tone="neutral">
            No class is reported as malignant, benign or staged: the labels available in the
            dataset describe the tumour grade group, and nothing in this pipeline predicts
            malignancy, clinical stage or prognosis. Training accuracy is never reported as test
            accuracy — every metric above names the split it came from.
          </InfoNote>
          <p className="text-xs text-clinical-500 dark:text-clinical-400">
            {available
              ? "Predictions on this page come from the served artefact and carry its recorded metrics."
              : "No prediction is shown while classification is disabled; nothing is filled in for display."}
          </p>
        </CardBody>
      </Card>

      <DisclaimerBanner
        text={
          card?.disclaimer ??
          "Research prototype. Outputs are AI-generated and are not intended to replace professional medical diagnosis."
        }
      />
      <p className="text-[11px] text-clinical-500 dark:text-clinical-400">
        Not evaluated is not the same as zero: {NOT_EVALUATED} means no artefact exists, while a
        measured value is always shown with the split it was measured on.
      </p>
    </div>
  );
}
