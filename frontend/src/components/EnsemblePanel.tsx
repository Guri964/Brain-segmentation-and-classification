import { useEffect, useState } from "react";
import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";
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
  Tooltip,
  ValueOrUnavailable,
} from "./ui";
import { FeatureBarChart } from "./charts";
import { api, ApiError } from "../services/api";
import {
  formatBytes,
  formatInt,
  formatNumber,
  NOT_AVAILABLE,
} from "../services/format";
import type {
  StackedEnsemblePayload,
  StackedEnsembleResult,
  StackedEnsembleStatus,
  StackedEnsembleUnavailable,
} from "../types/api";

const STATE_LABEL: Record<string, string> = {
  ready: "Installed and loadable",
  artifacts_missing: "Artefacts not installed",
  members_missing: "Segmentation members not loadable",
  incompatible: "Artefacts disagree with each other",
  unavailable: "Not wired into this deployment",
};

const ARTIFACT_STATE: Record<string, string> = {
  artifacts_missing: "amber",
  members_missing: "amber",
  incompatible: "rose",
  ready: "teal",
  unavailable: "neutral",
};

/**
 * The stacked ensemble of `FINAL ENSEMBLE NB.ipynb`.
 *
 * Three four-channel segmentation members contribute their maximum, mean and consensus-Dice
 * statistics, a 46-feature radiomics CNN contributes one binary score, and a gradient-boosted
 * head combines the ten columns. The notebook's target is `(mask pixel count > 100)`, an
 * image-level tumour-size proxy - **not** a WHO grade, not malignancy and not a tumour type - so
 * the panel carries the backend's own semantics and leakage text instead of inventing a label.
 *
 * Two substitutions are visible in every result because the notebook used ground truth: each
 * member's Dice is measured against the consensus mask, and the 46 features are extracted from
 * that mask. Both are stated in the payload and repeated here.
 */
export default function EnsemblePanel({
  payload,
  status: providedStatus,
}: {
  payload?: StackedEnsembleResult | null;
  status?: StackedEnsembleStatus | null;
}) {
  const [status, setStatus] = useState<StackedEnsembleStatus | null>(
    providedStatus ?? null,
  );
  const [loading, setLoading] = useState(!providedStatus);
  const [error, setError] = useState<ApiError | null>(null);

  // The endpoint re-reads the artefact table on every call, so one GET is a real re-check.
  const load = () => {
    setLoading(true);
    setError(null);
    api
      .stackedEnsemble()
      .then((value) => setStatus(value))
      .catch((cause) =>
        setError(
          cause instanceof ApiError
            ? cause
            : new ApiError(String(cause), "unknown", 0),
        ),
      )
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (providedStatus) return;
    load();
    // A single read is enough: the artefact table does not change while the page is open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providedStatus]);

  if (payload) {
    return payload.available === false ? (
      <UnavailableSection payload={payload} />
    ) : (
      <ResultSection payload={payload} />
    );
  }

  if (loading && !status) return <LoadingBlock label="Checking the stacked ensemble" />;

  if (error) {
    return (
      <Card>
        <CardHeader
          title="Stacked ensemble (FINAL ENSEMBLE NB.ipynb)"
          subtitle="Three segmentation members + radiomics CNN + gradient-boosted head"
        />
        <CardBody>
          <EmptyState
            title="The ensemble status could not be read"
            message={`${error.message}${error.code ? ` (${error.code})` : ""}`}
            action={
              <button type="button" className="btn-secondary" onClick={() => load()}>
                <RefreshCw className="h-3.5 w-3.5" />
                Try again
              </button>
            }
          />
        </CardBody>
      </Card>
    );
  }

  if (!status) return null;

  const available = status.available;
  const missing = status.artifacts.filter((entry) => !entry.present);

  return (
    <Card>
      <CardHeader
        title="Stacked ensemble (FINAL ENSEMBLE NB.ipynb)"
        subtitle="Not part of this analysis. It runs on the four-channel input when it is enabled in the run options; the state below is what the backend reports right now."
        actions={
          <button
            type="button"
            className="btn-secondary px-2 py-1 text-[11px]"
            onClick={() => load()}
            disabled={loading}
          >
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
            Re-check
          </button>
        }
      />
      <CardBody className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Pill
            tone={
              (ARTIFACT_STATE[status.state] ?? "neutral") as
                | "neutral"
                | "teal"
                | "amber"
                | "rose"
                | "violet"
            }
          >
            {STATE_LABEL[status.state] ?? status.state}
          </Pill>
          <span className="text-xs text-clinical-500 dark:text-clinical-400">
            {status.reason}
          </span>
        </div>

        <InfoNote tone="warn">
          <strong>This head does not predict a grade or a tumour type.</strong>{" "}
          {status.target_semantics}
        </InfoNote>

        <DataTable
          columns={["Artefact", "Role", "State", "Size"]}
          dense
          rows={status.artifacts.map((entry) => [
            <div key={entry.key}>
              <p className="font-medium text-clinical-900 dark:text-clinical-50">
                {entry.file}
              </p>
              <p className="text-[11px] text-clinical-500 dark:text-clinical-400">
                {entry.path}
              </p>
            </div>,
            entry.role,
            <Pill tone={entry.present ? "teal" : "amber"} key={`${entry.key}-s`}>
              {entry.present ? "present" : "missing"}
            </Pill>,
            entry.present ? formatBytes(entry.size_bytes) : NOT_AVAILABLE,
          ])}
        />

        <DataTable
          columns={["Member", "Status", "Architecture", "Parameters", "Message"]}
          dense
          rows={status.members.map((member) => [
            member.display_name,
            <Pill tone={member.available ? "teal" : "amber"} key={member.model_id}>
              {member.status}
            </Pill>,
            <span className="text-[11px]">{member.architecture}</span>,
            member.parameter_count ? formatInt(member.parameter_count) : NOT_AVAILABLE,
            <span className="text-[11px] text-clinical-500 dark:text-clinical-400">
              {member.message}
            </span>,
          ])}
        />

        {!available ? (
          <EmptyState
            title="The stacked ensemble cannot run yet"
            message={
              <span className="space-y-2">
                {missing.length ? (
                  <span className="block">
                    Missing files:{" "}
                    <code>{missing.map((entry) => entry.file).join(", ")}</code>
                  </span>
                ) : null}
                <span className="block">{status.install_note}</span>
              </span>
            }
          />
        ) : (
          <div className="grid gap-3 sm:grid-cols-3">
            <MetricCard
              label="Meta features"
              value={status.contracts.meta_features ?? NOT_AVAILABLE}
              hint={(status.meta_features ?? []).join(", ")}
            />
            <MetricCard
              label="Radiomics features"
              value={status.contracts.radiomics_features ?? NOT_AVAILABLE}
              hint="Column order taken from the saved schema, never re-derived"
            />
            <MetricCard
              label="Input"
              value={
                status.input_requirements
                  ? `${status.input_requirements.spatial_size.join(" × ")} × ${status.input_requirements.channels}`
                  : NOT_AVAILABLE
              }
              hint={status.input_requirements?.normalisation ?? ""}
            />
          </div>
        )}

        <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
          <p className="text-xs font-medium text-clinical-800 dark:text-clinical-100">
            Metrics printed by the notebook's own run
          </p>
          <div className="mt-2 grid gap-2 sm:grid-cols-3">
            {Object.entries(status.reported_metrics ?? {}).map(([key, value]) => (
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
          <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-400">
            {status.metrics_provenance}
          </p>
          <p className="mt-1 text-[11px] text-clinical-500 dark:text-clinical-400">
            No accuracy is claimed for this deployment: the split, the masks and the labels are
            not shipped with the weights, so nothing can be re-measured here.
            {status.checked_at ? ` Checked ${status.checked_at}.` : ""}
          </p>
        </div>

        <InfoNote tone="warn">
          <span className="inline-flex items-start gap-1.5">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5" />
            <span>{status.leakage_note}</span>
          </span>
        </InfoNote>

        <InfoNote tone="neutral">
          <strong>Separate from the other classifiers.</strong>{" "}
          {status.distinct_from_grade_and_type_models}
        </InfoNote>
      </CardBody>
    </Card>
  );
}

function ResultSection({ payload }: { payload: StackedEnsemblePayload }) {
  const labels = payload.class_labels ?? [];
  const predicted = labels[payload.prediction] ?? String(payload.prediction);
  const chartData = payload.class_probabilities
    ? payload.class_probabilities.map((value, index) => ({
        label: labels[index] ?? `class ${index}`,
        value,
      }))
    : [];
  const memberOrder = payload.member_statistics
    ? payload.member_statistics.map((entry) => entry.model_id)
    : payload.members.map((entry) => entry.model_id);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-4">
        <MetricCard
          label="Ensemble prediction"
          value={predicted.replace(/_/g, " ")}
          hint={`Class index ${payload.prediction} of ${labels.length}`}
          tooltip="The notebook's target is (mask pixel count > 100), an image-level size proxy. It is not a grade and not a tumour type."
        />
        <MetricCard
          label="Consensus region"
          value={formatInt(payload.consensus.foreground_pixels)}
          unit="px"
          hint={`${formatNumber(payload.consensus.foreground_fraction, 4)} of the slice at threshold ${payload.consensus.threshold}`}
        />
        <MetricCard
          label="Radiomics score"
          value={formatNumber(payload.radiomics.score, 4)}
          hint={`${payload.radiomics.feature_count} features → ${payload.radiomics.cnn}`}
        />
        <MetricCard
          label="Time"
          value={`${formatNumber(payload.inference_seconds, 3)} s`}
          hint={`${payload.member_statistics.filter((m) => m.reused_from_analysis).length}/${payload.member_statistics.length} members reused from the analysis`}
        />
      </div>

      <InfoNote tone="warn">
        <strong>Size proxy, not a grade.</strong> {payload.target_semantics}
      </InfoNote>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
            Head output probabilities
          </p>
          {chartData.length ? (
            <FeatureBarChart data={chartData} height={200} />
          ) : (
            <p className="text-xs text-clinical-500 dark:text-clinical-400">
              {NOT_AVAILABLE} — the saved head exposes no <code>predict_proba</code>.
            </p>
          )}
          {payload.meta_classes?.length ? (
            <p className="mt-2 text-[11px] text-clinical-500 dark:text-clinical-400">
              Class order recorded by the estimator: {payload.meta_classes.join(", ")}
            </p>
          ) : null}
        </div>

        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
            Member statistics (the meta-feature inputs)
          </p>
          <DataTable
            columns={["Member", "Dice vs consensus", "Max p", "Mean p", "Reused"]}
            dense
            rows={payload.member_statistics.map((entry) => [
              entry.model_id,
              formatNumber(entry.dice_vs_consensus, 4),
              formatNumber(entry.max_probability, 4),
              formatNumber(entry.mean_probability, 4),
              entry.reused_from_analysis ? "yes" : "computed here",
            ])}
          />
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
          The {payload.meta_feature_count} meta features handed to the head
        </p>
        <DataTable
          columns={["Column", "Value", "After the meta-scaler"]}
          dense
          rows={Object.entries(payload.meta_features).map(([name, value], index) => [
            name,
            formatNumber(value, 6),
            <ValueOrUnavailable
              key={name}
              value={
                payload.meta_features_scaled?.[index] !== undefined
                  ? formatNumber(payload.meta_features_scaled[index], 6)
                  : NOT_AVAILABLE
              }
              reason="The saved meta-scaler did not return this column."
            />,
          ])}
        />
        <p className="mt-2 text-[11px] text-clinical-500 dark:text-clinical-400">
          Order: {memberOrder.map((id) => `${id}.dice_vs_consensus, .max_probability, .mean_probability`).join(" · ")}
          {" · "}radiomics_prediction
        </p>
      </div>

      <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
        <p className="text-xs font-medium text-clinical-800 dark:text-clinical-100">
          What was substituted, because the notebook used ground truth
        </p>
        <p className="mt-1 text-[11px] text-clinical-500 dark:text-clinical-400">
          {payload.dice_note} Feature mask: {payload.feature_mask_source}.
        </p>
      </div>

      <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
        <p className="text-xs font-medium text-clinical-800 dark:text-clinical-100">
          Artefacts this result came from
        </p>
        <p className="mt-1 text-[11px] text-clinical-500 dark:text-clinical-400">
          {Object.entries(payload.artifact_digests ?? {})
            .map(
              ([key, digest]) =>
                `${key} ${digest} (${formatBytes(payload.artifact_sizes?.[key])})`,
            )
            .join(" · ")}
        </p>
        <p className="mt-1 text-[11px] text-clinical-500 dark:text-clinical-400">
          Raw feature digest {payload.radiomics.raw_feature_digest}, scaled digest{" "}
          {payload.radiomics.scaled_feature_digest}. Feature order from{" "}
          <code>{payload.radiomics.feature_names_from}</code>.
        </p>
      </div>

      <div className="rounded-lg border border-clinical-200 p-3 dark:border-clinical-800">
        <p className="text-xs font-medium text-clinical-800 dark:text-clinical-100">
          Metrics printed by the notebook's own run
        </p>
        <div className="mt-2 grid gap-2 sm:grid-cols-3">
          {Object.entries(payload.reported_metrics ?? {}).map(([key, value]) => (
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
        <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-400">
          {payload.metrics_provenance}
        </p>
      </div>

      <InfoNote tone="warn">
        <span className="inline-flex items-start gap-1.5">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5" />
          <span>{payload.leakage_note}</span>
        </span>
      </InfoNote>

      <p className="flex items-center gap-1.5 text-[11px] text-clinical-500 dark:text-clinical-400">
        <Tooltip text={payload.dice_note} />
        Dice reference: {payload.dice_reference}. Members and the radiomics branch were both run
        inside this analysis.
      </p>
    </div>
  );
}

/**
 * The ensemble was requested but the analysis could not run it.
 *
 * The backend returns this as a reported state rather than failing the whole analysis, so the
 * reason is shown verbatim and no score is displayed.
 */
function UnavailableSection({ payload }: { payload: StackedEnsembleUnavailable }) {
  const statuses = Object.entries(payload.model_status ?? {});
  return (
    <Card>
      <CardHeader
        title="Stacked ensemble (FINAL ENSEMBLE NB.ipynb)"
        subtitle="Requested for this analysis, but it could not run."
        actions={<Pill tone="amber">{payload.state.replace(/_/g, " ")}</Pill>}
      />
      <CardBody className="space-y-3">
        <InfoNote tone="warn">
          <strong>No ensemble output was produced.</strong> Nothing is substituted for it.
        </InfoNote>
        <ul className="list-disc space-y-1 pl-4 text-xs text-clinical-600 dark:text-clinical-300">
          {payload.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
        {payload.members_required?.length ? (
          <p className="text-xs text-clinical-500 dark:text-clinical-400">
            Members the meta-learner needs: {payload.members_required.join(", ")}.
          </p>
        ) : null}
        {statuses.length ? (
          <DataTable
            columns={["Member", "Status"]}
            dense
            rows={statuses.map(([modelId, state]) => [
              modelId,
              <Pill
                tone={state === "ready" ? "teal" : "amber"}
                key={`${modelId}-state`}
              >
                {state}
              </Pill>,
            ])}
          />
        ) : null}
        <p className="text-[11px] text-clinical-500 dark:text-clinical-400">
          The rest of the analysis (segmentation, radiomics, region measurements) is unaffected
          and is displayed on this page as usual.
        </p>
      </CardBody>
    </Card>
  );
}
