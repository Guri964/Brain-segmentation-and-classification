export interface InputSpec {
  channels: number;
  height: number;
  width: number;
  normalisation: string;
  dtype: string;
}

export interface FlaggedError {
  error: { code: string; message: string; details?: Record<string, unknown> };
}

export interface HealthCheckEntry {
  input: string;
  input_shape?: number[];
  raw_min?: number;
  raw_max?: number;
  raw_mean?: number;
  probability_min?: number;
  probability_max?: number;
  probability_mean?: number;
  probability_spread?: number;
  foreground_fraction_at_threshold?: number;
  threshold?: number;
  verdict?: string;
  explanation?: string;
  error?: string;
}

export interface TrainingProvenance {
  state: string;
  label: string;
  dataset?: string | null;
  dataset_root?: string | null;
  synthetic?: boolean | null;
  provenance_note?: string | null;
  trained_on?: string | null;
  evaluated_on?: string | null;
  split_method?: string | null;
  split_seed?: number | null;
  feature_source?: string | null;
  trainer?: string | null;
  trained_at?: string | null;
  produced_by?: string | null;
  exported_file?: string | null;
  best_validation_dice?: number | null;
  serving_note: string;
  note?: string | null;
}

export interface ModelHealth {
  model_id: string;
  status: string;
  verdict: string;
  device: string;
  threshold: number;
  reference: string;
  output_semantics: string;
  checks: HealthCheckEntry[];
  message: string;
  recommendation: string;
  checked_at: string;
  phantom_version: string;
  responsive: boolean;
  training?: TrainingProvenance;
}

export interface ModelInfo {
  model_id: string;
  display_name: string;
  task: string;
  architecture: string;
  path: string;
  input_spec: InputSpec;
  output_semantics: string;
  trainer: string;
  fixed_threshold: number;
  description: string;
  provenance: string;
  status: string;
  message: string;
  device: string;
  parameter_count: number | null;
  load_seconds: number | null;
  file_size_bytes: number | null;
  available: boolean;
  health?: ModelHealth;
  checkpoint_container?: string;
}

export interface EnsembleGroup {
  channels: number;
  height: number;
  width: number;
  normalisation: string;
  members: string[];
  ensemblable: boolean;
}

export interface EnsembleCompatibility {
  available: boolean;
  groups: EnsembleGroup[];
  model_status: Array<{
    model_id: string;
    status: string;
    message: string;
    input_spec?: InputSpec;
    health?: unknown;
  }>;
  reasons: string[];
}

export interface PerClassMetrics {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  support: number;
}

export interface ClassificationMetrics {
  accuracy?: number | null;
  macro_f1?: number | null;
  weighted_f1?: number | null;
  roc_auc?: number | null;
  roc_auc_average?: string | null;
  precision_macro?: number | null;
  recall_macro?: number | null;
  samples?: number | null;
  evaluated_model?: string | null;
  evaluation_split?: string | null;
  per_class?: Record<string, PerClassMetrics>;
  confusion_matrix?: number[][];
  confusion_matrix_order?: string[];
  class_distribution?: Record<string, number>;
  validation_accuracy?: number | null;
  validation_macro_f1?: number | null;
  validation_roc_auc?: number | null;
  baseline_test_metrics?: ClassificationMetrics | null;
  improved_test_metrics?: ClassificationMetrics | null;
}

export interface TrainedArtefact {
  name: string;
  task: string;
  label_space?: string | null;
  dataset?: string | null;
  dataset_synthetic?: boolean | null;
  state?: string | null;
  file?: string | null;
  served_by_application?: boolean;
  serving_note?: string | null;
}

export interface ClassificationStatus {
  available: boolean;
  reason?: string;
  model_id?: string;
  model_version?: string;
  trained_at?: string;
  label_mapping?: Record<string, string>;
  feature_count?: number;
  feature_schema?: string[];
  metrics?: ClassificationMetrics;
  model_comparison?: Record<string, { validation?: ClassificationMetrics | null; test?: ClassificationMetrics | null }>;
  selected_model?: string | null;
  splits?: Record<string, Record<string, number>>;
  preprocessing?: { type?: string | null; fit_on?: string | null; imputation?: string | null };
  caveats?: string[];
  trainer?: string;
  calibration?: Record<string, unknown>;
  target_description?: string;
  legacy_checkpoint_note?: string | null;
  required_artifacts?: string[];
  requirements?: string[];
  training?: TrainingProvenance;
  trained_artefacts?: TrainedArtefact[];
}

export interface ClassificationModelCard {
  status: ClassificationStatus;
  evaluation: {
    state?: string;
    reason?: string;
    model_id?: string;
    version?: string;
    target?: string | null;
    label_mapping?: Record<string, string>;
    comparison?: Record<
      string,
      { validation?: ClassificationMetrics | null; test?: ClassificationMetrics | null }
    >;
    selected_model?: string;
    splits?: Record<string, Record<string, number>>;
    feature_count?: number;
    baseline?: string;
    improved?: string;
    generated_at?: string;
    produced_by?: string;
    artefact?: string;
  };
  label_semantics: {
    target: string | null;
    explanation: string;
    derived_from_mask: boolean;
    note: string;
  };
  xai: {
    method: string;
    shap: { available: boolean; version?: string | null; reason?: string; fallback?: string };
  };
  training: {
    recommended: string;
    alternatives: string[];
    pipeline: string[];
    requirements: string[];
  };
  disclaimer: string;
}

export interface AssistantStatus {
  backend: "local" | "llm";
  local_backend_available: boolean;
  llm_requested: boolean;
  llm_key_present: boolean;
  llm_model: string | null;
  note: string;
  suggested_questions: string[];
  disclaimer: string;
}

export interface Capabilities {
  models: ModelInfo[];
  model_health: {
    phantom_version: string;
    generated_at: string;
    models: ModelHealth[];
    responsive_models: string[];
  };
  ensemble: EnsembleCompatibility;
  radiomics: {
    preference: string;
    pyradiomics_available: boolean;
    selected_backend: string;
    note: string;
  };
  classification: ClassificationStatus;
  image_classification: ImageClassificationSummary;
  assistant: AssistantStatus;
  explainability: {
    segmentation: Array<{
      model_id: string;
      display_name: string;
      available: boolean;
      method: string;
      target_layer: string | null;
      reason: string | null;
    }>;
    classification: {
      method: string;
      available: boolean;
      reason: string | null;
    };
    disclaimer: string;
  };
  evaluation: string;
  disclaimer: string;
}

export interface SegmentationSummary {
  model_id: string;
  model_display_name: string;
  device: string;
  threshold: number;
  input_shape: number[];
  output_semantics: string;
  foreground_pixels: number;
  foreground_fraction: number;
  probability_mean_in_mask: number | null;
  probability_max: number;
  probability_mean: number;
  inference_seconds: number;
  tta_used: boolean;
  mc_dropout_used: boolean;
  n_passes: number;
  warnings: string[];
  predictive_spread_mean?: number | null;
  predictive_spread_max?: number | null;
}

export interface ModelResult {
  available: boolean;
  status: string;
  message?: string;
  descriptor?: ModelInfo;
  summary?: SegmentationSummary;
  artifacts?: Record<string, string>;
  health?: { verdict: string; message: string; checks: HealthCheckEntry[] };
  is_reference_validated?: boolean;
}

export interface EnsembleResult {
  available: boolean;
  strategy: string;
  weights: Record<string, number>;
  weight_source?: string;
  members?: string[];
  threshold?: number;
  foreground_pixels?: number;
  foreground_fraction?: number;
  disagreement?: {
    mean_pairwise_dice_agreement: number | null;
    probability_std_mean: number;
    probability_std_max: number;
    pixels_with_majority_disagreement: number;
    pairwise: Record<
      string,
      {
        dice_agreement: number | null;
        iou_agreement: number | null;
        only_in_first: number;
        only_in_second: number;
      }
    >;
    note: string;
  } | null;
  reasons: string[];
  warnings?: string[];
  model_status?: Array<{
    model_id: string;
    status: string;
    message: string;
    health?: { verdict: string };
  }>;
  artifacts?: Record<string, string>;
}

export interface RadiomicsQuality {
  mask_source?: string;
  roi_voxel_count?: number;
  roi_bounding_box?: number[][];
  roi_extent?: number[];
  /** Mean coordinate per mask axis, same axis order as roi_bounding_box. */
  roi_centroid?: number[];
  roi_centroid_definition?: string;
  roi_touches_fov_border?: boolean | null;
  relative_roi_size?: number;
  note?: string;
}

/**
 * Measurements taken from the segmentation mask itself (pixel units). They
 * describe the region the model produced, not the patient: no anatomical,
 * malignancy or staging statement can be derived from them.
 */
export interface RegionMeasurements {
  detected: boolean;
  pixel_area: number;
  image_coverage: number;
  image_coverage_percent: number;
  bounding_box: {
    x: number;
    y: number;
    width: number;
    height: number;
  } | null;
  centroid: { x: number; y: number } | null;
  perimeter_px: number | null;
  equivalent_diameter_px: number | null;
  units: string;
  image_size: { width: number; height: number } | null;
  mask_source?: string | null;
  threshold?: number;
  note: string;
}

export interface RadiomicsFeatureSet {
  available: boolean;
  analysis_id?: string;
  backend?: string;
  backend_note?: string;
  dimensionality?: string;
  mask_source?: string;
  feature_count?: number;
  features?: Record<string, number | null>;
  categories?: Record<string, string[]>;
  spacing_available?: boolean;
  warnings?: string[];
  quality?: RadiomicsQuality;
  unavailable_reason?: string;
  descriptions?: Record<string, string>;
  artifacts?: Record<string, string>;
}

export interface ClassificationResult {
  available: boolean;
  reason: string;
  model_id?: string | null;
  model_version?: string | null;
  trainer?: string | null;
  label_mapping?: Record<string, string>;
  predicted_label?: string | null;
  predicted_index?: number | null;
  scores?: number[];
  calibrated?: boolean;
  calibration_method?: string | null;
  metrics?: Record<string, number | null> | null;
  feature_contributions?: Array<{
    feature: string;
    value?: number;
    weight?: number;
    contribution: number;
    method?: string;
  }>;
  missing_features?: string[];
  statement?: string;
  target_description?: string | null;
  model_comparison?: Record<
    string,
    { validation?: ClassificationMetrics | null; test?: ClassificationMetrics | null }
  >;
  caveats?: string[];
  evaluated_on?: string | null;
  training?: TrainingProvenance;
}

export interface ExplainabilityEntry {
  model_id: string;
  available: boolean;
  reason?: string;
  method?: string;
  target_layer?: string;
  target_score?: string;
  explanation?: string;
  activation_shape?: number[];
  heatmap_mean?: number;
  heatmap_max?: number;
  warnings?: string[];
  artifacts?: Record<string, string>;
  target_layer_semantics?: string;
}

export interface UncertaintyResult {
  available: boolean;
  method?: string;
  n_passes?: number;
  mean_dispersion?: number | null;
  max_dispersion?: number | null;
  p95_dispersion?: number | null;
  high_uncertainty_fraction?: number | null;
  boundary_dispersion?: number | null;
  interior_dispersion?: number | null;
  notes?: string[];
  unavailable_reason?: string | null;
  model_id?: string;
  artifact_path?: string;
  ensemble_disagreement?: Record<string, unknown> | null;
}

export interface ModelComparison {
  state: string;
  note: string;
  models: Array<Record<string, unknown>>;
  pairwise_agreement?: Record<
    string,
    {
      dice_agreement: number | null;
      iou_agreement: number | null;
      only_in_first: number;
      only_in_second: number;
    }
  >;
  ensemble?: Record<string, unknown>;
  mask_agreement_with_ensemble?: Record<
    string,
    {
      dice_agreement: number | null;
      iou_agreement: number | null;
      pixels_a: number;
      pixels_b: number;
    }
  >;
}

export interface AnalysisPayload {
  analysis_id: string;
  scan_identifier: string;
  analysis_type: string;
  created_at: string;
  device: string;
  threshold: number;
  preprocessing: {
    pipeline: string;
    tensor_shape: number[];
    channels: number;
    spatial_size: number[];
    normalisation: string;
    metadata: Record<string, unknown>;
  };
  inputs: Record<string, string>;
  models: Record<string, ModelResult>;
  ensemble: EnsembleResult;
  uncertainty: UncertaintyResult | null;
  reference_mask: {
    source: string;
    model_id: string | null;
    foreground_pixels: number;
  };
  region_measurements?: RegionMeasurements;
  radiomics: RadiomicsFeatureSet;
  explainability: ExplainabilityEntry[];
  classification: ClassificationResult;
  model_comparison: ModelComparison;
  artifacts_root: string;
  report?: {
    available: boolean;
    report_id?: string;
    relative_path?: string;
    size_bytes?: number;
    reason?: string;
    generator_version?: string;
  };
  disclaimer: string;
}

export interface AnalysisMetadata {
  analysis_id: string;
  scan_identifier: string;
  analysis_type: string;
  status: string;
  created_at: string;
  completed_at?: string | null;
  duration_seconds?: number | null;
  device?: string | null;
  notes?: string | null;
  error_message?: string | null;
  model_ids: string[];
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  preprocessing?: Record<string, unknown>;
}

export interface StoredRecords {
  segmentation: Array<Record<string, unknown>>;
  radiomics: Array<Record<string, unknown>>;
  classification: Record<string, unknown> | null;
  explainability: Array<Record<string, unknown>>;
  uncertainty: Record<string, unknown> | null;
  reports: Array<{
    report_id: string;
    path: string;
    size_bytes: number;
    created_at: string;
    generator_version?: string;
  }>;
}

export interface AnalysisDetail {
  metadata: AnalysisMetadata;
  stored_records: StoredRecords;
  payload_available: boolean;
  payload_source: string;
  result: AnalysisPayload | null;
}

export interface HistoryItem {
  analysis_id: string;
  scan_identifier: string;
  analysis_type: string;
  status: string;
  model_ids: string[];
  created_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
  device: string | null;
  notes: string | null;
  error_message: string | null;
  segmentation_count: number;
  radiomics_count: number;
  report_count: number;
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  filters: { statuses: string[]; analysis_types: string[]; models: string[] };
}

export interface Statistics {
  totals: {
    analyses: number;
    segmentations: number;
    radiomics: number;
    reports: number;
    classifications: number;
  };
  recent: HistoryItem[];
  status_counts: Record<string, number>;
  models: { ready: number; total: number };
  evaluation_state: string;
}

export interface PlanStep {
  step: string;
  model_id: string;
  status: string;
  runnable: boolean;
  health: string;
  message: string;
  input_spec: InputSpec;
}

export interface PlanResponse {
  analysis_type: string;
  requested_models: string[];
  steps: PlanStep[];
  ensemble: EnsembleCompatibility | null;
  radiomics: {
    runnable: boolean;
    backend: Record<string, unknown>;
    note: string;
  };
  uncertainty: { runnable: boolean; note: string };
  explainability: {
    runnable: boolean;
    targets: Array<{ model_id: string; target_layer: string | null }>;
  };
  classification: ClassificationStatus;
}

export interface AnalysisOptions {
  models: Array<{
    model_id: string;
    display_name: string;
    status: string;
    available: boolean;
    input_spec: InputSpec;
    output_semantics: string;
    description: string;
  }>;
  analysis_types: Array<{
    key: string;
    label: string;
    description: string;
    channels: number;
  }>;
  thresholds: { default: number; min: number; max: number; step: number };
  ensemble_strategies: Array<{
    key: string;
    label: string;
    description: string;
    usable: boolean;
    note: string;
  }>;
  radiomics_masks: Array<{ key: string; label: string; description: string }>;
  limits: {
    max_upload_bytes: number;
    max_files: number;
    accepted_extensions: string[];
  };
}

export interface RuntimeEntry {
  model_id: string;
  state: string;
  reason?: string;
  device?: string;
  input_shape?: number[];
  parameter_count?: number | null;
  seconds_per_image?: number;
  images_per_second?: number | null;
  note?: string;
}

export interface SegmentationMetricEntry {
  model_id: string;
  display_name: string;
  status: string;
  metrics: Record<string, number> | null;
  dataset?: Record<string, unknown>;
  runtime_seconds_per_image?: number | null;
  parameter_count?: number | null;
}

export interface EvaluationResponse {
  evaluation: {
    state: string;
    reason?: string;
    dataset?: Record<string, unknown>;
    segmentation?: Record<string, unknown>;
    classification?: Record<string, unknown>;
    runtime?: Record<string, unknown>;
    expected_artifact?: string;
    produced_by?: string;
    state_description?: string;
  };
  model_selection: {
    state: string;
    reason?: string;
    expected_artifact?: string;
    produced_by?: string;
  } & Record<string, unknown>;
  segmentation: {
    state: string;
    reason?: string;
    models: Array<{
      model_id: string;
      display_name: string;
      metrics: Record<string, number> | null;
      status: string;
    }>;
    dataset?: Record<string, unknown>;
    ensemble?: unknown;
    ablation?: Array<Record<string, unknown>>;
  };
  classification: Record<string, unknown> & { state?: string; reason?: string };
  runtime: { state: string; models?: RuntimeEntry[]; generated_at?: string };
  model_health: {
    models: ModelHealth[];
    responsive_models: string[];
    generated_at: string;
    phantom_version: string;
  };
  training_runs: Array<Record<string, unknown>>;
  protocol: {
    splits: string[];
    principles: string[];
    scripts: Record<string, string>;
  };
}

export interface AssistantAnswer {
  answer: string;
  topic: string;
  sources: string[];
  grounding: Record<string, unknown>;
  backend: string;
  suggestions: string[];
  disclaimers: string[];
  scope: string;
  analysis_id?: string | null;
}

export interface CompareResponse {
  analysis_type: string;
  threshold: number;
  input_preview: string;
  input_channels: Array<{ label: string; image: string }>;
  preprocessing: Record<string, unknown>;
  models: Array<{
    model_id: string;
    available: boolean;
    status: string;
    message?: string;
    summary?: SegmentationSummary;
    probability?: string;
    mask?: string;
    overlay?: string;
  }>;
  comparison: ModelComparison;
  ensemble?: EnsembleResult;
}

export interface ReportItem {
  report_id: string;
  analysis_id: string;
  scan_identifier: string;
  path: string;
  size_bytes: number;
  created_at: string;
  generator_version?: string;
}

export interface ReportListResponse {
  items: ReportItem[];
}

export interface SystemInfo {
  python: string;
  platform: string;
  torch_version: string;
  cuda_available: boolean;
  cuda_device_name: string | null;
  torch_threads: number;
  selected_device: string;
  config: Record<string, unknown>;
}

export interface HealthReport {
  models: ModelHealth[];
  responsive_models: string[];
  generated_at: string;
  phantom_version: string;
}

export interface FeatureRow {
  name: string;
  label?: string;
  value: number | null;
  category: string;
  description?: string;
  categoryLabel?: string;
}

export interface HealthStatus {
  status: string;
  environment: string;
  time: string;
  database: { path: string; schema_version: number };
}

export interface RadiomicsCatalogue {
  backend: Record<string, unknown>;
  categories: Array<{ key: string; label: string; description: string }>;
  descriptions: Record<string, string>;
  discretisation: Record<string, unknown>;
  notes: string[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  answer?: AssistantAnswer;
}

/** One Keras image-classification model: where it is looked for and whether it is usable. */
export interface ImageClassifierModel {
  model_id: string;
  display_name: string;
  task: string;
  framework: string;
  state:
    | "ready"
    | "checkpoint_missing"
    | "metadata_missing"
    | "framework_missing"
    | "incompatible"
    | "error";
  available: boolean;
  reason: string;
  model_file: string;
  model_path: string;
  model_file_present: boolean;
  metadata_file: string;
  metadata_present: boolean;
  framework_available: boolean | null;
  framework_version: string | null;
  framework_error: string | null;
  loaded: boolean;
  parameter_count: number | null;
  label_semantics: string;
  preprocessing: string | null;
  preprocessing_implementation: string;
  input_size: number[] | null;
  classes: string[];
  class_to_index: Record<string, number>;
  reported_metrics: Record<string, number> | null;
  metrics_provenance: string;
  training_config: Record<string, unknown> | null;
  hardware: Record<string, unknown> | null;
  architecture: string | null;
  framework_version_recorded: string | null;
  trainer: string;
  searched: string[];
  accepted_formats: string[];
}

export interface ImageClassificationSummary {
  available: boolean;
  state: string;
  model_id: string;
  display_name: string;
  reason: string;
  classes: string[];
  input_size: number[] | null;
  preprocessing: string | null;
  reported_metrics: Record<string, number> | null;
  metrics_provenance: string;
  model_path: string;
  metadata_path: string;
}

export interface ImageClassificationStatus {
  available: boolean;
  state: string;
  reason: string;
  models: ImageClassifierModel[];
  accepted_formats: string[];
  task: string;
  task_description: string;
  distinct_from_grade_model: string;
  metrics_provenance: string;
  install_note: string;
  disclaimer: string;
}

export interface ImageClassificationPrediction {
  model_id: string;
  display_name: string;
  available: boolean;
  task: string;
  filename: string | null;
  predicted_index: number;
  predicted_label: string;
  probability: number;
  classes: string[];
  probabilities: number[];
  scores: Array<{ index: number; label: string; probability: number }>;
  class_to_index: Record<string, number>;
  classes_source: string;
  output_transform: string;
  calibrated: boolean;
  calibration_note: string;
  input: {
    filename: string | null;
    original_shape: number[];
    model_input_size: number[];
    resize: string;
  };
  preprocessing: string | null;
  preprocessing_implementation: string;
  inference_seconds: number;
  architecture: string | null;
  framework_version_recorded: string | null;
  reported_metrics: Record<string, number> | null;
  metrics_provenance: string;
  training_config: Record<string, unknown> | null;
  label_semantics: string;
  disclaimer: string;
  upload?: { filename: string | null; bytes: number; decoded_shape: number[] };
}
