import type {
  AnalysisDetail,
  AnalysisOptions,
  AnalysisPayload,
  AssistantAnswer,
  AssistantStatus,
  Capabilities,
  ClassificationModelCard,
  ClassificationStatus,
  CompareResponse,
  EvaluationResponse,
  FlaggedError,
  HistoryResponse,
  HealthReport,
  HealthStatus,
  ModelHealth,
  ModelInfo,
  ReportListResponse,
  SystemInfo,
  PlanResponse,
  RadiomicsCatalogue,
  RadiomicsFeatureSet,
  Statistics,
} from "../types/api";

export class ApiError extends Error {
  code: string;
  status: number;
  details: Record<string, unknown>;

  constructor(
    message: string,
    code: string,
    status: number,
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body instanceof FormData
          ? {}
          : { "Content-Type": "application/json" }),
        ...(init?.headers ?? {}),
      },
    });
  } catch (cause) {
    throw new ApiError(
      "The backend could not be reached. Check that the API server is running.",
      "network_error",
      0,
      {
        cause: String(cause),
      },
    );
  }

  const text = await response.text();
  const payload = text ? safeJson(text) : null;

  if (!response.ok) {
    const flagged = payload as FlaggedError | null;
    const message =
      flagged?.error?.message ??
      `Request failed with status ${response.status}.`;
    throw new ApiError(
      message,
      flagged?.error?.code ?? "http_error",
      response.status,
      flagged?.error?.details ?? {},
    );
  }
  return payload as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text.slice(0, 500) };
  }
}

export const api = {
  health: () => request<HealthStatus>("/api/health"),
  system: () => request<SystemInfo>("/api/system"),
  capabilities: () => request<Capabilities>("/api/capabilities"),
  analysisOptions: () => request<AnalysisOptions>("/api/analysis-options"),

  models: () => request<{ device: string; models: ModelInfo[] }>("/api/models"),
  model: (modelId: string) => request<ModelInfo>(`/api/models/${modelId}`),
  modelHealth: (modelId: string, force = false) =>
    request<ModelHealth>(
      `/api/models/${modelId}/health${force ? "?force=true" : ""}`,
    ),
  runHealthChecks: (force = false) =>
    request<HealthReport>("/api/models/health-check", {
      method: "POST",
      body: JSON.stringify({ force }),
    }),

  plan: (body: Record<string, unknown>) =>
    request<PlanResponse>("/api/analysis/plan", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  createAnalysis: (form: FormData, onProgress?: (percent: number) => void) =>
    uploadWithProgress<AnalysisPayload>("/api/analysis", form, onProgress),
  analysis: (analysisId: string) =>
    request<AnalysisDetail>(`/api/analysis/${analysisId}`),
  deleteAnalysis: (analysisId: string) =>
    request<{ deleted: boolean; files_removed: number }>(
      `/api/analysis/${analysisId}`,
      { method: "DELETE" },
    ),
  artifactUrl: (analysisId: string, filename: string) =>
    `/api/analysis/${analysisId}/artifacts/${encodeURIComponent(filename)}`,
  inputUrl: (analysisId: string, index: number) =>
    `/api/analysis/${analysisId}/inputs/${index}`,
  inputPreviewUrl: (analysisId: string, index: number) =>
    `/api/analysis/${analysisId}/preview/${index}`,

  compare: (form: FormData) =>
    request<CompareResponse>("/api/models/compare", {
      method: "POST",
      body: form,
    }),

  history: (params: Record<string, string | number | undefined>) => {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== "" && value !== null)
        query.set(key, String(value));
    });
    return request<HistoryResponse>(`/api/history?${query.toString()}`);
  },
  statistics: () => request<Statistics>("/api/history/statistics"),
  exportAnalysisUrl: (analysisId: string) =>
    `/api/history/${analysisId}/export`,

  reports: () => request<ReportListResponse>("/api/reports"),
  reportUrl: (reportId: string) => `/api/reports/${reportId}`,
  regenerateReport: (analysisId: string) =>
    request<{ report_id: string; relative_path: string; size_bytes: number }>(
      `/api/reports/analysis/${analysisId}`,
      { method: "POST" },
    ),

  evaluation: () => request<EvaluationResponse>("/api/evaluation"),

  radiomicsCatalogue: () =>
    request<RadiomicsCatalogue>("/api/radiomics/catalogue"),
  analysisRadiomics: (analysisId: string) =>
    request<RadiomicsFeatureSet & { analysis_id: string }>(
      `/api/radiomics/analysis/${analysisId}`,
    ),
  extractRadiomics: (form: FormData) =>
    request<RadiomicsFeatureSet>("/api/radiomics/extract", {
      method: "POST",
      body: form,
    }),
  radiomicsCsvUrl: (analysisId: string) =>
    `/api/radiomics/analysis/${analysisId}/features.csv`,

  assistantStatus: () => request<AssistantStatus>("/api/assistant/status"),
  ask: (analysisId: string | null, question: string) =>
    request<AssistantAnswer>("/api/assistant/ask", {
      method: "POST",
      body: JSON.stringify({ analysis_id: analysisId, question }),
    }),
  assistantHistory: (analysisId: string) =>
    request<{
      messages: Array<{
        role: string;
        content: string;
        created_at: string;
        sources?: string[];
      }>;
    }>(`/api/assistant/history/${analysisId}`),

  classificationStatus: (): Promise<ClassificationStatus> =>
    request<Capabilities>("/api/capabilities").then(
      (capabilities) => capabilities.classification,
    ),
  classificationModelCard: () =>
    request<ClassificationModelCard>("/api/classification"),
};

function uploadWithProgress<T>(
  path: string,
  form: FormData,
  onProgress?: (percent: number) => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path, true);
    xhr.responseType = "text";
    xhr.timeout = 15 * 60 * 1000;

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress)
        onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onload = () => {
      const payload = xhr.responseText ? safeJson(xhr.responseText) : null;
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload as T);
        return;
      }
      const flagged = payload as FlaggedError | null;
      reject(
        new ApiError(
          flagged?.error?.message ?? `Upload failed with status ${xhr.status}.`,
          flagged?.error?.code ?? "http_error",
          xhr.status,
          flagged?.error?.details ?? {},
        ),
      );
    };
    xhr.onerror = () =>
      reject(
        new ApiError(
          "The upload failed because the backend could not be reached.",
          "network_error",
          0,
        ),
      );
    xhr.ontimeout = () =>
      reject(
        new ApiError(
          "The analysis timed out. Large inputs on CPU can take several minutes.",
          "timeout",
          0,
        ),
      );
    xhr.send(form);
  });
}
