import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, ApiError } from "../services/api";
import type {
  AnalysisDetail,
  AnalysisPayload,
  Capabilities,
} from "../types/api";

const STORAGE_KEY = "brain-current-analysis";

interface AnalysisContextValue {
  analysisId: string | null;
  detail: AnalysisDetail | null;
  result: AnalysisPayload | null;
  loading: boolean;
  error: ApiError | null;
  capabilities: Capabilities | null;
  select: (analysisId: string | null) => void;
  refresh: () => Promise<void>;
  clear: () => void;
}

const AnalysisContext = createContext<AnalysisContextValue>({
  analysisId: null,
  detail: null,
  result: null,
  loading: false,
  error: null,
  capabilities: null,
  select: () => undefined,
  refresh: async () => undefined,
  clear: () => undefined,
});

const CapabilityContext = createContext<{
  capabilities: Capabilities | null;
  reload: () => void;
}>({ capabilities: null, reload: () => undefined });

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [analysisId, setAnalysisId] = useState<string | null>(() =>
    localStorage.getItem(STORAGE_KEY),
  );
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);

  const loadCapabilities = useCallback(() => {
    api
      .capabilities()
      .then(setCapabilities)
      .catch(() => setCapabilities(null));
  }, []);

  useEffect(() => {
    loadCapabilities();
  }, [loadCapabilities]);

  const fetchDetail = useCallback(async (id: string | null) => {
    if (!id) {
      setDetail(null);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setDetail(await api.analysis(id));
    } catch (caught) {
      setDetail(null);
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError(String(caught), "unknown", 0),
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchDetail(analysisId);
  }, [analysisId, fetchDetail]);

  const select = useCallback((id: string | null) => {
    setAnalysisId(id);
    if (id) localStorage.setItem(STORAGE_KEY, id);
    else localStorage.removeItem(STORAGE_KEY);
  }, []);

  const value = useMemo<AnalysisContextValue>(
    () => ({
      analysisId,
      detail,
      result: detail?.result ?? null,
      loading,
      error,
      capabilities,
      select,
      refresh: async () => {
        await fetchDetail(analysisId);
        loadCapabilities();
      },
      clear: () => select(null),
    }),
    [
      analysisId,
      detail,
      loading,
      error,
      capabilities,
      select,
      fetchDetail,
      loadCapabilities,
    ],
  );

  return (
    <AnalysisContext.Provider value={value}>
      <CapabilityContext.Provider
        value={{ capabilities, reload: loadCapabilities }}
      >
        {children}
      </CapabilityContext.Provider>
    </AnalysisContext.Provider>
  );
}

export const useAnalysis = () => useContext(AnalysisContext);
export const useCapabilities = () => useContext(CapabilityContext);
