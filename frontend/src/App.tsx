import { Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "./layouts/AppLayout";
import OverviewPage from "./pages/OverviewPage";
import NewAnalysisPage from "./pages/NewAnalysisPage";
import SegmentationPage from "./pages/SegmentationPage";
import ModelComparisonPage from "./pages/ModelComparisonPage";
import RadiomicsPage from "./pages/RadiomicsPage";
import TumourClassificationPage from "./pages/TumorClassificationPage";
import ExplainabilityPage from "./pages/ExplainabilityPage";
import ClinicalSupportPage from "./pages/ClinicalSupportPage";
import HistoryPage from "./pages/HistoryPage";
import ReportsPage from "./pages/ReportsPage";
import EvaluationPage from "./pages/EvaluationPage";
import SettingsPage from "./pages/SettingsPage";
import NotFoundPage from "./pages/NotFoundPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<OverviewPage />} />
        <Route path="analysis/new" element={<NewAnalysisPage />} />
        <Route path="analysis/segmentation" element={<SegmentationPage />} />
        <Route path="models/compare" element={<ModelComparisonPage />} />
        <Route path="analysis/radiomics" element={<RadiomicsPage />} />
        <Route
          path="analysis/classification"
          element={<TumourClassificationPage />}
        />
        <Route
          path="analysis/explainability"
          element={<ExplainabilityPage />}
        />
        <Route path="analysis/clinical" element={<ClinicalSupportPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="history/:analysisId" element={<HistoryPage />} />
        <Route path="reports" element={<ReportsPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="overview" element={<Navigate to="/" replace />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
