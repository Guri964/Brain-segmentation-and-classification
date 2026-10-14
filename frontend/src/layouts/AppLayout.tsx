import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  Brain,
  ChevronLeft,
  FileText,
  FlaskConical,
  Gauge,
  History,
  Layers,
  Menu,
  Moon,
  Settings,
  Sparkles,
  Stethoscope,
  Sun,
  Microscope,
  Upload,
  X,
} from "lucide-react";
import { clsx } from "clsx";
import { useTheme } from "../components/AppProviders";
import { DisclaimerBanner, Toaster } from "../components/ui";
import { api } from "../services/api";
import type { Capabilities } from "../types/api";
const NAVIGATION = [
  { to: "/", label: "Overview", icon: Gauge, end: true },
  { to: "/analysis/new", label: "New MRI Analysis", icon: Upload },
  { to: "/analysis/segmentation", label: "Segmentation Results", icon: Layers },
  { to: "/models/compare", label: "Model Comparison", icon: Activity },
  {
    to: "/analysis/radiomics",
    label: "Quantitative Radiomics",
    icon: FlaskConical,
  },
  {
    to: "/analysis/classification",
    label: "Tumour Classification",
    icon: Microscope,
  },
  { to: "/analysis/explainability", label: "Explainable AI", icon: Sparkles },
  {
    to: "/analysis/clinical",
    label: "Clinical Decision Support",
    icon: Stethoscope,
  },
  { to: "/history", label: "Analysis History", icon: History },
  { to: "/reports", label: "Reports", icon: FileText },
  { to: "/evaluation", label: "Model Evaluation", icon: BarChart3 },
  { to: "/settings", label: "Settings", icon: Settings },
];
export default function AppLayout() {
  const { theme, toggle } = useTheme();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  useEffect(() => {
    let cancelled = false;
    api
      .capabilities()
      .then((payload) => {
        if (!cancelled) {
          setCapabilities(payload);
          setBackendDown(false);
        }
      })
      .catch(() => {
        if (!cancelled) setBackendDown(true);
      });
    return () => {
      cancelled = true;
    };
  }, [location.pathname]);
  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);
  const width = collapsed ? "lg:w-[68px]" : "lg:w-64";
  return (
    <div className="flex min-h-screen">
      <aside
        className={clsx(
          "fixed inset-y-0 left-0 z-40 flex flex-col border-r border-clinical-200 bg-white transition-[width,transform] dark:border-clinical-800 dark:bg-clinical-900 lg:static lg:translate-x-0",
          width,
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
          collapsed ? "w-[68px]" : "w-64",
        )}
        aria-label="Primary navigation"
      >
        <div className="flex h-14 items-center justify-between border-b border-clinical-200 px-3 dark:border-clinical-800">
          <div className="flex items-center gap-2 overflow-hidden">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-clinical-600 text-white">
              <Brain className="h-4 w-4" />
            </span>
            {!collapsed ? (
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold leading-tight text-clinical-900 dark:text-clinical-50">
                  Brain Abnormality
                </p>
                <p className="truncate text-[11px] leading-tight text-clinical-500 dark:text-clinical-400">
                  Radiomics research platform
                </p>
              </div>
            ) : null}
          </div>
          <button
            type="button"
            className="rounded p-1 text-clinical-400 hover:bg-clinical-100 dark:hover:bg-clinical-800 lg:hidden"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <nav className="flex-1 overflow-y-auto p-2">
          <ul className="space-y-0.5">
            {NAVIGATION.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    clsx(
                      "group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition",
                      isActive
                        ? "bg-clinical-100 font-medium text-clinical-900 dark:bg-clinical-800 dark:text-clinical-50"
                        : "text-clinical-600 hover:bg-clinical-50 hover:text-clinical-900 dark:text-clinical-300 dark:hover:bg-clinical-800/60 dark:hover:text-clinical-50",
                    )
                  }
                  title={collapsed ? item.label : undefined}
                >
                  <item.icon className="h-4 w-4 shrink-0" />
                  {!collapsed ? (
                    <span className="truncate">{item.label}</span>
                  ) : null}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="border-t border-clinical-200 p-2 dark:border-clinical-800">
          {!collapsed && capabilities ? (
            <div className="mb-2 space-y-1 px-1 text-[11px] text-clinical-500 dark:text-clinical-400">
              <p>
                Models ready:{" "}
                {capabilities.model_health.responsive_models.length}/
                {capabilities.models.length}
              </p>
              <p>Radiomics: {capabilities.radiomics.selected_backend}</p>
              {/* Two classifiers, two lines: the radiomics grade classifier and the Xception image
                  classifier are separate tasks that are enabled independently, so one line saying
                  "classification: disabled" would misdescribe both. */}
              <p>
                Grade classifier (radiomics):{" "}
                {capabilities.classification.available ? "enabled" : "disabled"}
              </p>
              <p>
                Image classifier (Xception):{" "}
                {capabilities.image_classification.available
                  ? "ready"
                  : (capabilities.image_classification.state ?? "unavailable").replace(/_/g, " ")}
              </p>
            </div>
          ) : null}
          <div className="flex items-center gap-1">
            <button
              type="button"
              className="btn-ghost flex-1 justify-start px-2 py-1.5 text-xs"
              onClick={toggle}
              aria-label="Toggle colour theme"
            >
              {theme === "dark" ? (
                <Sun className="h-4 w-4" />
              ) : (
                <Moon className="h-4 w-4" />
              )}
              {!collapsed ? (
                <span>{theme === "dark" ? "Light theme" : "Dark theme"}</span>
              ) : null}
            </button>
            <button
              type="button"
              className="hidden rounded p-1.5 text-clinical-400 hover:bg-clinical-100 dark:hover:bg-clinical-800 lg:block"
              onClick={() => setCollapsed((value) => !value)}
              aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              <ChevronLeft
                className={clsx(
                  "h-4 w-4 transition-transform",
                  collapsed && "rotate-180",
                )}
              />
            </button>
          </div>
        </div>
      </aside>
      {mobileOpen ? (
        <div
          className="fixed inset-0 z-30 bg-black/40 lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      ) : null}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-clinical-200 bg-white/85 px-4 backdrop-blur dark:border-clinical-800 dark:bg-clinical-900/85">
          <button
            type="button"
            className="rounded p-1.5 text-clinical-500 hover:bg-clinical-100 dark:hover:bg-clinical-800 lg:hidden"
            onClick={() => setMobileOpen(true)}
            aria-label="Open navigation"
          >
            <Menu className="h-4 w-4" />
          </button>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-semibold text-clinical-900 dark:text-clinical-50">
              {NAVIGATION.find((item) =>
                item.end
                  ? location.pathname === item.to
                  : location.pathname.startsWith(item.to),
              )?.label ?? "Brain Abnormality Research Platform"}
            </h1>
          </div>
          {backendDown ? (
            <span className="badge-rose">API unreachable</span>
          ) : capabilities ? (
            <span className="badge-teal">
              API connected · {capabilities.radiomics.selected_backend}
            </span>
          ) : (
            <span className="badge-neutral">Connecting…</span>
          )}
        </header>
        {backendDown ? (
          <div className="px-4 pt-4">
            <DisclaimerBanner text="The backend API is not reachable. Start it with: python backend/run.py (the interface will keep retrying automatically)." />
          </div>
        ) : null}
        <main className="min-w-0 flex-1 p-4 lg:p-6">
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.18 }}
            key={location.pathname}
          >
            <Outlet context={{ capabilities }} />
          </motion.div>
        </main>
        <footer className="border-t border-clinical-200 px-4 py-3 text-[11px] leading-relaxed text-clinical-500 dark:border-clinical-800 dark:text-clinical-400">
          Research prototype. Outputs are AI-generated and are not intended to
          replace professional medical diagnosis. Metrics are shown only when
          they were measured; everything else is reported as not available or
          not evaluated.
        </footer>
      </div>
      <Toaster />
    </div>
  );
}
