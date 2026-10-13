import { motion } from "framer-motion";
import { AlertTriangle, HelpCircle, Info, Loader2, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { clsx } from "clsx";
import { NOT_AVAILABLE } from "../services/format";
import { useToast } from "./AppProviders";
export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`card ${className}`}>{children}</div>;
}
export function CardHeader({
  title,
  subtitle,
  actions,
  tooltip,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  tooltip?: ReactNode;
}) {
  return (
    <div className="card-header">
      <div className="min-w-0">
        <div className="flex items-center gap-1.5">
          <h3 className="truncate text-sm font-semibold text-clinical-900 dark:text-clinical-50">
            {title}
          </h3>
          {tooltip ? <Tooltip text={tooltip} /> : null}
        </div>
        {subtitle ? (
          <p className="mt-0.5 text-xs text-clinical-500 dark:text-clinical-400">
            {subtitle}
          </p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex shrink-0 items-center gap-2">{actions}</div>
      ) : null}
    </div>
  );
}
export function CardBody({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`card-body ${className}`}>{children}</div>;
}
export function Tooltip({
  text,
  children,
}: {
  text: ReactNode;
  children?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={`Explanation: ${text}`}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((value) => !value)}
        className="inline-flex h-4 w-4 items-center justify-center rounded-full text-clinical-400 transition hover:text-clinical-600 dark:hover:text-clinical-200"
      >
        {children ?? <HelpCircle className="h-3.5 w-3.5" />}
      </button>
      {open ? (
        <span
          role="tooltip"
          className="absolute bottom-full left-1/2 z-40 mb-2 w-64 -translate-x-1/2 rounded-lg border border-clinical-200 bg-white p-2.5 text-xs font-normal leading-relaxed text-clinical-700 shadow-lg dark:border-clinical-700 dark:bg-clinical-900 dark:text-clinical-200"
        >
          {text}
        </span>
      ) : null}
    </span>
  );
}
export function StatusBadge({
  value,
  label,
}: {
  value: string;
  label?: string;
}) {
  const variant =
    value === "responsive" ||
    value === "ready" ||
    value === "completed" ||
    value === "ok"
      ? "badge-teal"
      : value === "failed" ||
          value === "error" ||
          value === "saturated" ||
          value === "degenerate"
        ? "badge-rose"
        : value === "no_foreground" ||
            value === "missing" ||
            value === "incompatible" ||
            value === "partial"
          ? "badge-amber"
          : "badge-neutral";
  return <span className={variant}>{(label ?? value).replace(/_/g, " ")}</span>;
}
export function Pill({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "teal" | "amber" | "rose" | "violet";
}) {
  const map = {
    neutral: "badge-neutral",
    teal: "badge-teal",
    amber: "badge-amber",
    rose: "badge-rose",
    violet: "badge-violet",
  };
  return <span className={map[tone]}>{children}</span>;
}
export function MetricCard({
  label,
  value,
  unit,
  tooltip,
  hint,
  tone = "default",
}: {
  label: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  tooltip?: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  const toneClass =
    tone === "good"
      ? "text-emerald-600 dark:text-emerald-400"
      : tone === "warn"
        ? "text-amber-600 dark:text-amber-400"
        : tone === "bad"
          ? "text-rose-600 dark:text-rose-400"
          : "text-clinical-900 dark:text-clinical-50";
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      className="card p-4"
    >
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-medium uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
          {label}
        </span>
        {tooltip ? <Tooltip text={tooltip} /> : null}
      </div>
      <div
        className={`mt-1.5 text-2xl font-semibold tabular-nums ${toneClass}`}
      >
        {value}
        {unit ? (
          <span className="ml-1 text-sm font-normal text-clinical-500 dark:text-clinical-400">
            {unit}
          </span>
        ) : null}
      </div>
      {hint ? (
        <div className="mt-1 text-xs text-clinical-500 dark:text-clinical-400">
          {hint}
        </div>
      ) : null}
    </motion.div>
  );
}
export function ValueOrUnavailable({
  value,
  reason,
}: {
  value?: ReactNode;
  reason?: ReactNode;
}) {
  if (
    value === null ||
    value === undefined ||
    value === "" ||
    value === NOT_AVAILABLE
  ) {
    return (
      <span className="inline-flex items-center gap-1 text-clinical-400 dark:text-clinical-500">
        {reason ? (
          <Tooltip text={reason}>
            <AlertTriangle className="h-3.5 w-3.5" />
          </Tooltip>
        ) : null}
        <span>{NOT_AVAILABLE}</span>
      </span>
    );
  }
  return <>{value}</>;
}
export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}
export function SkeletonCard() {
  return (
    <div className="card p-4">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="mt-3 h-7 w-32" />
      <Skeleton className="mt-2 h-3 w-40" />
    </div>
  );
}
export function LoadingBlock({ label = "Loading" }: { label?: string }) {
  return (
    <div
      className="flex items-center gap-2 py-6 text-sm text-clinical-500 dark:text-clinical-400"
      role="status"
    >
      <Loader2 className="h-4 w-4 animate-spin" />
      {label}…
    </div>
  );
}
export function EmptyState({
  title,
  message,
  action,
  icon,
}: {
  title: ReactNode;
  message: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-clinical-300 px-6 py-10 text-center dark:border-clinical-700">
      <div className="text-clinical-400 dark:text-clinical-500">
        {icon ?? <Info className="h-6 w-6" />}
      </div>
      <h4 className="text-sm font-semibold text-clinical-800 dark:text-clinical-100">
        {title}
      </h4>
      <div className="max-w-xl text-xs leading-relaxed text-clinical-500 dark:text-clinical-400">
        {message}
      </div>
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}
export function ErrorState({
  title = "Something went wrong",
  message,
  action,
}: {
  title?: ReactNode;
  message: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-rose-200 bg-rose-50/60 p-4 dark:border-rose-900/60 dark:bg-rose-950/30">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600 dark:text-rose-400" />
        <div className="min-w-0">
          <h4 className="text-sm font-semibold text-rose-800 dark:text-rose-200">
            {title}
          </h4>
          <div className="mt-1 text-xs leading-relaxed text-rose-700 dark:text-rose-300">
            {message}
          </div>
          {action ? <div className="mt-2">{action}</div> : null}
        </div>
      </div>
    </div>
  );
}
export function InfoNote({
  children,
  tone = "info",
}: {
  children: ReactNode;
  tone?: "info" | "warn" | "neutral";
}) {
  const styles =
    tone === "warn"
      ? "border-amber-200 bg-amber-50/70 text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200"
      : tone === "neutral"
        ? "border-clinical-200 bg-clinical-50 text-clinical-700 dark:border-clinical-800 dark:bg-clinical-900/60 dark:text-clinical-300"
        : "border-clinical-200 bg-white text-clinical-700 dark:border-clinical-800 dark:bg-clinical-900/60 dark:text-clinical-300";
  return (
    <div className={`rounded-lg border p-3 text-xs leading-relaxed ${styles}`}>
      {children}
    </div>
  );
}
export function DisclaimerBanner({ text }: { text: ReactNode }) {
  return (
    <div className="rounded-lg border border-amber-300/70 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-900 dark:border-amber-800/70 dark:bg-amber-950/40 dark:text-amber-200">
      {text}
    </div>
  );
}
export function Toaster() {
  const { toasts, dismiss } = useToast();
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
      {toasts.map((toast) => {
        const tone =
          toast.kind === "error"
            ? "border-rose-300 bg-rose-50 dark:border-rose-900 dark:bg-rose-950"
            : toast.kind === "warning"
              ? "border-amber-300 bg-amber-50 dark:border-amber-900 dark:bg-amber-950"
              : toast.kind === "success"
                ? "border-emerald-300 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950"
                : "border-clinical-300 bg-white dark:border-clinical-700 dark:bg-clinical-900";
        return (
          <motion.div
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0 }}
            className={`pointer-events-auto flex items-start gap-2 rounded-lg border p-3 shadow-lg ${tone}`}
            role="status"
            key={toast.id}
          >
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-clinical-900 dark:text-clinical-50">
                {toast.title}
              </p>
              {toast.message ? (
                <p className="mt-0.5 text-xs text-clinical-600 dark:text-clinical-300">
                  {toast.message}
                </p>
              ) : null}
            </div>
            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              className="rounded p-0.5 text-clinical-400 hover:bg-black/5"
              aria-label="Dismiss"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </motion.div>
        );
      })}
    </div>
  );
}
export function DataTable({
  columns,
  rows,
  empty,
  dense = false,
}: {
  columns: string[];
  rows: ReactNode[][];
  empty?: ReactNode;
  dense?: boolean;
}) {
  if (!rows.length) {
    return (
      <div className="py-4 text-xs text-clinical-500 dark:text-clinical-400">
        {empty ?? "No rows to display."}
      </div>
    );
  }
  return (
    <div className="-mx-1 overflow-x-auto">
      <table className={`table-base ${dense ? "text-xs" : ""}`}>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              className="hover:bg-clinical-50/60 dark:hover:bg-clinical-800/40"
              key={index}
            >
              {row.map((cell, cellIndex) => (
                <td
                  className={cellIndex === 0 ? "font-medium" : "tabular-nums"}
                  key={cellIndex}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: Array<{ id: string; label: ReactNode; badge?: ReactNode }>;
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div
      className="flex flex-wrap gap-1 border-b border-clinical-200 dark:border-clinical-800"
      role="tablist"
    >
      {tabs.map((tab) => {
        const selected = tab.id === active;
        return (
          <button
            role="tab"
            aria-selected={selected}
            type="button"
            onClick={() => onChange(tab.id)}
            className={`-mb-px inline-flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition ${
              selected
                ? "border-clinical-600 text-clinical-900 dark:border-clinical-300 dark:text-clinical-50"
                : "border-transparent text-clinical-500 hover:text-clinical-800 dark:text-clinical-400 dark:hover:text-clinical-100"
            }`}
            key={tab.id}
          >
            {tab.label}
            {tab.badge}
          </button>
        );
      })}
    </div>
  );
}
export { clsx };
