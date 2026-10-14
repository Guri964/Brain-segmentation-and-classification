export const NOT_AVAILABLE = "Not available";
export const NOT_EVALUATED = "Not evaluated";

export function formatNumber(
  value: number | null | undefined,
  digits = 4,
): string {
  if (value === null || value === undefined || Number.isNaN(value))
    return NOT_AVAILABLE;
  if (!Number.isFinite(value)) return NOT_AVAILABLE;
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e5 || magnitude < 1e-4) return value.toExponential(3);
  return Number(value.toFixed(digits)).toString();
}

export function formatMetric(
  value: number | null | undefined,
  digits = 4,
): string {
  if (value === null || value === undefined || Number.isNaN(value))
    return NOT_EVALUATED;
  return formatNumber(value, digits);
}

export function formatPercent(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || Number.isNaN(value))
    return NOT_AVAILABLE;
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value))
    return NOT_AVAILABLE;
  return value.toLocaleString("en-US");
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return NOT_AVAILABLE;
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let size = bytes / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(1)} ${units[index]}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return NOT_AVAILABLE;
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${(seconds - minutes * 60).toFixed(0)} s`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return NOT_AVAILABLE;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function humaniseFeature(name: string): string {
  const [prefix, ...rest] = name.split("_");
  const labels: Record<string, string> = {
    shape: "Shape",
    firstorder: "Intensity",
    glcm: "GLCM",
    glrlm: "Run-length",
    diagnostics: "Diagnostics",
  };
  const body = rest
    .join(" ")
    .replace(/\bmm3\b/, "volume (mm3)")
    .replace(/\bmm2\b/, "area (mm2)")
    .replace(/\bpx\b/, "pixels")
    .replace(/\bstd\b/, "standard deviation");
  return rest.length
    ? `${body.charAt(0).toUpperCase()}${body.slice(1)} · ${labels[prefix] ?? prefix}`
    : name;
}

export function humanise(value: string | null | undefined): string {
  if (!value) return NOT_AVAILABLE;
  return value.replace(/_/g, " ");
}

export function truncate(value: string, length = 90): string {
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

export function cosineClass(value: string): string {
  switch (value) {
    case "responsive":
    case "ready":
    case "completed":
    case "ok":
      return "text-emerald-600 dark:text-emerald-400";
    case "no_foreground":
    case "missing":
    case "incompatible":
    case "partial":
    case "not_loaded":
      return "text-amber-600 dark:text-amber-400";
    case "failed":
    case "error":
    case "saturated":
    case "degenerate":
    case "unsupported_container":
      return "text-rose-600 dark:text-rose-400";
    default:
      return "text-clinical-600 dark:text-clinical-300";
  }
}
