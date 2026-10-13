import { motion } from "framer-motion";
import { Download, Maximize2, X } from "lucide-react";
import { useState } from "react";
import { clsx } from "clsx";

export interface ImagePanel {
  src?: string;
  caption: string;
  note?: string;
  downloadName?: string;
  tone?: keyof typeof toneRing;
}
const toneRing = {
  input: "ring-clinical-300 dark:ring-clinical-700",
  mask: "ring-teal-300 dark:ring-teal-800",
  overlay: "ring-rose-300 dark:ring-rose-800",
  heatmap: "ring-amber-300 dark:ring-amber-800",
  uncertainty: "ring-violet-300 dark:ring-violet-800",
  neutral: "ring-clinical-200 dark:ring-clinical-800",
};
export function ImagePanelView({
  panel,
  onExpand,
}: {
  panel: ImagePanel;
  onExpand?: (panel: ImagePanel) => void;
}) {
  return (
    <motion.figure
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="group flex flex-col gap-2"
    >
      <div
        className={clsx(
          "relative aspect-square w-full overflow-hidden rounded-lg bg-clinical-950 ring-1",
          toneRing[panel.tone ?? "neutral"],
        )}
      >
        {panel.src ? (
          <img
            src={panel.src}
            alt={panel.caption}
            className="h-full w-full object-contain"
            loading="lazy"
            style={{
              imageRendering: panel.tone === "mask" ? "pixelated" : "auto",
            }}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center px-4 text-center text-xs text-clinical-400">
            Image not available
          </div>
        )}
        <div className="absolute right-1.5 top-1.5 flex gap-1 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
          {onExpand && panel.src ? (
            <button
              type="button"
              onClick={() => onExpand(panel)}
              className="rounded bg-black/60 p-1 text-white hover:bg-black/80"
              aria-label={`Expand ${panel.caption}`}
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          ) : null}
          {panel.src && panel.downloadName ? (
            <a
              href={panel.src}
              download={panel.downloadName}
              className="rounded bg-black/60 p-1 text-white hover:bg-black/80"
              aria-label={`Download ${panel.caption}`}
            >
              <Download className="h-3.5 w-3.5" />
            </a>
          ) : null}
        </div>
      </div>
      <figcaption className="text-xs leading-snug text-clinical-600 dark:text-clinical-300">
        <span className="font-medium text-clinical-800 dark:text-clinical-100">
          {panel.caption}
        </span>
        {panel.note ? (
          <span className="mt-0.5 block text-[11px] text-clinical-500 dark:text-clinical-400">
            {panel.note}
          </span>
        ) : null}
      </figcaption>
    </motion.figure>
  );
}
export function ImageGrid({
  panels,
  columns = 3,
}: {
  panels: ImagePanel[];
  columns?: 2 | 3 | 4;
}) {
  const [expanded, setExpanded] = useState<ImagePanel | null>(null);
  const grid = clsx(
    "grid gap-4",
    columns === 2 && "grid-cols-1 sm:grid-cols-2",
    columns === 3 && "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
    columns === 4 && "grid-cols-2 lg:grid-cols-4",
  );
  return (
    <>
      <div className={grid}>
        {panels.map((panel, index) => (
          <ImagePanelView
            panel={panel}
            onExpand={setExpanded}
            key={`${panel.caption}-${index}`}
          />
        ))}
      </div>
      {expanded ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          role="dialog"
          aria-modal="true"
          onClick={() => setExpanded(null)}
        >
          <div
            className="max-h-full w-full max-w-4xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-2 flex items-center justify-between text-white">
              <span className="text-sm font-medium">{expanded.caption}</span>
              <button
                type="button"
                onClick={() => setExpanded(null)}
                aria-label="Close"
                className="rounded p-1 hover:bg-white/10"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            {expanded.src ? (
              <img
                src={expanded.src}
                alt={expanded.caption}
                className="mx-auto max-h-[80vh] rounded-lg"
              />
            ) : null}
            {expanded.note ? (
              <p className="mt-2 text-center text-xs text-white/70">
                {expanded.note}
              </p>
            ) : null}
          </div>
        </div>
      ) : null}
    </>
  );
}
export function ChannelStrip({
  channels,
}: {
  channels: Array<{ label: string; src?: string | null }>;
}) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {channels.map((channel) => (
        <div className="flex flex-col gap-1.5" key={channel.label}>
          <div className="aspect-square overflow-hidden rounded-md bg-clinical-950 ring-1 ring-clinical-200 dark:ring-clinical-800">
            {channel.src ? (
              <img
                src={channel.src}
                alt={channel.label}
                className="h-full w-full object-contain"
              />
            ) : null}
          </div>
          <span className="text-center text-[11px] font-medium uppercase tracking-wide text-clinical-500 dark:text-clinical-400">
            {channel.label}
          </span>
        </div>
      ))}
    </div>
  );
}
