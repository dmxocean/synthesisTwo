"use client";

import { useState } from "react";
import { PageDetail, RegionView, MarkView, WordConfidence } from "@/lib/api";
import { confColor, confBg } from "@/lib/ui";

/**
 * Clean reading-order transcription: printed and handwritten shown as separate
 * sections assembled in reading order, with optional per-word confidence
 * colouring, a copy button per section, and a compact marks summary.
 */
export default function Transcription({ detail }: { detail: PageDetail }) {
  const [colorByConf, setColorByConf] = useState(true);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-auto bg-stage">
      <div className="flex items-center justify-between gap-3 border-b border-transparent px-5 py-3 bg-black/10">
        <label className="flex items-center gap-2 text-xs text-ink/60">
          <input
            type="checkbox"
            checked={colorByConf}
            onChange={(e) => setColorByConf(e.target.checked)}
            className="accent-accent"
          />
          colour words by confidence
        </label>
        <div className="flex items-center gap-2 text-[11px] text-ink/40">
          {[
            ["bg-moss", "high"],
            ["bg-warn", "medium"],
            ["bg-error", "low"],
          ].map(([c, label]) => (
            <span key={label} className="flex items-center gap-1">
              <span className={`inline-block h-2.5 w-2.5 rounded-sm ${c}`} />
              {label}
            </span>
          ))}
        </div>
      </div>

      <Section
        title="Printed"
        regions={detail.printed_regions}
        fallback={detail.printed_text}
        colorByConf={colorByConf}
      />
      <Section
        title="Handwritten"
        regions={detail.handwritten_regions}
        fallback={detail.handwritten_text}
        colorByConf={colorByConf}
      />
      <MarksSummary marks={detail.marks} />
    </div>
  );
}

function plainText(regions: RegionView[], fallback: string): string {
  const ordered = [...regions].sort((a, b) => a.reading_order - b.reading_order);
  const joined = ordered.map((r) => r.text).filter(Boolean).join("\n");
  return joined || fallback || "";
}

function Section({
  title,
  regions,
  fallback,
  colorByConf,
}: {
  title: string;
  regions: RegionView[];
  fallback: string;
  colorByConf: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const ordered = [...regions].sort((a, b) => a.reading_order - b.reading_order);
  const text = plainText(regions, fallback);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <div className="border-b border-transparent px-5 py-4">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-ink/40">{title}</p>
        <div className="flex items-center gap-3">
          <span className="text-[11px] text-ink/30">{ordered.length} regions</span>
          {text && (
            <button
              onClick={copy}
              className="text-[11px] text-ink/60 underline-offset-4 hover:text-ink hover:underline"
            >
              {copied ? "copied" : "copy"}
            </button>
          )}
        </div>
      </div>

      {ordered.length === 0 && !fallback && <p className="text-sm text-ink/20">none</p>}
      {ordered.length === 0 && fallback && <p className="text-sm leading-relaxed text-ink/80">{fallback}</p>}

      <div className="space-y-2">
        {ordered.map((r) => (
          <div key={r.region_id} className="flex gap-2.5 text-sm leading-relaxed text-ink/90">
            <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded bg-black/20 font-mono text-[10px] text-ink/60">
              {r.reading_order}
            </span>
            <p className="min-w-0">
              <span className={`mr-2 font-mono text-[10px] ${confColor(r.confidence)} opacity-80`}>
                {Math.round(r.confidence * 100)}%
              </span>
              <RegionText region={r} colorByConf={colorByConf} />
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

/** One region's text, optionally colouring each word by its own confidence. */
function RegionText({ region, colorByConf }: { region: RegionView; colorByConf: boolean }) {
  const words: WordConfidence[] = region.word_confidences ?? [];
  if (!colorByConf || words.length === 0) {
    return region.text ? <>{region.text}</> : <em className="text-sepia/50">empty</em>;
  }
  return (
    <>
      {words.map((w, i) => (
        <span key={i} className={confColor(w.confidence)} title={`${Math.round(w.confidence * 100)}%`}>
          {w.text}
          {i < words.length - 1 ? " " : ""}
        </span>
      ))}
    </>
  );
}

function MarksSummary({ marks }: { marks: MarkView[] }) {
  return (
    <div className="px-5 py-4">
      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-ink/40">Marks ({marks.length})</p>
      {marks.length === 0 && <p className="text-sm text-ink/20">none</p>}
      <div className="flex flex-wrap gap-2">
        {marks.map((m) => (
          <span key={m.mark_id} title={`bbox ${m.bbox.join(", ")}`} className={`chip border-transparent ${confBg(m.confidence)}`}>
            {m.mark_type} · {Math.round(m.confidence * 100)}%
          </span>
        ))}
      </div>
    </div>
  );
}
