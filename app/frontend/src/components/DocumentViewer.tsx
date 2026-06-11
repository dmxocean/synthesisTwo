"use client";

import { useMemo, useState } from "react";
import { PageDetail } from "@/lib/api";
import { api } from "@/lib/api";
import { confBorder, COMPOSITE_LAYERS, SWATCHES, LayerSpec } from "@/lib/ui";

interface LayerState {
  spec: LayerSpec;
  visible: boolean;
  color: string;
  opacity: number;
  url: string;
}

const ZOOM_MIN = 0.25;
const ZOOM_MAX = 3.0;
const ZOOM_STEP = 0.25;

export default function DocumentViewer({ doc, detail }: { doc: string; detail: PageDetail }) {
  const initial: LayerState[] = useMemo(
    () =>
      COMPOSITE_LAYERS.filter((s) => detail.layers[s.id]).map((spec) => ({
        spec,
        visible: true,
        color: spec.color,
        opacity: 0.85,
        url: detail.layers[spec.id],
      })),
    [detail.layers]
  );

  const [layers, setLayers] = useState<LayerState[]>(initial);
  const [rawVisible, setRawVisible] = useState(true);
  const [rawOpacity, setRawOpacity] = useState(1.0);
  const [showBoxes, setShowBoxes] = useState(false);
  const [zoom, setZoom] = useState(1.0);
  // Native page dimensions (300-DPI mask space) - the coordinate space of region bboxes.
  const [pageNat, setPageNat] = useState<{ w: number; h: number } | null>(null);

  const baseUrl = api.pageImageUrl(doc, detail.page_id);
  const printedRegions = detail.printed_regions.filter((r) => r.bbox && r.bbox.length === 4);
  const handwrittenRegions = detail.handwritten_regions.filter((r) => r.bbox && r.bbox.length === 4);
  const marks = (detail.marks ?? []).filter((m) => m.bbox && m.bbox.length === 4);

  function patch(i: number, p: Partial<LayerState>) {
    setLayers((ls) => ls.map((l, idx) => (idx === i ? { ...l, ...p } : l)));
  }

  function clampZoom(v: number) {
    return Math.round(Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v)) / ZOOM_STEP) * ZOOM_STEP;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-stage">
      {/* controls */}
      <div className="flex flex-col gap-3 border-b border-transparent px-4 py-3 bg-stage">

        {/* top bar: boxes toggle + zoom + PDF link */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setShowBoxes((v) => !v)}
            className={`toolbtn rounded ${
              showBoxes ? "bg-accent/20 text-accent" : "text-ink/60 hover:bg-black/20 hover:text-ink"
            }`}
          >
            Region boxes
          </button>

          {/* zoom controls */}
          <div className="ml-auto flex items-center gap-1">
            <button
              onClick={() => setZoom((z) => clampZoom(z - ZOOM_STEP))}
              disabled={zoom <= ZOOM_MIN}
              className="toolbtn bg-black/20 text-ink/60 hover:text-ink disabled:opacity-30"
            >
              −
            </button>
            <span className="min-w-[3.5rem] text-center font-mono text-xs text-ink/60">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom((z) => clampZoom(z + ZOOM_STEP))}
              disabled={zoom >= ZOOM_MAX}
              className="toolbtn bg-black/20 text-ink/60 hover:text-ink disabled:opacity-30"
            >
              +
            </button>
          </div>

          <a
            href={api.pdfUrl(doc)}
            target="_blank"
            className="text-xs text-ink/40 underline-offset-4 hover:text-accent hover:underline"
          >
            PDF ↗
          </a>
        </div>

        {/* layer cards grid - 4 cards: PR, HW, NS, Raw */}
        <div className="grid gap-2 sm:grid-cols-4">
          {/* tintable mask layers */}
          {layers.map((l, i) => (
            <div key={l.spec.id} className="rounded bg-black/20 p-2 shadow-sm">
              <div className="flex items-center justify-between gap-2">
                <button
                  onClick={() => patch(i, { visible: !l.visible })}
                  className="flex items-center gap-2 text-xs font-medium"
                >
                  <span
                    className={`inline-block h-3 w-3 rounded-sm ring-1 ring-line-dark ${l.visible ? "" : "opacity-30"}`}
                    style={{ backgroundColor: l.color }}
                  />
                  <span className={l.visible ? "text-ink/90" : "text-ink/30 line-through"}>{l.spec.label}</span>
                </button>
                <div className="flex gap-1">
                  {SWATCHES.map((c) => (
                    <button
                      key={c}
                      aria-label={`color ${c}`}
                      onClick={() => patch(i, { color: c, visible: true })}
                      className={`h-3 w-3 rounded-full ring-1 transition-transform hover:scale-110 ${
                        l.color === c ? "ring-white/40" : "ring-line-dark"
                      }`}
                      style={{ backgroundColor: c }}
                    />
                  ))}
                </div>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={l.opacity}
                onChange={(e) => patch(i, { opacity: Number(e.target.value), visible: true })}
                className="mt-2 w-full accent-accent"
              />
            </div>
          ))}

          {/* Raw base image card */}
          <div className="rounded bg-black/20 p-2 shadow-sm">
            <div className="flex items-center gap-2 text-xs font-medium">
              <button
                onClick={() => setRawVisible((v) => !v)}
                className="flex items-center gap-2"
              >
                <span
                  className={`inline-block h-3 w-3 rounded-sm bg-ink/20 ring-1 ring-line-dark ${rawVisible ? "" : "opacity-30"}`}
                />
                <span className={rawVisible ? "text-ink/90" : "text-ink/30 line-through"}>Raw</span>
              </button>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={rawOpacity}
              onChange={(e) => { setRawOpacity(Number(e.target.value)); setRawVisible(true); }}
              className="mt-2 w-full accent-accent"
            />
          </div>
        </div>
      </div>

      {/* stage */}
      <div className="flex-1 overflow-auto bg-stage p-4">
        {/* zoom container - width drives the zoom; absolute children scale with it */}
        <div
          className="relative mx-auto inline-block"
          style={{ width: `${zoom * 100}%` }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={baseUrl}
            alt={`${detail.page_id} raw`}
            className="block w-full rounded-lg shadow-card"
            style={{ opacity: rawVisible ? rawOpacity : 0 }}
          />

          {/* hidden probe: read the native page dimensions from a mask (bbox coordinate space) */}
          {layers[0] && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={layers[0].url}
              alt=""
              aria-hidden
              className="hidden"
              onLoad={(e) => setPageNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
            />
          )}

          {/* tinted mask layers */}
          {layers.map(
            (l) =>
              l.visible && (
                <div
                  key={l.spec.id}
                  aria-hidden
                  className="pointer-events-none absolute inset-0 rounded-lg"
                  style={{
                    backgroundColor: l.color,
                    opacity: l.opacity,
                    mixBlendMode: "screen",
                    WebkitMaskImage: `url(${l.url})`,
                    maskImage: `url(${l.url})`,
                    WebkitMaskSize: "100% 100%",
                    maskSize: "100% 100%",
                    WebkitMaskRepeat: "no-repeat",
                    maskRepeat: "no-repeat",
                    maskMode: "luminance",
                  }}
                />
              )
          )}

          {/* region boxes (normalized by native page size, not the lower-DPI base) */}
          {showBoxes && pageNat && (
            <>
              {/* PR boxes - solid border, confidence-colored */}
              {printedRegions.map((r) => {
                const [x, y, w, h] = r.bbox;
                return (
                  <div
                    key={r.region_id}
                    title={`PR·${r.reading_order} · ${Math.round(r.confidence * 100)}% · ${r.text?.slice(0, 60)}`}
                    className={`pointer-events-none absolute border-2 ${confBorder(r.confidence)}`}
                    style={{
                      left: `${(x / pageNat.w) * 100}%`,
                      top: `${(y / pageNat.h) * 100}%`,
                      width: `${(w / pageNat.w) * 100}%`,
                      height: `${(h / pageNat.h) * 100}%`,
                    }}
                  >
                    <span className="absolute left-0 top-0 rounded-br bg-highlight/70 px-1 py-px font-mono text-[9px] leading-none text-ink">
                      PR·{r.reading_order}
                    </span>
                  </div>
                );
              })}

              {/* HW boxes - dashed border, confidence-colored */}
              {handwrittenRegions.map((r) => {
                const [x, y, w, h] = r.bbox;
                return (
                  <div
                    key={r.region_id}
                    title={`HW·${r.reading_order} · ${Math.round(r.confidence * 100)}% · ${r.text?.slice(0, 60)}`}
                    className={`pointer-events-none absolute border-2 border-dashed ${confBorder(r.confidence)}`}
                    style={{
                      left: `${(x / pageNat.w) * 100}%`,
                      top: `${(y / pageNat.h) * 100}%`,
                      width: `${(w / pageNat.w) * 100}%`,
                      height: `${(h / pageNat.h) * 100}%`,
                    }}
                  >
                    <span className="absolute left-0 top-0 rounded-br bg-highlight/70 px-1 py-px font-mono text-[9px] leading-none text-ink">
                      HW·{r.reading_order}
                    </span>
                  </div>
                );
              })}

              {/* NS mark boxes - dotted sepia border */}
              {marks.map((m) => {
                const [x, y, w, h] = m.bbox;
                return (
                  <div
                    key={m.mark_id}
                    title={`NS · ${m.mark_type} · ${Math.round(m.confidence * 100)}%`}
                    className="pointer-events-none absolute border-2 border-dotted border-sepia/70"
                    style={{
                      left: `${(x / pageNat.w) * 100}%`,
                      top: `${(y / pageNat.h) * 100}%`,
                      width: `${(w / pageNat.w) * 100}%`,
                      height: `${(h / pageNat.h) * 100}%`,
                    }}
                  >
                    <span className="absolute left-0 top-0 rounded-br bg-sepia/70 px-1 py-px font-mono text-[9px] leading-none text-ink">
                      {m.mark_type}
                    </span>
                  </div>
                );
              })}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
