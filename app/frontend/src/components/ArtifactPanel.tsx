"use client";

import { useEffect, useState } from "react";
import { api, PageDetail } from "@/lib/api";
import DocumentViewer from "./DocumentViewer";
import Transcription from "./Transcription";

type Tab = "document" | "heatmap" | "transcription";

const TABS: { id: Tab; label: string }[] = [
  { id: "document", label: "Document" },
  { id: "heatmap", label: "Heatmap" },
  { id: "transcription", label: "Transcription" },
];

/**
 * The right-hand artifact: loads one page and shows it as a layered document,
 * an uncertainty heatmap, or a clean transcription. Used inside the assistant
 * (with onClose) and standalone in the document route (without).
 */
export default function ArtifactPanel({
  doc,
  page,
  onClose,
}: {
  doc: string;
  page: string;
  onClose?: () => void;
}) {
  const [detail, setDetail] = useState<PageDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("document");

  useEffect(() => {
    setDetail(null);
    setErr(null);
    api.page(doc, page).then(setDetail).catch((e) => setErr(String(e)));
  }, [doc, page]);

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-stage">
      {/* header */}
      <div className="flex items-center gap-3 border-b border-transparent px-4 py-2.5 bg-stage">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-ink" title={page}>
            {page}
          </p>
          <p className="truncate text-[11px] text-ink/50" title={doc}>
            {doc}
          </p>
        </div>
        <div className="ml-auto flex rounded-md bg-black/30 p-1 ring-1 ring-line-dark">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`toolbtn rounded ${tab === t.id ? "bg-accent text-ink shadow-sm" : "text-ink/60 hover:bg-black/20 hover:text-ink/90"}`}
            >
              {t.label}
            </button>
          ))}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            aria-label="close artifact"
            className="toolbtn border border-line-dark text-ink/60 hover:bg-black/20 hover:text-ink/90"
          >
            ✕
          </button>
        )}
      </div>

      {/* body */}
      {err && <p className="p-6 text-sm text-accent">Could not load page: {err}</p>}
      {!detail && !err && <p className="p-6 text-sm text-ink/40">Loading page…</p>}
      {detail && tab === "document" && <DocumentViewer doc={doc} detail={detail} />}
      {detail && tab === "transcription" && <Transcription detail={detail} />}
      {detail && tab === "heatmap" && <HeatmapTab detail={detail} />}
    </div>
  );
}

function HeatmapTab({ detail }: { detail: PageDetail }) {
  const url = api.heatmapUrl(detail.document_id, detail.page_id);
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-auto bg-stage p-4">
      {/* clean heatmap image - rendered from .conf.npy without matplotlib chrome */}
      <div className="mx-auto inline-block">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={url}
          alt={`${detail.page_id} uncertainty heatmap`}
          className="block max-w-full rounded-lg shadow-card"
        />
      </div>

      {/* inline colour legend (CSS gradient, no image) */}
      <div className="mx-auto mt-4 flex w-full max-w-sm flex-col gap-1">
        <div
          className="h-3 w-full rounded-full"
          style={{
            background: "linear-gradient(to right, #440154, #31688e, #35b779, #fde725)",
          }}
        />
        <div className="flex justify-between text-[10px] text-ink/50">
          <span>Confident</span>
          <span>Uncertain</span>
        </div>
      </div>
    </div>
  );
}
