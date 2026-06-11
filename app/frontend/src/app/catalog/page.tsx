"use client";

import { useEffect, useState } from "react";
import { api, DocumentSummary } from "@/lib/api";

export default function Catalog() {
  const [docs, setDocs] = useState<DocumentSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.documents().then(setDocs).catch((e) => setErr(String(e)));
  }, []);

  const totalPages = docs?.reduce((n, d) => n + d.n_pages, 0) ?? 0;

  return (
    <div className="flex-1 overflow-auto">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight">Document Catalog</h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-sepia">
            GUIRAD documents to view their layers, transcriptions and detected marks
          </p>
          {docs && docs.length > 0 && (
            <div className="mt-4 flex gap-2 text-xs text-sepia">
              <span className="chip border-line bg-surface">{docs.length} documents</span>
              <span className="chip border-line bg-surface">{totalPages} pages</span>
            </div>
          )}
        </header>

        {err && (
          <div className="card border-accent/40 p-4 text-sm text-accent">Backend Not Reachable: {err}</div>
        )}
        {!docs && !err && <SkeletonGrid />}
        {docs && docs.length === 0 && (
          <div className="card p-8 text-center text-sm text-sepia">
            No indexed documents yet. Run the indexing pipeline first
            <code className="ml-1 rounded bg-ink/60 px-1.5 py-0.5 font-mono text-paper">
              scripts/indexing/build_index.py
            </code>
            .
          </div>
        )}

        <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {docs?.map((d) => (
            <li
              key={d.document_id}
              className="card p-4 transition-transform duration-150 hover:-translate-y-0.5 hover:border-accent/50"
            >
              <div className="flex items-center gap-3">
                <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-accent-soft font-serif text-sm font-bold text-paper">
                  {d.document_id.replace(/[^0-9]/g, "").slice(-2) || "·"}
                </span>
                <div className="min-w-0">
                  <p className="truncate font-medium" title={d.document_id}>
                    {d.document_id}
                  </p>
                  <p className="text-xs text-sepia">{d.n_pages} pages</p>
                </div>
              </div>
              <div className="mt-3 flex flex-wrap gap-1">
                {d.pages.slice(0, 14).map((p) => (
                  <a
                    key={p}
                    href={`/documents/${encodeURIComponent(d.document_id)}/${encodeURIComponent(p)}`}
                    className="rounded-md bg-ink/50 px-2 py-0.5 font-mono text-[11px] text-sepia transition-colors hover:bg-accent/30 hover:text-paper"
                  >
                    {p.split("_").pop()}
                  </a>
                ))}
                {d.pages.length > 14 && (
                  <span className="px-1 text-[11px] text-sepia">+{d.pages.length - 14}</span>
                )}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function SkeletonGrid() {
  return (
    <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <li key={i} className="card h-28 animate-pulse p-4">
          <div className="h-9 w-9 rounded-lg bg-line" />
          <div className="mt-3 h-3 w-2/3 rounded bg-line" />
        </li>
      ))}
    </ul>
  );
}
