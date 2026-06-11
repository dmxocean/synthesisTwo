"use client";

import { useEffect, useRef, useState } from "react";
import { api, ChatResponse, ChatProvenance } from "@/lib/api";
import { confBg, confLevelToNum } from "@/lib/ui";

export interface SourceRef {
  doc: string;
  page: string;
}

interface Turn {
  question: string;
  response?: ChatResponse;
  error?: string;
}

const PLAN_LABEL: Record<string, string> = {
  vector: "semantic",
  hybrid: "hybrid (semantic + lexical)",
  metadata_lookup: "metadata filter",
  provenance_lookup: "provenance",
};

const SUGGESTIONS = [
  "pages with censorship stamps over text",
  "which records are uncertain and need human review",
  "documents that mention the República",
  "show the provenance for handwritten corrections",
];

/**
 * Claude-style assistant: a centred empty state, a scrolling conversation, and a
 * bottom composer. Provenance sources are clickable and open the artifact panel.
 */
export default function Chat({
  onOpenSource,
  activeSource,
  resetKey,
}: {
  onOpenSource: (s: SourceRef) => void;
  activeSource: SourceRef | null;
  resetKey: number;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // "New chat" from the nav rail bumps resetKey -> clear the conversation.
  useEffect(() => {
    setTurns([]);
    setQuestion("");
    inputRef.current?.focus();
  }, [resetKey]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  async function ask(q: string) {
    const query = q.trim();
    if (!query || loading) return;
    setQuestion("");
    setLoading(true);
    const idx = turns.length;
    setTurns((t) => [...t, { question: query }]);
    try {
      const response = await api.chat(query);
      setTurns((t) => t.map((turn, i) => (i === idx ? { ...turn, response } : turn)));
      const first = firstSource(response.provenance);
      if (first) onOpenSource(first);
    } catch (e) {
      setTurns((t) => t.map((turn, i) => (i === idx ? { ...turn, error: String(e) } : turn)));
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  const empty = turns.length === 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {empty ? (
        <div className="flex flex-1 flex-col items-center justify-center px-6">
          <h1 className="text-2xl font-semibold tracking-tight">Ask the archive</h1>
          <p className="mt-2 max-w-md text-center text-sm text-sepia">
            Query the Radio Barcelona records for answers that cite the pages they rely on, and open one to inspect its layers, uncertainty, and transcription
          </p>
          <div className="mt-6 w-full max-w-xl">
            <Composer
              value={question}
              setValue={setQuestion}
              onSubmit={() => ask(question)}
              loading={loading}
              inputRef={inputRef}
            />
            <div className="mt-3 flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => ask(s)}
                  className="chip border-line bg-surface text-sepia transition-colors hover:border-accent/50 hover:text-paper"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="flex-1 overflow-auto">
            <div className="mx-auto max-w-2xl space-y-6 px-5 py-6">
              {turns.map((t, i) => (
                <TurnView
                  key={i}
                  turn={t}
                  onOpenSource={onOpenSource}
                  activeSource={activeSource}
                />
              ))}
              {loading && (
                <div className="flex items-center gap-3 text-sepia">
                  <DotsLoader />
                  <span className="text-sm">Searching the archive…</span>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </div>
          <div className="border-t border-line px-5 py-3">
            <div className="mx-auto max-w-2xl">
              <Composer
                value={question}
                setValue={setQuestion}
                onSubmit={() => ask(question)}
                loading={loading}
                inputRef={inputRef}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function firstSource(prov: ChatProvenance[]): SourceRef | null {
  for (const p of prov) {
    if (p.document_id && p.page_id) return { doc: p.document_id, page: p.page_id };
  }
  return null;
}

function Composer({
  value,
  setValue,
  onSubmit,
  loading,
  inputRef,
}: {
  value: string;
  setValue: (v: string) => void;
  onSubmit: () => void;
  loading: boolean;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
  return (
    <div className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 shadow-card focus-within:border-accent/60">
      <textarea
        ref={inputRef}
        value={value}
        rows={1}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
        placeholder="Ask about the archive…"
        className="max-h-40 min-h-[2.25rem] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm placeholder:text-sepia/50 focus:outline-none"
      />
      <button
        onClick={onSubmit}
        disabled={loading || !value.trim()}
        className="toolbtn shrink-0 bg-accent/25 text-paper transition-all hover:bg-accent hover:text-ink disabled:opacity-20"
      >
        {loading ? "…" : "Ask"}
      </button>
    </div>
  );
}

function TurnView({
  turn,
  onOpenSource,
  activeSource,
}: {
  turn: Turn;
  onOpenSource: (s: SourceRef) => void;
  activeSource: SourceRef | null;
}) {
  const r = turn.response;
  return (
    <div className="space-y-3">
      {/* user */}
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-2xl bg-highlight/10 px-3.5 py-2 text-sm text-paper">{turn.question}</p>
      </div>

      {/* assistant */}
      {turn.error && <p className="text-sm text-accent">Error: {turn.error}</p>}
      {r && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-sepia">
            <span className="chip border-line text-sepia">{PLAN_LABEL[r.plan.mode] ?? r.plan.mode}</span>
            {r.confidence && <span className={`chip ${confBg(confLevelToNum(r.confidence))}`}>{r.confidence} confidence</span>}
            {r.needs_human_review && (
              <span className="chip border-warn/40 bg-warn/10 text-warn">human review</span>
            )}
            {r.insufficient_evidence && (
              <span className="chip border-error/40 bg-error/10 text-error">insufficient evidence</span>
            )}
          </div>

          <p className="whitespace-pre-wrap text-sm leading-relaxed">{r.answer}</p>

          {r.plan.rationale && <p className="text-xs text-sepia/70">Route: {r.plan.rationale}</p>}

          {r.uncertainties && r.uncertainties.length > 0 && (
            <ul className="list-inside list-disc text-xs text-sepia/80">
              {r.uncertainties.map((u, i) => (
                <li key={i}>{u}</li>
              ))}
            </ul>
          )}

          {r.provenance.length > 0 && (
            <div>
              <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-sepia">Sources</p>
              <div className="flex flex-wrap gap-1.5">
                {r.provenance
                  .filter((p) => p.document_id && p.page_id)
                  .map((p, i) => {
                    const active = activeSource?.doc === p.document_id && activeSource?.page === p.page_id;
                    return (
                      <button
                        key={i}
                        onClick={() => onOpenSource({ doc: p.document_id!, page: p.page_id! })}
                        title={p.record_id ?? undefined}
                        className={`chip border text-left font-mono text-[11px] transition-colors ${
                          active
                            ? "border-accent/60 bg-accent/20 text-paper"
                            : "border-line bg-surface text-sepia hover:border-accent/50 hover:text-paper"
                        }`}
                      >
                        {p.page_id}
                        {p.verification_status ? ` · ${p.verification_status}` : ""}
                      </button>
                    );
                  })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DotsLoader() {
  const r = 10;
  const cx = 13;
  const cy = 13;
  return (
    <div className="relative shrink-0" style={{ width: 26, height: 26 }}>
      {Array.from({ length: 8 }).map((_, i) => {
        const angle = (i * 45 * Math.PI) / 180;
        const x = cx + r * Math.sin(angle) - 2;
        const y = cy - r * Math.cos(angle) - 2;
        return (
          <span
            key={i}
            className="absolute block h-1 w-1 rounded-full bg-current"
            style={{
              left: x,
              top: y,
              animation: `dots-fade 1.2s ease-in-out ${(i * 0.15).toFixed(2)}s infinite`,
            }}
          />
        );
      })}
    </div>
  );
}
