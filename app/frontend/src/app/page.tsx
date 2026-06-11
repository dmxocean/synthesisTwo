"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Chat, { SourceRef } from "@/components/Chat";
import ArtifactPanel from "@/components/ArtifactPanel";
import { NEW_CHAT_EVENT } from "@/components/NavRail";

const CHAT_PCT_DEFAULT = 42;   // % of total width given to chat when artifact is open
const CHAT_PCT_MIN     = 25;
const CHAT_PCT_MAX     = 75;

export default function Assistant() {
  const [source, setSource] = useState<SourceRef | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [chatPct, setChatPct] = useState(CHAT_PCT_DEFAULT);
  const dragging = useRef(false);
  const startX   = useRef(0);
  const startPct = useRef(CHAT_PCT_DEFAULT);

  // "New chat" from the nav rail clears the conversation and the open artifact.
  useEffect(() => {
    function onNew() {
      setResetKey((k) => k + 1);
      setSource(null);
    }
    window.addEventListener(NEW_CHAT_EVENT, onNew);
    return () => window.removeEventListener(NEW_CHAT_EVENT, onNew);
  }, []);

  // Drag-resize divider handlers - attached to window while dragging
  const onDividerMouseDown = useCallback(
    (e: React.MouseEvent) => {
      dragging.current = true;
      startX.current   = e.clientX;
      startPct.current = chatPct;

      function onMove(ev: MouseEvent) {
        if (!dragging.current) return;
        const delta = ((ev.clientX - startX.current) / window.innerWidth) * 100;
        setChatPct(Math.max(CHAT_PCT_MIN, Math.min(CHAT_PCT_MAX, startPct.current + delta)));
      }
      function onUp() {
        dragging.current = false;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      }
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [chatPct]
  );

  return (
    <div className="flex min-h-0 flex-1 select-none">
      <section
        className="flex min-h-0 flex-col overflow-hidden"
        style={source ? { flex: `0 0 ${chatPct}%` } : { flex: "1 1 0%" }}
      >
        <Chat onOpenSource={setSource} activeSource={source} resetKey={resetKey} />
      </section>

      {source && (
        <>
          {/* drag handle */}
          <div
            onMouseDown={onDividerMouseDown}
            className="w-1 shrink-0 cursor-col-resize bg-sidebar/5 transition-colors hover:bg-accent/20 active:bg-accent/40"
          />

          <section className="flex min-h-0 flex-1 flex-col overflow-hidden bg-stage">
            <ArtifactPanel doc={source.doc} page={source.page} onClose={() => setSource(null)} />
          </section>
        </>
      )}
    </div>
  );
}
