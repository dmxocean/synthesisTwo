"use client";

import { usePathname, useRouter } from "next/navigation";

export const NEW_CHAT_EVENT = "radar:new-chat";

export default function NavRail() {
  const pathname = usePathname();
  const router = useRouter();

  function newChat() {
    if (pathname === "/") {
      window.dispatchEvent(new CustomEvent(NEW_CHAT_EVENT));
    } else {
      router.push("/");
    }
  }

  const onAssistant = pathname === "/";
  const onCatalog = pathname.startsWith("/catalog");

  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-black/10 bg-sidebar shadow-2xl">
      <a href="/" className="group flex items-center gap-2.5 px-4 py-6">
        <span className="grid h-7 w-7 place-items-center rounded bg-accent font-serif text-sm font-bold text-ink shadow-md">
          R
        </span>
        <span className="font-serif text-lg font-semibold tracking-wide text-ink group-hover:text-accent">
          RADAR
        </span>
      </a>

      <nav className="flex flex-col gap-1 px-3">
        <button
          onClick={newChat}
          className="flex items-center gap-2 rounded-lg bg-black/20 px-3 py-2 text-sm text-ink/80 transition-colors hover:bg-black/30 hover:text-ink"
        >
          <span className="text-accent">＋</span> New chat
        </button>

        <a
          href="/"
          className={`mt-2 rounded-lg px-3 py-2 text-sm transition-colors ${
            onAssistant ? "bg-accent/15 text-accent font-medium" : "text-ink/50 hover:bg-black/10 hover:text-ink/80"
          }`}
        >
          Assistant
        </a>
        <a
          href="/catalog"
          className={`rounded-lg px-3 py-2 text-sm transition-colors ${
            onCatalog ? "bg-accent/15 text-accent font-medium" : "text-ink/50 hover:bg-black/10 hover:text-ink/80"
          }`}
        >
          Catalog
        </a>
      </nav>

      <div className="mt-auto px-4 py-4 text-[10px] uppercase tracking-widest text-ink/20">
        GUIRAD · Radio Barcelona
      </div>
    </aside>
  );
  }
