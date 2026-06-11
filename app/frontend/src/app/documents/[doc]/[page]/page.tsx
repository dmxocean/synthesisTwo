"use client";

import { use } from "react";
import ArtifactPanel from "@/components/ArtifactPanel";

export default function ViewerPage({ params }: { params: Promise<{ doc: string; page: string }> }) {
  const { doc, page } = use(params);
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ArtifactPanel doc={decodeURIComponent(doc)} page={decodeURIComponent(page)} />
    </div>
  );
}
