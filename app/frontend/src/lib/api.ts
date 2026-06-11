// Typed client for the FastAPI backend (proxied at /api by next.config)

export interface DocumentSummary {
  document_id: string;
  n_pages: number;
  pages: string[];
}

export interface WordConfidence {
  text: string;
  confidence: number;
}

export interface RegionView {
  region_id: string;
  layer: string;
  text: string;
  confidence: number;
  verification_status: string;
  bbox: number[]; // [x, y, w, h] on the page
  reading_order: number;
  word_confidences: WordConfidence[];
}

export interface MarkView {
  mark_id: string;
  mark_type: string;
  bbox: number[]; // [x, y, w, h]
  confidence: number;
}

export interface PageDetail {
  document_id: string;
  page_id: string;
  title: string;
  date_created: string | null;
  language: string[];
  printed_text: string;
  handwritten_text: string;
  printed_regions: RegionView[];
  handwritten_regions: RegionView[];
  marks: MarkView[];
  forensic: Record<string, unknown>;
  confidence: number;
  layers: Record<string, string>; // name -> derived image url
}

// --- Chat (RAG) ---

export interface ChatProvenance {
  score: number | null;
  record_id: string | null;
  document_id: string | null;
  page_id: string | null;
  verification_status: string | null;
  human_review_required: string | null;
  forensic_confidence_score: number | null;
  alerts: string | null;
  source_image_path: string | null;
}

export interface ChatPlan {
  mode: string;        // vector | hybrid | metadata_lookup | provenance_lookup
  rationale: string;
}

export interface ChatResponse {
  answer: string;
  confidence?: string;             // high | medium | low
  needs_human_review?: boolean;
  insufficient_evidence?: boolean;
  uncertainties?: string[];
  provenance: ChatProvenance[];
  plan: ChatPlan;
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${url} -> ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  documents: () => getJSON<DocumentSummary[]>("/api/documents"),
  page: (doc: string, page: string) =>
    getJSON<PageDetail>(`/api/documents/${encodeURIComponent(doc)}/pages/${encodeURIComponent(page)}`),
  pdfUrl: (doc: string) => `/api/pdf/${encodeURIComponent(doc)}`,
  pageImageUrl: (doc: string, page: string) =>
    `/api/page-image/${encodeURIComponent(doc)}/${encodeURIComponent(page)}`,
  heatmapUrl: (doc: string, page: string) =>
    `/api/heatmap/${encodeURIComponent(doc)}/${encodeURIComponent(page)}`,
  chat: (question: string) => postJSON<ChatResponse>("/api/chat", { question }),
};
