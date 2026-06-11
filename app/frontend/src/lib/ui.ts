// Shared UI helpers and layer configuration for the viewer + chat.

/** Text color class by confidence (0..1). */
export function confColor(c: number): string {
  if (c >= 0.8) return "text-moss";
  if (c >= 0.55) return "text-warn";
  return "text-error";
}

/** Chip/badge background+border class by confidence (0..1). */
export function confBg(c: number): string {
  if (c >= 0.8) return "bg-moss/10 border-moss/30 text-moss";
  if (c >= 0.55) return "bg-warn/10 border-warn/30 text-warn";
  return "bg-error/10 border-error/30 text-error";
}

/** Region-box border class by confidence (0..1). */
export function confBorder(c: number): string {
  if (c >= 0.8) return "border-moss/60";
  if (c >= 0.55) return "border-warn/60";
  return "border-error/60";
}

/** Map the qualitative chat confidence to a numeric proxy for confBg(). */
export function confLevelToNum(level?: string): number {
  if (level === "high") return 0.9;
  if (level === "medium") return 0.6;
  return 0.3;
}

/** A tintable composite layer backed by a grayscale (white-on-black) mask image. */
export interface LayerSpec {
  id: string;            // key into PageDetail.layers
  label: string;
  color: string;         // default tint (hex)
}

// Default muted tints. Masks are white-on-black, so the tint shows on strokes.
export const COMPOSITE_LAYERS: LayerSpec[] = [
  { id: "printed",     label: "PR", color: "#E2D06D" },  // muted gold
  { id: "handwritten", label: "HW", color: "#2B4C9A" },  // delft navy
  { id: "noise",       label: "NS", color: "#94A3B8" },  // slate smoke
];

// A sophisticated palette offered in each layer's color swatch.
export const SWATCHES = ["#E2D06D", "#2B4C9A", "#7C2D12", "#B7A38A", "#6A704D", "#94A3B8"];
