import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // refined cohesive vintage porcelain palette
        ink: "#EBE8DF",           // chat background (slightly darker bone)
        surface: "#FCFBF9",       // cards / panels (clean porcelain)
        sidebar: "#1E2838",       // softer midnight navy (nav rail)
        line: "#E2E8F0",          // light hairline borders (light side only)
        "line-dark": "#141C29",   // deep navy borders (dark side)
        paper: "#2F2F2F",         // body text (charcoal glaze)
        sepia: "#5C5C5C",         // muted text
        accent: "#8B2635",        // brand / actions (vintage madder red)
        "accent-soft": "#F5EBEB", // very soft red wash (hovers)
        highlight: "#435C94",      // muted porcelain blue (headings / highlights)
        warn: "#D9A066",          // lighter, more saturated brass
        error: "#C0392B",         // more vibrant pomegranate red
        moss: "#88B04B",          // more vibrant glaze green
        stage: "#2D3C52",         // less dark navy for the right artifact stage
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"],
        serif: ['"Iowan Old Style"', "Palatino", '"Palatino Linotype"', "Georgia", "ui-serif", "serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        card: "0 1px 3px rgba(11,31,59,0.04), 0 12px 24px -12px rgba(11,31,59,0.08)",
        glow: "0 0 0 1px rgba(139,38,53,0.15)",
      },
      borderRadius: {
        xl: "0.9rem",
      },
    },
  },
  plugins: [],
};
export default config;
