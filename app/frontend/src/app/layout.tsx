import type { Metadata } from "next";
import "./globals.css";
import NavRail from "@/components/NavRail";

export const metadata: Metadata = {
  title: "RADAR",
  description: "Assistant + layer-separated viewer for the Radio Barcelona historical archive",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="flex h-screen overflow-hidden">
          <NavRail />
          <main className="flex min-h-0 min-w-0 flex-1 flex-col">{children}</main>
        </div>
      </body>
    </html>
  );
}
