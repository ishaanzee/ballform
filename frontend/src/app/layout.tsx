import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ballform — Shot mechanics & 1-on-1 review",
  description: "Basketball shooting mechanics and 1-on-1 shot-space analysis from video.",
  referrer: "no-referrer",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
