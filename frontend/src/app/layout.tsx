import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const baseUrl =
  process.env.NEXT_PUBLIC_APP_URL ||
  (typeof process.env.VERCEL_URL === "string"
    ? `https://${process.env.VERCEL_URL}`
    : "http://localhost:3000");

export const metadata: Metadata = {
  metadataBase: new URL(baseUrl),
  title: "PaperRAG",
  description: "AI-powered assistant for exploring research papers in AI and Machine Learning",
  openGraph: {
    title: "PaperRAG",
    description: "AI-powered assistant for exploring research papers in AI and Machine Learning",
    images: ["/Paper-Rag-logo.png"],
  },
  twitter: {
    card: "summary_large_image",
    title: "PaperRAG",
    description: "AI-powered assistant for exploring research papers in AI and Machine Learning",
    images: ["/Paper-Rag-logo.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link 
          href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" 
          rel="stylesheet" 
        />
      </head>
      <body className={`${jetbrainsMono.variable} font-mono antialiased`}>
        {/* Grid background */}
        <div 
          className="fixed inset-0 pointer-events-none grid-background"
          aria-hidden="true"
        />
        {/* Main content */}
        {children}
        {/* CRT Scanlines overlay */}
        <div className="scanlines" aria-hidden="true" />
      </body>
    </html>
  );
}
