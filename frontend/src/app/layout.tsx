import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "PaperRAG",
  description: "AI-powered assistant for exploring research papers in AI and Machine Learning",
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
