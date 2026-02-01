"use client";

import type { Citation } from "@/types/chat";
import { FileText, Minimize2, ExternalLink, Quote } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface InspectorProps {
  citation: Citation | null;
  onClose: () => void;
}

/**
 * Slide-in panel showing citation/paper details.
 */
export function Inspector({ citation, onClose }: InspectorProps) {
  if (!citation) return null;

  const source = citation.source;
  const relevancePercent = source?.relevance_score
    ? Math.round(source.relevance_score * 100)
    : null;

  return (
    <div className="w-[400px] border-l border-zinc-200 bg-white flex flex-col h-full absolute right-0 top-0 bottom-0 z-50 shadow-2xl shadow-zinc-900/20 animate-slide-in-right">
      {/* Header */}
      <div className="p-3 border-b border-zinc-200 flex items-center justify-between bg-zinc-50 shrink-0">
        <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider flex items-center gap-2">
          <FileText size={12} /> Source Details
        </span>
        <button
          onClick={onClose}
          className="text-zinc-400 hover:text-red-600 transition-colors"
        >
          <Minimize2 size={14} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 p-6 overflow-y-auto">
        {/* Reference badge */}
        <div className="inline-block px-2 py-0.5 border border-red-200 bg-red-50 text-red-600 text-[10px] font-mono mb-4 rounded">
          REF: [{citation.ref}]
        </div>

        {/* Title */}
        <h2 className="text-lg font-bold text-zinc-900 mb-2 leading-tight">
          {source?.title || `Source ${citation.ref}`}
        </h2>

        {/* Metadata */}
        <div className="text-xs text-zinc-500 mb-6 font-mono flex items-center gap-2">
          {relevancePercent !== null && (
            <span className="text-emerald-600 font-bold">{relevancePercent}% match</span>
          )}
          {source?.section && (
            <>
              <span className="text-zinc-300">•</span>
              <span>{source.section}</span>
            </>
          )}
        </div>

        <div className="space-y-6">
          {/* Claim */}
          {citation.claim && (
            <div>
              <div className="text-[10px] uppercase tracking-widest text-zinc-400 mb-2 font-bold">
                Claim
              </div>
              <div className="border-l-2 border-red-400 pl-4 py-1 text-sm text-zinc-700 leading-relaxed">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className="m-0">{children}</p>,
                    strong: ({ children }) => <strong className="font-semibold text-zinc-900">{children}</strong>,
                  }}
                >
                  {citation.claim}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Quote */}
          {citation.quote && (
            <div>
              <div className="text-[10px] uppercase tracking-widest text-zinc-400 mb-2 font-bold">
                Quote
              </div>
              <div className="flex gap-2 text-sm border-l-2 border-red-500 pl-4 bg-zinc-50/50 py-2 pr-2">
                <Quote size={14} className="text-zinc-400 shrink-0 mt-0.5" />
                <p className="text-zinc-600 italic">
                  &ldquo;{citation.quote}&rdquo;
                </p>
              </div>
            </div>
          )}

          {/* URL */}
          {source?.url && (
            <div>
              <div className="text-[10px] uppercase tracking-widest text-zinc-400 mb-2 font-bold">
                Source URL
              </div>
              <a
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary hover:underline flex items-center gap-1"
              >
                <ExternalLink size={12} />
                {source.url.length > 40 ? source.url.substring(0, 40) + "..." : source.url}
              </a>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
