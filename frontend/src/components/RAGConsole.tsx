"use client";

import { useRef, useEffect } from "react";
import { Terminal, Activity, Network } from "lucide-react";
import type { LogEntry } from "@/types/chat";

interface RAGConsoleProps {
  logs: LogEntry[];
  isOnline?: boolean;
}

/**
 * Right sidebar console showing RAG pipeline logs with latency.
 */
export function RAGConsole({ logs, isOnline = true }: RAGConsoleProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="w-[380px] bg-[#0c0c0c] border-l border-zinc-800 flex flex-col h-full font-mono text-xs shadow-2xl relative shrink-0 z-20">
      {/* Console Header */}
      <div className="h-10 border-b border-zinc-800 flex items-center justify-between px-4 bg-[#0c0c0c] shrink-0">
        <span className="text-zinc-400 font-bold tracking-wider flex items-center gap-2">
          <Terminal size={12} />
          RAG_KERNEL
        </span>
        <div className="flex items-center gap-3">
          <span className="text-zinc-600">v0.0.1</span>
          <span
            className={`flex items-center gap-1.5 text-[10px] ${
              isOnline ? "text-emerald-500" : "text-zinc-500"
            }`}
          >
            <div
              className={`w-1.5 h-1.5 rounded-full ${
                isOnline ? "bg-emerald-500 animate-pulse" : "bg-zinc-500"
              }`}
            />
            {isOnline ? "ONLINE" : "OFFLINE"}
          </span>
        </div>
      </div>

      {/* Logs Area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 space-y-3 console-scrollbar"
      >
        {logs.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center opacity-20 text-zinc-500">
            <Activity size={32} className="mb-2" />
            <span>Awaiting Query...</span>
          </div>
        )}

        {logs.map((log) => (
          <div
            key={log.id}
            className="grid grid-cols-[55px_1fr] gap-0 animate-fade-in"
          >
            {/* Timestamp */}
            <span className="text-zinc-600 font-light tabular-nums text-[10px] pt-0.5 border-r border-zinc-800 pr-2 mr-2">
              {log.timestamp}
            </span>

            {/* Log content */}
            <div className="flex flex-col pl-2">
              <div className="flex items-start gap-2">
                {/* Status indicator */}
                <div
                  className={`mt-1 w-2 h-2 shrink-0 ${
                    log.type === "success"
                      ? "bg-emerald-900 border border-emerald-500/50"
                      : log.type === "metric"
                      ? "bg-emerald-900 border border-emerald-500/50"
                      : log.type === "warning"
                      ? "bg-amber-900 border border-amber-500/50"
                      : log.type === "error"
                      ? "bg-red-900 border border-red-500/50"
                      : "bg-zinc-800 border border-zinc-700"
                  }`}
                />

                <div className="flex flex-col w-full">
                  {/* Message */}
                  <span
                    className={`leading-relaxed ${
                      log.type === "success" || log.type === "metric"
                        ? "text-emerald-400"
                        : log.type === "warning"
                        ? "text-amber-400"
                        : log.type === "error"
                        ? "text-red-400"
                        : "text-zinc-300"
                    }`}
                  >
                    {log.message}
                  </span>

                  {/* Latency */}
                  {log.latency && (
                    <span className="text-[10px] text-zinc-500 font-mono mt-0.5 flex items-center gap-1">
                      <Activity size={8} /> {log.latency}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
