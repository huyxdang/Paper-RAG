"use client";

import { STEP_LABELS } from "@/types/chat";
import { Loader2 } from "lucide-react";

interface StreamingStatusProps {
  step: string | null;
  latencyMs: number | null;
}

/**
 * Displays the current pipeline step with latency.
 * Shows: "[156ms] Reranking documents..."
 */
export function StreamingStatus({ step, latencyMs }: StreamingStatusProps) {
  if (!step) return null;

  const label = STEP_LABELS[step] || step;

  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground animate-pulse">
      <Loader2 className="h-4 w-4 animate-spin" />
      {latencyMs !== null && latencyMs > 0 ? (
        <span className="font-mono text-xs bg-muted px-1.5 py-0.5 rounded">
          {latencyMs}ms
        </span>
      ) : null}
      <span>{label}...</span>
    </div>
  );
}
