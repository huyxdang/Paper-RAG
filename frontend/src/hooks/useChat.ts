"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import type {
  Message,
  Citation,
  LogEntry,
  StatusEventData,
  TokenEventData,
  CitationsEventData,
  DoneEventData,
  ErrorEventData,
} from "@/types/chat";
import { buildStreamUrl, getSession, createSession } from "@/lib/api";

/**
 * Format granular log messages for each pipeline step.
 * Returns an array of log entries for multi-line console output.
 */
function formatLogMessages(
  step: string,
  details?: Record<string, unknown>
): Array<{ message: string; type: LogEntry["type"] }> {
  const logs: Array<{ message: string; type: LogEntry["type"] }> = [];

  switch (step) {
    case "processing":
      logs.push({ message: "Processing query (rewrite + route)...", type: "info" });
      const orig = details?.original as string | undefined;
      const rewr = details?.rewritten as string | undefined;
      const route = details?.route as string | undefined;
      const reason = details?.reasoning as string | undefined;
      if (orig) {
        logs.push({
          message: `Original: "${orig.slice(0, 50)}${orig.length > 50 ? "..." : ""}"`,
          type: "info",
        });
      }
      if (rewr) {
        logs.push({
          message: `Rewritten: "${rewr.slice(0, 50)}${rewr.length > 50 ? "..." : ""}"`,
          type: "success",
        });
      }
      if (route) {
        const routeLabels: Record<string, string> = {
          conversational: "CONVERSATIONAL",
          vectorstore: "VECTORSTORE",
          web_search: "WEB_SEARCH",
        };
        logs.push({
          message: `Routed → ${routeLabels[route] || route.toUpperCase()}`,
          type: "success",
        });
      }
      if (reason) {
        logs.push({ message: `Reason: ${reason}`, type: "info" });
      }
      break;

    case "rewriting":
      if (details?.changed) {
        logs.push({ message: "Optimizing query for search...", type: "info" });
        const original = details.original as string | undefined;
        const rewritten = details.rewritten as string | undefined;
        if (original) {
          logs.push({
            message: `Original: "${original.slice(0, 50)}${original.length > 50 ? "..." : ""}"`,
            type: "info",
          });
        }
        if (rewritten) {
          logs.push({
            message: `Rewritten: "${rewritten.slice(0, 50)}${rewritten.length > 50 ? "..." : ""}"`,
            type: "success",
          });
        }
      } else {
        logs.push({ message: "Query already optimized", type: "info" });
      }
      break;

    case "routing":
      const decision = details?.decision as string | undefined;
      const reasoning = details?.reasoning as string | undefined;
      logs.push({ message: "Analyzing query intent...", type: "info" });
      if (decision) {
        const routeLabels: Record<string, string> = {
          conversational: "CONVERSATIONAL",
          vectorstore: "VECTORSTORE",
          web_search: "WEB_SEARCH",
        };
        logs.push({
          message: `Routed → ${routeLabels[decision] || decision.toUpperCase()}`,
          type: "success",
        });
      }
      if (reasoning) {
        logs.push({
          message: `Reason: ${reasoning}`,
          type: "info",
        });
      }
      break;

    case "retrieving":
      const count = (details?.count as number) || 0;
      logs.push({ message: "Running hybrid search (vector + BM25)", type: "info" });
      logs.push({ message: `Retrieved ${count} documents`, type: "success" });
      break;

    case "reranking":
      const before = details?.before as number | undefined;
      const after = details?.after as number | undefined;
      logs.push({ message: "Applying Cohere reranking", type: "info" });
      if (before !== undefined && after !== undefined) {
        logs.push({ message: `Reranked: ${before} → ${after} documents`, type: "success" });
      } else {
        logs.push({ message: "Reranking complete", type: "success" });
      }
      break;

    case "grading_docs":
      const kept = details?.kept as number | undefined;
      const needsWebSearch = details?.needs_web_search as boolean | undefined;
      logs.push({ message: "Grading document relevance", type: "info" });
      if (kept !== undefined) {
        logs.push({ message: `Kept ${kept} relevant documents`, type: "success" });
      }
      if (needsWebSearch) {
        logs.push({ message: "Web search fallback triggered", type: "warning" });
      }
      break;

    case "web_search":
      if (details?.skipped) {
        logs.push({
          message: `Web search skipped: ${details.reason || "not configured"}`,
          type: "warning",
        });
      } else if (details?.error) {
        logs.push({ message: `Web search error: ${details.error}`, type: "error" });
      } else {
        const webCount = (details?.count as number) || 0;
        logs.push({ message: "Searching the web...", type: "info" });
        logs.push({ message: `Found ${webCount} web results`, type: "success" });
      }
      break;

    case "generating":
      if (details?.mode === "conversational") {
        logs.push({ message: "Generating conversational response...", type: "metric" });
      } else if (details?.status === "starting") {
        const docCount = details.doc_count as number | undefined;
        logs.push({
          message: `Generating answer from ${docCount || "?"} documents...`,
          type: "metric",
        });
      } else if (details?.status === "complete") {
        const tokens = details.tokens as number | undefined;
        logs.push({ message: `Generated ${tokens || "?"} tokens`, type: "success" });
      } else {
        logs.push({ message: "Generating answer...", type: "metric" });
      }
      break;

    case "extracting_citations":
      const citationCount = (details?.count as number) || 0;
      logs.push({ message: `Extracted ${citationCount} citations`, type: "success" });
      break;

    case "grading_gen":
      if (details?.skipped) {
        logs.push({ message: "Answer grading skipped", type: "warning" });
      } else {
        const grade = details?.grade as string | undefined;
        logs.push({
          message: `Answer grade: ${grade || "unknown"}`,
          type: grade === "useful" ? "success" : "warning",
        });
      }
      break;

    default:
      logs.push({ message: step, type: "info" });
  }

  return logs;
}

// Session storage key
const SESSION_KEY = "crag_session_id";

interface UseChatReturn {
  messages: Message[];
  logs: LogEntry[];
  isLoading: boolean;
  error: string | null;
  currentStep: string | null;
  stepLatency: number | null;
  sessionId: string | null;
  sendMessage: (content: string) => void;
  clearChat: () => void;
  clearLogs: () => void;
  stopGeneration: () => void;
}

/**
 * Custom hook for managing chat state and SSE streaming.
 */
export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [stepLatency, setStepLatency] = useState<number | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Reference to current EventSource for cleanup
  const eventSourceRef = useRef<EventSource | null>(null);
  // Reference to track if we're currently streaming
  const isStreamingRef = useRef(false);

  /**
   * Add a log entry to the console.
   */
  const addLog = useCallback(
    (message: string, type: LogEntry["type"] = "info", latency?: string, details?: Record<string, unknown>) => {
      const now = new Date();
      const timestamp = `${now.getHours().toString().padStart(2, "0")}:${now
        .getMinutes()
        .toString()
        .padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`;

      setLogs((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          timestamp,
          message,
          type,
          latency,
          details,
        },
      ]);
    },
    []
  );

  /**
   * Clear all logs.
   */
  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  // Load session from localStorage on mount
  useEffect(() => {
    const storedSessionId = localStorage.getItem(SESSION_KEY);
    if (storedSessionId) {
      // Verify session still exists
      getSession(storedSessionId)
        .then((session) => {
          if (session) {
            setSessionId(storedSessionId);
          } else {
            localStorage.removeItem(SESSION_KEY);
          }
        })
        .catch(() => {
          localStorage.removeItem(SESSION_KEY);
        });
    }
  }, []);

  // Save session to localStorage when it changes
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem(SESSION_KEY, sessionId);
    }
  }, [sessionId]);

  /**
   * Stop the current generation stream.
   */
  const stopGeneration = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    isStreamingRef.current = false;
    setIsLoading(false);
    setCurrentStep(null);
    setStepLatency(null);
  }, []);

  /**
   * Clear chat and start a new session.
   */
  const clearChat = useCallback(async () => {
    stopGeneration();
    setMessages([]);
    setError(null);
    
    // Create new session
    try {
      const session = await createSession();
      setSessionId(session.id);
    } catch {
      // If session creation fails, just clear the session ID
      setSessionId(null);
      localStorage.removeItem(SESSION_KEY);
    }
  }, [stopGeneration]);

  /**
   * Send a message and stream the response.
   */
  const sendMessage = useCallback(
    (content: string) => {
      if (isStreamingRef.current || !content.trim()) return;

      // Add user message
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: content.trim(),
        timestamp: new Date(),
      };

      // Create placeholder for assistant message
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsLoading(true);
      setError(null);
      setCurrentStep(null);
      setStepLatency(null);
      isStreamingRef.current = true;

      // Add initial log
      const queryPreview = content.length > 30 ? content.substring(0, 30) + "..." : content;
      addLog(`Received query: "${queryPreview}"`, "info");

      // Build SSE URL
      const url = buildStreamUrl(content, sessionId);

      // Create EventSource
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      let accumulatedContent = "";
      let citations: Citation[] = [];
      let newSessionId: string | null = null;

      // Handle status events
      eventSource.addEventListener("status", (event) => {
        try {
          const data: StatusEventData = JSON.parse(event.data);
          setCurrentStep(data.step);
          setStepLatency(data.latency_ms);

          const latencyStr = data.latency_ms ? `${data.latency_ms}ms` : undefined;

          // Get granular log messages for this step
          const logMessages = formatLogMessages(data.step, data.details);
          
          // Add all log messages, with latency only on the last one
          logMessages.forEach((log, idx) => {
            const showLatency = idx === logMessages.length - 1 ? latencyStr : undefined;
            addLog(log.message, log.type, showLatency, data.details);
          });
        } catch {
          console.error("Failed to parse status event:", event.data);
        }
      });

      // Handle token events
      eventSource.addEventListener("token", (event) => {
        try {
          const data: TokenEventData = JSON.parse(event.data);
          accumulatedContent += data.content;

          // Update assistant message with accumulated content
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessage.id
                ? { ...msg, content: accumulatedContent }
                : msg
            )
          );
        } catch {
          console.error("Failed to parse token event:", event.data);
        }
      });

      // Handle citations events
      eventSource.addEventListener("citations", (event) => {
        try {
          const data: CitationsEventData = JSON.parse(event.data);
          citations = data.citations;
        } catch {
          console.error("Failed to parse citations event:", event.data);
        }
      });

      // Handle done events
      eventSource.addEventListener("done", (event) => {
        try {
          const data: DoneEventData = JSON.parse(event.data);
          
          // Use final generation if provided (cleaned answer)
          const finalContent = data.generation || accumulatedContent;
          
          // Merge citations from done event if not already set
          if (data.citations && data.citations.length > 0 && citations.length === 0) {
            citations = data.citations;
          }

          // Update session ID
          if (data.session_id) {
            newSessionId = data.session_id;
            setSessionId(data.session_id);
          }

          // Finalize assistant message
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessage.id
                ? {
                    ...msg,
                    content: finalContent,
                    citations,
                    isStreaming: false,
                  }
                : msg
            )
          );

          // Add completion log
          addLog(
            `Query completed: ${citations.length} sources cited`,
            "success",
            `Total: ${data.total_ms}ms`
          );
        } catch {
          console.error("Failed to parse done event:", event.data);
        }

        // Cleanup
        eventSource.close();
        eventSourceRef.current = null;
        isStreamingRef.current = false;
        setIsLoading(false);
        setCurrentStep(null);
        setStepLatency(null);
      });

      // Handle error events
      eventSource.addEventListener("error", (event) => {
        // Check if this is a custom error event from our API
        if (event instanceof MessageEvent && event.data) {
          try {
            const data: ErrorEventData = JSON.parse(event.data);
            setError(data.message);
          } catch {
            setError("An error occurred while streaming the response.");
          }
        } else {
          // Connection error
          setError("Connection lost. Please try again.");
        }

        // Cleanup
        eventSource.close();
        eventSourceRef.current = null;
        isStreamingRef.current = false;
        setIsLoading(false);
        setCurrentStep(null);
        setStepLatency(null);

        // Mark message as not streaming
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessage.id
              ? { ...msg, isStreaming: false }
              : msg
          )
        );
      });

      // Handle connection errors
      eventSource.onerror = () => {
        if (eventSource.readyState === EventSource.CLOSED) {
          // Already handled by done event
          return;
        }

        setError("Connection lost. Please try again.");
        eventSource.close();
        eventSourceRef.current = null;
        isStreamingRef.current = false;
        setIsLoading(false);
        setCurrentStep(null);
        setStepLatency(null);
      };
    },
    [sessionId, addLog]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return {
    messages,
    logs,
    isLoading,
    error,
    currentStep,
    stepLatency,
    sessionId,
    sendMessage,
    clearChat,
    clearLogs,
    stopGeneration,
  };
}
