/**
 * TypeScript interfaces matching the CRAG backend API.
 */

// ============== Source & Citation Types ==============

export interface SourceInfo {
  title: string;
  url: string;
  section: string;
  relevance_score: number;
}

export interface Citation {
  ref: number;
  claim: string;
  quote?: string;
  source?: SourceInfo;
}

// ============== Message Types ==============

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  timestamp: Date;
  isStreaming?: boolean;
}

// ============== API Response Types ==============

export interface ChatResponse {
  answer: string;
  session_id: string;
  route_decision?: string;
  documents_used: number;
  generation_grade?: string;
  total_ms: number;
  citations: Citation[];
}

export interface HealthResponse {
  status: string;
  retriever: string;
  session_storage: string;
  pinecone_configured: boolean;
  redis_configured: boolean;
}

export interface SessionResponse {
  id: string;
  created_at: string;
  last_activity: string;
  turn_count: number;
}

// ============== SSE Event Types ==============

export type StreamEventType = "status" | "token" | "citations" | "error" | "done";

export interface StatusEventData {
  step: string;
  latency_ms: number;
  details?: Record<string, unknown>;
}

export interface TokenEventData {
  content: string;
}

export interface CitationsEventData {
  citations: Citation[];
}

export interface ErrorEventData {
  message: string;
  step?: string;
}

export interface DoneEventData {
  session_id: string;
  total_ms: number;
  grade?: string;
  generation?: string;
  citations?: Citation[];
}

export interface StreamEvent {
  event: StreamEventType;
  data: StatusEventData | TokenEventData | CitationsEventData | ErrorEventData | DoneEventData;
}

// ============== Pipeline Step Names ==============

export const STEP_LABELS: Record<string, string> = {
  rewriting: "Rewriting query",
  routing: "Analyzing question",
  retrieving: "Searching documents",
  reranking: "Reranking results",
  grading_docs: "Grading relevance",
  web_search: "Web search",
  generating: "Generating answer",
  extracting_citations: "Extracting citations",
  grading_gen: "Verifying answer",
};

// ============== Chat State ==============

export interface ChatState {
  messages: Message[];
  sessionId: string | null;
  isLoading: boolean;
  error: string | null;
  currentStep: string | null;
  stepLatency: number | null;
}

// ============== RAG Console Log Entry ==============

export type LogType = "info" | "success" | "warning" | "error" | "metric";

export interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  type: LogType;
  latency?: string;
  details?: Record<string, unknown>;
}
