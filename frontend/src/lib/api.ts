/**
 * API client for the CRAG backend.
 */

import type { HealthResponse, SessionResponse } from "@/types/chat";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Check backend health status.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

/**
 * Create a new session.
 */
export async function createSession(): Promise<SessionResponse> {
  const response = await fetch(`${API_URL}/sessions`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.status}`);
  }
  return response.json();
}

/**
 * Get session details.
 */
export async function getSession(sessionId: string): Promise<SessionResponse | null> {
  const response = await fetch(`${API_URL}/sessions/${sessionId}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to get session: ${response.status}`);
  }
  return response.json();
}

/**
 * Delete a session.
 */
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_URL}/sessions/${sessionId}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Failed to delete session: ${response.status}`);
  }
}

/**
 * Build the SSE stream URL for chat.
 */
export function buildStreamUrl(question: string, sessionId?: string | null): string {
  const params = new URLSearchParams({ question });
  if (sessionId) {
    params.set("session_id", sessionId);
  }
  return `${API_URL}/chat/stream?${params.toString()}`;
}

/**
 * Get the API base URL.
 */
export function getApiUrl(): string {
  return API_URL;
}
