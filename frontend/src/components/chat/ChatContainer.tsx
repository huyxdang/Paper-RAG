"use client";

import { useState } from "react";
import { useChat } from "@/hooks/useChat";
import { MessageList, HomeView } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { StreamingStatus } from "./StreamingStatus";
import { Header } from "@/components/Header";
import { RAGConsole } from "@/components/RAGConsole";
import { Inspector } from "@/components/Inspector";
import type { Citation } from "@/types/chat";
import { AlertCircle, ArrowRight } from "lucide-react";

/**
 * Main chat container with retro two-column layout.
 */
export function ChatContainer() {
  const {
    messages,
    logs,
    isLoading,
    error,
    currentStep,
    stepLatency,
    sendMessage,
    clearChat,
    clearLogs,
    stopGeneration,
  } = useChat();

  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [view, setView] = useState<"home" | "chat">("home");

  const handleSearch = (query: string) => {
    if (view === "home") {
      setView("chat");
    }
    sendMessage(query);
  };

  const handleNewChat = () => {
    clearChat();
    clearLogs();
    setView("home");
    setSelectedCitation(null);
  };

  return (
    <div className="h-screen w-screen flex flex-col text-zinc-900 font-mono overflow-hidden">
      {/* Header */}
      <Header onNewChat={handleNewChat} />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden relative z-10">
        {/* Left/Center Main Content */}
        <div className="flex-1 flex flex-col min-w-0 bg-[#fdfbf7] relative">
          {view === "home" ? (
            <HomeView onSearch={handleSearch} />
          ) : (
            <>
              {/* Messages */}
              <MessageList
                messages={messages}
                onCitationClick={(citation) => setSelectedCitation(citation)}
              />

              {/* Footer with input */}
              <div className="p-4 bg-white border-t border-zinc-200">
                <div className="max-w-4xl mx-auto space-y-3">
                  {/* Error display */}
                  {error && (
                    <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 px-3 py-2 rounded border border-red-200">
                      <AlertCircle className="h-4 w-4 shrink-0" />
                      <span>{error}</span>
                    </div>
                  )}

                  {/* Streaming status */}
                  {isLoading && (
                    <StreamingStatus step={currentStep} latencyMs={stepLatency} />
                  )}

                  {/* Input */}
                  <div className="relative">
                    <ChatInput
                      onSend={handleSearch}
                      onStop={stopGeneration}
                      isLoading={isLoading}
                    />
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Right Sidebar: RAG Console */}
        <RAGConsole logs={logs} isOnline={true} />

        {/* Floating Inspector Panel */}
        <Inspector
          citation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
        />
      </div>
    </div>
  );
}
