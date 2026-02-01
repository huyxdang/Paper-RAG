"use client";

import { useRef, useEffect } from "react";
import type { Message, Citation } from "@/types/chat";
import { MessageBubble } from "./MessageBubble";
import { ArrowRight } from "lucide-react";

interface MessageListProps {
  messages: Message[];
  onCitationClick?: (citation: Citation) => void;
}

/**
 * Scrollable list of chat messages.
 */
export function MessageList({ messages, onCitationClick }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 space-y-8 relative z-10 scroll-smooth">
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          onCitationClick={onCitationClick}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

interface HomeViewProps {
  onSearch: (query: string) => void;
}

/**
 * Home view with centered search.
 */
export function HomeView({ onSearch }: HomeViewProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-4 sm:p-8 animate-fade-in pb-32">
      {/* Centered Logo */}
      <div className="mb-10 text-center">
        <div className="font-pixel text-4xl md:text-5xl text-red-600 tracking-tighter filter drop-shadow-[2px_2px_0px_rgba(0,0,0,0.1)]">
          Paper<span className="text-zinc-900">RAG</span>
        </div>
      </div>

      {/* Search Card */}
      <div className="w-full max-w-3xl bg-white rounded-lg shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-zinc-200 p-2 transition-all focus-within:ring-1 focus-within:ring-red-500/20 focus-within:border-red-500/50">
        <SearchInput onSearch={onSearch} />
      </div>

      {/* Version info */}
      <div className="mt-8 text-center opacity-40 hover:opacity-100 transition-opacity">
        <div className="text-zinc-400 text-[10px] font-mono uppercase tracking-widest">
          5,876 Documents Indexed (NeurIPS '25)
        </div>
      </div>
    </div>
  );
}

function SearchInput({ onSearch }: { onSearch: (query: string) => void }) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [query]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (query.trim()) {
        onSearch(query.trim());
        setQuery("");
      }
    }
  };

  return (
    <div className="relative flex items-end">
      <textarea
        ref={textareaRef}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about AI Ideas, Trends, Limitations, Papers,..."
        className="w-full bg-transparent border-none outline-none text-zinc-900 text-lg resize-none overflow-hidden placeholder-zinc-300 font-mono py-3 pr-10 pl-2"
        rows={1}
        autoFocus
      />
      <button
        onClick={() => {
          if (query.trim()) {
            onSearch(query.trim());
            setQuery("");
          }
        }}
        disabled={!query.trim()}
        className={`absolute right-2 bottom-2 h-8 w-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
          query.trim()
            ? "bg-zinc-900 text-white shadow-md hover:bg-black"
            : "bg-transparent text-zinc-300"
        }`}
      >
        <ArrowRight size={18} />
      </button>
    </div>
  );
}

// Need to import useState
import { useState } from "react";
