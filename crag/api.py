"""
FastAPI Server for CRAG Chatbot.

Provides REST and streaming endpoints for the CRAG pipeline:
- POST /chat - Non-streaming chat endpoint
- GET /chat/stream - SSE streaming endpoint with latency metrics
- GET /health - Health check for Railway

Usage:
    # Development
    uvicorn crag.api:app --reload --port 8000
    
    # Production (Railway)
    uvicorn crag.api:app --host 0.0.0.0 --port $PORT

Environment Variables:
    MISTRAL_API_KEY - Required for LLM operations
    PINECONE_API_KEY - Required for vector search
    COHERE_API_KEY - Required for reranking
    REDIS_URL - Optional, for session persistence
    CORS_ORIGINS - Optional, comma-separated allowed origins
"""

import os
import json
import traceback

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .session import SessionManager
from .streaming import PipelineStreamer, StreamEvent, stream_crag_with_events
from .retrieval import PineconeHybridRetriever, InMemoryHybridRetriever


# ============== Configuration ==============

def get_cors_origins() -> list:
    """Get CORS origins from environment or use defaults."""
    origins = os.getenv("CORS_ORIGINS", "")
    if origins:
        return [o.strip() for o in origins.split(",")]
    # Default origins for development
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


# ============== Global State ==============

# These are initialized in lifespan
_retriever = None
_session_manager = None
_streamer = None


def get_retriever():
    """Get the retriever instance."""
    global _retriever
    if _retriever is None:
        index_name = os.getenv("PINECONE_INDEX", "neurips-2025-hybrid")
        if os.getenv("PINECONE_API_KEY"):
            _retriever = PineconeHybridRetriever(
                index_name=index_name,
                namespace="",
                create_index=False,
            )
        else:
            _retriever = InMemoryHybridRetriever()
    return _retriever


def get_session_manager():
    """Get the session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_streamer():
    """Get the pipeline streamer instance."""
    global _streamer
    if _streamer is None:
        _streamer = PipelineStreamer(
            retriever=get_retriever(),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "50")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "10")),
        )
    return _streamer


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    # Startup
    print("Starting CRAG API server...")
    
    # Check required API keys
    required_keys = ["MISTRAL_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(f"WARNING: Missing required API keys: {missing}")
    
    # Optional keys
    optional_keys = ["PINECONE_API_KEY", "COHERE_API_KEY", "TAVILY_API_KEY", "REDIS_URL"]
    for key in optional_keys:
        status = "set" if os.getenv(key) else "not set"
        print(f"  {key}: {status}")
    
    # Initialize components
    retriever = get_retriever()
    session_manager = get_session_manager()
    
    print(f"  Retriever: {type(retriever).__name__}")
    print(f"  Session storage: {session_manager.get_storage_type()}")
    print("CRAG API ready!")
    
    yield
    
    # Shutdown
    print("Shutting down CRAG API server...")


# ============== FastAPI App ==============

app = FastAPI(
    title="CRAG API",
    description="Corrective RAG API for NeurIPS 2025 Papers",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request/Response Models ==============

class ChatRequest(BaseModel):
    """Request body for non-streaming chat."""
    question: str
    session_id: Optional[str] = None


class CitationSource(BaseModel):
    """Source information for a citation."""
    title: str = ""
    url: str = ""
    section: str = ""
    relevance_score: float = 0.0


class CitationModel(BaseModel):
    """A structured citation."""
    ref: int
    claim: str
    quote: str = ""
    source: Optional[CitationSource] = None


class ChatResponse(BaseModel):
    """Response for non-streaming chat."""
    answer: str
    session_id: str
    route_decision: Optional[str] = None
    documents_used: int = 0
    generation_grade: Optional[str] = None
    total_ms: int = 0
    citations: list[CitationModel] = []


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    retriever: str
    session_storage: str
    pinecone_configured: bool
    redis_configured: bool


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Railway/container orchestration.
    
    Returns service status and configuration details.
    """
    return HealthResponse(
        status="healthy",
        retriever=type(get_retriever()).__name__,
        session_storage=get_session_manager().get_storage_type(),
        pinecone_configured=bool(os.getenv("PINECONE_API_KEY")),
        redis_configured=bool(os.getenv("REDIS_URL")),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat endpoint.
    
    Runs the full CRAG pipeline and returns the complete answer.
    Use /chat/stream for real-time streaming responses.
    """
    session_manager = get_session_manager()
    session = session_manager.get_or_create(request.session_id)
    
    # Collect all events to build response
    events = []
    answer = ""
    
    try:
        streamer = get_streamer()
        for event in streamer.stream(
            question=request.question,
            history=session.get_history(),
            session_id=session.id,
        ):
            events.append(event)
            if event.event == "token":
                answer += event.data.get("content", "")
        
        # Update session
        if answer:
            session.add_turn(request.question, answer)
            session_manager.save_session(session)
        
        # Extract metadata from events
        route_decision = None
        generation_grade = None
        documents_used = 0
        total_ms = 0
        citations = []
        
        for event in events:
            if event.event == "status":
                if event.data.get("step") == "routing":
                    route_decision = event.data.get("details", {}).get("decision")
                elif event.data.get("step") == "grading_docs":
                    documents_used = event.data.get("details", {}).get("kept", 0)
                elif event.data.get("step") == "grading_gen":
                    generation_grade = event.data.get("details", {}).get("grade")
            elif event.event == "citations":
                citations = event.data.get("citations", [])
            elif event.event == "done":
                total_ms = event.data.get("total_ms", 0)
                # Also get citations from done event if not already set
                if not citations:
                    citations = event.data.get("citations", [])
                # Get cleaned answer from done event
                if event.data.get("generation"):
                    answer = event.data["generation"]
        
        return ChatResponse(
            answer=answer,
            session_id=session.id,
            route_decision=route_decision,
            documents_used=documents_used,
            generation_grade=generation_grade,
            total_ms=total_ms,
            citations=citations,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stream")
async def chat_stream(
    question: str = Query(..., description="The question to ask"),
    session_id: Optional[str] = Query(None, description="Session ID for conversation history"),
):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    
    Streams pipeline status updates with latency metrics, followed by
    token-by-token answer generation.
    
    Event types:
    - status: Pipeline step completed (includes latency_ms)
    - token: A token from the answer
    - error: An error occurred
    - done: Pipeline completed (includes total_ms)
    
    Example events:
        {"event": "status", "data": {"step": "routing", "latency_ms": 45}}
        {"event": "token", "data": {"content": "GPT-4"}}
        {"event": "done", "data": {"session_id": "...", "total_ms": 1234}}
    """
    session_manager = get_session_manager()
    session = session_manager.get_or_create(session_id)
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events from the pipeline."""
        answer = ""
        
        try:
            streamer = get_streamer()
            
            # Run pipeline in thread pool to not block async loop
            # (the streaming is sync, so we yield events as they come)
            loop = asyncio.get_event_loop()
            
            def run_stream():
                return list(streamer.stream(
                    question=question,
                    history=session.get_history(),
                    session_id=session.id,
                ))
            
            # For now, run synchronously since the stream is a generator
            # In production, consider using asyncio.to_thread
            for event in streamer.stream(
                question=question,
                history=session.get_history(),
                session_id=session.id,
            ):
                # Collect answer tokens
                if event.event == "token":
                    answer += event.data.get("content", "")
                
                # Yield event with correct event type for SSE
                yield {
                    "event": event.event,  # Use actual event type: status, token, done, etc.
                    "data": json.dumps(event.data),
                }
            
            # Update session after completion
            if answer:
                session.add_turn(question, answer)
                session_manager.save_session(session)
                
        except Exception as e:
            print(f"Stream error: {e}")
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }
    
    return EventSourceResponse(event_generator())


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get session details including conversation history.
    
    Returns 404 if session not found or expired.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {
        "id": session.id,
        "turns": session.turn_count(),
        "history": session.get_history(),
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session_manager = get_session_manager()
    deleted = session_manager.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"deleted": True, "session_id": session_id}


@app.post("/sessions")
async def create_session():
    """Create a new session."""
    session_manager = get_session_manager()
    session = session_manager.create_session()
    
    return {
        "session_id": session.id,
        "created_at": session.created_at.isoformat(),
    }


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "crag.api:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "development") == "development",
    )
