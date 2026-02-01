"""
API Integration Tests for CRAG System.

Tests all components to verify API keys and integrations are working.

Usage:
    python -m crag.test_apis
    
    # Or run specific tests
    python -m crag.test_apis --component router
    python -m crag.test_apis --component rewriter
    python -m crag.test_apis --component doc_grader
    python -m crag.test_apis --component reranker
    python -m crag.test_apis --component gen_grader
    python -m crag.test_apis --component retriever_memory
    python -m crag.test_apis --component retriever_pinecone
    python -m crag.test_apis --component web_search
    python -m crag.test_apis --component full_pipeline
    python -m crag.test_apis --component pipeline_history
    
    # List all available tests
    python -m crag.test_apis --list
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    error: Optional[str] = None


def print_result(result: TestResult):
    """Print a test result with color coding."""
    status = "✓ PASS" if result.passed else "✗ FAIL"
    color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"\n{color}{status}{reset} - {result.name}")
    print(f"  {result.message}")
    if result.error:
        print(f"  Error: {result.error}")


def check_api_key(key_name: str) -> bool:
    """Check if an API key is set."""
    return bool(os.getenv(key_name))


# ============== Individual Component Tests ==============

def test_query_router() -> TestResult:
    """Test QueryRouter with Mistral API."""
    name = "QueryRouter (ministral-3b)"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.graders import QueryRouter
        
        router = QueryRouter()
        result = router.route("What is GPT-4's MMLU score?")
        
        assert result.datasource in ["vectorstore", "web_search"], f"Invalid datasource: {result.datasource}"
        assert result.reasoning, "No reasoning provided"
        
        return TestResult(
            name, 
            True, 
            f"Routed to '{result.datasource}' - {result.reasoning[:50]}..."
        )
    except Exception as e:
        return TestResult(name, False, "Failed to route query", str(e))


def test_query_rewriter() -> TestResult:
    """Test QueryRewriter with conversation history."""
    name = "QueryRewriter (mistral-small)"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.graders import QueryRewriter
        
        rewriter = QueryRewriter()
        
        history = [
            {"role": "user", "content": "Tell me about GPT-4"},
            {"role": "assistant", "content": "GPT-4 is a large multimodal model by OpenAI..."},
        ]
        
        result = rewriter.rewrite("What about its MMLU score?", history)
        
        assert result.query, "No rewritten query"
        assert "mmlu" in result.query.lower() or "gpt" in result.query.lower(), \
            f"Query doesn't seem to include context: {result.query}"
        
        return TestResult(
            name, 
            True, 
            f"Rewrote to: '{result.query}'"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to rewrite query", str(e))


def test_document_grader() -> TestResult:
    """Test DocumentGrader batch relevance grading."""
    name = "DocumentGrader (mistral-large)"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.graders import DocumentGrader
        from crag.retrieval import Document
        
        grader = DocumentGrader()
        
        docs = [
            Document(content="GPT-4 achieved 86.4% accuracy on the MMLU benchmark."),
            Document(content="The weather in Paris is sunny today."),  # Irrelevant
            Document(content="Claude 2 scored 78.5% on MMLU evaluation."),
        ]
        
        filtered, needs_web = grader.grade_documents(docs, "What is GPT-4's MMLU score?")
        
        assert len(filtered) >= 1, "Should keep at least the relevant document"
        assert len(filtered) <= 2, "Should filter out the weather document"
        
        return TestResult(
            name, 
            True, 
            f"Filtered {len(docs)} → {len(filtered)} docs, needs_web_search={needs_web}"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to grade documents", str(e))


def test_document_reranker() -> TestResult:
    """Test DocumentReranker with Cohere API."""
    name = "DocumentReranker (Cohere)"
    
    if not check_api_key("COHERE_API_KEY"):
        return TestResult(name, False, "COHERE_API_KEY not set (skipped)")
    
    try:
        from crag.graders import DocumentReranker
        from crag.retrieval import Document
        
        reranker = DocumentReranker()
        
        docs = [
            Document(content="The weather is nice today.", score=0.9),
            Document(content="GPT-4 achieved 86.4% on MMLU.", score=0.5),
            Document(content="Paris is the capital of France.", score=0.8),
            Document(content="MMLU tests broad knowledge across 57 subjects.", score=0.3),
        ]
        
        reranked = reranker.rerank(docs, "What is GPT-4's MMLU score?", top_k=2)
        
        assert len(reranked) == 2, f"Should return 2 docs, got {len(reranked)}"
        
        # The GPT-4 MMLU doc should be ranked higher after reranking
        top_content = reranked[0].content.lower()
        assert "gpt-4" in top_content or "mmlu" in top_content, \
            f"Top doc should be about GPT-4/MMLU, got: {reranked[0].content[:50]}"
        
        return TestResult(
            name, 
            True, 
            f"Reranked {len(docs)} → {len(reranked)} docs, top: '{reranked[0].content[:40]}...'"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to rerank with Cohere", str(e))


def test_generation_grader() -> TestResult:
    """Test GenerationGrader for hallucination and usefulness."""
    name = "GenerationGrader (mistral-small)"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.graders import GenerationGrader
        
        grader = GenerationGrader()
        
        documents = "GPT-4 achieved 86.4% accuracy on the MMLU benchmark."
        generation = "According to the document, GPT-4 scored 86.4% on MMLU."
        question = "What is GPT-4's MMLU score?"
        
        result = grader.grade(documents, generation, question)
        
        assert result in ["useful", "not supported", "not useful"], f"Invalid grade: {result}"
        
        # This should be graded as useful (grounded + answers question)
        expected = "useful"
        
        return TestResult(
            name, 
            True, 
            f"Graded as '{result}' (expected '{expected}')"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to grade generation", str(e))


def test_inmemory_retriever() -> TestResult:
    """Test InMemoryHybridRetriever."""
    name = "InMemoryHybridRetriever"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set (needed for embeddings)")
    
    try:
        from crag.retrieval import InMemoryHybridRetriever, Document
        
        retriever = InMemoryHybridRetriever()
        
        docs = [
            Document(content="GPT-4 achieved 86.4% on MMLU.", id="doc1"),
            Document(content="Claude 2 scored 78.5% on MMLU.", id="doc2"),
            Document(content="Llama 2 70B got 68.9% on MMLU.", id="doc3"),
        ]
        
        retriever.add_documents(docs)
        
        results = retriever.retrieve("What is GPT-4's MMLU score?", top_k=2)
        
        assert len(results) == 2, f"Should return 2 docs, got {len(results)}"
        assert any("gpt-4" in r.content.lower() for r in results), \
            "GPT-4 doc should be in top results"
        
        return TestResult(
            name, 
            True, 
            f"Retrieved {len(results)} docs, top: '{results[0].content[:40]}...'"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to retrieve documents", str(e))


def test_pinecone_retriever() -> TestResult:
    """Test PineconeHybridRetriever connection."""
    name = "PineconeHybridRetriever"
    
    if not check_api_key("PINECONE_API_KEY"):
        return TestResult(name, False, "PINECONE_API_KEY not set (skipped)")
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set (needed for embeddings)")
    
    try:
        from crag.retrieval import PineconeHybridRetriever
        
        # Just test connection, don't create a real index
        retriever = PineconeHybridRetriever(
            index_name="crag-test-connection",
            create_index=False  # Don't actually create
        )
        
        # Test that we can get Pinecone client
        pc = retriever._get_pinecone()
        indexes = [idx.name for idx in pc.list_indexes()]
        
        return TestResult(
            name, 
            True, 
            f"Connected to Pinecone, found {len(indexes)} indexes"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to connect to Pinecone", str(e))


def test_web_search() -> TestResult:
    """Test Tavily web search integration."""
    name = "TavilySearch"
    
    if not check_api_key("TAVILY_API_KEY"):
        return TestResult(name, False, "TAVILY_API_KEY not set (skipped)")
    
    try:
        from langchain_tavily import TavilySearch
        
        search = TavilySearch(max_results=2)
        results = search.invoke({"query": "OpenAI GPT-4 release date"})
        
        assert results, "No search results returned"
        
        if isinstance(results, list):
            count = len(results)
        else:
            count = 1
        
        return TestResult(
            name, 
            True, 
            f"Retrieved {count} web result(s)"
        )
    except Exception as e:
        return TestResult(name, False, "Failed to perform web search", str(e))


def test_full_pipeline() -> TestResult:
    """Test the full CRAG pipeline end-to-end."""
    name = "Full CRAG Pipeline"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag import run_crag, InMemoryHybridRetriever, Document
        
        # Use in-memory retriever for testing
        retriever = InMemoryHybridRetriever()
        
        docs = [
            Document(content="GPT-4 achieved 86.4% accuracy on the MMLU benchmark, significantly outperforming previous models."),
            Document(content="Claude 2 by Anthropic scored 78.5% on MMLU evaluation."),
            Document(content="The MMLU benchmark tests knowledge across 57 academic subjects."),
        ]
        retriever.add_documents(docs)
        
        # Run with minimal output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            result = run_crag(
                "What is GPT-4's MMLU score?",
                retriever=retriever,
                retrieval_top_k=3,  # Small for testing
                rerank_top_k=2,
            )
        
        assert result.get("generation"), "No generation produced"
        assert result.get("route_decision"), "No route decision"
        assert result.get("documents"), "No documents in final state"
        
        gen_preview = result["generation"][:80].replace('\n', ' ')
        
        return TestResult(
            name, 
            True, 
            f"Generated answer: '{gen_preview}...'"
        )
    except Exception as e:
        return TestResult(name, False, "Pipeline failed", str(e))


def test_full_pipeline_with_history() -> TestResult:
    """Test CRAG pipeline with conversation history."""
    name = "CRAG Pipeline with History"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag import run_crag, InMemoryHybridRetriever, Document
        
        retriever = InMemoryHybridRetriever()
        
        docs = [
            Document(content="GPT-4 achieved 86.4% accuracy on the MMLU benchmark."),
            Document(content="GPT-4 scored 95.3% on HellaSwag benchmark."),
            Document(content="Claude 2 scored 78.5% on MMLU."),
        ]
        retriever.add_documents(docs)
        
        history = [
            {"role": "user", "content": "Tell me about GPT-4 benchmarks"},
            {"role": "assistant", "content": "GPT-4 performs well on various benchmarks including MMLU and HellaSwag."},
        ]
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            result = run_crag(
                "What about HellaSwag specifically?",  # Contextual question
                retriever=retriever,
                history=history,
                retrieval_top_k=3,
                rerank_top_k=2,
            )
        
        assert result.get("generation"), "No generation produced"
        assert result.get("rewritten_query"), "Query should be rewritten"
        
        # The rewritten query should mention HellaSwag and possibly GPT-4
        rewritten = result["rewritten_query"].lower()
        assert "hellaswag" in rewritten, f"Rewritten query should mention HellaSwag: {result['rewritten_query']}"
        
        return TestResult(
            name, 
            True, 
            f"Rewritten: '{result['rewritten_query']}'"
        )
    except Exception as e:
        return TestResult(name, False, "Pipeline with history failed", str(e))


# ============== Session Management Tests ==============

def test_session_inmemory() -> TestResult:
    """Test in-memory session management."""
    name = "Session (InMemory)"
    
    try:
        from crag.session import Session, SessionManager, InMemorySessionStore
        
        # Create manager with in-memory store explicitly
        store = InMemorySessionStore()
        manager = SessionManager(store=store)
        
        # Test session creation
        session = manager.create_session(metadata={"user": "test"})
        assert session.id, "Session should have an ID"
        assert session.metadata.get("user") == "test", "Metadata should be stored"
        assert session.turn_count() == 0, "New session should have 0 turns"
        
        # Test adding turns
        session.add_turn("What is GPT-4?", "GPT-4 is a large language model by OpenAI.")
        assert session.turn_count() == 1, "Should have 1 turn after add_turn"
        assert len(session.get_history()) == 2, "History should have 2 messages"
        
        # Test save and retrieve
        manager.save_session(session)
        retrieved = manager.get_session(session.id)
        assert retrieved is not None, "Should retrieve saved session"
        assert retrieved.turn_count() == 1, "Retrieved session should have 1 turn"
        
        # Test get_or_create with existing
        same_session = manager.get_or_create(session.id)
        assert same_session.id == session.id, "get_or_create should return existing session"
        
        # Test get_or_create without ID
        new_session = manager.get_or_create(None)
        assert new_session.id != session.id, "get_or_create with None should create new session"
        
        # Test delete
        deleted = manager.delete_session(session.id)
        assert deleted, "delete_session should return True"
        assert manager.get_session(session.id) is None, "Deleted session should not be found"
        
        return TestResult(
            name,
            True,
            f"All session operations work correctly. Storage: {manager.get_storage_type()}"
        )
    except Exception as e:
        return TestResult(name, False, "Session test failed", str(e))


def test_session_history_limit() -> TestResult:
    """Test that session history is properly limited."""
    name = "Session History Limit"
    
    try:
        from crag.session import Session, SessionManager, InMemorySessionStore
        
        # Create manager with small max_turns for testing
        store = InMemorySessionStore()
        manager = SessionManager(store=store, max_turns=3)
        session = manager.create_session()
        
        # Add more turns than max_turns
        for i in range(5):
            session.add_turn(f"Question {i+1}", f"Answer {i+1}")
        
        # Should only keep last 3 turns (6 messages)
        history = session.get_history()
        assert len(history) == 6, f"History should have 6 messages (3 turns), got {len(history)}"
        assert session.turn_count() == 3, f"Turn count should be 3, got {session.turn_count()}"
        
        # Verify it's the LAST 3 turns
        assert history[0]["content"] == "Question 3", f"First message should be Question 3, got {history[0]['content']}"
        assert history[-1]["content"] == "Answer 5", f"Last message should be Answer 5, got {history[-1]['content']}"
        
        return TestResult(
            name,
            True,
            f"History correctly limited to 3 turns (oldest trimmed)"
        )
    except Exception as e:
        return TestResult(name, False, "History limit test failed", str(e))


def test_session_expiry() -> TestResult:
    """Test session expiration."""
    name = "Session Expiry"
    
    try:
        from crag.session import Session, SessionManager, InMemorySessionStore
        from datetime import datetime, timedelta
        
        store = InMemorySessionStore()
        manager = SessionManager(store=store, timeout_minutes=1)  # 1 minute timeout
        session = manager.create_session()
        session_id = session.id
        
        # Session should be valid immediately
        assert manager.get_session(session_id) is not None, "Fresh session should be valid"
        
        # Manually expire the session by setting last_activity in the past
        session.last_activity = datetime.now() - timedelta(minutes=5)
        manager.save_session(session)
        
        # Session should now be expired
        expired_session = manager.get_session(session_id)
        assert expired_session is None, "Expired session should return None"
        
        # Test is_expired method directly
        fresh_session = manager.create_session()
        assert not fresh_session.is_expired(1), "Fresh session should not be expired"
        
        fresh_session.last_activity = datetime.now() - timedelta(minutes=5)
        assert fresh_session.is_expired(1), "Old session should be expired"
        
        return TestResult(
            name,
            True,
            "Session expiry works correctly"
        )
    except Exception as e:
        return TestResult(name, False, "Expiry test failed", str(e))


def test_session_serialization() -> TestResult:
    """Test session serialization/deserialization."""
    name = "Session Serialization"
    
    try:
        from crag.session import Session
        
        # Create a session with data
        session = Session(
            id="test-123",
            max_turns=10,
            metadata={"user_id": "user-456", "preference": "dark_mode"}
        )
        session.add_turn("Hello", "Hi there!")
        session.add_turn("How are you?", "I'm doing well, thanks!")
        
        # Serialize to dict
        data = session.to_dict()
        assert data["id"] == "test-123", "ID should be preserved"
        assert len(data["history"]) == 4, "History should have 4 messages"
        assert data["metadata"]["user_id"] == "user-456", "Metadata should be preserved"
        
        # Deserialize from dict
        restored = Session.from_dict(data)
        assert restored.id == session.id, "Restored ID should match"
        assert restored.turn_count() == session.turn_count(), "Restored turn count should match"
        assert restored.metadata == session.metadata, "Restored metadata should match"
        
        return TestResult(
            name,
            True,
            "Session serialization/deserialization works correctly"
        )
    except Exception as e:
        return TestResult(name, False, "Serialization test failed", str(e))


def test_session_redis() -> TestResult:
    """Test Redis session storage (if REDIS_URL is set)."""
    name = "Session (Redis)"
    
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return TestResult(name, False, "REDIS_URL not set (skipped)")
    
    try:
        from crag.session import SessionManager, RedisSessionStore
        
        # Create manager with Redis store
        store = RedisSessionStore(redis_url=redis_url)
        manager = SessionManager(store=store)
        
        # Test basic operations
        session = manager.create_session(metadata={"test": "redis"})
        session.add_turn("Test question", "Test answer")
        manager.save_session(session)
        
        # Retrieve and verify
        retrieved = manager.get_session(session.id)
        assert retrieved is not None, "Should retrieve session from Redis"
        assert retrieved.turn_count() == 1, "Should have 1 turn"
        assert retrieved.metadata.get("test") == "redis", "Metadata should be preserved"
        
        # Cleanup
        manager.delete_session(session.id)
        
        return TestResult(
            name,
            True,
            f"Redis session storage works correctly"
        )
    except ImportError:
        return TestResult(name, False, "redis package not installed (skipped)")
    except Exception as e:
        return TestResult(name, False, "Redis session test failed", str(e))


# ============== Streaming Tests ==============

def test_stream_event_format() -> TestResult:
    """Test StreamEvent creation and SSE formatting."""
    name = "StreamEvent Format"
    
    try:
        from crag.streaming import (
            StreamEvent, status_event, token_event, error_event, done_event
        )
        import json
        
        # Test status event
        evt = status_event("routing", 45, {"decision": "vectorstore"})
        assert evt.event == "status", "Event type should be 'status'"
        assert evt.data["latency_ms"] == 45, "Latency should be 45"
        assert evt.data["details"]["decision"] == "vectorstore", "Details should be preserved"
        
        # Test SSE format
        sse = evt.to_sse()
        assert sse.startswith("data: "), "SSE should start with 'data: '"
        assert sse.endswith("\n\n"), "SSE should end with double newline"
        
        # Parse SSE JSON
        json_part = sse[6:-2]  # Remove 'data: ' and '\n\n'
        parsed = json.loads(json_part)
        assert parsed["event"] == "status", "Parsed event type should match"
        assert parsed["data"]["latency_ms"] == 45, "Parsed data should match"
        
        # Test token event
        evt = token_event("GPT-4")
        assert evt.event == "token", "Event type should be 'token'"
        assert evt.data["content"] == "GPT-4", "Content should be 'GPT-4'"
        
        # Test error event
        evt = error_event("Connection failed", step="retrieving")
        assert evt.event == "error", "Event type should be 'error'"
        assert evt.data["message"] == "Connection failed", "Message should match"
        assert evt.data["step"] == "retrieving", "Step should match"
        
        # Test done event
        evt = done_event("sess-123", 1500, grade="useful", generation="The answer")
        assert evt.event == "done", "Event type should be 'done'"
        assert evt.data["session_id"] == "sess-123", "Session ID should match"
        assert evt.data["total_ms"] == 1500, "Total ms should match"
        assert evt.data["grade"] == "useful", "Grade should match"
        
        return TestResult(
            name,
            True,
            "All event formats correct, SSE parsing works"
        )
    except Exception as e:
        return TestResult(name, False, "Event format test failed", str(e))


def test_stream_timer() -> TestResult:
    """Test Timer latency tracking."""
    name = "Streaming Timer"
    
    try:
        from crag.streaming import Timer
        import time
        
        timer = Timer()
        
        # Test elapsed timing
        timer.start()
        time.sleep(0.03)  # 30ms
        elapsed = timer.elapsed_ms()
        assert 20 <= elapsed <= 80, f"Expected ~30ms, got {elapsed}ms"
        
        # Test reset
        timer.start()
        time.sleep(0.02)  # 20ms
        elapsed2 = timer.elapsed_ms()
        assert 10 <= elapsed2 <= 60, f"Expected ~20ms after reset, got {elapsed2}ms"
        
        # Test total (should accumulate)
        total = timer.total_ms()
        assert total >= 40, f"Total should be >= 40ms, got {total}ms"
        
        return TestResult(
            name,
            True,
            f"Timer works: elapsed={elapsed2}ms, total={total}ms"
        )
    except Exception as e:
        return TestResult(name, False, "Timer test failed", str(e))


def test_pipeline_streamer_init() -> TestResult:
    """Test PipelineStreamer initialization."""
    name = "PipelineStreamer Init"
    
    if not check_api_key("MISTRAL_API_KEY"):
        return TestResult(name, False, "MISTRAL_API_KEY not set")
    
    try:
        from crag.streaming import PipelineStreamer
        from crag.retrieval import InMemoryHybridRetriever
        
        retriever = InMemoryHybridRetriever()
        
        # Test initialization with defaults
        streamer = PipelineStreamer(
            retriever=retriever,
            retrieval_top_k=10,
            rerank_top_k=5,
        )
        
        assert streamer.retriever is retriever, "Retriever should be set"
        assert streamer.retrieval_top_k == 10, "retrieval_top_k should be 10"
        assert streamer.rerank_top_k == 5, "rerank_top_k should be 5"
        assert streamer.router is not None, "Router should be auto-created"
        assert streamer.doc_grader is not None, "DocGrader should be auto-created"
        
        return TestResult(
            name,
            True,
            "PipelineStreamer initializes correctly with defaults"
        )
    except Exception as e:
        return TestResult(name, False, "PipelineStreamer init failed", str(e))


def test_api_health_endpoint() -> TestResult:
    """Test the FastAPI health endpoint."""
    name = "API Health Endpoint"
    
    try:
        from fastapi.testclient import TestClient
        from crag.api import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["status"] == "healthy", "Status should be 'healthy'"
        assert "retriever" in data, "Should include retriever type"
        assert "session_storage" in data, "Should include session storage type"
        
        return TestResult(
            name,
            True,
            f"Health endpoint works: {data['retriever']}, {data['session_storage']}"
        )
    except ImportError as e:
        return TestResult(name, False, f"FastAPI not installed: {e}")
    except Exception as e:
        return TestResult(name, False, "Health endpoint test failed", str(e))


def test_api_session_endpoints() -> TestResult:
    """Test the FastAPI session endpoints."""
    name = "API Session Endpoints"
    
    try:
        from fastapi.testclient import TestClient
        from crag.api import app
        
        client = TestClient(app)
        
        # Create session
        response = client.post("/sessions")
        assert response.status_code == 200, f"Create: expected 200, got {response.status_code}"
        data = response.json()
        session_id = data["session_id"]
        assert session_id, "Should return session_id"
        
        # Get session
        response = client.get(f"/sessions/{session_id}")
        assert response.status_code == 200, f"Get: expected 200, got {response.status_code}"
        data = response.json()
        assert data["id"] == session_id, "Session ID should match"
        assert data["turns"] == 0, "New session should have 0 turns"
        
        # Delete session
        response = client.delete(f"/sessions/{session_id}")
        assert response.status_code == 200, f"Delete: expected 200, got {response.status_code}"
        
        # Verify deleted
        response = client.get(f"/sessions/{session_id}")
        assert response.status_code == 404, f"After delete: expected 404, got {response.status_code}"
        
        return TestResult(
            name,
            True,
            "Session endpoints work: create, get, delete"
        )
    except ImportError as e:
        return TestResult(name, False, f"FastAPI not installed: {e}")
    except Exception as e:
        return TestResult(name, False, "Session endpoints test failed", str(e))


# ============== Test Runner ==============

ALL_TESTS = {
    "router": test_query_router,
    "rewriter": test_query_rewriter,
    "doc_grader": test_document_grader,
    "reranker": test_document_reranker,
    "gen_grader": test_generation_grader,
    "retriever_memory": test_inmemory_retriever,
    "retriever_pinecone": test_pinecone_retriever,
    "web_search": test_web_search,
    "full_pipeline": test_full_pipeline,
    "pipeline_history": test_full_pipeline_with_history,
    # Session Management Tests
    "session_inmemory": test_session_inmemory,
    "session_history_limit": test_session_history_limit,
    "session_expiry": test_session_expiry,
    "session_serialization": test_session_serialization,
    "session_redis": test_session_redis,
    # Streaming Tests
    "stream_event_format": test_stream_event_format,
    "stream_timer": test_stream_timer,
    "stream_pipeline_init": test_pipeline_streamer_init,
    # API Tests
    "api_health": test_api_health_endpoint,
    "api_sessions": test_api_session_endpoints,
}


def run_tests(components: Optional[list] = None) -> list:
    """Run specified tests or all tests."""
    if components:
        tests_to_run = {k: v for k, v in ALL_TESTS.items() if k in components}
    else:
        tests_to_run = ALL_TESTS
    
    results = []
    
    print("\n" + "=" * 60)
    print("CRAG API Integration Tests")
    print("=" * 60)
    
    # Check API keys first
    print("\nAPI Keys Status:")
    api_keys = ["MISTRAL_API_KEY", "PINECONE_API_KEY", "COHERE_API_KEY", "TAVILY_API_KEY"]
    for key in api_keys:
        status = "✓ Set" if check_api_key(key) else "✗ Not set"
        color = "\033[92m" if check_api_key(key) else "\033[93m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} - {key}")
    
    print("\n" + "-" * 60)
    print("Running Tests...")
    print("-" * 60)
    
    for name, test_fn in tests_to_run.items():
        print(f"\nTesting: {name}...")
        result = test_fn()
        results.append(result)
        print_result(result)
    
    return results


def print_summary(results: list):
    """Print test summary."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    skipped = sum(1 for r in results if not r.passed and "not set" in r.message.lower())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Total:   {len(results)}")
    print(f"  \033[92mPassed:  {passed}\033[0m")
    print(f"  \033[91mFailed:  {failed - skipped}\033[0m")
    print(f"  \033[93mSkipped: {skipped}\033[0m (missing API keys)")
    print("=" * 60)
    
    if failed - skipped > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed and "not set" not in r.message.lower():
                print(f"  - {r.name}: {r.error or r.message}")
    
    return failed - skipped == 0


def main():
    parser = argparse.ArgumentParser(description="Test CRAG API integrations")
    parser.add_argument(
        "--component",
        choices=list(ALL_TESTS.keys()),
        help="Test specific component only"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available tests:")
        for name in ALL_TESTS.keys():
            print(f"  - {name}")
        return
    
    components = [args.component] if args.component else None
    results = run_tests(components)
    success = print_summary(results)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
