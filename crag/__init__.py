"""
Corrective RAG (CRAG) with Hybrid Search using LangGraph.

A stateful RAG workflow that:
1. Routes queries to vectorstore or web search
2. Performs hybrid retrieval via Pinecone (BM25 + Semantic)
3. Grades document relevance with Mistral Large (batch mode)
4. Falls back to web search if context is insufficient
5. Validates generation for hallucinations and usefulness (single LLM call)

All LLM operations use Mistral Large (mistral-large-latest).
Vector storage uses Pinecone with persistent hybrid search.
Uses closure pattern for dependency injection - no global state.

Usage:
    >>> from crag import run_crag, PineconeHybridRetriever, Document
    >>> 
    >>> # Create retriever (connects to Pinecone index)
    >>> retriever = PineconeHybridRetriever(
    ...     index_name="my-crag-index",
    ...     namespace="my-project"
    ... )
    >>> 
    >>> # Add documents (one-time indexing)
    >>> docs = [
    ...     Document(content="GPT-4 achieved 86.4% on MMLU."),
    ...     Document(content="Claude 2 scored 78.5% on MMLU."),
    ... ]
    >>> retriever.add_documents(docs)
    >>> 
    >>> # Run CRAG (retriever is required)
    >>> result = run_crag("What is GPT-4's MMLU score?", retriever=retriever)
    >>> print(result["generation"])
"""

from .graph_state import GraphState
from .retrieval import (
    Document,
    HybridRetriever,
    PineconeHybridRetriever,
    InMemoryHybridRetriever,
)
from .graders import (
    QueryRouter,
    QueryRewriter,
    QueryRewriterAndRouter,
    DocumentGrader,
    DocumentReranker,
    GenerationGrader,
    GradeGeneration,
    GradeDocumentsBatch,
    RewrittenQuery,
    RewriteAndRoute,
)
from .config import (
    ROUTER_MODEL,
    CONVERSATIONAL_MODEL,
    REWRITER_MODEL,
    QUERY_PROCESSOR_MODEL,
    DOC_GRADER_MODEL,
    GEN_GRADER_MODEL,
    GENERATION_MODEL,
    EMBEDDING_MODEL,
    DEFAULT_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    MAX_GENERATION_TOKENS,
    # System Prompts
    CONVERSATIONAL_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    DOC_GRADER_SYSTEM_PROMPT,
    GEN_GRADER_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_SIMPLE_PROMPT,
    CITATION_SYSTEM_PROMPT,
)
from .graph_state import Message
from .graph import (
    create_crag_graph,
    compile_crag_graph,
    run_crag,
    stream_crag,
)
from .session import (
    Session,
    SessionManager,
    SessionStore,
    InMemorySessionStore,
    RedisSessionStore,
    create_session_manager,
)
from .streaming import (
    StreamEvent,
    PipelineStreamer,
    stream_crag_with_events,
    status_event,
    token_event,
    error_event,
    done_event,
    citation_event,
)
from .citations import (
    Citation,
    CitedResponse,
    CitationExtractor,
    SourceInfo,
    extract_citations,
    format_citations_for_display,
)

__all__ = [
    # State
    "GraphState",
    "Message",
    # Retrieval
    "Document",
    "HybridRetriever",
    "PineconeHybridRetriever",
    "InMemoryHybridRetriever",
    # Graders
    "QueryRouter",
    "QueryRewriter",
    "DocumentGrader",
    "DocumentReranker",
    "GenerationGrader",
    "GradeGeneration",
    "GradeDocumentsBatch",
    "RewrittenQuery",
    # Model configs
    "DEFAULT_MODEL",
    "ROUTER_MODEL",
    "REWRITER_MODEL",
    "QUERY_PROCESSOR_MODEL",
    "DOC_GRADER_MODEL",
    "GEN_GRADER_MODEL",
    # System Prompts
    "CONVERSATIONAL_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "REWRITER_SYSTEM_PROMPT",
    "DOC_GRADER_SYSTEM_PROMPT",
    "GEN_GRADER_SYSTEM_PROMPT",
    "GENERATION_SYSTEM_PROMPT",
    "GENERATION_SIMPLE_PROMPT",
    "CITATION_SYSTEM_PROMPT",
    # Graph
    "create_crag_graph",
    "compile_crag_graph",
    "run_crag",
    "stream_crag",
    # Session Management
    "Session",
    "SessionManager",
    "SessionStore",
    "InMemorySessionStore",
    "RedisSessionStore",
    "create_session_manager",
    # Streaming
    "StreamEvent",
    "PipelineStreamer",
    "stream_crag_with_events",
    "status_event",
    "token_event",
    "error_event",
    "done_event",
    "citation_event",
    # Citations
    "Citation",
    "CitedResponse",
    "CitationExtractor",
    "SourceInfo",
    "extract_citations",
    "format_citations_for_display",
]
