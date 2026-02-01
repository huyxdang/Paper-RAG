"""
CRAG - Corrective RAG for NeurIPS 2025 Papers.

Query 130K+ paper chunks from NeurIPS 2025 using hybrid search (BM25 + semantic).
All LLM operations use Mistral models.

Usage (From the repo root):

    # Single question
    python -m crag.main --pinecone --question "What are the latest advances in reinforcement learning?"

    # Interactive mode
    python -m crag.main --pinecone --mode interactive

Required environment variables:
    MISTRAL_API_KEY   - For LLM operations
    PINECONE_API_KEY  - For vector search
    COHERE_API_KEY    - For document reranking
    TAVILY_API_KEY    - (Optional) For web search fallback
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crag.retrieval import PineconeHybridRetriever, InMemoryHybridRetriever, Document
from crag.graph import run_crag, stream_crag
from crag.session import SessionManager


def create_sample_documents() -> list[Document]:
    """Create sample documents for demonstration."""
    return [
        Document(
            content="""GPT-4 Technical Report Summary
            
GPT-4 is a large multimodal model that accepts image and text inputs and produces text outputs. 
It exhibits human-level performance on various professional and academic benchmarks.

Key Results:
- MMLU (5-shot): 86.4% accuracy
- HellaSwag (10-shot): 95.3% accuracy  
- WinoGrande (5-shot): 87.5% accuracy
- ARC Challenge (25-shot): 96.3% accuracy

GPT-4 significantly outperforms existing large language models on many NLP benchmarks.
The model was trained using a combination of supervised learning and RLHF.""",
            metadata={"source": "gpt4_paper.pdf", "page": 1},
            id="doc_gpt4"
        ),
        Document(
            content="""Claude 2 Benchmark Results

Anthropic's Claude 2 model demonstrates strong performance across various benchmarks:

- MMLU: 78.5% accuracy
- HellaSwag: 89.2% accuracy
- TruthfulQA: 72.1% accuracy
- Codex HumanEval: 71.2% pass@1

Claude 2 shows particular strength in safety evaluations and honest responses.
The model uses Constitutional AI (CAI) for alignment.""",
            metadata={"source": "claude2_report.pdf", "page": 1},
            id="doc_claude2"
        ),
        Document(
            content="""Llama 2 Model Family

Meta's Llama 2 is a collection of pretrained and fine-tuned LLMs ranging from 7B to 70B parameters.

Llama 2 70B Results:
- MMLU (5-shot): 68.9% accuracy
- TriviaQA (1-shot): 85.0% accuracy
- Natural Questions: 33.0% accuracy

Llama 2 is released under a permissive license that allows commercial use.
The fine-tuned chat models (Llama 2-Chat) are optimized for dialogue use cases.""",
            metadata={"source": "llama2_paper.pdf", "page": 2},
            id="doc_llama2"
        ),
        Document(
            content="""Retrieval Augmented Generation (RAG) Overview

RAG combines the benefits of retrieval systems with generative models:

1. Query Processing: User query is embedded
2. Retrieval: Relevant documents are fetched from a knowledge base
3. Augmentation: Retrieved context is prepended to the prompt
4. Generation: LLM generates response grounded in retrieved documents

Benefits:
- Reduces hallucinations by grounding in facts
- Enables up-to-date knowledge without retraining
- Provides source attribution for answers

Common retrieval methods include BM25, dense retrieval, and hybrid approaches.""",
            metadata={"source": "rag_survey.pdf", "page": 5},
            id="doc_rag"
        ),
        Document(
            content="""MMLU Benchmark Description

The Massive Multitask Language Understanding (MMLU) benchmark tests models on 57 subjects:

Categories:
- STEM: Physics, Chemistry, Math, Computer Science, Engineering
- Humanities: History, Philosophy, Law
- Social Sciences: Economics, Psychology, Sociology
- Other: Professional Medicine, Business, etc.

Each question is multiple choice (A, B, C, or D).
Human expert baseline varies by subject (typically 75-90%).

MMLU is considered one of the most comprehensive benchmarks for evaluating broad knowledge.""",
            metadata={"source": "mmlu_paper.pdf", "page": 1},
            id="doc_mmlu"
        ),
    ]
    

def get_retriever(use_pinecone: bool = True, index_name: str = "neurips-2025-hybrid"):
    """
    Get the appropriate retriever.
    
    Args:
        use_pinecone: If True, use Pinecone (persistent). If False, use in-memory (testing).
        index_name: Pinecone index name
        
    Returns:
        Configured retriever
    """
    if use_pinecone and os.getenv("PINECONE_API_KEY"):
        print(f"Using Pinecone hybrid retriever (index: {index_name})")
        return PineconeHybridRetriever(
            index_name=index_name,
            namespace="",  # Use default namespace (where your data lives)
            create_index=False  # Don't try to create, use existing
        )
    else:
        print("Using in-memory retriever (Pinecone not configured)")
        return InMemoryHybridRetriever()


def demo_basic_query(use_pinecone: bool = False):
    """Demonstrate basic CRAG query."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Query (Vectorstore Route)")
    print("=" * 70)
    
    # Create retriever with sample documents
    retriever = get_retriever(use_pinecone)
    retriever.add_documents(create_sample_documents())
    
    # Run CRAG
    result = run_crag(
        question="What is GPT-4's performance on the MMLU benchmark?",
        retriever=retriever
    )
    
    print("\n--- FINAL ANSWER ---")
    print(result["generation"])
    
    return result


def demo_comparison_query(use_pinecone: bool = False):
    """Demonstrate comparison query."""
    print("\n" + "=" * 70)
    print("DEMO 2: Comparison Query")
    print("=" * 70)
    
    retriever = get_retriever(use_pinecone)
    retriever.add_documents(create_sample_documents())
    
    result = run_crag(
        question="Compare GPT-4 and Claude 2 MMLU scores",
        retriever=retriever
    )
    
    print("\n--- FINAL ANSWER ---")
    print(result["generation"])
    
    return result


def demo_web_search_query(use_pinecone: bool = False):
    """Demonstrate query that needs web search."""
    print("\n" + "=" * 70)
    print("DEMO 3: Current Events Query (Web Search Route)")
    print("=" * 70)
    
    retriever = get_retriever(use_pinecone)
    retriever.add_documents(create_sample_documents())
    
    result = run_crag(
        question="What are the latest developments in AI regulation in 2024?",
        retriever=retriever
    )
    
    print("\n--- FINAL ANSWER ---")
    print(result["generation"])
    
    return result


def demo_streaming(use_pinecone: bool = False):
    """Demonstrate streaming execution."""
    print("\n" + "=" * 70)
    print("DEMO 4: Streaming Execution")
    print("=" * 70)
    
    retriever = get_retriever(use_pinecone)
    retriever.add_documents(create_sample_documents())
    
    print("\nStreaming through nodes:")
    for node_name, state in stream_crag("What is RAG?", retriever):
        doc_count = len(state.get("documents", []))
        gen_preview = state.get("generation", "")[:50]
        print(f"  [{node_name}] documents={doc_count}, generation='{gen_preview}...'")


def interactive_mode(use_pinecone: bool = False, index_name: str = "neurips-2025-hybrid"):
    """Interactive query mode with session-based conversation history."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE (with session history)")
    print("=" * 70)
    print("\nEnter questions to query the CRAG system.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'new' or 'clear' - Start a new conversation (clear history)")
    print("  'history' - Show conversation history")
    print()
    
    # Initialize retriever (uses existing Pinecone index)
    retriever = get_retriever(use_pinecone, index_name)
    
    # Initialize session manager (auto-detects Redis vs in-memory)
    session_manager = SessionManager()
    session = session_manager.create_session()
    
    print(f"Session ID: {session.id[:8]}...")
    print(f"Storage: {session_manager.get_storage_type()}")
    
    # Show index stats
    if use_pinecone:
        try:
            stats = retriever.get_stats()
            total = stats.get('total_vector_count', 0)
            print(f"Connected to index with {total:,} vectors.")
        except Exception:
            print("Connected to Pinecone index.")
    
    print()
    
    while True:
        try:
            # Show turn count in prompt
            turn_count = session.turn_count()
            prompt = f"[Turn {turn_count + 1}] Question: "
            question = input(prompt).strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if question.lower() in ('new', 'clear'):
                session = session_manager.create_session()
                print(f"\n--- New session started (ID: {session.id[:8]}...) ---\n")
                continue
            
            if question.lower() == 'history':
                history = session.get_history()
                if not history:
                    print("\n--- No conversation history ---\n")
                else:
                    print(f"\n--- Conversation History ({session.turn_count()} turns) ---")
                    for i, msg in enumerate(history):
                        role = msg['role'].upper()
                        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        print(f"  [{role}]: {content}")
                    print("-" * 40 + "\n")
                continue
            
            # Run CRAG with session history
            result = run_crag(
                question, 
                retriever=retriever,
                history=session.get_history()  # Pass conversation history
            )
            
            # Update session with this turn
            session.add_turn(question, result["generation"])
            session_manager.save_session(session)
            
            # Display results
            print("\n--- ANSWER ---")
            print(result["generation"])
            print("-" * 40)
            print(f"Route: {result['route_decision']}")
            print(f"Documents used: {len(result['documents'])}")
            print(f"Generation attempts: {result['generation_attempts']}")
            print(f"Generation grade: {result.get('generation_grade', 'N/A')}")
            print(f"Web search done: {result.get('web_search_done', False)}")
            if result.get('rewritten_query') and result['rewritten_query'] != question:
                print(f"Rewritten query: {result['rewritten_query']}")
            print(f"Session turns: {session.turn_count()}")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Check for required API keys
    if not os.getenv("MISTRAL_API_KEY"):
        print("ERROR: MISTRAL_API_KEY not set. All LLM operations will fail.")
        return
    if not os.getenv("PINECONE_API_KEY"):
        print("Warning: PINECONE_API_KEY not set. Using in-memory retriever (not persistent).")
    if not os.getenv("COHERE_API_KEY"):
        print("Warning: COHERE_API_KEY not set. Document reranking will fail.")
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not set. Web search fallback will fail.")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="CRAG - Corrective RAG with Hybrid Search (Mistral + Pinecone)")
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive", "basic", "compare", "web", "stream"],
        default="demo",
        help="Execution mode (default: demo)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to answer"
    )
    parser.add_argument(
        "--pinecone",
        action="store_true",
        help="Use Pinecone for persistent storage (requires PINECONE_API_KEY)"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="neurips-2025-hybrid",
        help="Pinecone index name (default: neurips-2025-hybrid)"
    )
    
    args = parser.parse_args()
    use_pinecone = args.pinecone and os.getenv("PINECONE_API_KEY")
    
    print("\n" + "=" * 70)
    print("CRAG - Corrective RAG with Hybrid Search")
    print("  LLM: Mistral Large (mistral-large-latest)")
    print(f"  Retriever: {'Pinecone Hybrid' if use_pinecone else 'In-Memory (testing)'}")
    print("=" * 70)
    
    if args.question:
        # Single question mode
        retriever = get_retriever(use_pinecone, args.index)
        result = run_crag(args.question, retriever=retriever)
        print("\n--- ANSWER ---")
        print(result["generation"])
        
    elif args.mode == "interactive":
        interactive_mode(use_pinecone, args.index)
        
    elif args.mode == "basic":
        demo_basic_query(use_pinecone)
        
    elif args.mode == "compare":
        demo_comparison_query(use_pinecone)
        
    elif args.mode == "web":
        demo_web_search_query(use_pinecone)
        
    elif args.mode == "stream":
        demo_streaming(use_pinecone)
        
    else:  # demo mode - run all demos
        demo_basic_query(use_pinecone)
        demo_comparison_query(use_pinecone)
        # demo_web_search_query(use_pinecone)  # Requires Tavily API key
        demo_streaming(use_pinecone)
        
        print("\n" + "=" * 70)
        print("All demos complete!")
        print("Run with --mode interactive for custom queries.")
        print("Run with --pinecone for persistent Pinecone storage.")
        print("=" * 70)


if __name__ == "__main__":
    main()
