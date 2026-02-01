"""
Centralized configuration for CRAG pipeline models.

Edit this file to change models used across the pipeline.
All model assignments are in one place for easy tuning.
"""

# ============== Model Configurations ==============

# Routing & lightweight tasks (fast, cheap)
ROUTER_MODEL = "ministral-3b-latest"
CONVERSATIONAL_MODEL = "mistral-small-latest"

# Query processing
REWRITER_MODEL = "mistral-small-latest"

# Document grading (needs good context understanding)
DOC_GRADER_MODEL = "mistral-large-latest"

# Generation grading
GEN_GRADER_MODEL = "mistral-small-latest"

# Research generation (best quality for citations)
GENERATION_MODEL = "mistral-large-latest"

# Embeddings
EMBEDDING_MODEL = "mistral-embed"

# Legacy alias for backwards compatibility
DEFAULT_MODEL = GENERATION_MODEL

# ============== Pipeline Settings ==============

RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 10
MAX_GENERATION_TOKENS = 4000

# Document processing limits
MAX_DOC_LENGTH = 2000       # For grading
MAX_RERANK_LENGTH = 4000    # For Cohere reranking
MAX_HISTORY_LENGTH = 500    # For query rewriting

# ============== System Prompts ==============

CONVERSATIONAL_SYSTEM_PROMPT = """You are PaperRAG, a friendly research assistant
 specializing in Artificial Intelligence and Machine Learning

You have access to ~6,000 indexed NeurIPS papers including, but NOT LIMITED TO:
- Transformers, attention mechanisms, and LLMs
- Reinforcement learning and decision-making
- Diffusion models and generative AI
- Computer vision and multimodal learning
- Optimization, theory, and benchmarks
- And many other cross-disciplinary topics in the field of Artificial Intelligence and Machine Learning

INSTRUCTIONS:
- Respond warmly and welcomingly
- If greeted, introduce yourself and your capabilities
- Respond to all queries in a respective way that is suitable for the query and the user's intent.
- Suggest relevant research topics the user might explore
- Keep responses concise (2-3 sentences max for greetings)
- Do NOT use any emojis or em-dashes
- Always end with an engaging question, but still related to their query, to guide them toward research
"""
