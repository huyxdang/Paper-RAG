"""
Structured Citations for CRAG.

Provides structured citation extraction and mapping from LLM responses
to source documents. Enables machine-readable references for frontend display.

Usage:
    from crag.citations import CitationExtractor, Citation, CitedResponse
    
    extractor = CitationExtractor()
    response = extractor.generate_with_citations(
        question="What is GPT-4's MMLU score?",
        documents=documents,
        mistral_client=client,
    )
    
    print(response.answer)  # "GPT-4 achieved 86.4% on MMLU [1]."
    print(response.citations[0].source_title)  # "GPT-4 Technical Report"
"""

import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple

from .config import GENERATION_SYSTEM_PROMPT


# ============== Data Classes ==============

@dataclass
class SourceInfo:
    """Information about a source document."""
    title: str
    url: str = ""
    section: str = ""
    keywords: str = ""
    relevance_score: float = 0.0
    doc_index: int = 0  # Index in the documents list (0-based)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    """A single citation linking a claim to a source."""
    ref: int  # Reference number as shown in text [1], [2], etc.
    claim: str  # The claim being cited
    quote: str = ""  # Optional exact quote from source
    source: Optional[SourceInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "ref": self.ref,
            "claim": self.claim,
        }
        if self.quote:
            result["quote"] = self.quote
        if self.source:
            result["source"] = self.source.to_dict()
        return result


@dataclass
class CitedResponse:
    """A response with structured citations."""
    answer: str  # The answer text with [1], [2] references
    citations: List[Citation] = field(default_factory=list)
    sources_used: int = 0
    total_sources: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources_used": self.sources_used,
            "total_sources": self.total_sources,
        }


# ============== Citation Prompts ==============

# GENERATION_SYSTEM_PROMPT is imported from config.py

CITATION_USER_PROMPT = """## Documents:

{context}

## Question:
{question}

## Instructions:
1. Answer the question using the documents above
2. Cite sources using [1], [2], etc.
3. End with a JSON block listing your citations

## Answer:"""


# ============== Citation Extractor ==============

class CitationExtractor:
    """
    Extracts structured citations from LLM responses.
    
    Can either:
    1. Parse citations from a response that already contains them
    2. Generate a new response with citations using Mistral
    """
    
    def __init__(self, model: str = "mistral-large-latest"):
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy init Mistral client."""
        if self._client is None:
            from mistralai import Mistral
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found")
            self._client = Mistral(api_key=api_key)
        return self._client
    
    def extract_citations_from_text(
        self,
        text: str,
        documents: List[Any],
    ) -> CitedResponse:
        """
        Extract citations from text that contains [N] references.
        
        Parses both inline [N] references and optional JSON citation block.
        Maps references to source documents.
        
        Args:
            text: The response text with [1], [2], etc. references
            documents: List of Document objects
            
        Returns:
            CitedResponse with parsed citations
        """
        # Try to extract JSON citation block
        json_citations = self._extract_json_citations(text)
        
        # Clean answer (remove JSON block if present)
        answer = self._clean_answer(text)
        
        # Find all [N] references in the text
        ref_numbers = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))
        
        # Build citations
        citations = []
        for ref_num in sorted(ref_numbers):
            doc_idx = ref_num - 1  # Convert to 0-based index

            # Get claim and quote from JSON if available
            json_cite = next((c for c in json_citations if c.get("ref") == ref_num), {})
            claim = json_cite.get("claim", "")
            quote = json_cite.get("quote", "")

            # If no claim from JSON, extract context around the reference
            if not claim:
                claim = self._extract_claim_context(answer, ref_num)

            # Build source info from document
            source = None
            doc_content = ""
            if 0 <= doc_idx < len(documents):
                doc = documents[doc_idx]
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                doc_content = doc.content if hasattr(doc, 'content') else str(doc)
                source = SourceInfo(
                    title=metadata.get("title", f"Document {ref_num}"),
                    url=metadata.get("url", ""),
                    section=metadata.get("section", ""),
                    keywords=metadata.get("keywords", ""),
                    relevance_score=doc.score if hasattr(doc, 'score') else 0.0,
                    doc_index=doc_idx,
                )

            # If no quote from LLM, extract the most relevant snippet from the source
            if not quote and doc_content and claim:
                quote = self._extract_best_snippet(doc_content, claim)

            citations.append(Citation(
                ref=ref_num,
                claim=claim,
                quote=quote,
                source=source,
            ))
        
        return CitedResponse(
            answer=answer,
            citations=citations,
            sources_used=len(citations),
            total_sources=len(documents),
        )
    
    def _extract_json_citations(self, text: str) -> List[Dict]:
        """Extract citations from JSON block in text."""
        # Look for ```json ... ``` block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data.get("citations", [])
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object at end of text
        json_match = re.search(r'\{[^{}]*"citations"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("citations", [])
            except json.JSONDecodeError:
                pass
        
        return []
    
    def _clean_answer(self, text: str) -> str:
        """Remove JSON block from answer text."""
        # Remove ```json ... ``` block
        text = re.sub(r'```json\s*\{.*?\}\s*```', '', text, flags=re.DOTALL)
        
        # Remove trailing JSON object
        text = re.sub(r'\{[^{}]*"citations"[^{}]*\[.*?\][^{}]*\}\s*$', '', text, flags=re.DOTALL)
        
        return text.strip()
    
    def _extract_claim_context(self, text: str, ref_num: int) -> str:
        """Extract the sentence/claim containing a reference."""
        # Find the reference marker position
        ref_pattern = r'\[' + str(ref_num) + r'\]'
        match = re.search(ref_pattern, text)
        if not match:
            return ""
        
        ref_start = match.start()
        ref_end = match.end()
        
        # Find sentence boundaries (handling decimal numbers by looking for ". " or ".\n")
        # Look backwards for sentence start
        sentence_start = 0
        for i in range(ref_start - 1, -1, -1):
            if text[i] in '.!?' and (i + 1 >= len(text) or text[i + 1] in ' \n\t'):
                sentence_start = i + 1
                break
        
        # Look forwards for sentence end
        sentence_end = len(text)
        for i in range(ref_end, len(text)):
            if text[i] in '.!?' and (i + 1 >= len(text) or text[i + 1] in ' \n\t'):
                sentence_end = i + 1
                break
        
        claim = text[sentence_start:sentence_end].strip()
        # Remove the reference marker from the claim
        claim = re.sub(rf'\s*{ref_pattern}\s*', ' ', claim).strip()
        return claim
    
    def _extract_best_snippet(self, doc_content: str, claim: str, max_len: int = 300) -> str:
        """
        Extract the most relevant snippet from a document given a claim.

        Uses keyword overlap to find the sentence(s) in the document
        that best match the claim.
        """
        # Split document into sentences
        sentences = re.split(r'(?<=[.!?])\s+', doc_content.strip())
        if not sentences:
            return ""

        # Extract keywords from the claim (words 4+ chars, lowercased)
        claim_words = set(
            w.lower() for w in re.findall(r'\b\w{4,}\b', claim)
        )
        if not claim_words:
            # Fall back to first sentence if claim has no meaningful keywords
            return sentences[0][:max_len]

        # Score each sentence by keyword overlap
        scored = []
        for sent in sentences:
            sent_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', sent))
            overlap = len(claim_words & sent_words)
            if overlap > 0:
                scored.append((overlap, sent))

        if not scored:
            # No keyword match — return first sentence as fallback
            return sentences[0][:max_len]

        # Sort by overlap descending, take the best match
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1].strip()

        # If the best sentence is short, try to include the next sentence for context
        if len(best) < 100 and len(scored) > 1:
            best = best + " " + scored[1][1].strip()

        return best[:max_len]

    def generate_with_citations(
        self,
        question: str,
        documents: List[Any],
        temperature: float = 0,
        max_tokens: int = 2000,
    ) -> CitedResponse:
        """
        Generate an answer with structured citations.
        
        Uses a specialized prompt to get the LLM to output citations
        in a structured format.
        
        Args:
            question: The user's question
            documents: List of Document objects
            temperature: LLM temperature
            max_tokens: Max tokens for response
            
        Returns:
            CitedResponse with answer and citations
        """
        if not documents:
            return CitedResponse(
                answer="I don't have enough information to answer this question.",
                citations=[],
                sources_used=0,
                total_sources=0,
            )
        
        # Format documents as context
        context_parts = []
        for i, doc in enumerate(documents):
            content = doc.content if hasattr(doc, 'content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            title = metadata.get("title", "")
            
            header = f"[Document {i+1}]"
            if title:
                header += f" - {title}"
            
            context_parts.append(f"{header}\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate response
        client = self._get_client()
        
        response = client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": CITATION_USER_PROMPT.format(
                    context=context,
                    question=question,
                )}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # Extract citations from response
        return self.extract_citations_from_text(raw_response, documents)
    
    def generate_with_citations_streaming(
        self,
        question: str,
        documents: List[Any],
        temperature: float = 0,
        max_tokens: int = 2000,
    ):
        """
        Generate answer with citations, streaming tokens.
        
        Yields tokens as they come, then returns the final CitedResponse.
        
        Args:
            question: The user's question
            documents: List of Document objects
            temperature: LLM temperature
            max_tokens: Max tokens for response
            
        Yields:
            str tokens during generation
            
        Returns:
            CitedResponse after generation completes (access via .send(None) after exhausting generator)
        """
        if not documents:
            yield "I don't have enough information to answer this question."
            return CitedResponse(
                answer="I don't have enough information to answer this question.",
                citations=[],
                sources_used=0,
                total_sources=0,
            )
        
        # Format context
        context_parts = []
        for i, doc in enumerate(documents):
            content = doc.content if hasattr(doc, 'content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            title = metadata.get("title", "")
            
            header = f"[Document {i+1}]"
            if title:
                header += f" - {title}"
            
            context_parts.append(f"{header}\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Stream response
        client = self._get_client()
        
        stream = client.chat.stream(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": CITATION_USER_PROMPT.format(
                    context=context,
                    question=question,
                )}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.data.choices and chunk.data.choices[0].delta.content:
                token = chunk.data.choices[0].delta.content
                full_response += token
                yield token
        
        # Extract citations from complete response
        cited_response = self.extract_citations_from_text(full_response, documents)
        
        # Store for retrieval (hacky but works with generators)
        self._last_response = cited_response
        
        return cited_response


# ============== Convenience Functions ==============

def extract_citations(
    text: str,
    documents: List[Any],
) -> CitedResponse:
    """
    Extract citations from response text.
    
    Convenience wrapper around CitationExtractor.
    
    Args:
        text: Response text with [1], [2], etc.
        documents: Source documents
        
    Returns:
        CitedResponse with parsed citations
    """
    extractor = CitationExtractor()
    return extractor.extract_citations_from_text(text, documents)


def format_citations_for_display(cited_response: CitedResponse) -> str:
    """
    Format citations for human-readable display.
    
    Args:
        cited_response: The cited response to format
        
    Returns:
        Formatted string with answer and sources
    """
    lines = [cited_response.answer, "", "---", "Sources:"]
    
    for citation in cited_response.citations:
        source = citation.source
        if source:
            line = f"  [{citation.ref}] {source.title}"
            if source.section:
                line += f" (section: {source.section})"
            if source.url:
                line += f"\n      {source.url}"
            lines.append(line)
        else:
            lines.append(f"  [{citation.ref}] Unknown source")
    
    return "\n".join(lines)
