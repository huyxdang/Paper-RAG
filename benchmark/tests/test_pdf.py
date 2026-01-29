"""
Test PDF extraction and Mistral vision on a sample document.
Runs multiple prompts to explore the model's understanding.

Usage:
    python test_pdf.py                          # Uses folder 1's PDF
    python test_pdf.py data/0/P19-1598.pdf      # Custom PDF path
    python test_pdf.py --model ministral-8b-latest  # Use smaller model
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from pdf_extract import extract_pdf, get_document_summary

load_dotenv(Path(__file__).parent.parent / ".env")


# Test prompts to explore document understanding
TEST_PROMPTS = [
    {
        "name": "Summary",
        "prompt": "Provide a concise summary of this document in 3-4 sentences. What is it about?"
    },
    {
        "name": "Key Findings",
        "prompt": "What are the main findings or conclusions of this paper?"
    },
    {
        "name": "Describe Figures",
        "prompt": "Describe each figure/chart in the document. What data does each one visualize? What patterns or trends do you see?"
    },
    {
        "name": "Methodology",
        "prompt": "What methodology or approach did the authors use? Summarize in 2-3 sentences."
    },
    {
        "name": "Data Used",
        "prompt": "What dataset(s) were used in this work? How much data was there?"
    },
    {
        "name": "Tables",
        "prompt": "Are there any tables in this document? If so, describe what information they contain and highlight key numbers."
    },
    {
        "name": "Authors & Venue",
        "prompt": "Who wrote this paper? Where was it published?"
    },
]


def run_prompt(client, model: str, doc_content: list[dict], prompt: str) -> str:
    """Run a single prompt against the document."""
    user_content = doc_content.copy()
    user_content.append({
        "type": "text",
        "text": f"\n## Task:\n{prompt}"
    })
    
    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant analyzing academic documents."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        max_tokens=800,
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()


def build_doc_content(doc, max_images: int = 10) -> list[dict]:
    """Build multimodal content from extracted document."""
    content = []
    
    # Add text with page markers
    text_parts = []
    for page in doc.pages:
        if page.text.strip():
            text_parts.append(f"[Page {page.page_num}]\n{page.text.strip()}")
    
    if text_parts:
        content.append({
            "type": "text",
            "text": "## Document:\n\n" + "\n\n---\n\n".join(text_parts)
        })
    
    # Add images
    image_count = 0
    for page in doc.pages:
        for img in page.images:
            if image_count >= max_images:
                break
            
            content.append({
                "type": "text",
                "text": f"[Figure from page {img.page_num}]"
            })
            
            mime_type = f"image/{img.image_type}"
            if img.image_type == "jpg":
                mime_type = "image/jpeg"
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img.image_b64}"
                }
            })
            image_count += 1
        
        if image_count >= max_images:
            break
    
    return content, image_count


def main():
    # Parse args
    pdf_path = "data/1/W18-4401.pdf"
    model = "mistral-large-latest"
    
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model = arg.split("=")[1]
        elif arg.startswith("--model"):
            continue
        elif not arg.startswith("-"):
            pdf_path = arg
    
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model = sys.argv[idx + 1]
    
    print(f"\n{'='*70}")
    print(f"  PDF Test: {pdf_path}")
    print(f"  Model: {model}")
    print('='*70)
    
    # Extract PDF
    print("\n📄 Extracting PDF...")
    doc = extract_pdf(pdf_path)
    summary = get_document_summary(doc)
    
    print(f"   Pages: {summary['total_pages']}")
    print(f"   Images: {summary['total_images']}")
    print(f"   Text: {summary['total_text_chars']:,} chars")
    
    # Build content
    doc_content, image_count = build_doc_content(doc)
    print(f"   Images included: {image_count}")
    
    # Initialize client
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    # Run each prompt
    print(f"\n{'='*70}")
    print("  Running test prompts...")
    print('='*70)
    
    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n\n{'─'*70}")
        print(f"  [{i}/{len(TEST_PROMPTS)}] {test['name']}")
        print(f"{'─'*70}")
        print(f"  Prompt: {test['prompt'][:60]}...")
        print()
        
        try:
            response = run_prompt(client, model, doc_content, test['prompt'])
            print(response)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n\n{'='*70}")
    print("  ✅ Done!")
    print('='*70)


if __name__ == "__main__":
    main()
