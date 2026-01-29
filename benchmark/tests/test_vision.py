"""
Test vision capabilities with Mistral Large on a PDF.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from pdf_extract import extract_pdf, get_document_summary, build_mistral_content

load_dotenv(Path(__file__).parent.parent / ".env")


def test_image_description(pdf_path: str, model: str = "mistral-large-latest"):
    """Test that the model can see and describe images from a PDF."""
    
    print(f"\n{'='*60}")
    print(f"Testing Vision: {pdf_path}")
    print(f"Model: {model}")
    print('='*60)
    
    # 1. Extract PDF
    doc = extract_pdf(pdf_path)
    summary = get_document_summary(doc)
    
    print(f"\n📄 Document Summary:")
    print(f"   Pages: {summary['total_pages']}")
    print(f"   Images: {summary['total_images']}")
    print(f"   Text: {summary['total_text_chars']} chars")
    
    if summary['total_images'] == 0:
        print("\n⚠️  No images found in PDF!")
        return
    
    # 2. List images
    print(f"\n🖼️  Images found:")
    for page in doc.pages:
        for img in page.images:
            print(f"   - Page {img.page_num}: {img.image_type} {img.width}x{img.height}")
    
    # 3. Build content asking to describe images
    content = []
    
    # Add images with annotations
    for page in doc.pages:
        for img in page.images:
            content.append({
                "type": "text",
                "text": f"[Image from page {img.page_num}]"
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
    
    # Add question
    content.append({
        "type": "text",
        "text": f"""I've shown you {summary['total_images']} image(s) from this PDF document.

Please describe each image in detail:
1. What type of visualization is it? (chart, table, diagram, photo, etc.)
2. What information does it show?
3. Any key data points or trends you can identify?

Number your descriptions to match the image order."""
    })
    
    # 4. Call Mistral
    print(f"\n🚀 Sending to {model}...")
    
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze images and documents."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=1000,
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    
    print(f"\n📝 Model Response:")
    print("-" * 40)
    print(answer)
    print("-" * 40)
    
    # Token usage
    if hasattr(response, 'usage') and response.usage:
        print(f"\n💰 Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
    
    return answer


if __name__ == "__main__":
    import sys
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/1/W18-4401.pdf"
    model = sys.argv[2] if len(sys.argv) > 2 else "mistral-large-latest"
    
    test_image_description(pdf_path, model)
