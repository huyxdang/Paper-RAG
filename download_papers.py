#!/usr/bin/env python3
"""Download NeurIPS PDFs with auto-resume and retry support."""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# --- DEFAULTS ---
DEFAULT_CSV = "neurips_2025.csv"
DEFAULT_OUTPUT_DIR = "NeurIPS_2025_PDFs"
DEFAULT_WORKERS = 10
DEFAULT_TIMEOUT = 30
FAILED_LOG = "download_failures.json"


def clean_filename(title: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9 \-_]', '', str(title))
    return clean.replace(' ', '_')[:150]


def load_failures(log_path: str) -> dict:
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return {}


def save_failures(log_path: str, failures: dict) -> None:
    with open(log_path, 'w') as f:
        json.dump(failures, f, indent=2)


def download_paper(row: dict, output_dir: str, timeout: int, max_retries: int = 3, force: bool = False) -> tuple[str, bool, str]:
    """Download a single paper with retries. Returns: (note_id, success, error_message)"""
    note_id = row['note_id']
    title = row['title']
    
    pdf_url = f"https://openreview.net/pdf?id={note_id}"
    filename = f"{clean_filename(title)}.pdf"
    filepath = os.path.join(output_dir, filename)
    
    # Skip if already exists and has content (unless force mode)
    if not force and os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return (note_id, True, "")
    
    last_error = ""
    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, stream=True, timeout=timeout)
            if response.status_code == 200:
                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'pdf' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                    last_error = f"Not a PDF (content-type: {content_type})"
                    continue
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify file size (PDFs should be at least a few KB)
                if os.path.getsize(filepath) < 1000:
                    os.remove(filepath)
                    last_error = "Downloaded file too small (likely error page)"
                    continue
                    
                return (note_id, True, "")
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
                last_error = f"Rate limited (429), waited {wait_time}s"
            else:
                last_error = f"HTTP {response.status_code}"
        except requests.Timeout:
            last_error = "Timeout"
        except requests.ConnectionError as e:
            last_error = f"Connection error: {e}"
        except Exception as e:
            last_error = str(e)
        
        # Wait before retry
        if attempt < max_retries - 1:
            time.sleep(1 * (attempt + 1))
    
    return (note_id, False, last_error)


def main():
    parser = argparse.ArgumentParser(description="Download NeurIPS PDFs with resume support")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to CSV file")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS, help="Parallel workers")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, help="Request timeout (seconds)")
    parser.add_argument("--limit", "-n", type=int, help="Only download first N papers (for testing)")
    parser.add_argument("--retry-failed", action="store_true", help="Only retry previously failed downloads")
    parser.add_argument("--retries", type=int, default=3, help="Max retries per paper")
    parser.add_argument("--failures-log", default=FAILED_LOG, help="Path to failures log file")
    parser.add_argument("--force", action="store_true", help="Re-download all papers (ignore existing files)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(args.csv)
    rows = df.to_dict('records')
    
    # Load previous failures
    failures = load_failures(args.failures_log)
    
    total_papers = len(rows)
    
    if args.retry_failed:
        # Only retry previously failed papers
        failed_ids = set(failures.keys())
        rows = [r for r in rows if r['note_id'] in failed_ids]
        print(f"Retrying {len(rows)} previously failed downloads...")
    elif args.force:
        print(f"Force mode: re-downloading all {len(rows)} papers...")
    else:
        # Filter out already downloaded papers (auto-resume)
        already_downloaded = set()
        for r in rows:
            filepath = os.path.join(args.output, f"{clean_filename(r['title'])}.pdf")
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                already_downloaded.add(r['note_id'])
        
        rows = [r for r in rows if r['note_id'] not in already_downloaded]
        
        if already_downloaded:
            print(f"Resuming: {len(already_downloaded)}/{total_papers} already downloaded")
        else:
            print(f"Starting fresh: {total_papers} papers to download")
    
    # Apply limit
    if args.limit:
        rows = rows[:args.limit]
    
    if not rows:
        print("Nothing to download!")
        return
    
    print(f"Downloading {len(rows)} papers with {args.workers} workers...")
    
    new_failures = {}
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_paper, row, args.output, args.timeout, args.retries, args.force): row
            for row in rows
        }
        
        with tqdm(total=len(rows), desc="Downloading") as pbar:
            for future in as_completed(futures):
                note_id, success, error = future.result()
                if success:
                    success_count += 1
                    # Remove from failures if it was there
                    failures.pop(note_id, None)
                else:
                    new_failures[note_id] = error
                    failures[note_id] = error
                pbar.update(1)
    
    # Save updated failures
    save_failures(args.failures_log, failures)
    
    # Summary
    print(f"\nDone! {success_count}/{len(rows)} successful")
    if new_failures:
        print(f"Failed: {len(new_failures)} papers (logged to {args.failures_log})")
        print("Run with --retry-failed to retry them")


if __name__ == "__main__":
    main()