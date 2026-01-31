#!/usr/bin/env python3
"""Download NeurIPS PDFs with auto-resume and retry support."""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Defaults
DEFAULT_CSV = "neurips_2025.csv"
DEFAULT_OUTPUT_DIR = "papers"
DEFAULT_WORKERS = 10
DEFAULT_TIMEOUT = 30
FAILED_LOG = "download_failures.json"
MIN_FILE_SIZE = 1000  # bytes


def clean_filename(title: str, max_len: int = 150) -> str:
    """Sanitize title for use as filename."""
    clean = re.sub(r'[^a-zA-Z0-9 \-_]', '', str(title))
    return clean.replace(' ', '_')[:max_len]


def load_failures(path: Path) -> dict:
    """Load previously failed downloads."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_failures(path: Path, failures: dict) -> None:
    """Save failed downloads."""
    path.write_text(json.dumps(failures, indent=2))


def download_paper(
    note_id: str,
    title: str,
    output_dir: Path,
    timeout: int,
    max_retries: int = 3,
    force: bool = False
) -> tuple[str, bool, str]:
    """Download a single paper. Returns (note_id, success, error)."""
    filepath = output_dir / f"{clean_filename(title)}.pdf"
    
    # Skip if exists (unless force)
    if not force and filepath.exists() and filepath.stat().st_size > MIN_FILE_SIZE:
        return (note_id, True, "")
    
    url = f"https://openreview.net/pdf?id={note_id}"
    last_error = ""
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            
            if resp.status_code == 200:
                content_type = resp.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'octet-stream' not in content_type:
                    last_error = f"Not PDF: {content_type}"
                    continue
                
                filepath.write_bytes(resp.content)
                
                if filepath.stat().st_size < MIN_FILE_SIZE:
                    filepath.unlink()
                    last_error = "File too small"
                    continue
                
                return (note_id, True, "")
            
            elif resp.status_code == 429:
                wait = 5 * (attempt + 1)
                time.sleep(wait)
                last_error = f"Rate limited, waited {wait}s"
            else:
                last_error = f"HTTP {resp.status_code}"
                
        except requests.Timeout:
            last_error = "Timeout"
        except requests.ConnectionError:
            last_error = "Connection error"
        except Exception as e:
            last_error = str(e)
        
        if attempt < max_retries - 1:
            time.sleep(attempt + 1)
    
    return (note_id, False, last_error)


def main():
    parser = argparse.ArgumentParser(description="Download NeurIPS PDFs")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--limit", "-n", type=int, help="Download first N only")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    failures_path = Path(FAILED_LOG)
    
    # Load data
    df = pd.read_csv(args.csv)
    papers = [(r['note_id'], r['title']) for _, r in df.iterrows()]
    failures = load_failures(failures_path)
    
    # Filter
    if args.retry_failed:
        papers = [(nid, t) for nid, t in papers if nid in failures]
        print(f"Retrying {len(papers)} failed...")
    elif not args.force:
        already = {nid for nid, t in papers 
                   if (output_dir / f"{clean_filename(t)}.pdf").exists()}
        papers = [(nid, t) for nid, t in papers if nid not in already]
        if already:
            print(f"Resuming: {len(already)} already downloaded")
    
    if args.limit:
        papers = papers[:args.limit]
    
    if not papers:
        print("Nothing to download!")
        return
    
    print(f"Downloading {len(papers)} papers ({args.workers} workers)...")
    
    success = 0
    new_failures = {}
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_paper, nid, title, output_dir, 
                          args.timeout, args.retries, args.force): nid
            for nid, title in papers
        }
        
        with tqdm(total=len(papers)) as pbar:
            for future in as_completed(futures):
                nid, ok, err = future.result()
                if ok:
                    success += 1
                    failures.pop(nid, None)
                else:
                    new_failures[nid] = err
                    failures[nid] = err
                pbar.update(1)
    
    save_failures(failures_path, failures)
    
    print(f"\nDone! {success}/{len(papers)} successful")
    if new_failures:
        print(f"Failed: {len(new_failures)} (see {FAILED_LOG})")


if __name__ == "__main__":
    main()