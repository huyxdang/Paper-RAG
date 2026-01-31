#!/bin/bash
# Full pipeline: Extract PDFs → Embed → Upload to Pinecone
# Run overnight with: nohup ./run_pipeline.sh > pipeline.log 2>&1 &

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"
VENV="./venv/bin/python"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Starting NeurIPS 2025 Pipeline"
log "=========================================="

# Check dependencies
if [ ! -f "$VENV" ]; then
    log "ERROR: venv not found at $VENV"
    exit 1
fi

if [ ! -d "papers" ]; then
    log "ERROR: papers/ directory not found"
    exit 1
fi

PDF_COUNT=$(ls papers/*.pdf 2>/dev/null | wc -l | tr -d ' ')
log "Found $PDF_COUNT PDFs in papers/"

# Check GROBID
log "Checking GROBID..."
if ! curl -s http://localhost:8070/api/isalive > /dev/null 2>&1; then
    log "ERROR: GROBID not running. Start it with:"
    log "  docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0"
    exit 1
fi
log "GROBID is running"

# Step 1: Extract and embed
log ""
log "=========================================="
log "Step 1: Extracting PDFs and generating embeddings"
log "=========================================="

START_TIME=$(date +%s)

$VENV extract.py \
    --pdf-dir papers \
    --output chunks.jsonl \
    --csv neurips_2025.csv \
    2>&1 | tee -a "$LOG_FILE"

EXTRACT_TIME=$(($(date +%s) - START_TIME))
log "Extraction completed in $((EXTRACT_TIME / 60)) minutes"

# Check output
if [ ! -f "chunks.jsonl" ]; then
    log "ERROR: chunks.jsonl not created"
    exit 1
fi

CHUNK_COUNT=$(wc -l < chunks.jsonl | tr -d ' ')
log "Generated $CHUNK_COUNT chunks"

# Step 2: Upload to Pinecone
log ""
log "=========================================="
log "Step 2: Uploading to Pinecone"
log "=========================================="

START_TIME=$(date +%s)

$VENV upload_to_pinecone.py \
    --input chunks.jsonl \
    2>&1 | tee -a "$LOG_FILE"

UPLOAD_TIME=$(($(date +%s) - START_TIME))
log "Upload completed in $((UPLOAD_TIME / 60)) minutes"

# Summary
log ""
log "=========================================="
log "Pipeline Complete!"
log "=========================================="
log "PDFs processed: $PDF_COUNT"
log "Chunks generated: $CHUNK_COUNT"
log "Log file: $LOG_FILE"
log "=========================================="
