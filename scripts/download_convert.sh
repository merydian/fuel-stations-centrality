#!/bin/bash

# --- Functions ---
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# --- Check arguments ---
if [ "$#" -ne 1 ]; then
    error_exit "Usage: $0 <URL_to_download>"
fi

URL="$1"
FILENAME=$(basename "$URL")
DATA_DIR="data"

# --- Start processing ---
log "Starting script"

# Ensure data directory exists
mkdir -p "$DATA_DIR" || error_exit "Failed to create data directory"
cd "$DATA_DIR" || error_exit "Failed to enter data directory"

# Check if .pbf exists and is not corrupted
if [ -f "$FILENAME" ]; then
    log "$FILENAME exists, checking integrity..."
    if osmium fileinfo "$FILENAME" >/dev/null 2>&1; then
        log "$FILENAME exists and is valid, skipping download"
    else
        log "$FILENAME exists but is corrupted, re-downloading"
        curl -L -o "$FILENAME" "$URL" || error_exit "Download failed"
    fi
else
    log "Downloading $URL"
    curl -L -o "$FILENAME" "$URL" || error_exit "Download failed"
fi

# Filter highways and write PBF (overwrite if exists)
FILTERED_PBF="${FILENAME%.osm.pbf}_filtered.osm.pbf"
log "Filtering highways to $FILTERED_PBF"
osmium tags-filter "$FILENAME" \
    w/highway=motorway,trunk,primary,secondary,tertiary \
    n/amenity=fuel \
    w/amenity=fuel \
    r/amenity=fuel \
    -o "$FILTERED_PBF" --overwrite \
    || error_exit "Osmium filtering failed"

# Convert filtered PBF to XML (overwrite if exists)
OUTPUT_XML="${FILENAME%.osm.pbf}.osm"
log "Converting $FILTERED_PBF to XML $OUTPUT_XML"
osmium cat "$FILTERED_PBF" -o "$OUTPUT_XML" --overwrite || error_exit "Conversion to XML failed"

log