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

# Download file (overwrite if exists)
log "Downloading $URL"
curl -L -o "$FILENAME" "$URL" || error_exit "Download failed"

# Filter highways and write PBF (overwrite if exists)
FILTERED_PBF="${FILENAME%.osm.pbf}_filtered.osm.pbf"
log "Filtering highways to $FILTERED_PBF"
osmium tags-filter "$FILENAME" w/highway=motorway,trunk,primary,secondary,tertiary -o "$FILTERED_PBF" -f pbf --overwrite \
    || error_exit "Osmium filtering failed"

# Convert filtered PBF to XML (overwrite if exists)
OUTPUT_XML="${FILENAME%.osm.pbf}.osm"
log "Converting $FILTERED_PBF to XML $OUTPUT_XML"
osmium cat "$FILTERED_PBF" -o "$OUTPUT_XML" --overwrite || error_exit "Conversion to XML failed"

log "Script completed successfully"
