#!/bin/bash

# Default region
REGION="${1:-turkmenistan}"

echo "Downloading and converting OSM data for: $REGION"
echo "URL: https://download.geofabrik.de/asia/${REGION}-latest.osm.pbf"
cd data || exit 1

curl -O "https://download.geofabrik.de/asia/${REGION}-latest.osm.pbf"
osmium cat "${REGION}-latest.osm.pbf" -o "${REGION}-latest.osm"

echo "Copying converted OSM data to remote server..."
scp "${REGION}-latest.osm" helix:ma/fuel-stations-centrality/data/

echo "Download and conversion complete for $REGION"