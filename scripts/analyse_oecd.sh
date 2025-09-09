#!/bin/bash

#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=05:30:00

# Exit on any error except for main.py failures
set -e

# Record start time
START_TIME=$(date +%s)
START_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

# Global configuration variables
MAX_DISTANCE=150000
N_REMOVE=120
MAX_STATIONS=1000000
EPSG_CODE=4087

# Array of configurations: "place_name|url"
configs=(
    # Europe
    #"austria|https://download.geofabrik.de/europe/austria-latest.osm.pbf"
    #"belgium|https://download.geofabrik.de/europe/belgium-latest.osm.pbf"
    #"czech-republic|https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf"
    #"denmark|https://download.geofabrik.de/europe/denmark-latest.osm.pbf"
    #"estonia|https://download.geofabrik.de/europe/estonia-latest.osm.pbf"
    #"finland|https://download.geofabrik.de/europe/finland-latest.osm.pbf"
    "france|https://download.geofabrik.de/europe/france-latest.osm.pbf"
    "germany|https://download.geofabrik.de/europe/germany-latest.osm.pbf"
    "greece|https://download.geofabrik.de/europe/greece-latest.osm.pbf"
    "hungary|https://download.geofabrik.de/europe/hungary-latest.osm.pbf"
    "iceland|https://download.geofabrik.de/europe/iceland-latest.osm.pbf"
    "ireland-and-northern-ireland|https://download.geofabrik.de/europe/ireland-and-northern-ireland-latest.osm.pbf"
    "italy|https://download.geofabrik.de/europe/italy-latest.osm.pbf"
    "latvia|https://download.geofabrik.de/europe/latvia-latest.osm.pbf"
    "lithuania|https://download.geofabrik.de/europe/lithuania-latest.osm.pbf"
    "luxembourg|https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf"
    "netherlands|https://download.geofabrik.de/europe/netherlands-latest.osm.pbf"
    "norway|https://download.geofabrik.de/europe/norway-latest.osm.pbf"
    "poland|https://download.geofabrik.de/europe/poland-latest.osm.pbf"
    "portugal|https://download.geofabrik.de/europe/portugal-latest.osm.pbf"
    "slovakia|https://download.geofabrik.de/europe/slovakia-latest.osm.pbf"
    "slovenia|https://download.geofabrik.de/europe/slovenia-latest.osm.pbf"
    "spain|https://download.geofabrik.de/europe/spain-latest.osm.pbf"
    "sweden|https://download.geofabrik.de/europe/sweden-latest.osm.pbf"
    "switzerland|https://download.geofabrik.de/europe/switzerland-latest.osm.pbf"
    "great-britain|https://download.geofabrik.de/europe/great-britain-latest.osm.pbf"
    
    # North America
    "canada|https://download.geofabrik.de/north-america/canada-latest.osm.pbf"
    "mexico|https://download.geofabrik.de/north-america/mexico-latest.osm.pbf"
    "us|https://download.geofabrik.de/north-america/us-latest.osm.pbf"
    
    # Asia-Pacific
    "australia|https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf"
    "japan|https://download.geofabrik.de/asia/japan-latest.osm.pbf"
    "south-korea|https://download.geofabrik.de/asia/south-korea-latest.osm.pbf"
    "new-zealand|https://download.geofabrik.de/australia-oceania/new-zealand-latest.osm.pbf"
    
    # South America
    "chile|https://download.geofabrik.de/south-america/chile-latest.osm.pbf"
    "colombia|https://download.geofabrik.de/south-america/colombia-latest.osm.pbf"
    
    # Middle East
    "israel-and-palestine|https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf"
    "turkey|https://download.geofabrik.de/europe/turkey-latest.osm.pbf"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Arrays to track results
successful_analyses=()
failed_analyses=()

echo -e "${GREEN}Starting OECD fuel station analyses...${NC}"
echo -e "${BLUE}Start time: $START_TIME_HUMAN${NC}"
echo "Total configurations: ${#configs[@]}"
echo -e "${BLUE}Global settings:${NC}"
echo "  Max Distance: $MAX_DISTANCE"
echo "  Stations to Remove: $N_REMOVE"
echo "  Max Stations: $MAX_STATIONS"
echo "  EPSG Code: $EPSG_CODE"
echo ""

# Counter for progress
counter=1
total=${#configs[@]}

# Loop through each configuration
for config in "${configs[@]}"; do
    # Split the configuration string
    IFS='|' read -r place url <<< "$config"
    
    echo -e "${YELLOW}[$counter/$total] Processing: $place${NC}"
    echo "URL: $url"
    echo "EPSG Code: $EPSG_CODE"
    echo ""
    
    # Use download_convert.sh to download and process the file
    data_file="data/${place}-latest.osm"
    if [ ! -f "$data_file" ]; then
        echo -e "${YELLOW}Downloading and processing $place data...${NC}"
        fuel-stations-centrality/scripts/download_convert.sh "$url"
        
        # Rename the output file to match expected name
        original_filename=$(basename "$url")
        converted_file="data/${original_filename%.osm.pbf}.osm"
        if [ -f "$converted_file" ] && [ "$converted_file" != "$data_file" ]; then
            mv "$converted_file" "$data_file"
        fi
        
        echo -e "${GREEN}Download and conversion complete: $data_file${NC}"
    else
        echo -e "${GREEN}Data file already exists: $data_file${NC}"
    fi
    
    # Run the analysis (disable exit on error for this command)
    echo -e "${YELLOW}Running analysis for $place...${NC}"
    set +e  # Disable exit on error temporarily
    python fuel-stations-centrality/main.py \
        --place "$place" \
        --max-distance "$MAX_DISTANCE" \
        --n-remove "$N_REMOVE" \
        --epsg-code "$EPSG_CODE" \
        --max-stations "$MAX_STATIONS"
    
    exit_code=$?
    set -e  # Re-enable exit on error
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Analysis completed successfully for $place${NC}"
        successful_analyses+=("$place")
    else
        echo -e "${RED}✗ Analysis failed for $place (exit code: $exit_code)${NC}"
        failed_analyses+=("$place")
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
    
    ((counter++))
done

echo -e "${GREEN}All OECD analyses processing completed!${NC}"
echo ""

# Summary of results
echo "=========================================="
echo -e "${BLUE}OECD ANALYSIS SUMMARY${NC}"
echo "=========================================="
echo "Total configurations processed: $total"
echo -e "${GREEN}Successful: ${#successful_analyses[@]}${NC}"
echo -e "${RED}Failed: ${#failed_analyses[@]}${NC}"
echo ""

if [ ${#successful_analyses[@]} -gt 0 ]; then
    echo -e "${GREEN}Successful analyses:${NC}"
    for place in "${successful_analyses[@]}"; do
        echo "  ✓ $place - output/$place/"
    done
    echo ""
fi

if [ ${#failed_analyses[@]} -gt 0 ]; then
    echo -e "${RED}Failed analyses:${NC}"
    for place in "${failed_analyses[@]}"; do
        echo "  ✗ $place"
    done
    echo ""
    echo -e "${YELLOW}Note: Check logs above for specific error details for failed analyses.${NC}"
    echo ""
fi

# Calculate and display execution time
END_TIME=$(date +%s)
END_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_TIME=$((END_TIME - START_TIME))

# Convert seconds to human readable format
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "=========================================="
echo -e "${BLUE}EXECUTION TIME SUMMARY${NC}"
echo "=========================================="
echo -e "${BLUE}Start time:  ${NC}$START_TIME_HUMAN"
echo -e "${BLUE}End time:    ${NC}$END_TIME_HUMAN"
echo -e "${BLUE}Total time:  ${NC}${HOURS}h ${MINUTES}m ${SECONDS}s (${TOTAL_TIME} seconds)"
echo "=========================================="