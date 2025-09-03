#!/bin/bash

#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=03:30:00

# Exit on any error except for main.py failures
set -e

# Record start time
START_TIME=$(date +%s)
START_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

# Global configuration variables
MAX_DISTANCE=150000
N_REMOVE=120
MAX_STATIONS=1000000

# Array of configurations: "place_name|url|epsg_code"
configs=(
    # Africa
    "angola|https://download.geofabrik.de/africa/angola-latest.osm.pbf|9159"
    "benin|https://download.geofabrik.de/africa/benin-latest.osm.pbf|32631"
    "burkina-faso|https://download.geofabrik.de/africa/burkina-faso-latest.osm.pbf|32630"
    "burundi|https://download.geofabrik.de/africa/burundi-latest.osm.pbf|32736"
    "central-african-republic|https://download.geofabrik.de/africa/central-african-republic-latest.osm.pbf|0000"
    "chad|https://download.geofabrik.de/africa/chad-latest.osm.pbf|0000"
    "comores|https://download.geofabrik.de/africa/comores-latest.osm.pbf|0000"
    "congo-democratic-republic|https://download.geofabrik.de/africa/congo-democratic-republic-latest.osm.pbf|0000"
    "djibouti|https://download.geofabrik.de/africa/djibouti-latest.osm.pbf|0000"
    "eritrea|https://download.geofabrik.de/africa/eritrea-latest.osm.pbf|0000"
    "ethiopia|https://download.geofabrik.de/africa/ethiopia-latest.osm.pbf|0000"
    "senegal-and-gambia|https://download.geofabrik.de/africa/senegal-and-gambia-latest.osm.pbf|0000"
    "guinea|https://download.geofabrik.de/africa/guinea-latest.osm.pbf|0000"
    "guinea-bissau|https://download.geofabrik.de/africa/guinea-bissau-latest.osm.pbf|0000"
    "lesotho|https://download.geofabrik.de/africa/lesotho-latest.osm.pbf|0000"
    "liberia|https://download.geofabrik.de/africa/liberia-latest.osm.pbf|0000"
    "madagascar|https://download.geofabrik.de/africa/madagascar-latest.osm.pbf|0000"
    "malawi|https://download.geofabrik.de/africa/malawi-latest.osm.pbf|0000"
    "mali|https://download.geofabrik.de/africa/mali-latest.osm.pbf|0000"
    "mauritania|https://download.geofabrik.de/africa/mauritania-latest.osm.pbf|0000"
    "mozambique|https://download.geofabrik.de/africa/mozambique-latest.osm.pbf|0000"
    "niger|https://download.geofabrik.de/africa/niger-latest.osm.pbf|0000"
    "rwanda|https://download.geofabrik.de/africa/rwanda-latest.osm.pbf|0000"
    "sierra-leone|https://download.geofabrik.de/africa/sierra-leone-latest.osm.pbf|0000"
    "somalia|https://download.geofabrik.de/africa/somalia-latest.osm.pbf|0000"
    "south-sudan|https://download.geofabrik.de/africa/south-sudan-latest.osm.pbf|0000"
    "sudan|https://download.geofabrik.de/africa/sudan-latest.osm.pbf|0000"
    "togo|https://download.geofabrik.de/africa/togo-latest.osm.pbf|0000"
    "uganda|https://download.geofabrik.de/africa/uganda-latest.osm.pbf|0000"
    "tanzania|https://download.geofabrik.de/africa/tanzania-latest.osm.pbf|0000"
    "zambia|https://download.geofabrik.de/africa/zambia-latest.osm.pbf|0000"
    
    # Asia
    "afghanistan|https://download.geofabrik.de/asia/afghanistan-latest.osm.pbf|0000"
    "bangladesh|https://download.geofabrik.de/asia/bangladesh-latest.osm.pbf|0000"
    "cambodia|https://download.geofabrik.de/asia/cambodia-latest.osm.pbf|0000"
    "laos|https://download.geofabrik.de/asia/laos-latest.osm.pbf|0000"
    "myanmar|https://download.geofabrik.de/asia/myanmar-latest.osm.pbf|0000"
    "nepal|https://download.geofabrik.de/asia/nepal-latest.osm.pbf|0000"
    "yemen|https://download.geofabrik.de/asia/yemen-latest.osm.pbf|0000"
    "east-timor|https://download.geofabrik.de/asia/east-timor-latest.osm.pbf|0000"
    
    # Caribbean
    "haiti-and-domrep|https://download.geofabrik.de/central-america/haiti-and-domrep-latest.osm.pbf|0000"
    
    # Pacific
    "kiribati|https://download.geofabrik.de/australia-oceania/kiribati-latest.osm.pbf|0000"
    "solomon-islands|https://download.geofabrik.de/australia-oceania/solomon-islands-latest.osm.pbf|0000"
    "tuvalu|https://download.geofabrik.de/australia-oceania/tuvalu-latest.osm.pbf|0000"
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

echo -e "${GREEN}Starting multiple fuel station analyses...${NC}"
echo -e "${BLUE}Start time: $START_TIME_HUMAN${NC}"
echo "Total configurations: ${#configs[@]}"
echo -e "${BLUE}Global settings:${NC}"
echo "  Max Distance: $MAX_DISTANCE"
echo "  Stations to Remove: $N_REMOVE"
echo "  Max Stations: $MAX_STATIONS"
echo ""

# Counter for progress
counter=1
total=${#configs[@]}

# Loop through each configuration
for config in "${configs[@]}"; do
    # Split the configuration string
    IFS='|' read -r place url epsg_code <<< "$config"
    
    echo -e "${YELLOW}[$counter/$total] Processing: $place${NC}"
    echo "URL: $url"
    echo "EPSG Code: $epsg_code"
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
        --epsg-code "$epsg_code" \
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

echo -e "${GREEN}All analyses processing completed!${NC}"
echo ""

# Summary of results
echo "=========================================="
echo -e "${BLUE}ANALYSIS SUMMARY${NC}"
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