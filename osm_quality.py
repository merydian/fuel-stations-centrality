import geopandas as gpd
import requests
import json
from datetime import datetime
import pandas as pd
import geopandas as gpd
import requests
import json
import os

def generate_latex_table(completeness_stats, dataset_name):
    """
    Generate LaTeX table with completeness statistics
    
    Args:
        completeness_stats (dict): Completeness statistics
        dataset_name (str): Name of the dataset
        
    Returns:
        str: LaTeX table code
    """
    if not completeness_stats or not completeness_stats['countries']:
        print("No completeness data available for LaTeX table generation")
        return None
    
    # Sort countries by completeness value (descending)
    sorted_countries = sorted(completeness_stats['countries'], key=lambda x: x['value'], reverse=True)
    
    # Generate LaTeX table
    latex_code = []
    latex_code.append("% LaTeX table of OSM road completeness statistics")
    latex_code.append("% Requires packages: longtable, booktabs")
    latex_code.append(f"% Usage: \\input{{completeness_table_{dataset_name}.tex}}")
    latex_code.append("")
    latex_code.append("\\begin{longtable}{@{}p{3.5cm}r@{\\,}>{\\centering\\arraybackslash}p{2.5cm}@{}}")
    latex_code.append(f"\\caption{{Road Network Completeness Statistics for {dataset_name.upper()} Countries}}")
    latex_code.append(f"\\label{{tab:road_completeness_{dataset_name}}} \\\\")
    latex_code.append("\\toprule")
    latex_code.append("Country & Completeness & Quality Class \\\\")
    latex_code.append(" & (\\%) & \\\\")
    latex_code.append("\\midrule")
    latex_code.append("\\endfirsthead")
    latex_code.append("")
    latex_code.append("\\multicolumn{3}{c}{{\\tablename\\ \\thetable{} -- continued from previous page}} \\\\")
    latex_code.append("\\toprule")
    latex_code.append("Country & Completeness & Quality Class \\\\")
    latex_code.append(" & (\\%) & \\\\")
    latex_code.append("\\midrule")
    latex_code.append("\\endhead")
    latex_code.append("")
    latex_code.append("\\midrule")
    latex_code.append("\\multicolumn{3}{r@{}}{{Continued on next page}} \\\\")
    latex_code.append("\\endfoot")
    latex_code.append("")
    latex_code.append("\\bottomrule")
    latex_code.append("\\endlastfoot")
    
    # Add country rows
    for country in sorted_countries:
        country_name = country['name'].replace('&', '\\&')  # Escape ampersands
        completeness_pct = f"{country['value']*100:.1f}"
        quality_class = country.get('class', 'N/A') if country.get('class') else 'N/A'
        
        latex_code.append(f"{country_name} & {completeness_pct} & {quality_class} \\\\")
    
    # Add summary statistics section
    latex_code.append("\\midrule")
    latex_code.append("\\multicolumn{3}{@{}l@{}}{\\textbf{Summary Statistics}} \\\\")
    latex_code.append("\\midrule")
    latex_code.append(f"Average & {completeness_stats['average']*100:.1f} & N/A \\\\")
    latex_code.append(f"Minimum & {completeness_stats['min']*100:.1f} & N/A \\\\")
    latex_code.append(f"Maximum & {completeness_stats['max']*100:.1f} & N/A \\\\")
    latex_code.append(f"Total Countries & {completeness_stats['count']} & N/A \\\\")
    
    latex_code.append("\\end{longtable}")
    
    # Join all lines
    latex_table = "\n".join(latex_code)
    
    # Save to file
    latex_output_file = f"output/osm_quality/completeness_table_{dataset_name}.tex"
    try:
        os.makedirs(os.path.dirname(latex_output_file), exist_ok=True)
        with open(latex_output_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_output_file}")
    except Exception as e:
        print(f"Error saving LaTeX table: {e}")
    
    return latex_table

def generate_currentness_latex_table(currentness_stats, dataset_name):
    """
    Generate LaTeX table with gas station currentness statistics
    
    Args:
        currentness_stats (dict): Currentness statistics
        dataset_name (str): Name of the dataset
        
    Returns:
        str: LaTeX table code
    """
    if not currentness_stats or not currentness_stats['countries']:
        print("No currentness data available for LaTeX table generation")
        return None
    
    # Sort countries by currentness value (descending)
    sorted_countries = sorted(currentness_stats['countries'], key=lambda x: x['currentness'], reverse=True)
    
    # Generate LaTeX table
    latex_code = []
    latex_code.append("% LaTeX table of OSM gas station currentness statistics")
    latex_code.append("% Requires packages: longtable, booktabs")
    latex_code.append(f"% Usage: \\input{{currentness_table_{dataset_name}.tex}}")
    latex_code.append("")
    latex_code.append("\\begin{longtable}{@{}p{3.5cm}r@{\\,}r@{\\,}>{\\centering\\arraybackslash}p{2.5cm}@{}}")
    latex_code.append(f"\\caption{{Gas Station Currentness Statistics for {dataset_name.upper()} Countries}}")
    latex_code.append(f"\\label{{tab:gas_currentness_{dataset_name}}} \\\\")
    latex_code.append("\\toprule")
    latex_code.append("Country & Currentness & Stations & Quality Class \\\\")
    latex_code.append(" & (\\%) & Count & \\\\")
    latex_code.append("\\midrule")
    latex_code.append("\\endfirsthead")
    latex_code.append("")
    latex_code.append("\\multicolumn{4}{c}{{\\tablename\\ \\thetable{} -- continued from previous page}} \\\\")
    latex_code.append("\\toprule")
    latex_code.append("Country & Currentness & Stations & Quality Class \\\\")
    latex_code.append(" & (\\%) & Count & \\\\")
    latex_code.append("\\midrule")
    latex_code.append("\\endhead")
    latex_code.append("")
    latex_code.append("\\midrule")
    latex_code.append("\\multicolumn{4}{r@{}}{{Continued on next page}} \\\\")
    latex_code.append("\\endfoot")
    latex_code.append("")
    latex_code.append("\\bottomrule")
    latex_code.append("\\endlastfoot")
    
    # Add country rows
    for country in sorted_countries:
        country_name = country['name'].replace('&', '\\&')  # Escape ampersands
        currentness_pct = f"{country['currentness']*100:.1f}"
        station_count = country.get('count', 0)
        quality_class = country.get('class', 'N/A') if country.get('class') else 'N/A'
        
        latex_code.append(f"{country_name} & {currentness_pct} & {station_count} & {quality_class} \\\\")
    
    # Add summary statistics section
    latex_code.append("\\midrule")
    latex_code.append("\\multicolumn{4}{@{}l@{}}{\\textbf{Summary Statistics}} \\\\")
    latex_code.append("\\midrule")
    latex_code.append(f"Average & {currentness_stats['average']*100:.1f} & {currentness_stats['total_stations']} & N/A \\\\")
    latex_code.append(f"Minimum & {currentness_stats['min']*100:.1f} & -- & N/A \\\\")
    latex_code.append(f"Maximum & {currentness_stats['max']*100:.1f} & -- & N/A \\\\")
    latex_code.append(f"Total Countries & {currentness_stats['count']} & -- & N/A \\\\")
    
    latex_code.append("\\end{longtable}")
    
    # Join all lines
    latex_table = "\n".join(latex_code)
    
    # Save to file
    latex_output_file = f"output/osm_quality/currentness_table_{dataset_name}.tex"
    try:
        os.makedirs(os.path.dirname(latex_output_file), exist_ok=True)
        with open(latex_output_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX currentness table saved to: {latex_output_file}")
    except Exception as e:
        print(f"Error saving LaTeX currentness table: {e}")
    
    return latex_table

def get_gas_station_currentness(gpkg_path, layer_name=None, dataset_name=None):
    """
    Get gas station currentness data from ohsome quality API for geometry in GeoPackage
    
    Args:
        gpkg_path (str): Path to the GeoPackage file
        layer_name (str): Layer name in GeoPackage (optional, uses first layer if None)
        dataset_name (str): Name of the dataset for output files
    
    Returns:
        dict: ohsome quality response data for currentness
    """
    
    # Read geometry from GeoPackage
    try:
        if layer_name:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
        else:
            gdf = gpd.read_file(gpkg_path)
    except Exception as e:
        print(f"Error reading GeoPackage: {e}")
        return None
    
    if gdf.empty:
        print("No geometries found in the GeoPackage")
        return None
    
    # Ensure CRS is WGS84
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Convert geometry to GeoJSON format
    geometry_geojson = json.loads(gdf.to_json())
    
    # ohsome quality API endpoint for currentness
    url = "https://api.quality.ohsome.org/v1/indicators/currentness"
    
    # Prepare the request data for gas stations
    data = {
        "bpolys": geometry_geojson,
        "topic": "amenity=fuel",  # Gas/fuel stations
        "includeFigure": False
    }
    
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"Fetching gas station currentness data...")
        print(f"Topic: amenity=fuel")
        print(f"Area bounds: {gdf.total_bounds}")
        
        response = requests.post(url, json=data, headers=headers)
        
        # Print response details for debugging
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            print(f"Response text: {response.text}")
            return None
        
        result = response.json()
        
        # Save results to file
        output_dir = "output/osm_quality"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/gas_currentness_{dataset_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Gas station currentness results saved to: {output_file}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ohsome quality API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.text}")
        return None

def analyze_gas_station_currentness(results):
    """
    Analyze gas station currentness results
    
    Args:
        results (dict): Results from ohsome quality API currentness endpoint
        
    Returns:
        dict: Currentness statistics
    """
    if not results:
        print("No currentness results to analyze")
        return None
    
    currentness_values = []
    currentness_stats = {
        'average': None,
        'min': None,
        'max': None,
        'count': 0,
        'total_stations': 0,
        'values': [],
        'countries': []
    }
    
    try:
        print("\n=== GAS STATION CURRENTNESS ANALYSIS ===")
        
        # The results structure for currentness
        if 'result' in results and isinstance(results['result'], list):
            for country_result in results['result']:
                if 'result' in country_result:
                    country_data = country_result['result']
                    
                    # Get currentness value (ratio of current to total)
                    if 'currentnessRatio' in country_data:
                        currentness = country_data['currentnessRatio']
                    elif 'current' in country_data and 'total' in country_data:
                        total = country_data['total']
                        current = country_data['current']
                        currentness = current / total if total > 0 else 0
                    else:
                        continue
                    
                    if currentness is not None:
                        currentness_values.append(float(currentness))
                        country_name = country_result.get('NAME', 'Unknown')
                        total_stations = country_data.get('total', 0)
                        current_stations = country_data.get('current', 0)
                        
                        currentness_stats['countries'].append({
                            'name': country_name,
                            'currentness': float(currentness),
                            'total': total_stations,
                            'current': current_stations,
                            'count': total_stations,
                            'iso_a2': country_result.get('iso_a2', ''),
                            'class': country_data.get('class', None)
                        })
                        
                        currentness_stats['total_stations'] += total_stations
        
        # Calculate statistics if we found currentness values
        if currentness_values:
            currentness_stats['values'] = currentness_values
            currentness_stats['count'] = len(currentness_values)
            currentness_stats['average'] = sum(currentness_values) / len(currentness_values)
            currentness_stats['min'] = min(currentness_values)
            currentness_stats['max'] = max(currentness_values)
            
            print(f"Found {currentness_stats['count']} countries with gas station currentness data")
            print(f"Total gas stations analyzed: {currentness_stats['total_stations']}")
            print(f"Average currentness: {currentness_stats['average']:.3f} ({currentness_stats['average']*100:.1f}%)")
            print(f"Min currentness: {currentness_stats['min']:.3f} ({currentness_stats['min']*100:.1f}%)")
            print(f"Max currentness: {currentness_stats['max']:.3f} ({currentness_stats['max']*100:.1f}%)")
            
            # Show top and bottom performers
            sorted_countries = sorted(currentness_stats['countries'], key=lambda x: x['currentness'], reverse=True)
            print(f"\nTop 5 performers (currentness):")
            for country in sorted_countries[:5]:
                print(f"  {country['name']} ({country['iso_a2']}): {country['currentness']*100:.1f}% ({country['current']}/{country['total']} stations)")
            
            print(f"\nBottom 5 performers (currentness):")
            for country in sorted_countries[-5:]:
                print(f"  {country['name']} ({country['iso_a2']}): {country['currentness']*100:.1f}% ({country['current']}/{country['total']} stations)")
                
            # Show countries with most gas stations
            sorted_by_count = sorted(currentness_stats['countries'], key=lambda x: x['total'], reverse=True)
            print(f"\nCountries with most gas stations:")
            for country in sorted_by_count[:5]:
                print(f"  {country['name']} ({country['iso_a2']}): {country['total']} stations ({country['currentness']*100:.1f}% current)")
                
        else:
            print("No currentness values found in the results")
    
    except Exception as e:
        print(f"Error analyzing gas station currentness: {e}")
        return None
    
    return currentness_stats

def get_road_comparison_quality(gpkg_path, layer_name=None, include_figure=True, dataset_name=None):
    """
    Get road comparison quality data from ohsome quality API for geometry in GeoPackage
    
    Args:
        gpkg_path (str): Path to the GeoPackage file
        layer_name (str): Layer name in GeoPackage (optional, uses first layer if None)
        include_figure (bool): Whether to include figure in response
    
    Returns:
        dict: ohsome quality response data
    """
    
    # Read geometry from GeoPackage
    try:
        if layer_name:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
        else:
            gdf = gpd.read_file(gpkg_path)
    except Exception as e:
        print(f"Error reading GeoPackage: {e}")
        return None
    
    if gdf.empty:
        print("No geometries found in the GeoPackage")
        return None
    
    # Ensure CRS is WGS84
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Convert geometry to GeoJSON format
    geometry_geojson = json.loads(gdf.to_json())
    
    # ohsome quality API endpoint for road comparison
    url = "https://api.quality.ohsome.org/v1/indicators/road-comparison"
    
    # Prepare the request data according to API specification
    data = {
        "bpolys": geometry_geojson,
        "topic": "roads-all-highways",
        "includeFigure": include_figure,
        "ohsomedb": False
    }
    
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"Fetching road comparison data...")
        print(f"Include figure: {include_figure}")
        print(f"Area bounds: {gdf.total_bounds}")
        
        response = requests.post(url, json=data, headers=headers)
        
        # Print response details for debugging
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            print(f"Response text: {response.text}")
            return None
        
        result = response.json()
        
        # Save results to file
        output_file = f"output/osm_quality/road_comparison_{dataset_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ohsome quality API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.text}")
        return None
    
def calculate_average_completeness(results):
    """
    Calculate average completeness from ohsome quality road comparison results
    
    Args:
        results (dict): Results from ohsome quality API
    
    Returns:
        dict: Dictionary containing completeness statistics
    """
    if not results:
        print("No results to analyze for completeness")
        return None
    
    completeness_values = []
    completeness_stats = {
        'average': None,
        'min': None,
        'max': None,
        'count': 0,
        'values': [],
        'countries': []
    }
    
    try:
        # The results structure shows an array of country results
        if 'result' in results and isinstance(results['result'], list):
            for country_result in results['result']:
                if 'result' in country_result and 'value' in country_result['result']:
                    value = country_result['result']['value']
                    # Skip null values (like Japan and South Korea)
                    if value is not None:
                        completeness_values.append(float(value))
                        country_name = country_result.get('NAME', 'Unknown')
                        completeness_stats['countries'].append({
                            'name': country_name,
                            'value': float(value),
                            'iso_a2': country_result.get('iso_a2', ''),
                            'class': country_result['result'].get('class', None)
                        })
        
        # Calculate statistics if we found completeness values
        if completeness_values:
            completeness_stats['values'] = completeness_values
            completeness_stats['count'] = len(completeness_values)
            completeness_stats['average'] = sum(completeness_values) / len(completeness_values)
            completeness_stats['min'] = min(completeness_values)
            completeness_stats['max'] = max(completeness_values)
            
            print(f"\n=== COMPLETENESS ANALYSIS ===")
            print(f"Found {completeness_stats['count']} countries with completeness values")
            print(f"Average completeness: {completeness_stats['average']:.3f} ({completeness_stats['average']*100:.1f}%)")
            print(f"Min completeness: {completeness_stats['min']:.3f} ({completeness_stats['min']*100:.1f}%)")
            print(f"Max completeness: {completeness_stats['max']:.3f} ({completeness_stats['max']*100:.1f}%)")
            
            # Show top and bottom performers
            sorted_countries = sorted(completeness_stats['countries'], key=lambda x: x['value'], reverse=True)
            print(f"\nTop 5 performers:")
            for country in sorted_countries[:5]:
                print(f"  {country['name']} ({country['iso_a2']}): {country['value']*100:.1f}%")
            
            print(f"\nBottom 5 performers:")
            for country in sorted_countries[-5:]:
                print(f"  {country['name']} ({country['iso_a2']}): {country['value']*100:.1f}%")
                
        else:
            print("\n=== COMPLETENESS ANALYSIS ===")
            print("No completeness values found in the results")
    
    except Exception as e:
        print(f"Error calculating completeness: {e}")
        return None
    
    return completeness_stats

def analyze_road_comparison_results(results):
    """
    Analyze and display road comparison results
    
    Args:
        results (dict): Results from ohsome quality API
        
    Returns:
        dict: Completeness statistics
    """
    if not results:
        print("No results to analyze")
        return None
    
    print("\n=== ROAD COMPARISON ANALYSIS ===")
    
    # Check different possible result structures
    if 'result' in results:
        print("Found 'result' section in response")
        result_data = results['result']
        
        # Display indicators if they exist
        if isinstance(result_data, dict):
            for key, value in result_data.items():
                print(f"\n{key}: {value}")
    
    if 'metadata' in results:
        print("\n--- METADATA ---")
        metadata = results['metadata']
        for key, value in metadata.items():
            print(f"{key}: {value}")
    
    if 'figure' in results:
        print("\n--- FIGURE ---")
        print("Figure data included in response (base64 encoded)")
    
    # Calculate average completeness
    completeness_stats = calculate_average_completeness(results)
    
    # Print full structure for debugging
    print("\n--- FULL RESPONSE STRUCTURE ---")
    print(json.dumps(results, indent=2)[:1000] + "..." if len(json.dumps(results)) > 1000 else json.dumps(results, indent=2))
    
    return completeness_stats

def main():
    """
    Main function to run the script
    """
    print("=== OHSOME QUALITY ANALYSIS ===\n")
    
    # Configuration
    for dataset_name in ["ldcs", "oecds"]:
        gpkg_path = f"/home/till/Documents/ma/data/{dataset_name}.gpkg"
        layer_name = "world_map"
        
        # Check if file exists
        if not os.path.exists(gpkg_path):
            print(f"File not found: {gpkg_path}")
            continue
        
        print(f"\n=== PROCESSING DATASET: {dataset_name.upper()} ===")
        
        # 1. Road comparison analysis
        print("\n1. Running road comparison analysis...")
        road_result = get_road_comparison_quality(gpkg_path, layer_name, include_figure=True, dataset_name=dataset_name)
        if road_result:
            completeness_stats = analyze_road_comparison_results(road_result)
            
            if completeness_stats and completeness_stats['average'] is not None:
                # Save completeness stats to JSON
                completeness_output_file = f"output/osm_quality/completeness_stats_{dataset_name}.json"
                try:
                    os.makedirs(os.path.dirname(completeness_output_file), exist_ok=True)
                    with open(completeness_output_file, 'w') as f:
                        json.dump(completeness_stats, f, indent=2)
                    print(f"Completeness statistics saved to: {completeness_output_file}")
                except Exception as e:
                    print(f"Error saving completeness stats: {e}")
                
                # Generate and save LaTeX table
                latex_table = generate_latex_table(completeness_stats, dataset_name)
                if latex_table:
                    print("Road completeness LaTeX table generated successfully!")
        
        # 2. Gas station currentness analysis
        print("\n2. Running gas station currentness analysis...")
        currentness_result = get_gas_station_currentness(gpkg_path, layer_name, dataset_name=dataset_name)
        if currentness_result:
            currentness_stats = analyze_gas_station_currentness(currentness_result)
            
            if currentness_stats and currentness_stats['average'] is not None:
                # Save currentness stats to JSON
                currentness_output_file = f"output/osm_quality/currentness_stats_{dataset_name}.json"
                try:
                    os.makedirs(os.path.dirname(currentness_output_file), exist_ok=True)
                    with open(currentness_output_file, 'w') as f:
                        json.dump(currentness_stats, f, indent=2)
                    print(f"Currentness statistics saved to: {currentness_output_file}")
                except Exception as e:
                    print(f"Error saving currentness stats: {e}")
                
                # Generate and save LaTeX table for currentness
                currentness_latex_table = generate_currentness_latex_table(currentness_stats, dataset_name)
                if currentness_latex_table:
                    print("Gas station currentness LaTeX table generated successfully!")
        
        print(f"\n=== SUMMARY FOR {dataset_name.upper()} ===")
        print("Analysis complete! Check the output/osm_quality/ directory for results.")

if __name__ == "__main__":
    main()