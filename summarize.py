import os
import pandas as pd
import argparse
from pathlib import Path

def find_csv_files(input_dir, filename_prefix="graph_info_table_"):
    """Find all CSV files that start with the specified prefix in input_dir and subdirectories."""
    csv_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith(filename_prefix) and file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def load_and_combine_data(csv_files):
    """Load all CSV files and combine them with source directory information."""
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add source directory as a column
            df['Source_Directory'] = os.path.dirname(csv_file)
            df['Source_File'] = csv_file
            # Extract country/region from filename
            filename = os.path.basename(csv_file)
            if filename.startswith('graph_info_table_'):
                country = filename.replace('graph_info_table_', '').replace('.csv', '')
                df['Country'] = country
            else:
                df['Country'] = 'unknown'
            all_data.append(df)
            print(f"Loaded: {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def generate_summary(combined_df):
    """Generate summary statistics from the combined data."""
    if combined_df.empty:
        print("No data found to summarize.")
        return
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL graph_info_table_*.csv FILES")
    print("="*80)
    
    # Basic info
    print(f"\nTotal files processed: {combined_df['Source_File'].nunique()}")
    print(f"Total scenarios found: {len(combined_df)}")
    print(f"Countries/regions found: {', '.join(sorted(combined_df['Country'].unique()))}")
    
    # Scenarios breakdown by country
    print(f"\nScenarios by country and type:")
    print("-" * 60)
    scenario_country_table = combined_df.groupby(['Country', 'Graph Scenario']).size().unstack(fill_value=0)
    print(scenario_country_table)
    
    # Overall scenarios breakdown
    print(f"\nOverall scenarios by type:")
    scenario_counts = combined_df['Graph Scenario'].value_counts()
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count} instances")
    
    # Detailed comparison between scenarios
    print(f"\nDETAILED SCENARIO COMPARISON:")
    print("="*80)
    
    numeric_columns = ['Nodes', 'Edges', 'Total Length (km)', 'Avg Edge Length (m)', 'Density', 'Components']
    
    # Create comparison table
    scenario_summary = []
    for scenario in sorted(combined_df['Graph Scenario'].unique()):
        scenario_data = combined_df[combined_df['Graph Scenario'] == scenario]
        row = {'Scenario': scenario, 'Count': len(scenario_data)}
        
        for col in numeric_columns:
            if col in scenario_data.columns:
                values = pd.to_numeric(scenario_data[col], errors='coerce')
                row[f'{col}_mean'] = values.mean()
                row[f'{col}_std'] = values.std()
        
        scenario_summary.append(row)
    
    summary_df = pd.DataFrame(scenario_summary)
    
    # Print detailed statistics for each scenario
    for scenario in sorted(combined_df['Graph Scenario'].unique()):
        scenario_data = combined_df[combined_df['Graph Scenario'] == scenario]
        print(f"\n{scenario.upper()} SCENARIO ({len(scenario_data)} instances):")
        print("-" * 50)
        
        # Show by country
        country_breakdown = scenario_data['Country'].value_counts()
        print(f"Countries: {', '.join([f'{country}({count})' for country, count in country_breakdown.items()])}")
        
        for col in numeric_columns:
            if col in scenario_data.columns:
                values = pd.to_numeric(scenario_data[col], errors='coerce')
                print(f"  {col}:")
                print(f"    Mean: {values.mean():.2f} ± {values.std():.2f}")
                print(f"    Range: {values.min():.2f} - {values.max():.2f}")
                if len(values) > 1:
                    print(f"    CV: {(values.std()/values.mean()*100):.1f}%")
    
    # Scenario comparison table
    print(f"\nSCENARIO COMPARISON TABLE:")
    print("="*80)
    comparison_cols = ['Scenario', 'Count'] + [f'{col}_mean' for col in numeric_columns if f'{col}_mean' in summary_df.columns]
    display_df = summary_df[comparison_cols].round(2)
    print(display_df.to_string(index=False))
    
    # Relative changes from Original scenario
    if 'Original' in combined_df['Graph Scenario'].values:
        print(f"\nRELATIVE CHANGES FROM ORIGINAL SCENARIO:")
        print("="*60)
        original_data = combined_df[combined_df['Graph Scenario'] == 'Original']
        
        for col in numeric_columns:
            if col in original_data.columns:
                original_mean = pd.to_numeric(original_data[col], errors='coerce').mean()
                print(f"\n{col} (Original mean: {original_mean:.2f}):")
                
                for scenario in sorted(combined_df['Graph Scenario'].unique()):
                    if scenario != 'Original':
                        scenario_data = combined_df[combined_df['Graph Scenario'] == scenario]
                        if col in scenario_data.columns:
                            scenario_mean = pd.to_numeric(scenario_data[col], errors='coerce').mean()
                            change_pct = ((scenario_mean - original_mean) / original_mean) * 100
                            print(f"  {scenario}: {scenario_mean:.2f} ({change_pct:+.1f}%)")

def save_combined_data(combined_df, output_file="combined_graph_summary.csv"):
    """Save the combined data to a CSV file."""
    if not combined_df.empty:
        combined_df.to_csv(output_file, index=False)
        print(f"\nCombined data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Summarize all graph_info_table_*.csv files")
    parser.add_argument("input_dir", help="Input directory to search for CSV files")
    parser.add_argument("--output", "-o", help="Output file for combined data", default="combined_graph_summary.csv")
    parser.add_argument("--prefix", "-p", help="CSV filename prefix to search for", default="graph_info_table_")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
    
    print(f"Searching for CSV files starting with '{args.prefix}' in: {args.input_dir}")
    
    # Find all CSV files
    csv_files = find_csv_files(args.input_dir, args.prefix)
    
    if not csv_files:
        print(f"No CSV files starting with '{args.prefix}' found in {args.input_dir} or its subdirectories.")
        return
    
    print(f"Found {len(csv_files)} files:")
    for f in csv_files:
        print(f"  {f}")
    
    # Load and combine data
    combined_df = load_and_combine_data(csv_files)
    
    # Generate summary
    generate_summary(combined_df)
    
    # Save combined data
    save_combined_data(combined_df, args.output)

if __name__ == "__main__":
    main()