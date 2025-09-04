import os
import pandas as pd
import argparse
from pathlib import Path
from config import Config

def find_csv_files(input_dir, filename_prefix="graph_info_table_"):
    """Find all CSV files that start with the specified prefix in input_dir and subdirectories."""
    csv_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith(filename_prefix) and file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def load_and_combine_data(csv_files, dataset_name="Dataset"):
    """Load all CSV files and combine them with source file information."""
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Remove Density column if it exists
            if 'Density' in df.columns:
                df = df.drop('Density', axis=1)
            
            # Add only the metadata we want
            df['Dataset'] = dataset_name
            
            # Extract country from the parent directory name
            country = Path(csv_file).parent.name
            df['Country'] = country
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns to put metadata at the end
        data_columns = [col for col in combined_df.columns if col not in ['Dataset', 'Country']]
        final_columns = data_columns + ['Dataset', 'Country']
        combined_df = combined_df[final_columns]
        
        return combined_df
    else:
        return pd.DataFrame()

def compare_datasets(df1, df2, dataset1_name="Dataset 1", dataset2_name="Dataset 2"):
    """Compare two datasets and generate comparison statistics."""
    
    print(f"\n" + "="*80)
    print(f"COMPARISON: {dataset1_name} vs {dataset2_name}")
    print("="*80)
    
    # Basic comparison
    print(f"\nBasic Statistics:")
    print(f"  {dataset1_name}: {len(df1)} scenarios from {df1['Country'].nunique()} countries")
    print(f"  {dataset2_name}: {len(df2)} scenarios from {df2['Country'].nunique()} countries")
    
    # Find common countries
    countries1 = set(df1['Country'].unique())
    countries2 = set(df2['Country'].unique())
    common_countries = countries1.intersection(countries2)
    
    print(f"\nCountry overlap:")
    print(f"  Common countries ({len(common_countries)}): {', '.join(sorted(common_countries))}")
    print(f"  Only in {dataset1_name} ({len(countries1 - countries2)}): {', '.join(sorted(countries1 - countries2))}")
    print(f"  Only in {dataset2_name} ({len(countries2 - countries1)}): {', '.join(sorted(countries2 - countries1))}")
    
    # Numeric columns for comparison (removed Density)
    numeric_columns = ['Nodes', 'Edges', 'Total Length (km)', 'Avg Edge Length (m)', 'Components']
    
    # Create comparison dataframes for common countries and scenarios
    comparison_results = []
    
    for country in sorted(common_countries):
        for scenario in sorted(set(df1['Graph Scenario'].unique()).intersection(set(df2['Graph Scenario'].unique()))):
            data1 = df1[(df1['Country'] == country) & (df1['Graph Scenario'] == scenario)]
            data2 = df2[(df2['Country'] == country) & (df2['Graph Scenario'] == scenario)]
            
            if len(data1) > 0 and len(data2) > 0:
                row = {
                    'Country': country,
                    'Scenario': scenario,
                    'Dataset_1_Name': dataset1_name,
                    'Dataset_2_Name': dataset2_name
                }
                
                for col in numeric_columns:
                    if col in data1.columns and col in data2.columns:
                        val1 = pd.to_numeric(data1[col].iloc[0], errors='coerce')
                        val2 = pd.to_numeric(data2[col].iloc[0], errors='coerce')
                        
                        if pd.notna(val1) and pd.notna(val2):
                            row[f'{col}_{dataset1_name}'] = val1
                            row[f'{col}_{dataset2_name}'] = val2
                            
                            # Calculate relative change
                            if val1 != 0:
                                change_pct = ((val2 - val1) / val1) * 100
                                row[f'{col}_Change_Pct'] = change_pct
                            else:
                                row[f'{col}_Change_Pct'] = float('inf') if val2 > 0 else 0
                
                comparison_results.append(row)
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS BY SCENARIO:")
    print("-" * 60)
    
    summary_stats = []
    for scenario in sorted(comparison_df['Scenario'].unique()):
        scenario_data = comparison_df[comparison_df['Scenario'] == scenario]
        row = {'Scenario': scenario, 'Countries': len(scenario_data)}
        
        for col in numeric_columns:
            change_col = f'{col}_Change_Pct'
            if change_col in scenario_data.columns:
                changes = scenario_data[change_col].dropna()
                if len(changes) > 0:
                    row[f'{col}_Mean_Change'] = changes.mean()
                    row[f'{col}_Std_Change'] = changes.std()
        
        summary_stats.append(row)
    
    summary_df = pd.DataFrame(summary_stats)
    
    return comparison_df, summary_df

def save_to_latex(df, filename, caption="", label=""):
    """Save DataFrame to LaTeX table format using pandas built-in escaping."""
    
    # Convert to Path object if it isn't already
    filename = Path(filename)
    
    # Generate the table content with proper escaping
    latex_content = df.to_latex(
        index=False,
        float_format='{:.2f}'.format,
        caption=caption,
        label=label,
        longtable=True,
        escape=True,  # Let pandas handle all escaping automatically
        column_format='l' * len(df.columns)  # Left-align all columns
    )
    
    # Create complete LaTeX document with landscape orientation
    full_latex = f"""\\documentclass{{article}}
\\usepackage{{longtable}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{geometry}}
\\usepackage{{pdflscape}}
\\usepackage{{afterpage}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\geometry{{margin=0.5in}}

\\begin{{document}}

\\afterpage{{\\clearpage}}
\\begin{{landscape}}

{latex_content}

\\end{{landscape}}

\\end{{document}}
"""
    
    # Create table-only filename
    table_only_filename = filename.with_name(filename.stem + '_table_only' + filename.suffix)
    
    # Save complete document
    with open(filename, 'w') as f:
        f.write(full_latex)
    
    # Save table-only version
    table_header = """% Required packages:
% \\usepackage{longtable}
% \\usepackage{booktabs}
% \\usepackage{array}
% \\usepackage{pdflscape}
% \\usepackage{afterpage}
% \\usepackage{amsmath}
% \\usepackage{amssymb}

\\afterpage{\\clearpage}
\\begin{landscape}

"""
    
    table_footer = """
\\end{landscape}
"""
    
    with open(table_only_filename, 'w') as f:
        f.write(table_header + latex_content + table_footer)
    
    print(f"Complete LaTeX document saved to: {filename}")
    print(f"Table-only version saved to: {table_only_filename}")

def generate_metric_specific_tables(df1, df2, dataset1_name, dataset2_name, output_dir):
    """Generate separate tables for each key metric."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define metrics to create tables for
    metrics = ['Nodes', 'Edges', 'Total Length (km)', 'Avg Edge Length (m)']
    
    # Get common countries and scenarios
    countries1 = set(df1['Country'].unique())
    countries2 = set(df2['Country'].unique())
    common_countries = countries1.intersection(countries2)
    
    scenarios1 = set(df1['Graph Scenario'].unique())
    scenarios2 = set(df2['Graph Scenario'].unique())
    common_scenarios = scenarios1.intersection(scenarios2)
    
    for metric in metrics:
        print(f"\nGenerating table for: {metric}")
        
        # Create comparison data for this metric
        metric_data = []
        
        for country in sorted(common_countries):
            for scenario in sorted(common_scenarios):
                data1 = df1[(df1['Country'] == country) & (df1['Graph Scenario'] == scenario)]
                data2 = df2[(df2['Country'] == country) & (df2['Graph Scenario'] == scenario)]
                
                if len(data1) > 0 and len(data2) > 0 and metric in data1.columns and metric in data2.columns:
                    val1 = pd.to_numeric(data1[metric].iloc[0], errors='coerce')
                    val2 = pd.to_numeric(data2[metric].iloc[0], errors='coerce')
                    
                    if pd.notna(val1) and pd.notna(val2):
                        # Calculate change
                        if val1 != 0:
                            change_pct = ((val2 - val1) / val1) * 100
                            change_abs = val2 - val1
                        else:
                            change_pct = float('inf') if val2 > 0 else 0
                            change_abs = val2
                        
                        row = {
                            'Country': country,
                            'Scenario': scenario,
                            f'{dataset1_name}': val1,
                            f'{dataset2_name}': val2,
                            'Absolute Change': change_abs,
                            'Percent Change': change_pct
                        }
                        metric_data.append(row)
        
        if metric_data:
            metric_df = pd.DataFrame(metric_data)
            
            # Clean metric name for filename
            metric_clean = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            
            # Save CSV
            csv_filename = output_dir / f"{metric_clean}_comparison.csv"
            metric_df.to_csv(csv_filename, index=False)
            
            # Save LaTeX
            latex_filename = output_dir / f"{metric_clean}_comparison.tex"
            save_to_latex(
                metric_df,
                latex_filename,
                caption=f"{metric} Comparison: {dataset1_name} vs {dataset2_name}",
                label=f"tab:{metric_clean}_comparison"
            )
            
            print(f"  Saved: {csv_filename}")
            print(f"  Saved: {latex_filename}")
            
            # Generate summary statistics for this metric
            summary_stats = {
                'Metric': metric,
                'Total Comparisons': len(metric_df),
                'Mean Absolute Change': metric_df['Absolute Change'].mean(),
                'Mean Percent Change': metric_df['Percent Change'].mean(),
                'Std Percent Change': metric_df['Percent Change'].std(),
                'Min Percent Change': metric_df['Percent Change'].min(),
                'Max Percent Change': metric_df['Percent Change'].max(),
            }
            
            # Save metric summary
            summary_df = pd.DataFrame([summary_stats])
            summary_csv = output_dir / f"{metric_clean}_summary.csv"
            summary_df.to_csv(summary_csv, index=False)
            
            summary_latex = output_dir / f"{metric_clean}_summary.tex"
            save_to_latex(
                summary_df,
                summary_latex,
                caption=f"{metric} Summary Statistics: {dataset1_name} vs {dataset2_name}",
                label=f"tab:{metric_clean}_summary"
            )

def generate_comparison_report(df1, df2, dataset1_name, dataset2_name, output_dir):
    """Generate comprehensive comparison report with CSV and LaTeX outputs."""
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare datasets
    comparison_df, summary_df = compare_datasets(df1, df2, dataset1_name, dataset2_name)
    
    # Generate metric-specific tables
    generate_metric_specific_tables(df1, df2, dataset1_name, dataset2_name, output_dir)
    
    # Save combined datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_csv = output_dir / "combined_datasets.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"Combined datasets saved to: {combined_csv}")
    
    # Save comparison results
    if not comparison_df.empty:
        # CSV files
        comparison_csv = output_dir / "dataset_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Detailed comparison saved to: {comparison_csv}")
        
        summary_csv = output_dir / "comparison_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary statistics saved to: {summary_csv}")
        
        # LaTeX tables
        comparison_latex = output_dir / "dataset_comparison.tex"
        save_to_latex(
            comparison_df,
            comparison_latex,
            caption=f"Detailed Comparison: {dataset1_name} vs {dataset2_name}",
            label="tab:detailed_comparison"
        )
        
        summary_latex = output_dir / "comparison_summary.tex"
        save_to_latex(
            summary_df,
            summary_latex,
            caption=f"Summary Statistics: {dataset1_name} vs {dataset2_name}",
            label="tab:summary_comparison"
        )
        
        # Generate a simplified comparison table for key metrics
        numeric_columns = ['Nodes', 'Edges', 'Total Length (km)']
        simplified_df = comparison_df[['Country', 'Scenario'] + 
                                    [f'{col}_Change_Pct' for col in numeric_columns if f'{col}_Change_Pct' in comparison_df.columns]]
        
        if not simplified_df.empty:
            simplified_latex = output_dir / "simplified_comparison.tex"
            save_to_latex(
                simplified_df,
                simplified_latex,
                caption=f"Key Metrics Comparison (\\% Change): {dataset1_name} vs {dataset2_name}",
                label="tab:simplified_comparison"
            )
    
    # Save individual dataset summaries
    for df, name in [(df1, dataset1_name), (df2, dataset2_name)]:
        if not df.empty:
            # CSV
            dataset_csv = output_dir / f"{name.lower().replace(' ', '_')}_summary.csv"
            df.to_csv(dataset_csv, index=False)
            
            # LaTeX
            dataset_latex = output_dir / f"{name.lower().replace(' ', '_')}_summary.tex"
            save_to_latex(
                df,
                dataset_latex,
                caption=f"Dataset Summary: {name}",
                label=f"tab:{name.lower().replace(' ', '_')}_summary"
            )
    
    print(f"\nAll outputs saved to: {output_dir}")
    return comparison_df, summary_df

def main():
    parser = argparse.ArgumentParser(description="Compare graph_info_table_*.csv files from two directories")
    parser.add_argument("dir1", help="First input directory")
    parser.add_argument("dir2", help="Second input directory")
    parser.add_argument("--name1", help="Name for first dataset", default="Dataset 1")
    parser.add_argument("--name2", help="Name for second dataset", default="Dataset 2")
    parser.add_argument("--prefix", "-p", help="CSV filename prefix to search for", default="graph_info_table_")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: config.OUTPUT_DIR)", default=None)
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Output to /output directory
        output_dir = Path("output") / "comparison_results"
    
    # Validate input directories
    for dir_path, name in [(args.dir1, args.name1), (args.dir2, args.name2)]:
        if not os.path.exists(dir_path):
            print(f"Error: Input directory '{dir_path}' does not exist.")
            return
    
    print(f"Comparing directories:")
    print(f"  {args.name1}: {args.dir1}")
    print(f"  {args.name2}: {args.dir2}")
    print(f"Output directory: {output_dir}")
    
    # Find CSV files in both directories
    csv_files1 = find_csv_files(args.dir1, args.prefix)
    csv_files2 = find_csv_files(args.dir2, args.prefix)
    
    print(f"\nFound files:")
    print(f"  {args.name1}: {len(csv_files1)} files")
    print(f"  {args.name2}: {len(csv_files2)} files")
    
    if not csv_files1 and not csv_files2:
        print(f"No CSV files starting with '{args.prefix}' found in either directory.")
        return
    
    # Load and combine data from both directories
    print(f"\nLoading data from {args.name1}...")
    df1 = load_and_combine_data(csv_files1, args.name1)
    
    print(f"\nLoading data from {args.name2}...")
    df2 = load_and_combine_data(csv_files2, args.name2)
    
    if df1.empty and df2.empty:
        print("No data loaded from either directory.")
        return
    
    # Generate comparison report
    print(f"\nGenerating comparison report...")
    comparison_df, summary_df = generate_comparison_report(df1, df2, args.name1, args.name2, output_dir)
    
    print(f"\nComparison complete!")

if __name__ == "__main__":
    main()