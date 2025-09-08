import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import argparse
import logging
from tabulate import tabulate

logger = logging.getLogger(__name__)

class GraphComparison:
    def __init__(self, dir1, dir2, output_dir="comparison_output"):
        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define consistent colors for scenarios and datasets
        self.scenario_colors = {
            'Original': '#006D77',
            'Original Pruned': '#83C5BE',
            'KNN Filtered': '#CD6C73',
            'Randomized Filtered': '#769D9A'
        }
        
        self.dataset_colors = {
            'ldcs_150000': '#F4E8C1',
            'oecd_150000': '#A0C1B9'
        }
        
        # Configure matplotlib for better plots
        plt.style.use('fast')
        
    def load_data(self):
        """Load all graph_info_table CSV files from both directories."""
        logger.info("Loading data from both directories...")
        
        def load_from_dir(directory, label):
            data = []
            csv_files = list(directory.rglob("graph_info_table_*.csv"))
            logger.debug(f"Found {len(csv_files)} CSV files in {directory}")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Extract country name from filename instead of parent directory
                    # Format: graph_info_table_iceland.csv -> iceland
                    filename = csv_file.stem  # Remove .csv extension
                    country = filename.replace('graph_info_table_', '')
                    
                    df['Country'] = country
                    df['Dataset'] = label
                    
                    # Get number of stations from all_stations_nodes.gpkg
                    stations_file = csv_file.parent / "all_stations_nodes.gpkg"
                    stations_count = 0
                    
                    if stations_file.exists():
                        try:
                            import geopandas as gpd
                            stations_gdf = gpd.read_file(stations_file)
                            stations_count = len(stations_gdf)
                            logger.debug(f"Found {stations_count} stations in {stations_file}")
                        except Exception as e:
                            logger.warning(f"Error reading stations file {stations_file}: {e}")
                            # Try alternative approach if geopandas fails
                            try:
                                import fiona
                                with fiona.open(stations_file) as src:
                                    stations_count = len(list(src))
                                logger.debug(f"Found {stations_count} stations using fiona for {stations_file}")
                            except Exception as e2:
                                logger.warning(f"Could not read stations file with fiona either: {e2}")
                    else:
                        logger.warning(f"Stations file not found: {stations_file}")
                    
                    df['Stations_Used'] = stations_count
                    data.append(df)
                    logger.debug(f"Loaded {csv_file} with {len(df)} rows, {stations_count} stations for country: {country}")
                    
                except Exception as e:
                    logger.warning(f"Error loading {csv_file}: {e}")
            
            return pd.concat(data, ignore_index=True) if data else pd.DataFrame()
        
        self.data1 = load_from_dir(self.dir1, self.dir1.name)
        self.data2 = load_from_dir(self.dir2, self.dir2.name)
        
        # Check if we have any data
        if self.data1.empty and self.data2.empty:
            raise ValueError(f"No data found in either directory:\n- {self.dir1}\n- {self.dir2}")
        
        # Combine datasets
        data_to_combine = []
        if not self.data1.empty:
            data_to_combine.append(self.data1)
        if not self.data2.empty:
            data_to_combine.append(self.data2)
        
        self.combined_data = pd.concat(data_to_combine, ignore_index=True)
        
        # Verify we have the expected columns
        expected_columns = ['Graph Scenario', 'Nodes', 'Edges', 'Total Length (km)', 'Country', 'Dataset', 'Stations_Used']
        missing_columns = [col for col in expected_columns if col not in self.combined_data.columns]
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            logger.info(f"Available columns: {list(self.combined_data.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Successfully loaded data for {len(self.combined_data['Country'].unique())} countries")
        logger.info(f"Available scenarios: {self.combined_data['Graph Scenario'].unique()}")
        logger.info(f"Datasets: {self.combined_data['Dataset'].unique()}")
        logger.info(f"Station counts range: {self.combined_data['Stations_Used'].min()} - {self.combined_data['Stations_Used'].max()}")
        
        return self.combined_data
    
    def plot_scenario_differences(self, combined_data, output_dir):
        """
        Create histograms showing percentage differences from Original for each scenario,
        comparing between datasets.
        """
        logger.info("Creating scenario difference histograms")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate percentage differences for each country and scenario
        def calculate_percentage_diff(group):
            # Use 'Original' as baseline
            original = group[group['Graph Scenario'] == 'Original']['Total Length (km)'].iloc[0]
            group['Pct_Diff'] = ((group['Total Length (km)'] - original) / original) * 100
            return group
        
        # Group by Country and Dataset, then calculate differences
        data_with_diff = combined_data.groupby(['Country', 'Dataset']).apply(calculate_percentage_diff).reset_index(drop=True)
        
        # Updated scenario names to match the new CSV format
        scenarios_to_plot = ['Original Pruned', 'KNN Filtered', 'Randomized Filtered']
        plot_data = data_with_diff[data_with_diff['Graph Scenario'].isin(scenarios_to_plot)]
        
        # Create subplots - one for each scenario
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        datasets = plot_data['Dataset'].unique()
        
        for idx, scenario in enumerate(scenarios_to_plot):
            ax = axes[idx]
            scenario_data = plot_data[plot_data['Graph Scenario'] == scenario]
            
            # Create histogram for each dataset
            for dataset in datasets:
                dataset_scenario_data = scenario_data[scenario_data['Dataset'] == dataset]
                pct_diffs = dataset_scenario_data['Pct_Diff'].values
                
                if len(pct_diffs) > 0:  # Only plot if we have data
                    ax.hist(pct_diffs, 
                        alpha=0.7, 
                        label=f'{dataset} (n={len(pct_diffs)})', 
                        bins=20,
                        color=self.dataset_colors.get(dataset, '#CCCCCC'),
                        edgecolor='black',
                        linewidth=0.5)
            
            ax.set_xlabel('Percentage Change from Original (%)')
            ax.set_ylabel('Frequency (Number of Countries)')
            ax.set_title(f'{scenario} vs Original')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add summary statistics
            stats_text = ""
            for dataset in datasets:
                dataset_scenario_data = scenario_data[scenario_data['Dataset'] == dataset]
                if len(dataset_scenario_data) > 0:
                    pct_diffs = dataset_scenario_data['Pct_Diff']
                    mean_val = pct_diffs.mean()
                    std_val = pct_diffs.std()
                    stats_text += f"{dataset}: μ={mean_val:.1f}%, σ={std_val:.1f}%\n"
            
            # Position text box in corner
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Remove the empty subplot
        if len(scenarios_to_plot) < 4:
            fig.delaxes(axes[3])
        
        plt.suptitle('Total Length Percentage Changes by Scenario and Dataset', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / 'scenario_differences_histograms.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Scenario difference histograms saved to: {output_file}")
        
        # plt.show()
        
        # Create a summary table
        summary_stats = []
        for scenario in scenarios_to_plot:
            scenario_data = plot_data[plot_data['Graph Scenario'] == scenario]
            for dataset in datasets:
                dataset_scenario_data = scenario_data[scenario_data['Dataset'] == dataset]
                if len(dataset_scenario_data) > 0:
                    pct_diffs = dataset_scenario_data['Pct_Diff']
                    summary_stats.append({
                        'Scenario': scenario,
                        'Dataset': dataset,
                        'Mean_Pct_Change': pct_diffs.mean(),
                        'Std_Pct_Change': pct_diffs.std(),
                        'Min_Pct_Change': pct_diffs.min(),
                        'Max_Pct_Change': pct_diffs.max(),
                        'Countries_Count': len(pct_diffs)
                    })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save summary table
        summary_file = output_path / 'scenario_differences_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✓ Summary statistics saved to: {summary_file}")
        
        return output_file, summary_file

    def plot_stations_scatter(self, combined_data, output_dir):
        """
        Create a simple scatter plot of stations vs total length.
        """
        logger.info("Creating simple stations scatter plot")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Filter to only Original scenario
        original_data = combined_data[combined_data['Graph Scenario'] == 'Original'].copy()
        
        # Create simple scatter plot
        plt.figure(figsize=(10, 6))
        
        datasets = original_data['Dataset'].unique()
        
        for dataset in datasets:
            data_subset = original_data[original_data['Dataset'] == dataset]
            plt.scatter(data_subset['Stations_Used'], data_subset['Total Length (km)'], 
                    label=dataset, 
                    color=self.dataset_colors.get(dataset, '#CCCCCC'),
                    alpha=0.7,
                    s=50)
        
        plt.xlabel('Number of Stations')
        plt.ylabel('Total Length (km)')
        plt.title('Stations vs Total Road Network Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_file = output_path / 'stations_scatter.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Stations scatter plot saved to: {output_file}")
        
        # plt.show()
        
        return output_file

    def plot_length_differences_by_dataset(self, combined_data, output_dir):
        """
        Create separate bar plots for each dataset showing total length differences by country.
        """
        logger.info("Creating length differences bar plots per dataset")
        
        combined_data = combined_data[combined_data["Country"] != "iceland"]

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate percentage differences for each country and scenario
        def calculate_percentage_diff(group):
            # Use 'Original' as baseline
            original = group[group['Graph Scenario'] == 'Original']['Total Length (km)'].iloc[0]
            group['Length_Diff_Pct'] = ((group['Total Length (km)'] - original) / original) * 100
            return group
        
        # Group by Country and Dataset, then calculate differences
        data_with_diff = combined_data.groupby(['Country', 'Dataset']).apply(calculate_percentage_diff).reset_index(drop=True)
        
        # Filter out 'Original' scenario since it will always be 0%
        scenarios_to_plot = ['Original Pruned', 'KNN Filtered', 'Randomized Filtered']
        plot_data = data_with_diff[data_with_diff['Graph Scenario'].isin(scenarios_to_plot)]
        
        # Get unique datasets
        datasets = sorted(plot_data['Dataset'].unique())
        
        # Create separate plot for each dataset
        output_files = []
        for dataset in datasets:
            dataset_data = plot_data[plot_data['Dataset'] == dataset]
            all_countries = sorted(dataset_data['Country'].unique())
            
            # Filter countries with difference > 0.5% in any scenario
            countries_to_show = []
            for country in all_countries:
                country_data = dataset_data[dataset_data['Country'] == country]
                max_diff = country_data['Length_Diff_Pct'].abs().max()
                if max_diff > 0.5:
                    countries_to_show.append(country)
            
            if not countries_to_show:
                logger.info(f"No countries with >0.5% difference found in {dataset}")
                continue
            
            logger.info(f"Showing {len(countries_to_show)} countries with >0.5% difference in {dataset}")
            
            # Set up the plot
            fig, ax = plt.subplots(figsize=(max(15, len(countries_to_show) * 0.8), 8))
            
            # Set bar width and positions
            bar_width = 0.25
            x = np.arange(len(countries_to_show))
            
            # Track countries with zero differences for note
            zero_diff_countries = []
            
            # Plot bars for each scenario
            for s_idx, scenario in enumerate(scenarios_to_plot):
                scenario_data = dataset_data[dataset_data['Graph Scenario'] == scenario]
                
                # Get values for filtered countries only
                values = []
                for country in countries_to_show:
                    country_data = scenario_data[scenario_data['Country'] == country]
                    if len(country_data) > 0:
                        diff_pct = country_data['Length_Diff_Pct'].iloc[0]
                        # Use absolute value of the percentage difference
                        values.append(abs(diff_pct))
                        
                        # Check if difference is exactly 0 for this scenario
                        if diff_pct == 0 and country not in zero_diff_countries:
                            zero_diff_countries.append(country)
                    else:
                        values.append(0)
                
                # Calculate bar position
                offset = (s_idx - 1) * bar_width
                
                # Plot bars
                ax.bar(x + offset, values, 
                    bar_width, 
                    label=scenario,
                    color=self.scenario_colors.get(scenario, '#CCCCCC'),
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5)
            
            # Customize the plot
            ax.set_yscale('log')
            ax.set_xlabel('Country')
            ax.set_ylabel('Absolute Length Difference from Original (%)')
            ax.set_title(f'Total Length Changes by Country - {dataset.upper()}\n(Countries with >0.5% difference)')
            ax.set_xticks(x)
            ax.set_xticklabels(countries_to_show, rotation=45, ha='right')
            ax.set_ylim(top=100)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
            
            plt.tight_layout()
            
            # Save the plot
            safe_dataset_name = dataset.replace('_', '-')
            output_file = output_path / f'length_differences_{safe_dataset_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Length differences bar plot for {dataset} saved to: {output_file}")
            
            # Log countries with zero differences
            if zero_diff_countries:
                logger.info(f"Countries with 0% difference in {dataset}: {zero_diff_countries}")
            
            output_files.append(output_file)
            
            # plt.show()
        
        return output_files

    def create_summary_tables(self):
        """Create summary tables with statistics."""
        logger.info("Creating summary tables...")
        
        # Calculate percentage differences for each country and scenario
        def calculate_percentage_diff(group):
            original = group[group['Graph Scenario'] == 'Original']['Total Length (km)'].iloc[0]
            original_edges = group[group['Graph Scenario'] == 'Original']['Edges'].iloc[0]
            
            group['Length_Diff_Pct'] = ((group['Total Length (km)'] - original) / original) * 100
            group['Edge_Diff_Pct'] = ((group['Edges'] - original_edges) / original_edges) * 100
            return group
        
        # Group by Country and Dataset, then calculate differences
        data_with_diff = self.combined_data.groupby(['Country', 'Dataset']).apply(calculate_percentage_diff).reset_index(drop=True)
        
        # Filter out 'Original' scenario
        scenarios_to_plot = ['Original Pruned', 'KNN Filtered', 'Randomized Filtered']
        differences_data = data_with_diff[data_with_diff['Graph Scenario'].isin(scenarios_to_plot)]
        
        # Summary statistics by dataset and scenario
        summary_stats = differences_data.groupby(['Dataset', 'Graph Scenario']).agg({
            'Edge_Diff_Pct': ['mean', 'std', 'min', 'max', 'count'],
            'Length_Diff_Pct': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats = summary_stats.reset_index()
        
        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY STATISTICS: FILTERING EFFECTS")
        print("="*80)
        print(tabulate(summary_stats, headers='keys', tablefmt='grid', showindex=False))
        
        # Save to CSV
        summary_stats.to_csv(self.output_dir / 'summary_statistics.csv', index=False)
        
        # Country-by-country comparison
        country_comparison = differences_data.pivot_table(
            index=['Country', 'Dataset'], 
            columns='Graph Scenario', 
            values=['Edge_Diff_Pct', 'Length_Diff_Pct'],
            aggfunc='first'
        ).round(2)
        
        # Save detailed country comparison
        country_comparison.to_csv(self.output_dir / 'country_comparison.csv')
        
        logger.info(f"Summary tables saved to {self.output_dir}")
        
        return summary_stats, country_comparison
    
    def create_latex_table(self, combined_data, output_dir):
        """
        Create a LaTeX table from combined_data showing only specified columns,
        with country names displayed only once per country-dataset combination.
        """
        logger.info("Creating LaTeX table...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Select and prepare the data
        table_data = combined_data.copy()
        
        # Calculate average edge length
        table_data['Avg Edge Length (km)'] = table_data['Total Length (km)'] / table_data['Edges']
        table_data['Avg Edge Length (km)'] = table_data['Avg Edge Length (km)'].round(2)
        
        # Round other numeric columns
        table_data['Total Length (km)'] = table_data['Total Length (km)'].round(0).astype(int)
        table_data['Edges'] = table_data['Edges'].astype(int)
        
        # Select only required columns
        columns_to_show = [
            'Country', 'Dataset', 'Graph Scenario', 'Edges', 
            'Total Length (km)', 'Avg Edge Length (km)', 'Stations_Used'
        ]
        table_data = table_data[columns_to_show]
        
        # Sort data for better organization
        table_data = table_data.sort_values(['Country', 'Dataset', 'Graph Scenario'])
        
        # Create longtable version
        latex_content = []
        latex_content.append("\\begin{longtable}{@{}p{1.8cm}p{1.2cm}p{1.4cm}r@{\\,}r@{\\,}r@{\\,}r@{}}")
        latex_content.append("\\caption{Graph Statistics by Country, Dataset and Scenario}")
        latex_content.append("\\label{tab:graph_statistics} \\\\")
        latex_content.append("\\toprule")
        latex_content.append("Country & Dataset & Scenario & Edges & Total Length & Avg Edge & Stations \\\\")
        latex_content.append(" & & & & (km) & Length (km) & Used \\\\")
        latex_content.append("\\midrule")
        latex_content.append("\\endfirsthead")
        latex_content.append("")
        latex_content.append("\\multicolumn{7}{c}{{\\tablename\\ \\thetable{} -- continued from previous page}} \\\\")
        latex_content.append("\\toprule")
        latex_content.append("Country & Dataset & Scenario & Edges & Total Length & Avg Edge & Stations \\\\")
        latex_content.append(" & & & & (km) & Length (km) & Used \\\\")
        latex_content.append("\\midrule")
        latex_content.append("\\endhead")
        latex_content.append("")
        latex_content.append("\\midrule")
        latex_content.append("\\multicolumn{7}{r@{}}{{Continued on next page}} \\\\")
        latex_content.append("\\endfoot")
        latex_content.append("")
        latex_content.append("\\bottomrule")
        latex_content.append("\\endlastfoot")
        
        # Track previous values to avoid repetition
        prev_country = None
        prev_dataset = None
        prev_stations = None
        
        # Helper function to format dataset names with math mode (remove 150000)
        def format_dataset_name(dataset_raw):
            if '_' in dataset_raw:
                parts = dataset_raw.split('_')
                # Just return the first part (LDCS or OECD) without the number
                return parts[0].upper()
            return dataset_raw.upper()
        
        # Process each row
        for _, row in table_data.iterrows():
            # Format country name (replace hyphens with double hyphens)
            country_raw = row['Country']
            country = country_raw.replace('-', '--').title()
            
            dataset_raw = row['Dataset']
            scenario = row['Graph Scenario']
            
            # Format numbers
            edges = f"{row['Edges']:,}"
            total_length = f"{row['Total Length (km)']:,}"
            avg_length = f"{row['Avg Edge Length (km)']:.2f}"
            stations = row['Stations_Used']
            
            # Format dataset name without the number part
            dataset = format_dataset_name(dataset_raw)
            
            # Only show values if different from previous
            country_display = country if country_raw != prev_country else ""
            dataset_display = dataset if (dataset_raw != prev_dataset or country_raw != prev_country) else ""
            stations_display = f"{stations:,}" if (stations != prev_stations or country_raw != prev_country or dataset_raw != prev_dataset) else ""
            
            # Create table row
            latex_row = f"{country_display} & {dataset_display} & {scenario} & {edges} & {total_length} & {avg_length} & {stations_display} \\\\"
            latex_content.append(latex_row)
            
            # Update previous values
            prev_country = country_raw
            prev_dataset = dataset_raw
            prev_stations = stations
        
        latex_content.append("\\end{longtable}")
        
        # Join all content
        latex_table = "\n".join(latex_content)
        
        # Save to file
        latex_file = output_path / 'graph_statistics_table.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("% LaTeX table of graph statistics\n")
            f.write("% Requires packages: longtable, booktabs\n")
            f.write("% Usage: \\input{graph_statistics_table.tex}\n\n")
            f.write(latex_table)
        
        logger.info(f"✓ LaTeX table saved to: {latex_file}")
        
        # Also create a compact version with smaller font
        self._create_compact_longtable(table_data, output_path)
        
        return latex_file

    def _create_compact_longtable(self, table_data, output_path):
        """Create a compact version of the longtable with smaller font and columns."""
        compact_latex = []
        compact_latex.append("\\footnotesize")
        compact_latex.append("\\begin{longtable}{@{}p{1.5cm}p{1cm}p{1.2cm}r@{\\,}r@{\\,}r@{\\,}r@{}}")
        compact_latex.append("\\caption{Graph Statistics (Compact Version)}")
        compact_latex.append("\\label{tab:graph_statistics_compact} \\\\")
        compact_latex.append("\\toprule")
        compact_latex.append("Country & Dataset & Scenario & Edges & Length & Avg & Stations \\\\")
        compact_latex.append(" & & & & (km) & (km) & \\\\")
        compact_latex.append("\\midrule")
        compact_latex.append("\\endfirsthead")
        compact_latex.append("")
        compact_latex.append("\\multicolumn{7}{c}{{\\tablename\\ \\thetable{} -- continued}} \\\\")
        compact_latex.append("\\toprule")
        compact_latex.append("Country & Dataset & Scenario & Edges & Length & Avg & Stations \\\\")
        compact_latex.append(" & & & & (km) & (km) & \\\\")
        compact_latex.append("\\midrule")
        compact_latex.append("\\endhead")
        compact_latex.append("")
        compact_latex.append("\\midrule")
        compact_latex.append("\\multicolumn{7}{r@{}}{{Continued}} \\\\")
        compact_latex.append("\\endfoot")
        compact_latex.append("")
        compact_latex.append("\\bottomrule")
        compact_latex.append("\\endlastfoot")
        
        # Helper functions for compact formatting
        def format_dataset_compact(dataset_raw):
            if '_' in dataset_raw:
                parts = dataset_raw.split('_')
                # Return just the first letter for maximum compactness
                return parts[0][0].upper()  # L for LDCS, O for OECD
            return dataset_raw[0].upper()
        
        def shorten_scenario(scenario):
            name_map = {
                'Original': 'Orig',
                'Original Pruned': 'Pruned',
                'KNN Filtered': 'KNN',
                'Randomized Filtered': 'Random'
            }
            return name_map.get(scenario, scenario[:6])
        
        def shorten_country(country):
            if len(country) > 12:
                # Shorten very long country names
                parts = country.split('--')
                if len(parts) > 1:
                    return parts[0][:8] + "."
                return country[:10] + "."
            return country
        
        # Track previous values
        prev_country = None
        prev_dataset = None
        prev_stations = None
        
        # Process data with compact formatting
        for _, row in table_data.iterrows():
            country_raw = row['Country']
            country = shorten_country(country_raw.replace('-', '--').title())
            dataset_raw = row['Dataset']
            scenario = shorten_scenario(row['Graph Scenario'])
            
            # Compact number formatting
            edges = f"{row['Edges']/1000:.0f}k" if row['Edges'] >= 1000 else f"{row['Edges']}"
            total_length = f"{row['Total Length (km)']/1000:.0f}k" if row['Total Length (km)'] >= 1000 else f"{row['Total Length (km)']:.0f}"
            avg_length = f"{row['Avg Edge Length (km)']:.1f}"
            stations = f"{row['Stations_Used']/1000:.1f}k" if row['Stations_Used'] >= 1000 else f"{row['Stations_Used']}"
            
            dataset = format_dataset_compact(dataset_raw)
            
            country_display = country if country_raw != prev_country else ""
            dataset_display = dataset if (dataset_raw != prev_dataset or country_raw != prev_country) else ""
            stations_display = stations if (row['Stations_Used'] != prev_stations or country_raw != prev_country or dataset_raw != prev_dataset) else ""
            
            latex_row = f"{country_display} & {dataset_display} & {scenario} & {edges} & {total_length} & {avg_length} & {stations_display} \\\\"
            compact_latex.append(latex_row)
            
            prev_country = country_raw
            prev_dataset = dataset_raw
            prev_stations = row['Stations_Used']
        
        compact_latex.append("\\end{longtable}")
        
        # Save compact version
        compact_latex_content = "\n".join(compact_latex)
        compact_latex_file = output_path / 'graph_statistics_table_compact.tex'
        with open(compact_latex_file, 'w', encoding='utf-8') as f:
            f.write("% Compact LaTeX longtable of graph statistics\n")
            f.write("% Requires packages: longtable, booktabs\n")
            f.write("% Usage: \\input{graph_statistics_table_compact.tex}\n\n")
            f.write(compact_latex_content)
        
        logger.info(f"✓ Compact LaTeX longtable saved to: {compact_latex_file}")

    def run_full_analysis(self):
        """Run the complete comparison analysis."""
        logger.info(f"Starting full analysis: {self.dir1.name} vs {self.dir2.name}")
        
        # Load and process data
        self.load_data()
        
        self.combined_data.to_csv(self.output_dir / 'combined_data.csv', index=False)

        # Create all plots
        self.plot_scenario_differences(self.combined_data, self.output_dir)
        self.plot_stations_scatter(self.combined_data, self.output_dir)
        self.plot_length_differences_by_dataset(self.combined_data, self.output_dir)

        # Generate summary tables
        summary_stats, country_comparison = self.create_summary_tables()
        
        # Create LaTeX table
        latex_files = self.create_latex_table(self.combined_data, self.output_dir)
        
        logger.info(f"Analysis complete. Results saved to {self.output_dir}")
        
        return {
            'summary_stats': summary_stats,
            'country_comparison': country_comparison,
            'combined_data': self.combined_data,
            'latex_files': latex_files
        }



def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Compare graph filtering results between two datasets')
    parser.add_argument('--dir1', type=str, help='First directory containing analysis results', default="/home/till/Music/ldcs_150000/")
    parser.add_argument('--dir2', type=str, help='Second directory containing analysis results', default="/home/till/Music/oecd_150000/")
    parser.add_argument('--output', type=str, default='output/comparison_output', 
                       help='Output directory for comparison results')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run comparison
    comparison = GraphComparison(args.dir1, args.dir2, args.output)
    results = comparison.run_full_analysis()
    
    print(f"\nComparison analysis complete!")
    print(f"Results saved to: {comparison.output_dir}")


if __name__ == "__main__":
    main()
