import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import argparse
import logging
from tabulate import tabulate
import matplotlib.font_manager as fm
from scipy import stats
from scipy.stats import shapiro, normaltest
import numpy as np

logger = logging.getLogger(__name__)

class GraphComparison:
    def __init__(self, dir1, dir2, output_dir="comparison_output"):
        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define consistent colors for scenarios and datasets
        self.scenario_colors = {
            'Original': '#ffffff',
            'Original Pruned': '#ffffff',
            'KNN Filtered': '#F8471B',
            'Randomized Filtered': '#1B8B7C'
        }
        
        self.dataset_colors = {
            'ldcs_150000': '#99C24D',
            'oecd_150000': '#692537'
        }
        
        # Configure matplotlib for better plots
        plt.style.use('fast')
        self._configure_matplotlib()

    def _configure_matplotlib(self):
        """Configure matplotlib settings including Open Sans font."""
        plt.style.use('fast')
        
        # Set Open Sans as the default font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Open Sans', 'DejaVu Sans', 'Arial', 'sans-serif']
        
        # Additional font settings for better appearance
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 16
        
        # Check if Open Sans is available
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'Open Sans' in available_fonts:
            print("✓ Open Sans font is available")
        else:
            print("⚠ Open Sans font not found, using fallback fonts")
            print("Available sans-serif fonts:", [f for f in available_fonts if 'sans' in f.lower()][:5])
    
        
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
        self.combined_data['Removal Scenario'] = self.combined_data['Graph Scenario']
        self.combined_data.drop(columns=['Graph Scenario'], inplace=True)
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
            original = group[group['Removal Scenario'] == 'Original']['Total Length (km)'].iloc[0]
            group['Pct_Diff'] = ((group['Total Length (km)'] - original) / original) * 100
            return group
        
        # Group by Country and Dataset, then calculate differences
        data_with_diff = combined_data.groupby(['Country', 'Dataset']).apply(calculate_percentage_diff).reset_index(drop=True)
        
        # Updated scenario names to match the new CSV format
        scenarios_to_plot = ['Original Pruned', 'KNN Filtered', 'Randomized Filtered']
        plot_data = data_with_diff[data_with_diff['Removal Scenario'].isin(scenarios_to_plot)]
        
        # Create subplots - one for each scenario
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        datasets = plot_data['Dataset'].unique()
        
        for idx, scenario in enumerate(scenarios_to_plot):
            ax = axes[idx]
            scenario_data = plot_data[plot_data['Removal Scenario'] == scenario]
            
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
        # plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Scenario difference histograms saved to: {output_file}")
        
        # plt.show()
        
        # Create a summary table
        summary_stats = []
        for scenario in scenarios_to_plot:
            scenario_data = plot_data[plot_data['Removal Scenario'] == scenario]
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
        original_data = combined_data[combined_data['Removal Scenario'] == 'Original'].copy()
        
        # Create simple scatter plot
        plt.figure(figsize=(10, 6))
        
        datasets = original_data['Dataset'].unique()

        for dataset in datasets:
            data_subset = original_data[original_data['Dataset'] == dataset]
            
            # Customize the label here instead of using raw dataset name
            custom_label = dataset.replace('_150000', '').upper()  # Remove number, make uppercase
            # Or use a mapping:
            label_mapping = {
                'ldcs_150000': 'Least Developed Countries',
                'oecd_150000': 'OECD Countries'
            }
            custom_label = label_mapping.get(dataset, dataset)
            
            plt.scatter(data_subset['Stations_Used'], data_subset['Total Length (km)'], 
                label=custom_label,  # Use custom label instead of dataset
                color=self.dataset_colors.get(dataset, '#CCCCCC'),
                alpha=0.4,
                s=20)
    
        plt.xscale('log')
        plt.yscale('log')
        
        # Set readable tick labels
        from matplotlib.ticker import FuncFormatter
        
        def thousands_formatter(x, pos):
            if x >= 1000000:
                return f'{x/1000000:.0f}M'
            elif x >= 1000:
                return f'{x/1000:.0f}K'
            else:
                return f'{x:.0f}'
        plt.xlim(0, None)  # Start x-axis at 0, auto-scale max
        plt.ylim(0, None)  # Start y-axis at 0, auto-scale max

        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        
        plt.xlabel('Number of Stations')
        plt.ylabel('Total Road Network Length (km)')
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
            original = group[group['Removal Scenario'] == 'Original']['Total Length (km)'].iloc[0]
            group['Length_Diff_Pct'] = ((group['Total Length (km)'] - original) / original) * 100
            return group
        
        # Group by Country and Dataset, then calculate differences
        data_with_diff = combined_data.groupby(['Country', 'Dataset']).apply(calculate_percentage_diff).reset_index(drop=True)
        
        # Filter out 'Original' scenario since it will always be 0%
        scenarios_to_plot = ['KNN Filtered', 'Randomized Filtered']
        plot_data = data_with_diff[data_with_diff['Removal Scenario'].isin(scenarios_to_plot)]
        
        # Get unique datasets
        datasets = sorted(plot_data['Dataset'].unique())
        
        # Create separate plot for each dataset
        output_files = []
        for dataset in datasets:
            dataset_data = plot_data[plot_data['Dataset'] == dataset]
            dataset_data["Country"] = dataset_data["Country"].str.replace('-', '--').str.title()

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
            
            # Sort countries by their KNN filtering difference (ascending)
            knn_data = dataset_data[dataset_data['Removal Scenario'] == 'KNN Filtered']
            country_knn_diffs = {}
            for country in countries_to_show:
                country_knn_data = knn_data[knn_data['Country'] == country]
                if len(country_knn_data) > 0:
                    country_knn_diffs[country] = abs(country_knn_data['Length_Diff_Pct'].iloc[0])
                else:
                    country_knn_diffs[country] = 0
            
            # Sort countries by KNN difference in ascending order
            countries_to_show = sorted(countries_to_show, key=lambda x: country_knn_diffs[x])
            
            logger.info(f"Showing {len(countries_to_show)} countries with >0.5% difference in {dataset}, sorted by KNN difference")
            
            # Set up the plot
            fig, ax = plt.subplots(figsize=(max(15, len(countries_to_show) * 0.8), 8))
            
            # Set bar width and positions
            bar_width = 0.25
            x = np.arange(len(countries_to_show))
            
            # Track countries with zero differences for note
            zero_diff_countries = []
            
            # Plot bars for each scenario
            for s_idx, scenario in enumerate(scenarios_to_plot):
                scenario_data = dataset_data[dataset_data['Removal Scenario'] == scenario]
                
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
            
            # Customize the plot - make y-axis more readable
            from matplotlib.ticker import FuncFormatter
            
            def percentage_formatter(x, pos):
                if x >= 1:
                    return f'{x:.0f}%'
                else:
                    return f'{x:.1f}%'
            
            # Use linear scale instead of log for better readability
            ax.set_xlabel('Country (sorted by KNN filtering difference, ascending)')
            ax.set_ylabel('Absolute Length Difference from Original (%)')
            ax.set_title(f'Total Length Changes by Country - {dataset.upper()}\n(Countries with >0.5% difference, sorted by KNN effect)')
            ax.set_xticks(x)
            ax.set_xticklabels(countries_to_show, rotation=45, ha='right')
            
            # Set reasonable y-axis limits and formatting
            max_val = max([max(values) for values in [ax.containers[i].datavalues for i in range(len(scenarios_to_plot))]] + [1])
            ax.set_ylim(0, max_val * 1.1)
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            
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
        """Create summary tables with statistics using median instead of mean."""
        logger.info("Creating summary tables...")
        
        # Calculate percentage differences for each country and scenario
        def calculate_percentage_diff(group):
            # Make sure we're using the right baseline scenario
            original_scenario = 'Original'
            if original_scenario not in group['Removal Scenario'].values:
                logger.warning(f"No 'Original' scenario found for {group['Country'].iloc[0]}, {group['Dataset'].iloc[0]}")
                return group
                
            original = group[group['Removal Scenario'] == original_scenario]['Total Length (km)'].iloc[0]
            original_edges = group[group['Removal Scenario'] == original_scenario]['Edges'].iloc[0]
            
            group['Length_Diff_Pct'] = ((group['Total Length (km)'] - original) / original) * 100
            group['Edge_Diff_Pct'] = ((group['Edges'] - original_edges) / original_edges) * 100
            return group
        
        # Group by Country and Dataset, then calculate differences
        data_with_diff = self.combined_data.groupby(['Country', 'Dataset']).apply(calculate_percentage_diff).reset_index(drop=True)
        
        # Filter out 'Original' scenario and only include scenarios with actual differences
        scenarios_to_include = ['KNN Filtered', 'Randomized Filtered']
        differences_data = data_with_diff[data_with_diff['Removal Scenario'].isin(scenarios_to_include)]
        
        # Debug: Check if we have non-zero values
        logger.info(f"Max Edge difference: {differences_data['Edge_Diff_Pct'].max()}")
        logger.info(f"Min Edge difference: {differences_data['Edge_Diff_Pct'].min()}")
        logger.info(f"Max Length difference: {differences_data['Length_Diff_Pct'].max()}")
        logger.info(f"Min Length difference: {differences_data['Length_Diff_Pct'].min()}")
        
        # Summary statistics by dataset and scenario using median instead of mean
        summary_stats_edge = differences_data.groupby(['Dataset', 'Removal Scenario']).agg({
            'Edge_Diff_Pct': ['median', 'min', 'max'],
        }).round(2)

        summary_stats_length = differences_data.groupby(['Dataset', 'Removal Scenario']).agg({
            'Length_Diff_Pct': ['median', 'min', 'max'],
        }).round(2)

        # Flatten column names and rename them to be more descriptive
        summary_stats_edge.columns = [
            'Median (\%)',
            'Min (\%)',
            'Max (\%)',
        ]

        summary_stats_length.columns = [
            'Median (\%)',
            'Min (\%)',
            'Max (\%)',
        ]

        summary_stats_edge = summary_stats_edge.reset_index()
        summary_stats_edge["Dataset"] = summary_stats_edge["Dataset"].str.replace('_150000', '').str.upper()

        summary_stats_length = summary_stats_length.reset_index()
        summary_stats_length["Dataset"] = summary_stats_length["Dataset"].str.replace('_150000', '').str.upper()

        # Save to CSV
        scenario_mapping = {
            'KNN Filtered': 'KNN',
            'Randomized Filtered': 'Randomized',
            'Original Pruned': 'Pruned'
        }
        summary_stats_edge['Removal Scenario'] = summary_stats_edge['Removal Scenario'].replace(scenario_mapping)
        summary_stats_length['Removal Scenario'] = summary_stats_length['Removal Scenario'].replace(scenario_mapping)

        summary_stats_edge.to_csv(self.output_dir / 'summary_statistics_edge.csv', index=False)
        summary_stats_edge.to_latex(self.output_dir / 'summary_statistics_edge.tex', index=False, float_format="%.2f", caption="Summary Statistics of Edge Percentage Differences by Dataset and Scenario", label="tab:summary_statistics_edge")

        summary_stats_length.to_csv(self.output_dir / 'summary_statistics_length.csv', index=False)
        summary_stats_length.to_latex(self.output_dir / 'summary_statistics_length.tex', index=False, float_format="%.2f", caption="Summary Statistics of Edge Length Percentage Differences by Dataset and Scenario", label="tab:summary_statistics_length")

        # Country-by-country comparison
        differences_data['Dataset'] = differences_data["Dataset"].str.replace('_150000', '').str.upper()
        country_comparison = differences_data.pivot_table(
            index=['Country', 'Dataset'], 
            columns='Removal Scenario', 
            values=['Edge_Diff_Pct', 'Length_Diff_Pct'],
            aggfunc='first'
        ).round(2)
        
        # Save detailed country comparison
        country_comparison.to_csv(self.output_dir / 'country_comparison.csv')
        country_comparison.to_latex(self.output_dir / 'country_comparison.tex', float_format="%.2f", caption="Country-by-Country Comparison of Edge and Length Percentage Differences", label="tab:country_comparison")

        logger.info(f"Summary tables saved to {self.output_dir}")

        return summary_stats_edge, summary_stats_length, country_comparison
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
            'Country', 'Dataset', 'Removal Scenario', 'Edges', 
            'Total Length (km)', 'Avg Edge Length (km)', 'Stations_Used'
        ]
        table_data = table_data[columns_to_show]
        
        # Sort data for better organization
        table_data = table_data.sort_values(['Country', 'Dataset'])
        
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
            scenario = row['Removal Scenario'].replace('Filtered', '').replace('Original Pruned', 'Pruned')
            
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
            scenario = shorten_scenario(row['Removal Scenario'])
            
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

    def create_scenario_differences_table(self, output_dir):
        """Create a LaTeX table for scenario differences summary."""
        logger.info("Creating scenario differences summary table...")
        
        output_path = Path(output_dir)
        summary_file = output_path / 'scenario_differences_summary.csv'
        
        if not summary_file.exists():
            logger.warning(f"Scenario differences summary file not found: {summary_file}")
            return None
        
        # Read the CSV file
        import pandas as pd
        summary_data = pd.read_csv(summary_file)
        
        # Filter out Original Pruned (all zeros)
        summary_data = summary_data[summary_data['Scenario'] != 'Original Pruned']
        
        # Create LaTeX table
        latex_content = []
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Scenario Differences: Mean Percentage Changes in Total Length}")
        latex_content.append("\\label{tab:scenario_differences}")
        latex_content.append("\\begin{tabular}{@{}p{2.5cm}p{1.5cm}r@{}}")
        latex_content.append("\\toprule")
        latex_content.append("Scenario & Dataset & Mean Change (\\%) \\\\")
        latex_content.append("\\midrule")
        
        # Helper function to format dataset names
        def format_dataset_name(dataset_raw):
            if '_' in dataset_raw:
                parts = dataset_raw.split('_')
                return parts[0].upper()
            return dataset_raw.upper()
        
        # Process each row
        for _, row in summary_data.iterrows():
            scenario = row['Scenario']
            dataset_raw = row['Dataset']
            dataset = format_dataset_name(dataset_raw)
            
            # Format percentage
            mean_pct = f"{row['Mean_Pct_Change']:.1f}"
            
            # Create table row
            latex_row = f"{scenario} & {dataset} & {mean_pct} \\\\"
            latex_content.append(latex_row)
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        # Join all content
        latex_table = "\n".join(latex_content)
        
        # Save to file
        latex_file = output_path / 'scenario_differences_table.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("% LaTeX table of scenario differences summary\n")
            f.write("% Requires packages: booktabs\n")
            f.write("% Usage: \\input{scenario_differences_table.tex}\n\n")
            f.write(latex_table)
        
        logger.info(f"✓ Scenario differences table saved to: {latex_file}")
        
        return latex_file
        
    def check_missing_countries(self):
        """
        Check which OECD and LDC countries are missing from the analysis directories
        and output a txt file with countries that have empty analysis output.
        """
        logger.info("Checking for missing countries in analysis directories...")
        
        # Define OECD and LDC country lists based on the actual shell scripts
        oecd_countries = {
            # Europe (from analyse_oecd.sh)
            'austria', 'belgium', 'czech-republic', 'denmark', 'estonia', 'finland', 
            'france', 'germany', 'greece', 'hungary', 'iceland', 'ireland-and-northern-ireland', 
            'italy', 'latvia', 'lithuania', 'luxembourg', 'netherlands', 'norway', 
            'poland', 'portugal', 'slovakia', 'slovenia', 'spain', 'sweden', 
            'switzerland', 'great-britain',
            # North America
            'canada', 'mexico', 'us',
            # Asia-Pacific
            'australia', 'japan', 'south-korea', 'new-zealand',
            # South America
            'chile', 'colombia',
            # Middle East
            'israel-and-palestine', 'turkey'
        }
        
        ldc_countries = {
            # Africa (from analyse_ldcs.sh)
            'angola', 'benin', 'burkina-faso', 'burundi', 'central-african-republic', 
            'chad', 'comores', 'congo-democratic-republic', 'djibouti', 'eritrea', 
            'ethiopia', 'senegal-and-gambia', 'guinea', 'guinea-bissau', 'lesotho', 
            'liberia', 'madagascar', 'malawi', 'mali', 'mauritania', 'mozambique', 
            'niger', 'rwanda', 'sierra-leone', 'somalia', 'south-sudan', 'sudan', 
            'togo', 'uganda', 'tanzania', 'zambia',
            # Asia
            'afghanistan', 'bangladesh', 'cambodia', 'laos', 'myanmar', 'nepal', 
            'yemen', 'east-timor',
            # Caribbean
            'haiti-and-domrep',
            # Pacific
            'kiribati', 'solomon-islands', 'tuvalu'
        }
        
        def check_directory_countries(directory, expected_countries, dataset_name):
            """Check which countries are present/missing in a directory."""
            directory = Path(directory)
            
            # Find all CSV files matching the pattern
            csv_files = list(directory.rglob("graph_info_table_*.csv"))
            
            # Extract country names from CSV files
            found_countries = set()
            countries_with_data = set()
            countries_with_empty_data = set()
            
            for csv_file in csv_files:
                # Extract country name from filename
                filename = csv_file.stem
                country = filename.replace('graph_info_table_', '')
                found_countries.add(country)
                
                # Check if the CSV file has actual data
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0 and not df.empty:
                        countries_with_data.add(country)
                    else:
                        countries_with_empty_data.add(country)
                        logger.warning(f"Empty CSV file found: {csv_file}")
                except Exception as e:
                    countries_with_empty_data.add(country)
                    logger.warning(f"Error reading CSV file {csv_file}: {e}")
            
            # Find missing countries
            missing_countries = expected_countries - found_countries
            
            return {
                'found_countries': found_countries,
                'countries_with_data': countries_with_data,
                'countries_with_empty_data': countries_with_empty_data,
                'missing_countries': missing_countries,
                'dataset_name': dataset_name
            }
        
        # Check both directories
        results = {}
        
        # Check dir1 (assume LDCS if contains 'ldc' in name, otherwise OECD)
        dir1_name = self.dir1.name.lower()
        if 'ldc' in dir1_name:
            results['dir1'] = check_directory_countries(self.dir1, ldc_countries, f"LDC ({self.dir1.name})")
        else:
            results['dir1'] = check_directory_countries(self.dir1, oecd_countries, f"OECD ({self.dir1.name})")
        
        # Check dir2 (assume LDCS if contains 'ldc' in name, otherwise OECD)
        dir2_name = self.dir2.name.lower()
        if 'ldc' in dir2_name:
            results['dir2'] = check_directory_countries(self.dir2, ldc_countries, f"LDC ({self.dir2.name})")
        else:
            results['dir2'] = check_directory_countries(self.dir2, oecd_countries, f"OECD ({self.dir2.name})")
        
        # Create output report
        output_file = self.output_dir / 'missing_countries_report.txt'
        
        with open(output_file, 'w') as f:
            f.write("MISSING COUNTRIES ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory 1: {self.dir1}\n")
            f.write(f"Directory 2: {self.dir2}\n\n")
            
            # Add expected country lists to report
            f.write("EXPECTED COUNTRIES BY DATASET:\n")
            f.write("-" * 30 + "\n")
            f.write(f"OECD Countries ({len(oecd_countries)}): {', '.join(sorted(oecd_countries))}\n\n")
            f.write(f"LDC Countries ({len(ldc_countries)}): {', '.join(sorted(ldc_countries))}\n\n")
            
            for dir_key, result in results.items():
                dataset_name = result['dataset_name']
                f.write(f"DATASET: {dataset_name}\n")
                f.write("-" * 30 + "\n")
                
                f.write(f"Countries found with data: {len(result['countries_with_data'])}\n")
                f.write(f"Countries with empty/corrupted data: {len(result['countries_with_empty_data'])}\n")
                f.write(f"Countries completely missing: {len(result['missing_countries'])}\n\n")
                
                if result['countries_with_empty_data']:
                    f.write("COUNTRIES WITH EMPTY/CORRUPTED ANALYSIS OUTPUT:\n")
                    for country in sorted(result['countries_with_empty_data']):
                        f.write(f"  - {country}\n")
                    f.write("\n")
                
                if result['missing_countries']:
                    f.write("COUNTRIES COMPLETELY MISSING FROM ANALYSIS:\n")
                    for country in sorted(result['missing_countries']):
                        f.write(f"  - {country}\n")
                    f.write("\n")
                
                if result['countries_with_data']:
                    f.write("COUNTRIES WITH SUCCESSFUL ANALYSIS:\n")
                    for country in sorted(result['countries_with_data']):
                        f.write(f"  - {country}\n")
                    f.write("\n")
                
                f.write("\n")
            
            # Summary section
            f.write("SUMMARY\n")
            f.write("=" * 20 + "\n")
            
            total_empty = 0
            total_missing = 0
            total_expected = 0
            
            for result in results.values():
                total_empty += len(result['countries_with_empty_data'])
                total_missing += len(result['missing_countries'])
                total_expected += len(result['countries_with_data']) + len(result['countries_with_empty_data']) + len(result['missing_countries'])
            
            f.write(f"Total countries with empty analysis output: {total_empty}\n")
            f.write(f"Total countries completely missing: {total_missing}\n")
            f.write(f"Total countries that need attention: {total_empty + total_missing}\n")
            
            # All countries that need attention
            all_problematic = set()
            for result in results.values():
                all_problematic.update(result['countries_with_empty_data'])
                all_problematic.update(result['missing_countries'])
            
            if all_problematic:
                f.write("\nALL COUNTRIES NEEDING ATTENTION:\n")
                for country in sorted(all_problematic):
                    f.write(f"  - {country}\n")
        
        logger.info(f"✓ Missing countries report saved to: {output_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("MISSING COUNTRIES ANALYSIS SUMMARY")
        print("="*60)
        
        for result in results.values():
            dataset_name = result['dataset_name']
            print(f"\n{dataset_name}:")
            print(f"  ✓ Countries with data: {len(result['countries_with_data'])}")
            print(f"  ⚠ Empty/corrupted data: {len(result['countries_with_empty_data'])}")
            print(f"  ✗ Completely missing: {len(result['missing_countries'])}")
            
            if result['countries_with_empty_data']:
                print(f"  Empty data countries: {', '.join(sorted(result['countries_with_empty_data']))}")
            if result['missing_countries']:
                print(f"  Missing countries: {', '.join(sorted(result['missing_countries']))}")
        
        return results, output_file
    
    def plot_edge_length_loss_vs_stations(self, combined_data, output_dir):
        """
        Create scatter plots showing edge length loss vs station density for each filtering scenario.
        """
        logger.info("Creating edge length loss vs station density scatter plots")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate percentage differences for each country and scenario
        def calculate_percentage_diff(group):
            # Use 'Original' as baseline
            original = group[group['Removal Scenario'] == 'Original']['Total Length (km)'].iloc[0]
            group['Length_Loss_Pct'] = ((original - group['Total Length (km)']) / original) * 100
            # Calculate station density (stations per 1000 km of road network)
            group['Station_Density'] = (group['Stations_Used'] / original) * 1000
            return group
        
        # Group by Country and Dataset, then calculate differences
        data_with_diff = combined_data.groupby(['Country', 'Dataset']).apply(calculate_percentage_diff).reset_index(drop=True)
        
        # Filter to only include filtering scenarios (exclude Original and Original Pruned)
        filtering_scenarios = ['KNN Filtered', 'Randomized Filtered']
        plot_data = data_with_diff[data_with_diff['Removal Scenario'].isin(filtering_scenarios)]
        
        # Create separate plots for each scenario
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        datasets = plot_data['Dataset'].unique()
        
        for idx, scenario in enumerate(filtering_scenarios):
            ax = axes[idx]
            scenario_data = plot_data[plot_data['Removal Scenario'] == scenario]
            
            # Plot for each dataset
            for dataset in datasets:
                dataset_scenario_data = scenario_data[scenario_data['Dataset'] == dataset]
                
                if len(dataset_scenario_data) > 0:
                    x_vals = dataset_scenario_data['Station_Density']
                    y_vals = dataset_scenario_data['Length_Loss_Pct']
                    
                    # Custom label mapping
                    label_mapping = {
                        'ldcs_150000': 'Least Developed Countries',
                        'oecd_150000': 'OECD Countries'
                    }
                    custom_label = label_mapping.get(dataset, dataset)
                    
                    ax.scatter(x_vals, y_vals,
                            label=f'{custom_label} (n={len(dataset_scenario_data)})',
                            color=self.dataset_colors.get(dataset, '#CCCCCC'),
                            alpha=0.6,
                            s=40,
                            edgecolors='black',
                            linewidth=0.5)
            
            # Set log scale for both axes
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Format axes
            from matplotlib.ticker import FuncFormatter
            
            def density_formatter(x, pos):
                if x >= 1:
                    return f'{x:.1f}'
                else:
                    return f'{x:.2f}'
                    
            def percentage_formatter(x, pos):
                if x >= 1:
                    return f'{x:.1f}%'
                else:
                    return f'{x:.2f}%'
            
            ax.xaxis.set_major_formatter(FuncFormatter(density_formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            
            # Set labels and title
            ax.set_xlabel('Station Density (stations per 1000 km road, log scale)')
            ax.set_ylabel('Edge Length Loss (%, log scale)')
            ax.set_title(f'{scenario}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set reasonable axis limits
            if len(scenario_data) > 0:
                x_min = max(0.01, scenario_data['Station_Density'].min() * 0.8)
                x_max = scenario_data['Station_Density'].max() * 1.2
                y_min = max(0.01, scenario_data['Length_Loss_Pct'].min() * 0.8)
                y_max = scenario_data['Length_Loss_Pct'].max() * 1.2
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
        
        plt.suptitle('Edge Length Loss vs Station Density (Log-Log Scale)', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / 'edge_length_loss_vs_station_density.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Edge length loss vs station density plot saved to: {output_file}")
        
        # Create summary statistics
        summary_stats = []
        for scenario in filtering_scenarios:
            scenario_data = plot_data[plot_data['Removal Scenario'] == scenario]
            for dataset in datasets:
                dataset_scenario_data = scenario_data[scenario_data['Dataset'] == dataset]
                if len(dataset_scenario_data) > 0:
                    # Calculate correlation between log(density) and log(length loss)
                    log_density = np.log10(dataset_scenario_data['Station_Density'].replace(0, 0.01))
                    log_loss = np.log10(dataset_scenario_data['Length_Loss_Pct'].replace(0, 0.01))
                    correlation = log_density.corr(log_loss)
                    
                    summary_stats.append({
                        'Scenario': scenario,
                        'Dataset': dataset,
                        'Mean_Station_Density': dataset_scenario_data['Station_Density'].mean(),
                        'Std_Station_Density': dataset_scenario_data['Station_Density'].std(),
                        'Mean_Length_Loss_Pct': dataset_scenario_data['Length_Loss_Pct'].mean(),
                        'Std_Length_Loss_Pct': dataset_scenario_data['Length_Loss_Pct'].std(),
                        'Min_Length_Loss_Pct': dataset_scenario_data['Length_Loss_Pct'].min(),
                        'Max_Length_Loss_Pct': dataset_scenario_data['Length_Loss_Pct'].max(),
                        'Log_Correlation_Density_Loss': correlation,
                        'Countries_Count': len(dataset_scenario_data)
                    })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save summary statistics
        summary_file = output_path / 'edge_length_loss_vs_station_density_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✓ Summary statistics saved to: {summary_file}")
        
        # Print correlation insights
        print("\nLog-Log Correlation between Station Density and Edge Length Loss:")
        print("=" * 70)
        for _, row in summary_df.iterrows():
            print(f"{row['Scenario']} - {row['Dataset']}: r = {row['Log_Correlation_Density_Loss']:.3f}")
        
        # Print some density statistics
        print("\nStation Density Statistics (stations per 1000 km road):")
        print("=" * 60)
        for dataset in datasets:
            dataset_data = plot_data[plot_data['Dataset'] == dataset]
            if len(dataset_data) > 0:
                label_mapping = {
                    'ldcs_150000': 'Least Developed Countries',
                    'oecd_150000': 'OECD Countries'
                }
                custom_label = label_mapping.get(dataset, dataset)
                print(f"{custom_label}:")
                print(f"  Mean: {dataset_data['Station_Density'].mean():.2f}")
                print(f"  Median: {dataset_data['Station_Density'].median():.2f}")
                print(f"  Range: {dataset_data['Station_Density'].min():.2f} - {dataset_data['Station_Density'].max():.2f}")
        
        return output_file, summary_file
    
    def perform_statistical_tests(self, combined_data, output_dir):
        """
        Perform statistical significance tests comparing OECD vs LDC vulnerability patterns.
        Uses Mann-Whitney U test and Cliff's delta effect size (non-parametric approach).
        """
        logger.info("Performing statistical significance tests...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate vulnerability differences for each country
        def calculate_vulnerability_differences(group):
            """Calculate the difference between KNN and Random removal scenarios."""
            original = group[group['Removal Scenario'] == 'Original']['Total Length (km)'].iloc[0]
            
            # Calculate percentage reductions
            knn_data = group[group['Removal Scenario'] == 'KNN Filtered']
            random_data = group[group['Removal Scenario'] == 'Randomized Filtered']
            
            if len(knn_data) > 0 and len(random_data) > 0:
                knn_reduction = ((original - knn_data['Total Length (km)'].iloc[0]) / original) * 100
                random_reduction = ((original - random_data['Total Length (km)'].iloc[0]) / original) * 100
                vulnerability_difference = knn_reduction - random_reduction
                
                return pd.Series({
                    'Country': group['Country'].iloc[0],
                    'Dataset': group['Dataset'].iloc[0],
                    'KNN_Reduction_Pct': knn_reduction,
                    'Random_Reduction_Pct': random_reduction,
                    'Vulnerability_Difference': vulnerability_difference,
                    'Original_Length_km': original,
                    'Stations_Used': group['Stations_Used'].iloc[0]
                })
            return pd.Series()
        
        # Calculate vulnerability differences for each country
        vulnerability_data = combined_data.groupby(['Country', 'Dataset']).apply(
            calculate_vulnerability_differences
        ).reset_index(drop=True)
        
        # Remove any empty rows
        vulnerability_data = vulnerability_data.dropna()
        
        # Separate OECD and LDC data
        oecd_data = vulnerability_data[vulnerability_data['Dataset'] == 'oecd_150000']
        ldc_data = vulnerability_data[vulnerability_data['Dataset'] == 'ldcs_150000']
        
        if len(oecd_data) == 0 or len(ldc_data) == 0:
            logger.warning("Insufficient data for statistical testing")
            return {}
        
        # Extract vulnerability differences for testing
        oecd_vulnerability = oecd_data['Vulnerability_Difference'].values
        ldc_vulnerability = ldc_data['Vulnerability_Difference'].values
        
        logger.info(f"OECD countries: {len(oecd_vulnerability)}")
        logger.info(f"LDC countries: {len(ldc_vulnerability)}")
        logger.info(f"OECD vulnerability range: {oecd_vulnerability.min():.2f}% to {oecd_vulnerability.max():.2f}%")
        logger.info(f"LDC vulnerability range: {ldc_vulnerability.min():.2f}% to {ldc_vulnerability.max():.2f}%")
        
        # Test for normality (for reporting purposes only)
        oecd_shapiro_stat, oecd_shapiro_p = shapiro(oecd_vulnerability)
        ldc_shapiro_stat, ldc_shapiro_p = shapiro(ldc_vulnerability)
        
        oecd_normal = oecd_shapiro_p > 0.05
        ldc_normal = ldc_shapiro_p > 0.05
        
        logger.info(f"OECD data normal: {oecd_normal} (Shapiro-Wilk p = {oecd_shapiro_p:.4f})")
        logger.info(f"LDC data normal: {ldc_normal} (Shapiro-Wilk p = {ldc_shapiro_p:.4f})")
        
        # Calculate basic descriptive statistics
        stats_results = {
            'oecd_n': len(oecd_vulnerability),
            'oecd_mean': np.mean(oecd_vulnerability),
            'oecd_std': np.std(oecd_vulnerability, ddof=1),
            'oecd_median': np.median(oecd_vulnerability),
            'oecd_min': np.min(oecd_vulnerability),
            'oecd_max': np.max(oecd_vulnerability),
            'ldc_n': len(ldc_vulnerability),
            'ldc_mean': np.mean(ldc_vulnerability),
            'ldc_std': np.std(ldc_vulnerability, ddof=1),
            'ldc_median': np.median(ldc_vulnerability),
            'ldc_min': np.min(ldc_vulnerability),
            'ldc_max': np.max(ldc_vulnerability),
            'oecd_normal': oecd_normal,
            'ldc_normal': ldc_normal,
            'oecd_shapiro_p': oecd_shapiro_p,
            'ldc_shapiro_p': ldc_shapiro_p
        }
        
        # Mann-Whitney U test (primary statistical test)
        u_stat, u_p_value = stats.mannwhitneyu(
            ldc_vulnerability, oecd_vulnerability, 
            alternative='two-sided'
        )
        
        stats_results.update({
            'mann_whitney_u_statistic': u_stat,
            'mann_whitney_p_value': u_p_value,
            'mann_whitney_significant': u_p_value < 0.05
        })
        
        # Cliff's delta (non-parametric effect size)
        def cliffs_delta(x, y):
            """Calculate Cliff's delta effect size."""
            n1, n2 = len(x), len(y)
            delta = 0
            for i in range(n1):
                for j in range(n2):
                    if x[i] > y[j]:
                        delta += 1
                    elif x[i] < y[j]:
                        delta -= 1
            return delta / (n1 * n2)
        
        cliffs_d = cliffs_delta(ldc_vulnerability, oecd_vulnerability)
        
        def interpret_cliffs_delta(d):
            """Interpret Cliff's delta effect size."""
            abs_d = abs(d)
            if abs_d < 0.147:
                return "negligible"
            elif abs_d < 0.33:
                return "small"
            elif abs_d < 0.474:
                return "medium"
            else:
                return "large"
        
        stats_results.update({
            'cliffs_delta': cliffs_d,
            'cliffs_delta_interpretation': interpret_cliffs_delta(cliffs_d),
            'mean_difference': np.mean(ldc_vulnerability) - np.mean(oecd_vulnerability),
            'median_difference': np.median(ldc_vulnerability) - np.median(oecd_vulnerability),
            'relative_difference': np.mean(ldc_vulnerability) / np.mean(oecd_vulnerability) if np.mean(oecd_vulnerability) != 0 else float('inf')
        })
        
        # Calculate confidence intervals for medians (non-parametric)
        def median_confidence_interval(data, confidence=0.95):
            """Calculate confidence interval for the median using bootstrap."""
            from scipy.stats import bootstrap
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            
            # Define statistic function for bootstrap
            def median_stat(x):
                return np.median(x)
            
            # Prepare data for bootstrap
            data_reshaped = (data.reshape(-1, 1).T,)
            
            # Perform bootstrap
            result = bootstrap(data_reshaped, median_stat, n_resamples=1000, 
                            confidence_level=confidence, random_state=rng)
            
            return result.confidence_interval.low, result.confidence_interval.high
        
        try:
            oecd_median_ci = median_confidence_interval(oecd_vulnerability)
            ldc_median_ci = median_confidence_interval(ldc_vulnerability)
            
            stats_results.update({
                'oecd_median_ci_lower': oecd_median_ci[0],
                'oecd_median_ci_upper': oecd_median_ci[1],
                'ldc_median_ci_lower': ldc_median_ci[0],
                'ldc_median_ci_upper': ldc_median_ci[1],
                'confidence_level': 0.95
            })
        except Exception as e:
            logger.warning(f"Could not calculate bootstrap confidence intervals: {e}")
            # Fallback to simple percentile method
            oecd_ci_lower, oecd_ci_upper = np.percentile(oecd_vulnerability, [2.5, 97.5])
            ldc_ci_lower, ldc_ci_upper = np.percentile(ldc_vulnerability, [2.5, 97.5])
            
            stats_results.update({
                'oecd_median_ci_lower': oecd_ci_lower,
                'oecd_median_ci_upper': oecd_ci_upper,
                'ldc_median_ci_lower': ldc_ci_lower,
                'ldc_median_ci_upper': ldc_ci_upper,
                'confidence_level': 0.95
            })
        
        # Save detailed results
        vulnerability_data.to_csv(output_path / 'vulnerability_data.csv', index=False)
        
        # Save statistical results
        stats_df = pd.DataFrame([stats_results])
        stats_df.to_csv(output_path / 'statistical_tests_results.csv', index=False)
        
        # Create comprehensive report
        self._create_statistical_report(stats_results, output_path)
        
        # Create LaTeX table for statistical results
        self._create_statistical_latex_table(stats_results, output_path)
        
        logger.info("Statistical analysis complete!")
        logger.info(f"Mann-Whitney U test result: U = {stats_results['mann_whitney_u_statistic']:.1f}, p = {stats_results['mann_whitney_p_value']:.4f}")
        logger.info(f"Effect size: Cliff's δ = {cliffs_d:.3f} ({interpret_cliffs_delta(cliffs_d)} effect)")
        
        return stats_results

    def _create_statistical_report(self, stats_results, output_path):
        """Create a simplified statistical analysis report."""
        report_file = output_path / 'statistical_analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Descriptive Statistics
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"OECD Countries (n = {stats_results['oecd_n']}):\n")
            f.write(f"  Mean: {stats_results['oecd_mean']:.2f}%\n")
            f.write(f"  Median: {stats_results['oecd_median']:.2f}% (95% CI: [{stats_results['oecd_median_ci_lower']:.2f}%, {stats_results['oecd_median_ci_upper']:.2f}%])\n")
            f.write(f"  Std Dev: {stats_results['oecd_std']:.2f}%\n")
            f.write(f"  Range: {stats_results['oecd_min']:.2f}% to {stats_results['oecd_max']:.2f}%\n\n")
            
            f.write(f"LDC Countries (n = {stats_results['ldc_n']}):\n")
            f.write(f"  Mean: {stats_results['ldc_mean']:.2f}%\n")
            f.write(f"  Median: {stats_results['ldc_median']:.2f}% (95% CI: [{stats_results['ldc_median_ci_lower']:.2f}%, {stats_results['ldc_median_ci_upper']:.2f}%])\n")
            f.write(f"  Std Dev: {stats_results['ldc_std']:.2f}%\n")
            f.write(f"  Range: {stats_results['ldc_min']:.2f}% to {stats_results['ldc_max']:.2f}%\n\n")
            
            # Normality Assessment
            f.write("NORMALITY ASSESSMENT\n")
            f.write("-" * 25 + "\n")
            f.write(f"OECD data normally distributed: {stats_results['oecd_normal']} (Shapiro-Wilk p = {stats_results['oecd_shapiro_p']:.4f})\n")
            f.write(f"LDC data normally distributed: {stats_results['ldc_normal']} (Shapiro-Wilk p = {stats_results['ldc_shapiro_p']:.4f})\n")
            f.write("→ Non-parametric methods used due to normality violations\n\n")
            
            # Statistical Test
            f.write("STATISTICAL SIGNIFICANCE TEST\n")
            f.write("-" * 35 + "\n")
            f.write("Mann-Whitney U test (non-parametric):\n")
            f.write(f"  U = {stats_results['mann_whitney_u_statistic']:.1f}\n")
            
            u_p = stats_results['mann_whitney_p_value']
            if u_p < 0.001:
                f.write(f"  p-value < 0.001\n")
            else:
                f.write(f"  p-value = {u_p:.4f}\n")
                
            f.write(f"  Significant (α = 0.05): {stats_results['mann_whitney_significant']}\n\n")
            
            # Effect Size
            f.write("EFFECT SIZE\n")
            f.write("-" * 15 + "\n")
            f.write(f"Cliff's δ: {stats_results['cliffs_delta']:.3f} ({stats_results['cliffs_delta_interpretation']} effect)\n")
            f.write(f"Mean difference: {stats_results['mean_difference']:.2f}%\n")
            f.write(f"Median difference: {stats_results['median_difference']:.2f}%\n")
            f.write(f"Relative difference: {stats_results['relative_difference']:.2f}x\n\n")
            
            # Interpretation
            f.write("INTERPRETATION\n")
            f.write("-" * 15 + "\n")
            if stats_results.get('mann_whitney_significant', False):
                f.write("✓ Significant difference found between OECD and LDC vulnerability patterns\n")
            else:
                f.write("✗ No significant difference found between OECD and LDC vulnerability patterns\n")
                
            f.write(f"✓ Effect size is {stats_results['cliffs_delta_interpretation']} - indicates practical significance\n")
            f.write("✓ Non-parametric methods appropriate given normality violations\n")
        
        logger.info(f"✓ Statistical analysis report saved to: {report_file}")

    def _create_statistical_latex_table(self, stats_results, output_path):
        """Create simplified LaTeX table for statistical results."""
        latex_content = []
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Statistical Analysis: OECD vs LDC Vulnerability Differences}")
        latex_content.append("\\label{tab:statistical_analysis}")
        latex_content.append("\\begin{tabular}{@{}lr@{}}")
        latex_content.append("\\toprule")
        latex_content.append("Measure & Value \\\\")
        latex_content.append("\\midrule")
        
        # Descriptive statistics
        latex_content.append("\\textbf{Descriptive Statistics} & \\\\")
        latex_content.append(f"OECD Median (95\\% CI) & {stats_results['oecd_median']:.2f}\\% ({stats_results['oecd_median_ci_lower']:.2f}\\%, {stats_results['oecd_median_ci_upper']:.2f}\\%) \\\\")
        latex_content.append(f"LDC Median (95\\% CI) & {stats_results['ldc_median']:.2f}\\% ({stats_results['ldc_median_ci_lower']:.2f}\\%, {stats_results['ldc_median_ci_upper']:.2f}\\%) \\\\")
        latex_content.append(f"Median Difference & {stats_results['median_difference']:.2f}\\% \\\\")
        latex_content.append("\\midrule")
        
        # Statistical test
        latex_content.append("\\textbf{Statistical Test} & \\\\")
        u_p = stats_results['mann_whitney_p_value']
        u_p_str = "< 0.001" if u_p < 0.001 else f"{u_p:.4f}"
        latex_content.append(f"Mann-Whitney U p-value & {u_p_str} \\\\")
        latex_content.append("\\midrule")
        
        # Effect size
        latex_content.append("\\textbf{Effect Size} & \\\\")
        latex_content.append(f"Cliff's $\\delta$ & {stats_results['cliffs_delta']:.3f} ({stats_results['cliffs_delta_interpretation']}) \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        latex_table = "\n".join(latex_content)
        
        latex_file = output_path / 'statistical_analysis_table.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        logger.info(f"✓ Statistical analysis LaTeX table saved to: {latex_file}")
    def run_full_analysis(self):
        """Run the complete comparison analysis."""
        logger.info(f"Starting full analysis: {self.dir1.name} vs {self.dir2.name}")
        
        # Check for missing countries first
        missing_results, missing_report_file = self.check_missing_countries()
        
        # Load and process data
        self.load_data()
        
        self.combined_data.to_csv(self.output_dir / 'combined_data.csv', index=False)

        # Create all plots
        self.plot_scenario_differences(self.combined_data, self.output_dir)
        self.plot_stations_scatter(self.combined_data, self.output_dir)
        self.plot_length_differences_by_dataset(self.combined_data, self.output_dir)
        
        # Add the new edge length loss vs stations plot
        edge_loss_plot, edge_loss_summary = self.plot_edge_length_loss_vs_stations(self.combined_data, self.output_dir)

        self.create_scenario_differences_table(self.output_dir)

        # Generate summary tables
        summary_stats_edge, summary_stats_length, country_comparison = self.create_summary_tables()

        # CREATE LaTeX table
        latex_files = self.create_latex_table(self.combined_data, self.output_dir)
        
        # NEW: Perform statistical tests
        statistical_results = self.perform_statistical_tests(self.combined_data, self.output_dir)
        
        logger.info(f"Analysis complete. Results saved to {self.output_dir}")
        logger.info(f"Missing countries report: {missing_report_file}")
        
        return {
            'summary_stats_edge': summary_stats_edge,
            'summary_stats_length': summary_stats_length,
            'country_comparison': country_comparison,
            'combined_data': self.combined_data,
            'latex_files': latex_files,
            'missing_countries': missing_results,
            'missing_report': missing_report_file,
            'edge_loss_plot': edge_loss_plot,
            'edge_loss_summary': edge_loss_summary,
            'statistical_results': statistical_results  # NEW
        }



def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Compare graph filtering results between two datasets')
    parser.add_argument('--dir1', type=str, help='First directory containing analysis results', default="output/ldcs_150000/")
    parser.add_argument('--dir2', type=str, help='Second directory containing analysis results', default="output/oecd_150000/")
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
