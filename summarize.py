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
        
        # Configure matplotlib for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
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
                    # Extract country name from file path
                    country = csv_file.parent.name
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
                    logger.debug(f"Loaded {csv_file} with {len(df)} rows, {stations_count} stations")
                    
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
    
    def _safe_float_conversion(self, value):
        """Safely convert value to float, handling both strings and numeric types."""
        if isinstance(value, str):
            # Remove commas if it's a string
            return float(value.replace(',', ''))
        else:
            # Already numeric
            return float(value)
    
    def calculate_differences(self):
        """Calculate percentage differences from Original to filtered versions."""
        logger.info("Calculating percentage differences...")
        
        results = []
        
        for dataset in self.combined_data['Dataset'].unique():
            dataset_data = self.combined_data[self.combined_data['Dataset'] == dataset]
            
            for country in dataset_data['Country'].unique():
                country_data = dataset_data[dataset_data['Country'] == country]
                
                # Get original values
                original = country_data[country_data['Graph Scenario'] == 'Original']
                knn = country_data[country_data['Graph Scenario'] == 'KNN Filtered']
                random = country_data[country_data['Graph Scenario'] == 'Randomized Filtered']
                
                if len(original) == 0:
                    logger.warning(f"No original data found for {country} in {dataset}")
                    continue
                
                original_edges = self._safe_float_conversion(original['Edges'].iloc[0])
                original_length = self._safe_float_conversion(original['Total Length (km)'].iloc[0])
                
                for scenario, filtered_data in [('KNN', knn), ('Random', random)]:
                    if len(filtered_data) == 0:
                        continue
                        
                    filtered_edges = self._safe_float_conversion(filtered_data['Edges'].iloc[0])
                    filtered_length = self._safe_float_conversion(filtered_data['Total Length (km)'].iloc[0])
                    
                    # Calculate percentage changes
                    edge_change = ((filtered_edges - original_edges) / original_edges) * 100
                    length_change = ((filtered_length - original_length) / original_length) * 100
                    
                    results.append({
                        'Dataset': dataset,
                        'Country': country,
                        'Scenario': scenario,
                        'Edge_Change_Pct': edge_change,
                        'Length_Change_Pct': length_change,
                        'Original_Edges': original_edges,
                        'Filtered_Edges': filtered_edges,
                        'Original_Length': original_length,
                        'Filtered_Length': filtered_length
                    })
        
        self.differences = pd.DataFrame(results)
        return self.differences
    
    def create_comparison_plots(self):
        """Create comparison plots between the two datasets."""
        logger.info("Creating comparison plots...")
        
        # Set up the plotting style
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Graph Filtering Comparison: {self.dir1.name} vs {self.dir2.name}', fontsize=16)
        
        # Plot 1: Edge reduction comparison
        ax1 = axes[0, 0]
        knn_data = self.differences[self.differences['Scenario'] == 'KNN']
        random_data = self.differences[self.differences['Scenario'] == 'Random']
        
        datasets = knn_data['Dataset'].unique()
        x = np.arange(len(datasets))
        width = 0.35
        
        knn_means = [knn_data[knn_data['Dataset'] == d]['Edge_Change_Pct'].mean() for d in datasets]
        random_means = [random_data[random_data['Dataset'] == d]['Edge_Change_Pct'].mean() for d in datasets]
        
        ax1.bar(x - width/2, knn_means, width, label='KNN Filtered', alpha=0.8)
        ax1.bar(x + width/2, random_means, width, label='Random Filtered', alpha=0.8)
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Average Edge Reduction (%)')
        ax1.set_title('Edge Reduction Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Length reduction comparison
        ax2 = axes[0, 1]
        knn_length_means = [knn_data[knn_data['Dataset'] == d]['Length_Change_Pct'].mean() for d in datasets]
        random_length_means = [random_data[random_data['Dataset'] == d]['Length_Change_Pct'].mean() for d in datasets]
        
        ax2.bar(x - width/2, knn_length_means, width, label='KNN Filtered', alpha=0.8)
        ax2.bar(x + width/2, random_length_means, width, label='Random Filtered', alpha=0.8)
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Average Length Reduction (%)')
        ax2.set_title('Total Length Reduction Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot - Edge vs Length reduction for KNN
        ax3 = axes[1, 0]
        for dataset in datasets:
            data = knn_data[knn_data['Dataset'] == dataset]
            ax3.scatter(data['Edge_Change_Pct'], data['Length_Change_Pct'], 
                       label=dataset, alpha=0.7, s=50)
        
        ax3.set_xlabel('Edge Reduction (%)')
        ax3.set_ylabel('Length Reduction (%)')
        ax3.set_title('KNN Filtering: Edge vs Length Reduction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plot showing distribution of reductions
        ax4 = axes[1, 1]
        plot_data = []
        labels = []
        
        for dataset in datasets:
            for scenario in ['KNN', 'Random']:
                data = self.differences[(self.differences['Dataset'] == dataset) & 
                                      (self.differences['Scenario'] == scenario)]
                plot_data.append(data['Edge_Change_Pct'].values)
                labels.append(f'{dataset}\n{scenario}')
        
        ax4.boxplot(plot_data, labels=labels)
        ax4.set_ylabel('Edge Reduction (%)')
        ax4.set_title('Distribution of Edge Reductions')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / 'comparison_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'comparison_plots.pdf', bbox_inches='tight')
        logger.info(f"Comparison plots saved to {plot_path}")
        
        plt.show()
    
    def create_summary_tables(self):
        """Create summary tables with statistics."""
        logger.info("Creating summary tables...")
        
        # Summary statistics by dataset and scenario
        summary_stats = self.differences.groupby(['Dataset', 'Scenario']).agg({
            'Edge_Change_Pct': ['mean', 'std', 'min', 'max', 'count'],
            'Length_Change_Pct': ['mean', 'std', 'min', 'max']
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
        country_comparison = self.differences.pivot_table(
            index=['Country', 'Dataset'], 
            columns='Scenario', 
            values=['Edge_Change_Pct', 'Length_Change_Pct'],
            aggfunc='first'
        ).round(2)
        
        # Save detailed country comparison
        country_comparison.to_csv(self.output_dir / 'country_comparison.csv')
        
        # Create difference between KNN and Random
        knn_vs_random = []
        for country in self.differences['Country'].unique():
            for dataset in self.differences['Dataset'].unique():
                country_data = self.differences[
                    (self.differences['Country'] == country) & 
                    (self.differences['Dataset'] == dataset)
                ]
                
                knn_edge = country_data[country_data['Scenario'] == 'KNN']['Edge_Change_Pct']
                random_edge = country_data[country_data['Scenario'] == 'Random']['Edge_Change_Pct']
                
                if len(knn_edge) > 0 and len(random_edge) > 0:
                    knn_vs_random.append({
                        'Country': country,
                        'Dataset': dataset,
                        'Edge_Diff_KNN_vs_Random': knn_edge.iloc[0] - random_edge.iloc[0],
                        'KNN_More_Effective': knn_edge.iloc[0] < random_edge.iloc[0]  # More negative = more reduction
                    })
        
        effectiveness_df = pd.DataFrame(knn_vs_random)
        
        # Summary of effectiveness
        if len(effectiveness_df) > 0:
            effectiveness_summary = effectiveness_df.groupby('Dataset').agg({
                'Edge_Diff_KNN_vs_Random': ['mean', 'std'],
                'KNN_More_Effective': ['sum', 'count']
            }).round(3)
            
            print("\n" + "="*60)
            print("KNN vs RANDOM EFFECTIVENESS COMPARISON")
            print("="*60)
            print("Negative Edge_Diff means KNN removes more edges than Random")
            print(tabulate(effectiveness_summary, headers='keys', tablefmt='grid'))
            
            effectiveness_df.to_csv(self.output_dir / 'knn_vs_random_effectiveness.csv', index=False)
        
        logger.info(f"Summary tables saved to {self.output_dir}")
        
        return summary_stats, country_comparison, effectiveness_df
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for publication."""
        logger.info("Generating LaTeX tables...")
        
        # Summary statistics table
        summary_stats = self.differences.groupby(['Dataset', 'Scenario']).agg({
            'Edge_Change_Pct': ['mean', 'std'],
            'Length_Change_Pct': ['mean', 'std']
        }).round(2)
        
        # Create a cleaner version for LaTeX
        latex_data = []
        for dataset in summary_stats.index.get_level_values(0).unique():
            for scenario in ['KNN', 'Random']:
                if (dataset, scenario) in summary_stats.index:
                    row = summary_stats.loc[(dataset, scenario)]
                    latex_data.append({
                        'Dataset': dataset,
                        'Scenario': scenario,
                        'Edge Reduction (%)': f"{row[('Edge_Change_Pct', 'mean')]:.1f} ± {row[('Edge_Change_Pct', 'std')]:.1f}",
                        'Length Reduction (%)': f"{row[('Length_Change_Pct', 'mean')]:.1f} ± {row[('Length_Change_Pct', 'std')]:.1f}"
                    })
        
        latex_df = pd.DataFrame(latex_data)
        
        # Generate LaTeX table
        latex_table = latex_df.to_latex(
            index=False,
            column_format='llcc',
            caption='Comparison of Graph Filtering Effects: Mean ± Standard Deviation',
            label='tab:filtering_comparison',
            escape=False
        )
        
        # Save LaTeX table
        latex_path = self.output_dir / 'comparison_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"LaTeX table saved to {latex_path}")
        
        return latex_table

    def plot_scenario_differences(self, combined_data, output_dir):
        """
        Create histograms showing percentage differences from Original for each scenario,
        comparing between datasets.
        
        Parameters
        ----------
        combined_data : pd.DataFrame
            DataFrame with columns: 'Total Length (km)', 'Dataset', 'Graph Scenario', 'Country'
        output_dir : str or Path
            Directory to save the plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from pathlib import Path
        
        logger.info("Creating scenario difference histograms")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate percentage differences for each country and scenario
        def calculate_percentage_diff(group):
            original = group[group['Graph Scenario'] == 'Original']['Total Length (km)'].iloc[0]
            group['Pct_Diff'] = ((group['Total Length (km)'] - original) / original) * 100
            return group
        
        # Group by Country and Dataset, then calculate differences
        data_with_diff = combined_data.groupby(['Country', 'Dataset']).apply(calculate_percentage_diff).reset_index(drop=True)
        
        # Filter out 'Original' scenario since it will always be 0%
        scenarios_to_plot = ['Original Pruned', 'KNN Filtered', 'Randomized Filtered']
        plot_data = data_with_diff[data_with_diff['Graph Scenario'].isin(scenarios_to_plot)]
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'wheat']
        
        # Create subplots - one for each scenario
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        datasets = plot_data['Dataset'].unique()
        
        for idx, scenario in enumerate(scenarios_to_plot):
            ax = axes[idx]
            scenario_data = plot_data[plot_data['Graph Scenario'] == scenario]
            
            # Create histogram for each dataset
            for i, dataset in enumerate(datasets):
                dataset_scenario_data = scenario_data[scenario_data['Dataset'] == dataset]
                pct_diffs = dataset_scenario_data['Pct_Diff'].values
                
                if len(pct_diffs) > 0:  # Only plot if we have data
                    ax.hist(pct_diffs, 
                        alpha=0.7, 
                        label=f'{dataset} (n={len(pct_diffs)})', 
                        bins=20,
                        color=colors[i % len(colors)],
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
        
        plt.show()
        
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
        
        import pandas as pd
        summary_df = pd.DataFrame(summary_stats)
        
        # Save summary table
        summary_file = output_path / 'scenario_differences_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✓ Summary statistics saved to: {summary_file}")
        
        # Create box plots for additional comparison
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, scenario in enumerate(scenarios_to_plot):
            ax = axes2[idx]
            scenario_data = plot_data[plot_data['Graph Scenario'] == scenario]
            
            # Prepare data for box plot
            box_data = []
            box_labels = []
            for dataset in datasets:
                dataset_scenario_data = scenario_data[scenario_data['Dataset'] == dataset]
                if len(dataset_scenario_data) > 0:
                    box_data.append(dataset_scenario_data['Pct_Diff'].values)
                    box_labels.append(dataset)
            
            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_ylabel('Percentage Change from Original (%)')
            ax.set_title(f'{scenario} vs Original')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Total Length Percentage Changes - Box Plot Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save box plot
        boxplot_file = output_path / 'scenario_differences_boxplots.png'
        plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Scenario difference box plots saved to: {boxplot_file}")
        
        plt.show()
    
        return output_file, boxplot_file, summary_file
    

    def plot_stations_scatter(self, combined_data, output_dir):
        """
        Create a simple scatter plot of stations vs total length.
        
        Parameters
        ----------
        combined_data : pd.DataFrame
            DataFrame with columns: 'Stations_Used', 'Total Length (km)', 'Dataset'
        output_dir : str or Path
            Directory to save the plot
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        logger.info("Creating simple stations scatter plot")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Filter to only Original scenario
        original_data = combined_data[combined_data['Graph Scenario'] == 'Original'].copy()
        
        # Create simple scatter plot
        plt.figure(figsize=(10, 6))
        
        datasets = original_data['Dataset'].unique()
        colors = ['blue', 'red']
        
        for i, dataset in enumerate(datasets):
            data_subset = original_data[original_data['Dataset'] == dataset]
            plt.scatter(data_subset['Stations_Used'], data_subset['Total Length (km)'], 
                    label=dataset, 
                    color=colors[i % len(colors)],
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
        
        plt.show()
        
        return output_file

    def run_full_analysis(self):
        """Run the complete comparison analysis."""
        logger.info(f"Starting full analysis: {self.dir1.name} vs {self.dir2.name}")
        
        # Load and process data
        self.load_data()
        
        self.combined_data.to_csv(self.output_dir / 'combined_data.csv', index=False)

        self.plot_scenario_differences(self.combined_data, self.output_dir)
        self.plot_stations_scatter(self.combined_data, self.output_dir)

        self.calculate_differences()
        
        # Generate outputs
        self.create_comparison_plots()
        summary_stats, country_comparison, effectiveness = self.create_summary_tables()
        latex_table = self.generate_latex_tables()
        
        logger.info(f"Analysis complete. Results saved to {self.output_dir}")
        
        return {
            'summary_stats': summary_stats,
            'country_comparison': country_comparison,
            'effectiveness': effectiveness,
            'latex_table': latex_table
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