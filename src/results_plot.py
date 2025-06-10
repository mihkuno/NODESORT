import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import json
from cluster_specs import plot_data_distribution

# Ensure the output directory for charts exists
OUTPUT_DIR = 'out_matplotlib_charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a consistent, professional color palette
COLORS = {
    'hpsrs': '#2E86AB',      # Deep Blue
    'hpslp': '#A23B72',      # Deep Magenta
    'accent': '#F18F01',     # Orange
    'success': '#2A9D8F',    # Teal
    'warning': '#E76F51',    # Coral
    'neutral': '#6C757D',    # Gray
    'background': '#FFFFFF', # White
    'grid': '#E9ECEF',       # Light Gray
    'text': '#212529'        # Dark Gray
}

def setup_plot_style():
    """Configure matplotlib and seaborn for consistent, beautiful plots."""
    plt.style.use('default')
    
    # Configure matplotlib parameters
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['background'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.5,
        'axes.edgecolor': COLORS['grid'],
        'grid.alpha': 0.4,
        'grid.linewidth': 1.0,
        'grid.color': COLORS['grid'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold'
    })
    
    # Set seaborn style
    sns.set_palette([COLORS['hpsrs'], COLORS['hpslp'], COLORS['accent'], 
                     COLORS['success'], COLORS['warning'], COLORS['neutral']])

def format_large_numbers(x, pos):
    """Format large numbers with K, M suffixes for readability."""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{int(x)}'

def create_scatter_plot(ax, x_data, y_hpsrs, y_hpslp, df, title, ylabel):
    """Create a consistent scatter plot with proper styling."""
    # Plot scatter points with better styling
    ax.scatter(x_data, df[y_hpsrs], color=COLORS['hpsrs'], label='H-PSRS', 
               alpha=0.8, s=60, edgecolors='white', linewidth=1)
    ax.scatter(x_data, df[y_hpslp], color=COLORS['hpslp'], label='H-PSLP', 
               alpha=0.8, s=60, edgecolors='white', linewidth=1)
    
    # Styling
    ax.set_title(title, pad=20)
    ax.set_xlabel('Simulation Run')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))

def create_bar_plot(ax, y_hpsrs, y_hpslp, df, title, ylabel=None):
    """Create a consistent bar plot with proper styling."""
    avg_vals = df[[y_hpsrs, y_hpslp]].mean()
    
    # Create bars with better styling
    bars = ax.bar(['H-PSRS', 'H-PSLP'], avg_vals, 
                  color=[COLORS['hpsrs'], COLORS['hpslp']], 
                  alpha=0.9, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, pad=20)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

def plot_aggregated_metrics(df, filename='agg_metrics_comparison.png'):
    """Plot comprehensive performance comparison with improved styling."""
    setup_plot_style()
    
    if df.empty:
        print("Warning: No data for aggregated metrics plot. Skipping.")
        return

    runs = df['run']
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Performance Analysis: Individual Runs vs Average Performance', 
                 fontsize=22, fontweight='bold', y=0.98)

    # Left Column: Scatter Plots (All Runs)
    create_scatter_plot(axes[0, 0], runs, 'hpsrs_makespan', 'hpslp_makespan', df,
                       'Makespan Comparison (All Runs)', 'Makespan (µs)')
    
    create_scatter_plot(axes[1, 0], runs, 'hpsrs_cost', 'hpslp_cost', df,
                       'Total Cost Comparison (All Runs)', 'Cost ($)')
    
    create_scatter_plot(axes[2, 0], runs, 'hpsrs_mem_util', 'hpslp_mem_util', df,
                       'Memory Utilization (All Runs)', 'Utilization (%)')

    # Right Column: Bar Charts (Averages)
    create_bar_plot(axes[0, 1], 'hpsrs_makespan', 'hpslp_makespan', df,
                   'Average Makespan', 'Makespan (µs)')
    
    create_bar_plot(axes[1, 1], 'hpsrs_cost', 'hpslp_cost', df,
                   'Average Total Cost', 'Cost ($)')
    
    create_bar_plot(axes[2, 1], 'hpsrs_mem_util', 'hpslp_mem_util', df,
                   'Average Memory Utilization', 'Utilization (%)')

    # Special formatting for cost plot
    axes[1, 0].set_yscale('log')
    axes[1, 1].set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"✓ Saved aggregated metrics chart: {filename}")
    plt.close(fig)

def plot_lp_prediction_accuracy(df, filename='lp_prediction_accuracy.png'):
    """Plot LP model prediction accuracy with enhanced visualization."""
    setup_plot_style()
    
    df_filtered = df[
        (df['hpslp_predicted_makespan'] > 0) & 
        (df['hpslp_actual_sort_makespan'] > 0) & 
        (df['hpslp_actual_sort_cost'] > 0)
    ].copy()
    
    if df_filtered.empty:
        print("Warning: No prediction data for LP accuracy plot. Skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Linear Programming Model Prediction Accuracy', 
                 fontsize=20, fontweight='bold', y=0.98)

    # Makespan scatter plot
    sns.scatterplot(data=df_filtered, x='hpslp_predicted_makespan', 
                   y='hpslp_actual_sort_makespan', hue='distribution', 
                   style='nodes', ax=axes[0, 0], s=80, alpha=0.8)
    
    # Perfect prediction line
    lims_makespan = [0, max(df_filtered['hpslp_predicted_makespan'].max(), 
                           df_filtered['hpslp_actual_sort_makespan'].max()) * 1.05]
    axes[0, 0].plot(lims_makespan, lims_makespan, 'k--', alpha=0.7, 
                    linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_title('Sort Phase Makespan Prediction', pad=15)
    axes[0, 0].set_xlabel('Predicted Makespan (µs)')
    axes[0, 0].set_ylabel('Actual Makespan (µs)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cost scatter plot
    sns.scatterplot(data=df_filtered, x='hpslp_predicted_cost', 
                   y='hpslp_actual_sort_cost', hue='distribution', 
                   style='nodes', ax=axes[0, 1], s=80, alpha=0.8)
    
    lims_cost = [0, max(df_filtered['hpslp_predicted_cost'].max(), 
                       df_filtered['hpslp_actual_sort_cost'].max()) * 1.05]
    axes[0, 1].plot(lims_cost, lims_cost, 'k--', alpha=0.7, 
                    linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_title('Sort Phase Cost Prediction', pad=15)
    axes[0, 1].set_xlabel('Predicted Cost ($)')
    axes[0, 1].set_ylabel('Actual Cost ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Calculate MAPE
    mape_makespan = np.mean(np.abs((df_filtered['hpslp_actual_sort_makespan'] - 
                                   df_filtered['hpslp_predicted_makespan']) / 
                                  df_filtered['hpslp_actual_sort_makespan'])) * 100
    mape_cost = np.mean(np.abs((df_filtered['hpslp_actual_sort_cost'] - 
                               df_filtered['hpslp_predicted_cost']) / 
                              df_filtered['hpslp_actual_sort_cost'])) * 100

    # Makespan comparison bars
    avg_predicted_makespan = df_filtered['hpslp_predicted_makespan'].mean()
    avg_actual_makespan = df_filtered['hpslp_actual_sort_makespan'].mean()
    
    bars1 = axes[1, 0].bar(['Predicted', 'Actual'], 
                          [avg_predicted_makespan, avg_actual_makespan], 
                          color=[COLORS['accent'], COLORS['hpslp']], 
                          alpha=0.9, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars1, [avg_predicted_makespan, avg_actual_makespan]):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    axes[1, 0].set_title(f'Average Makespan Comparison\n(MAPE: {mape_makespan:.1f}%)', pad=15)
    axes[1, 0].set_ylabel('Makespan (µs)')
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # Cost comparison bars
    avg_predicted_cost = df_filtered['hpslp_predicted_cost'].mean()
    avg_actual_cost = df_filtered['hpslp_actual_sort_cost'].mean()
    
    bars2 = axes[1, 1].bar(['Predicted', 'Actual'], 
                          [avg_predicted_cost, avg_actual_cost], 
                          color=[COLORS['accent'], COLORS['hpslp']], 
                          alpha=0.9, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars2, [avg_predicted_cost, avg_actual_cost]):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    axes[1, 1].set_title(f'Average Cost Comparison\n(MAPE: {mape_cost:.1f}%)', pad=15)
    axes[1, 1].set_ylabel('Cost ($)')
    axes[1, 1].grid(True, axis='y', alpha=0.3)

    # Format all y-axes
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"✓ Saved LP accuracy chart: {filename}")
    plt.close(fig)

def plot_scalability(df, filename='scalability_analysis.png'):
    """Generate scalability analysis with clean line plots and confidence intervals."""
    setup_plot_style()
    
    if df.empty:
        print("Warning: No data for scalability plots. Skipping.")
        return

    # Define colors for different combinations
    colors = {
        'H-PSRS (uniform)': COLORS['hpsrs'],
        'H-PSLP (uniform)': COLORS['hpslp'],
        'H-PSRS (gaussian)': COLORS['accent'],
        'H-PSLP (gaussian)': COLORS['warning']
    }
    
    markers = {
        'H-PSRS (uniform)': 'o',
        'H-PSLP (uniform)': 's',
        'H-PSRS (gaussian)': '^',
        'H-PSLP (gaussian)': 'D'
    }

    dataset_sizes = sorted(df['dataset_size'].unique())
    fig, axes = plt.subplots(2, len(dataset_sizes), figsize=(20, 12))
    
    if len(dataset_sizes) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Scalability Analysis Across Dataset Sizes and Node Counts', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    for j, size in enumerate(dataset_sizes):
        df_size = df[df['dataset_size'] == size]
        
        # Makespan Plot
        ax_makespan = axes[0, j]
        for dist in ['uniform', 'gaussian']:
            subset = df_size[df_size['distribution'] == dist]
            if subset.empty:
                continue
            
            # Group by nodes and calculate statistics
            grouped = subset.groupby('nodes').agg({
                'hpsrs_makespan': ['mean', 'std'],
                'hpslp_makespan': ['mean', 'std']
            }).reset_index()
            
            grouped.columns = ['nodes', 'hpsrs_makespan_mean', 'hpsrs_makespan_std', 
                              'hpslp_makespan_mean', 'hpslp_makespan_std']

            # Plot H-PSRS
            series_name_hpsrs = f'H-PSRS ({dist})'
            ax_makespan.plot(grouped['nodes'], grouped['hpsrs_makespan_mean'], 
                           marker=markers[series_name_hpsrs], linestyle='-', 
                           label=series_name_hpsrs, color=colors[series_name_hpsrs], 
                           linewidth=2.5, markersize=8)
            
            # Add confidence interval
            ax_makespan.fill_between(grouped['nodes'], 
                                   grouped['hpsrs_makespan_mean'] - grouped['hpsrs_makespan_std'],
                                   grouped['hpsrs_makespan_mean'] + grouped['hpsrs_makespan_std'], 
                                   color=colors[series_name_hpsrs], alpha=0.2)
            
            # Plot H-PSLP
            series_name_hpslp = f'H-PSLP ({dist})'
            ax_makespan.plot(grouped['nodes'], grouped['hpslp_makespan_mean'], 
                           marker=markers[series_name_hpslp], linestyle='-', 
                           label=series_name_hpslp, color=colors[series_name_hpslp], 
                           linewidth=2.5, markersize=8)
            
            ax_makespan.fill_between(grouped['nodes'], 
                                   grouped['hpslp_makespan_mean'] - grouped['hpslp_makespan_std'],
                                   grouped['hpslp_makespan_mean'] + grouped['hpslp_makespan_std'], 
                                   color=colors[series_name_hpslp], alpha=0.2)

        ax_makespan.set_title(f'Dataset Size: {size:,} Elements', pad=15)
        ax_makespan.set_ylabel('Makespan (µs)' if j == 0 else '')
        ax_makespan.grid(True, alpha=0.3)
        ax_makespan.set_xticks(sorted(df['nodes'].unique()))
        
        # Cost Plot
        ax_cost = axes[1, j]
        for dist in ['uniform', 'gaussian']:
            subset = df_size[df_size['distribution'] == dist]
            if subset.empty:
                continue
            
            grouped = subset.groupby('nodes').agg({
                'hpsrs_cost': ['mean', 'std'],
                'hpslp_cost': ['mean', 'std']
            }).reset_index()
            
            grouped.columns = ['nodes', 'hpsrs_cost_mean', 'hpsrs_cost_std', 
                              'hpslp_cost_mean', 'hpslp_cost_std']

            # Plot H-PSRS
            series_name_hpsrs = f'H-PSRS ({dist})'
            ax_cost.plot(grouped['nodes'], grouped['hpsrs_cost_mean'], 
                        marker=markers[series_name_hpsrs], linestyle='-', 
                        label=series_name_hpsrs, color=colors[series_name_hpsrs], 
                        linewidth=2.5, markersize=8)
            
            ax_cost.fill_between(grouped['nodes'], 
                               grouped['hpsrs_cost_mean'] - grouped['hpsrs_cost_std'],
                               grouped['hpsrs_cost_mean'] + grouped['hpsrs_cost_std'], 
                               color=colors[series_name_hpsrs], alpha=0.2)
            
            # Plot H-PSLP
            series_name_hpslp = f'H-PSLP ({dist})'
            ax_cost.plot(grouped['nodes'], grouped['hpslp_cost_mean'], 
                        marker=markers[series_name_hpslp], linestyle='-', 
                        label=series_name_hpslp, color=colors[series_name_hpslp], 
                        linewidth=2.5, markersize=8)
            
            ax_cost.fill_between(grouped['nodes'], 
                               grouped['hpslp_cost_mean'] - grouped['hpslp_cost_std'],
                               grouped['hpslp_cost_mean'] + grouped['hpslp_cost_std'], 
                               color=colors[series_name_hpslp], alpha=0.2)
            
        ax_cost.set_xlabel('Number of Nodes')
        ax_cost.set_ylabel('Total Cost ($)' if j == 0 else '')
        ax_cost.grid(True, alpha=0.3)
        ax_cost.set_xticks(sorted(df['nodes'].unique()))

    # Format all axes
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.94), 
               frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"✓ Saved scalability chart: {filename}")
    plt.close(fig)

def plot_dataset_examples(example_datasets, filename='dataset_examples.png'):
    """Create beautiful dataset visualization with enhanced styling."""
    setup_plot_style()
    
    if not example_datasets:
        print("Warning: No example datasets provided. Skipping dataset examples plot.")
        return

    num_rows = len(example_datasets)
    if num_rows == 0:
        return
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Dataset Examples: Distribution Characteristics', 
                 fontsize=20, fontweight='bold', y=0.98)

    for i, (key, details) in enumerate(sorted(example_datasets.items())):
        if not details:
            continue
            
        dist_type, data_size_str = key.split('_')
        data = details.get('data', [])
        seed = details.get('seed', 'N/A')
        n = len(data)

        if n == 0:
            continue

        # Statistics text box
        mean_val = np.mean(data)
        std_val = np.std(data)
        stats_text = (f'{dist_type.upper()} Distribution\n'
                     f'Size: {n:,} elements\n'
                     f'Seed: {seed}\n'
                     f'Range: [{min(data):,}, {max(data):,}]\n'
                     f'Mean: {mean_val:,.1f}\n'
                     f'Std: {std_val:,.1f}')
        
        # Original data scatter plot
        axes[i, 0].scatter(range(n), data, s=2, alpha=0.6, color=COLORS['hpsrs'], 
                          edgecolors='none')
        axes[i, 0].set_title('Original Data Distribution', pad=15)
        axes[i, 0].set_xlabel('Index')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Add statistics box
        axes[i, 0].text(0.02, 0.98, stats_text, transform=axes[i, 0].transAxes, 
                       ha='left', va='top', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                alpha=0.9, edgecolor=COLORS['grid']))

        # Sorted data
        sorted_data = sorted(data)
        axes[i, 1].plot(range(n), sorted_data, color=COLORS['accent'], 
                       linewidth=2, alpha=0.8)
        axes[i, 1].set_title('Sorted Data Profile', pad=15)
        axes[i, 1].set_xlabel('Rank')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Histogram
        axes[i, 2].hist(data, bins=50, density=True, alpha=0.8, 
                       color=COLORS['hpslp'], edgecolor='white', linewidth=0.5)
        axes[i, 2].set_title('Value Distribution Histogram', pad=15)
        axes[i, 2].set_xlabel('Value')
        axes[i, 2].set_ylabel('Density')
        axes[i, 2].grid(True, alpha=0.3)
    
    # Format all y-axes
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"✓ Saved dataset examples chart: {filename}")
    plt.close(fig)

def generate_all_plots(results_file, example_datasets):
    """Main function to generate all enhanced visualizations."""
    print(f"\n🎨 Generating Enhanced Visualization Suite")
    print(f"📁 Input file: {results_file}")
    print(f"📊 Output directory: {OUTPUT_DIR}/")
    print("=" * 70)
    
    try:
        df = pd.read_json(results_file)
        print(f"✓ Successfully loaded {len(df)} experimental results")
    except Exception as e:
        print(f"❌ Error reading results file '{results_file}': {e}")
        return

    if df.empty:
        print("❌ Results dataframe is empty. No plots generated.")
        return
        
    print(f"\n📊 Generating Performance Analysis Charts...")
    plot_aggregated_metrics(df)
    plot_lp_prediction_accuracy(df)

    print(f"\n📈 Generating Scalability Analysis...")
    plot_scalability(df)
    
    print(f"\n📋 Generating Dataset Visualizations...")
    plot_dataset_examples(example_datasets)
    
    print(f"\n🎉 Visualization suite completed successfully!")
    print(f"   📁 All charts saved in: {OUTPUT_DIR}/")
    print(f"   📈 Generated {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])} charts")
    print("=" * 70)

if __name__ == '__main__':
    # Main execution
    generate_all_plots('results.json', example_datasets={})