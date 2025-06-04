import matplotlib.pyplot as plt
import numpy as np
import os

# Create an 'out' directory if it doesn't exist
output_dir = 'out_matplotlib_charts'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Data based on aggregated results from the research paper ---
# (Avg of Runs 1 & 2 for 4-node uniform, Avg of Runs 5 & 6 for 4-node Gaussian)
# (Avg of Runs 3 & 4 for 8-node uniform, Avg of Runs 7 & 8 for 8-node Gaussian)

# Makespan Data (µs)
makespan_data = {
    'Uniform': {
        '4 Nodes': {'H-PSRS': 36345.8, 'H-PSLP': 25213.5},
        '8 Nodes': {'H-PSRS': 359099.1, 'H-PSLP': 212877.4}
    },
    'Gaussian': {
        '4 Nodes': {'H-PSRS': 38199.5, 'H-PSLP': 26066.8},
        '8 Nodes': {'H-PSRS': 250614.6, 'H-PSLP': 229910.6} # Avg of Runs 7 & 8
    }
}

# Cost Data ($)
cost_data = {
    'Uniform': {
        '4 Nodes': {'H-PSRS': 43359.34, 'H-PSLP': 21172.96},
        '8 Nodes': {'H-PSRS': 561310.60, 'H-PSLP': 197406.66}
    },
    'Gaussian': {
        '4 Nodes': {'H-PSRS': 43856.09, 'H-PSLP': 19433.99},
        '8 Nodes': {'H-PSRS': 649911.91, 'H-PSLP': 250626.07} # Avg of Runs 7 & 8
    }
}

# Memory Utilization Example Data (Run 1, Node 3)
memory_limit_node3_run1 = 19744
hpsrs_items_node3_run1 = 84000
hpslp_items_node3_run1 = 19732

# --- Plotting Functions ---

def plot_bar_comparison(data_dict, title, ylabel, filename):
    """Plots a bar chart comparing H-PSRS and H-PSLP."""
    labels = list(data_dict.keys()) # Should be H-PSRS, H-PSLP
    values = list(data_dict.values())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    # Using distinct colors for H-PSRS (e.g., orange) and H-PSLP (e.g., blue)
    rects = ax.bar(x, values, width, label='Values', color=['#ff7f0e', '#1f77b4'])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Corrected legend to match colors
    ax.legend(['H-PSRS', 'H-PSLP'])


    def autolabel(rects_to_label):
        for rect in rects_to_label:
            height = rect.get_height()
            ax.annotate(f'{height:,.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved chart: {filename}")
    plt.close(fig)

def plot_memory_utilization(limit, hpsrs_val, hpslp_val, title, filename):
    """Plots memory utilization for a specific node."""
    labels = ['H-PSRS Items', 'H-PSLP Items']
    values = [hpsrs_val, hpslp_val]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, values, width, color=['#ff7f0e', '#1f77b4'])
    ax.axhline(limit, color='red', linestyle='--', label=f'Memory Limit ({limit:,.0f})')

    ax.set_ylabel('Number of Items')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects_to_label):
        for rect in rects_to_label:
            height = rect.get_height()
            ax.annotate(f'{height:,.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel(bars)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved chart: {filename}")
    plt.close(fig)

def plot_line_scaling(data_hpsrs, data_hpslp, nodes, title, ylabel, filename):
    """Plots a line chart showing scaling with node count."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(nodes, data_hpsrs, marker='o', linestyle='-', color='#ff7f0e', label='H-PSRS')
    ax.plot(nodes, data_hpslp, marker='s', linestyle='-', color='#1f77b4', label='H-PSLP')

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(nodes) # Ensure ticks are at 4 and 8
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))) # Format y-axis with commas
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

    # Annotate points
    for i, txt in enumerate(data_hpsrs):
        ax.annotate(f'{txt:,.0f}', (nodes[i], data_hpsrs[i]), textcoords="offset points", xytext=(0,5), ha='center', color='#ff7f0e', fontsize=8)
    for i, txt in enumerate(data_hpslp):
        ax.annotate(f'{txt:,.0f}', (nodes[i], data_hpslp[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='#1f77b4', fontsize=8)


    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved chart: {filename}")
    plt.close(fig)

# --- Generate Bar Charts ---
# Makespan - Uniform, 4 Nodes
plot_bar_comparison(makespan_data['Uniform']['4 Nodes'],
                    'Makespan Comparison - Uniform Data, 4 Nodes (Average)',
                    'Makespan (µs)', 'perf_makespan_uniform_4nodes.png')

# Cost - Uniform, 4 Nodes
plot_bar_comparison(cost_data['Uniform']['4 Nodes'],
                    'Cost Comparison - Uniform Data, 4 Nodes (Average)',
                    'Total Cost ($)', 'perf_cost_uniform_4nodes.png')

# Makespan - Gaussian, 4 Nodes
plot_bar_comparison(makespan_data['Gaussian']['4 Nodes'],
                    'Makespan Comparison - Gaussian Data, 4 Nodes (Average)',
                    'Makespan (µs)', 'perf_makespan_gaussian_4nodes.png')

# Cost - Gaussian, 4 Nodes
plot_bar_comparison(cost_data['Gaussian']['4 Nodes'],
                    'Cost Comparison - Gaussian Data, 4 Nodes (Average)',
                    'Total Cost ($)', 'perf_cost_gaussian_4nodes.png')

# Memory Utilization Example
plot_memory_utilization(memory_limit_node3_run1, hpsrs_items_node3_run1, hpslp_items_node3_run1,
                        'Memory Utilization Example - Run 1 (Uniform, 4N, Seed 4), Node 3',
                        'perf_memory_util_example.png')

# --- Generate Line Charts ---
node_counts = [4, 8]

# Makespan vs. Node Count - Uniform
hpsrs_makespan_uniform = [makespan_data['Uniform']['4 Nodes']['H-PSRS'], makespan_data['Uniform']['8 Nodes']['H-PSRS']]
hpslp_makespan_uniform = [makespan_data['Uniform']['4 Nodes']['H-PSLP'], makespan_data['Uniform']['8 Nodes']['H-PSLP']]
plot_line_scaling(hpsrs_makespan_uniform, hpslp_makespan_uniform, node_counts,
                  'Makespan vs. Node Count - Uniform Data', 'Makespan (µs)',
                  'scale_makespan_uniform_nodes.png')

# Cost vs. Node Count - Uniform
hpsrs_cost_uniform = [cost_data['Uniform']['4 Nodes']['H-PSRS'], cost_data['Uniform']['8 Nodes']['H-PSRS']]
hpslp_cost_uniform = [cost_data['Uniform']['4 Nodes']['H-PSLP'], cost_data['Uniform']['8 Nodes']['H-PSLP']]
plot_line_scaling(hpsrs_cost_uniform, hpslp_cost_uniform, node_counts,
                  'Cost vs. Node Count - Uniform Data', 'Total Cost ($)',
                  'scale_cost_uniform_nodes.png')

# Makespan vs. Node Count - Gaussian
hpsrs_makespan_gaussian = [makespan_data['Gaussian']['4 Nodes']['H-PSRS'], makespan_data['Gaussian']['8 Nodes']['H-PSRS']]
hpslp_makespan_gaussian = [makespan_data['Gaussian']['4 Nodes']['H-PSLP'], makespan_data['Gaussian']['8 Nodes']['H-PSLP']]
plot_line_scaling(hpsrs_makespan_gaussian, hpslp_makespan_gaussian, node_counts,
                  'Makespan vs. Node Count - Gaussian Data', 'Makespan (µs)',
                  'scale_makespan_gaussian_nodes.png')

# Cost vs. Node Count - Gaussian
hpsrs_cost_gaussian = [cost_data['Gaussian']['4 Nodes']['H-PSRS'], cost_data['Gaussian']['8 Nodes']['H-PSRS']]
hpslp_cost_gaussian = [cost_data['Gaussian']['4 Nodes']['H-PSLP'], cost_data['Gaussian']['8 Nodes']['H-PSLP']]
plot_line_scaling(hpsrs_cost_gaussian, hpslp_cost_gaussian, node_counts,
                  'Cost vs. Node Count - Gaussian Data', 'Total Cost ($)',
                  'scale_cost_gaussian_nodes.png')

print(f"\nAll Matplotlib charts saved to '{output_dir}' directory.")

