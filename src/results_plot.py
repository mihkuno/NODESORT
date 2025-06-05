import matplotlib.pyplot as plt
import numpy as np
import os

# Create an 'out' directory if it doesn't exist
output_dir = 'out_matplotlib_charts'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Data based on aggregated results & individual runs from the research paper ---

# Makespan Data (µs) - Averages for 4-node, individual for 8-node to combine later
avg_hpsrs_makespan_gaussian_8nodes = (446139.2 + 178432.2) / 2
avg_hpslp_makespan_gaussian_8nodes = (319017.1 + 124284.2) / 2

makespan_data = {
    'Uniform': {
        '4 Nodes': {'H-PSRS': 36345.8, 'H-PSLP (Heap)': 25213.5},
        '8 Nodes': {'H-PSRS': 359099.1, 'H-PSLP (Heap)': 212877.4}
    },
    'Gaussian': {
        '4 Nodes': {'H-PSRS': 38199.5, 'H-PSLP (Heap)': 26066.8},
        '8 Nodes': {'H-PSRS': avg_hpsrs_makespan_gaussian_8nodes, 'H-PSLP (Heap)': avg_hpslp_makespan_gaussian_8nodes}
    }
}

# Cost Data ($) - Averages for 4-node, individual for 8-node to combine later
avg_hpsrs_cost_gaussian_8nodes = (839765.87 + 428716.62) / 2
avg_hpslp_cost_gaussian_8nodes = (292759.19 + 154602.50) / 2

cost_data = {
    'Uniform': {
        '4 Nodes': {'H-PSRS': 43359.34, 'H-PSLP (Heap)': 21172.96},
        '8 Nodes': {'H-PSRS': 561310.60, 'H-PSLP (Heap)': 197406.66}
    },
    'Gaussian': {
        '4 Nodes': {'H-PSRS': 43856.09, 'H-PSLP (Heap)': 19433.99},
        '8 Nodes': {'H-PSRS': avg_hpsrs_cost_gaussian_8nodes, 'H-PSLP (Heap)': avg_hpslp_cost_gaussian_8nodes}
    }
}

# Memory Utilization Example Data (Run 1, Node 3)
memory_limit_node3_run1 = 19744
hpsrs_items_node3_run1 = 84000
hpslp_items_node3_run1 = 19732

# Predicted vs. Actual Data for H-PSLP Sort Phase (from individual run tables)
# (run_label, lp_pred_makespan, actual_max_local_sort_time, lp_pred_cost, actual_sum_local_sort_costs)
predicted_vs_actual_data = [
    # Run 1: Uniform, 4N, S4
    {'run_label': 'R1 U4S4', 'pred_makespan': 14044.3, 'actual_makespan': 16422.7, 'pred_cost': 17655.42, 'actual_cost': 21189.53},
    # Run 2: Uniform, 4N, S5
    {'run_label': 'R2 U4S5', 'pred_makespan': 9145.8, 'actual_makespan': 9790.4, 'pred_cost': 9921.22, 'actual_cost': 10049.08},
    # Run 3: Uniform, 8N, S4
    {'run_label': 'R3 U8S4', 'pred_makespan': 125524.1, 'actual_makespan': 122323.7, 'pred_cost': 303521.59, 'actual_cost': 274580.60},
    # Run 4: Uniform, 8N, S5
    {'run_label': 'R4 U8S5', 'pred_makespan': 47603.4, 'actual_makespan': 43836.4, 'pred_cost': 136047.90, 'actual_cost': 87424.32},
    # Run 5: Gaussian, 4N, S4
    {'run_label': 'R5 G4S4', 'pred_makespan': 13854.7, 'actual_makespan': 18009.5, 'pred_cost': 17417.16, 'actual_cost': 21842.52},
    # Run 6: Gaussian, 4N, S5
    {'run_label': 'R6 G4S5', 'pred_makespan': 9556.7, 'actual_makespan': 11038.9, 'pred_cost': 10366.95, 'actual_cost': 12135.29},
    # Run 7: Gaussian, 8N, S4
    {'run_label': 'R7 G8S4', 'pred_makespan': 125524.1, 'actual_makespan': 122323.7, 'pred_cost': 303521.59, 'actual_cost': 274580.60},
    # Run 8: Gaussian, 8N, S5
    {'run_label': 'R8 G8S5', 'pred_makespan': 48478.5, 'actual_makespan': 44617.9, 'pred_cost': 138548.89, 'actual_cost': 126599.78},
]


# --- Plotting Functions ---

def plot_bar_comparison(data_dict, title, ylabel, filename):
    labels = list(data_dict.keys()) 
    values = list(data_dict.values())
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    rects = ax.bar(x, values, width, color=['#ff7f0e', '#1f77b4'])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(['H-PSRS', 'H-PSLP']) # Updated to reflect H-PSLP (not H-PSLP (Heap))
    def autolabel(rects_to_label):
        for rect in rects_to_label:
            height = rect.get_height()
            ax.annotate(f'{height:,.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    autolabel(rects)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved chart: {filename}")
    plt.close(fig)

def plot_memory_utilization(limit, hpsrs_val, hpslp_val, title, filename):
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
            ax.annotate(f'{height:,.0f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    autolabel(bars)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved chart: {filename}")
    plt.close(fig)

def plot_line_scaling(data_hpsrs, data_hpslp, nodes, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(nodes, data_hpsrs, marker='o', linestyle='-', color='#ff7f0e', label='H-PSRS')
    ax.plot(nodes, data_hpslp, marker='s', linestyle='-', color='#1f77b4', label='H-PSLP') # Updated label
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(nodes)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    for i, txt in enumerate(data_hpsrs):
        ax.annotate(f'{txt:,.0f}', (nodes[i], data_hpsrs[i]), textcoords="offset points", xytext=(0,5), ha='center', color='#ff7f0e', fontsize=8)
    for i, txt in enumerate(data_hpslp):
        ax.annotate(f'{txt:,.0f}', (nodes[i], data_hpslp[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='#1f77b4', fontsize=8)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved chart: {filename}")
    plt.close(fig)

def plot_predicted_vs_actual_grouped_bar(runs_data, data_key_predicted, data_key_actual, title, ylabel, filename):
    """Plots a grouped bar chart for predicted vs. actual values for multiple runs."""
    run_labels = [r['run_label'] for r in runs_data]
    predicted_values = [r[data_key_predicted] for r in runs_data]
    actual_values = [r[data_key_actual] for r in runs_data]

    x = np.arange(len(run_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7)) # Wider for more runs
    rects1 = ax.bar(x - width/2, predicted_values, width, label='LP Predicted', color='#17becf') # Teal
    rects2 = ax.bar(x + width/2, actual_values, width, label='Actual Observed', color='#9467bd') # Purple

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=45, ha="right")
    ax.legend()
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda val, loc: "{:,}".format(int(val))))


    def autolabel_bars(rects_to_label, va_pos='bottom'):
        for rect in rects_to_label:
            height = rect.get_height()
            ax.annotate(f'{height:,.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 if va_pos=='bottom' else -3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va=va_pos, fontsize=7)

    autolabel_bars(rects1)
    autolabel_bars(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved chart: {filename}")
    plt.close(fig)

# --- Generate Bar Charts ---
plot_bar_comparison(makespan_data['Uniform']['4 Nodes'],
                    'Makespan Comparison - Uniform Data, 4 Nodes (Average)',
                    'Makespan (µs)', 'perf_makespan_uniform_4nodes.png')
plot_bar_comparison(cost_data['Uniform']['4 Nodes'],
                    'Cost Comparison - Uniform Data, 4 Nodes (Average)',
                    'Total Cost ($)', 'perf_cost_uniform_4nodes.png')
plot_bar_comparison(makespan_data['Gaussian']['4 Nodes'],
                    'Makespan Comparison - Gaussian Data, 4 Nodes (Average)',
                    'Makespan (µs)', 'perf_makespan_gaussian_4nodes.png')
plot_bar_comparison(cost_data['Gaussian']['4 Nodes'],
                    'Cost Comparison - Gaussian Data, 4 Nodes (Average)',
                    'Total Cost ($)', 'perf_cost_gaussian_4nodes.png')
plot_memory_utilization(memory_limit_node3_run1, hpsrs_items_node3_run1, hpslp_items_node3_run1,
                        'Memory Utilization Example - Run 1 (Uniform, 4N, Seed 4), Node 3',
                        'perf_memory_util_example.png')

# --- Generate Line Charts ---
node_counts = [4, 8]
hpsrs_makespan_uniform = [makespan_data['Uniform']['4 Nodes']['H-PSRS'], makespan_data['Uniform']['8 Nodes']['H-PSRS']]
hpslp_makespan_uniform = [makespan_data['Uniform']['4 Nodes']['H-PSLP (Heap)'], makespan_data['Uniform']['8 Nodes']['H-PSLP (Heap)']]
plot_line_scaling(hpsrs_makespan_uniform, hpslp_makespan_uniform, node_counts,
                  'Makespan vs. Node Count - Uniform Data', 'Makespan (µs)',
                  'scale_makespan_uniform_nodes.png')
hpsrs_cost_uniform = [cost_data['Uniform']['4 Nodes']['H-PSRS'], cost_data['Uniform']['8 Nodes']['H-PSRS']]
hpslp_cost_uniform = [cost_data['Uniform']['4 Nodes']['H-PSLP (Heap)'], cost_data['Uniform']['8 Nodes']['H-PSLP (Heap)']]
plot_line_scaling(hpsrs_cost_uniform, hpslp_cost_uniform, node_counts,
                  'Cost vs. Node Count - Uniform Data', 'Total Cost ($)',
                  'scale_cost_uniform_nodes.png')
hpsrs_makespan_gaussian = [makespan_data['Gaussian']['4 Nodes']['H-PSRS'], makespan_data['Gaussian']['8 Nodes']['H-PSRS']]
hpslp_makespan_gaussian = [makespan_data['Gaussian']['4 Nodes']['H-PSLP (Heap)'], makespan_data['Gaussian']['8 Nodes']['H-PSLP (Heap)']]
plot_line_scaling(hpsrs_makespan_gaussian, hpslp_makespan_gaussian, node_counts,
                  'Makespan vs. Node Count - Gaussian Data', 'Makespan (µs)',
                  'scale_makespan_gaussian_nodes.png')
hpsrs_cost_gaussian = [cost_data['Gaussian']['4 Nodes']['H-PSRS'], cost_data['Gaussian']['8 Nodes']['H-PSRS']]
hpslp_cost_gaussian = [cost_data['Gaussian']['4 Nodes']['H-PSLP (Heap)'], cost_data['Gaussian']['8 Nodes']['H-PSLP (Heap)']]
plot_line_scaling(hpsrs_cost_gaussian, hpslp_cost_gaussian, node_counts,
                  'Cost vs. Node Count - Gaussian Data', 'Total Cost ($)',
                  'scale_cost_gaussian_nodes.png')

# --- Generate Predicted vs. Actual Charts for H-PSLP Sort Phase ---
plot_predicted_vs_actual_grouped_bar(predicted_vs_actual_data, 'pred_makespan', 'actual_makespan',
                                     'H-PSLP: LP Predicted vs. Actual Sort Makespan (All Runs)',
                                     'Makespan (µs)', 'pred_vs_actual_makespan_hpslp.png')

plot_predicted_vs_actual_grouped_bar(predicted_vs_actual_data, 'pred_cost', 'actual_cost',
                                     'H-PSLP: LP Predicted vs. Actual Sort Cost (All Runs)',
                                     'Cost ($)', 'pred_vs_actual_cost_hpslp.png')

print(f"\nAll Matplotlib charts saved to '{output_dir}' directory.")

