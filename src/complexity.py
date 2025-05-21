import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
import matplotlib.gridspec as gridspec
import matplotlib as mpl

# Define the Catppuccin color palette
catppuccin = {
    'rosewater': '#f5e0dc',
    'flamingo': '#f2cdcd',
    'pink': '#f5c2e7',
    'mauve': '#cba6f7',
    'red': '#f38ba8',
    'maroon': '#eba0ac',
    'peach': '#fab387',
    'yellow': '#f9e2af',
    'green': '#a6e3a1',
    'teal': '#94e2d5',
    'sky': '#89dceb',
    'sapphire': '#74c7ec',
    'blue': '#89b4fa',
    'lavender': '#b4befe',
    'text': '#cdd6f4',
    'subtext1': '#bac2de',
    'subtext0': '#a6adc8',
    'overlay2': '#9399b2',
    'overlay1': '#7f849c',
    'overlay0': '#6c7086',
    'surface2': '#585b70',
    'surface1': '#45475a',
    'surface0': '#313244',
    'base': '#1e1e2e',
    'mantle': '#181825',
    'crust': '#11111b'
}

# Apply Catppuccin theme to matplotlib
plt.style.use('dark_background')
mpl.rcParams['figure.facecolor'] = catppuccin['base']
mpl.rcParams['axes.facecolor'] = catppuccin['base']
mpl.rcParams['text.color'] = catppuccin['text']
mpl.rcParams['axes.labelcolor'] = catppuccin['text']
mpl.rcParams['xtick.color'] = catppuccin['subtext1']
mpl.rcParams['ytick.color'] = catppuccin['subtext1']
mpl.rcParams['axes.edgecolor'] = catppuccin['surface0']
mpl.rcParams['grid.color'] = catppuccin['surface0']

# Function to calculate average-case time complexity of Heterogeneous PSRS (Baseline)
def time_complexity_baseline_avg(n, p, c_sort=1, c_comm=0.1, c_pivot=0.01):
    # c_sort: constant factor for sorting operations
    # c_comm: constant factor for communication operations
    # c_pivot: constant factor for pivot selection operations
    
    # Local sorting: O((n/p) log(n/p))
    local_sorting = c_sort * (n/p) * np.log2(n/p)
    
    # Pivot selection: O(p^2 log(p))
    pivot_selection = c_pivot * (p**2) * np.log2(p)
    
    # Data redistribution: O(n)
    redistribution = c_comm * n
    
    # Final sorting: O((n/p) log(n/p))
    final_sorting = c_sort * (n/p) * np.log2(n/p)
    
    return local_sorting + pivot_selection + redistribution + final_sorting

# Function to calculate worst-case time complexity of Heterogeneous PSRS (Baseline)
def time_complexity_baseline_worst(n, p, c_sort=1, c_comm=0.1, c_pivot=0.01):
    # Worst case for quicksort: O(n²) instead of O(n log n)
    # Poor pivot selection can lead to imbalanced partitions
    
    # Local sorting: O((n/p)²) in worst case
    local_sorting = c_sort * (n/p)**2
    
    # Pivot selection: O(p^2 log(p))
    pivot_selection = c_pivot * (p**2) * np.log2(p)
    
    # Data redistribution: O(n)
    redistribution = c_comm * n
    
    # Final sorting: O((n/p)²) in worst case
    final_sorting = c_sort * (n/p)**2
    
    return local_sorting + pivot_selection + redistribution + final_sorting

# Function to calculate average-case time complexity of LP-Guided Hybrid (Proposed, Option B)
def time_complexity_proposed_avg(n, p, c_lp=10, c_sort=1, c_comm=0.1, c_minmax=0.05):
    # c_lp: constant factor for LP solving
    # c_sort: constant factor for sorting operations
    # c_comm: constant factor for communication operations
    # c_minmax: constant factor for min-max finding operations
    
    # Static partitioning (LP solving): O(p^3)
    # static_partitioning = c_lp * (p**3)
    
    # Block distribution: O(n)
    block_distribution = c_comm * n
    
    # Local sorting: O((n/p) log(n/p))
    local_sorting = c_sort * (n/p) * np.log2(n/p)
    
    # Coordinated Min-Max Pop: O(n * p)
    coordinated_minmax = c_minmax * n * p
    
    return block_distribution + local_sorting + coordinated_minmax


def time_complexity_proposed_worst(n, p, c_lp=10, c_sort=1, c_comm=0.1, c_minmax=0.05):
    # c_lp: constant factor for LP solving
    # c_sort: constant factor for sorting operations
    # c_comm: constant factor for communication operations
    # c_minmax: constant factor for min-max finding operations
    
    # Static partitioning (LP solving): O(p^3)
    static_partitioning = c_lp * (p**3)
    
    # Block distribution: O(n)
    block_distribution = c_comm * n
    
    # Local sorting: O((n/p) log(n/p))
    local_sorting = c_sort * (n/p) * np.log2(n/p)
    
    # Coordinated Min-Max Pop: O(n * p)
    coordinated_minmax = c_minmax * n * p
    
    return static_partitioning + block_distribution + local_sorting + coordinated_minmax

# Main visualization function
def visualize_complexity():
    # Set up the figure with better proportions for sliders
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(bottom=0.15)  # Add margin at the bottom
    
    # Main plot area
    ax_plot = plt.subplot(gs[0])
    
    # Initialize parameters
    n_initial = 10000
    p_initial = 4
    c_sort_initial = 1.0
    c_comm_initial = 0.1
    c_pivot_initial = 0.01
    c_lp_initial = 10.0
    c_minmax_initial = 0.05
    
    # Generate initial data
    n_values = np.linspace(1000, n_initial*2, 100)
    
    baseline_avg_times = [time_complexity_baseline_avg(n, p_initial, c_sort_initial, c_comm_initial, c_pivot_initial) for n in n_values]
    baseline_worst_times = [time_complexity_baseline_worst(n, p_initial, c_sort_initial, c_comm_initial, c_pivot_initial) for n in n_values]
    proposed_avg_times = [time_complexity_proposed_avg(n, p_initial, c_lp_initial, c_sort_initial, c_comm_initial, c_minmax_initial) for n in n_values]
    proposed_worst_times = [time_complexity_proposed_worst(n, p_initial, c_lp_initial, c_sort_initial, c_comm_initial, c_minmax_initial) for n in n_values]
    
    # Create plot with Catppuccin colors
    line_baseline_avg, = ax_plot.plot(n_values, baseline_avg_times, color=catppuccin['blue'], linewidth=2.5, label='Baseline (Avg Case)')
    line_baseline_worst, = ax_plot.plot(n_values, baseline_worst_times, color=catppuccin['sapphire'], linewidth=2, linestyle='--', label='Baseline (Worst Case)')
    line_proposed_avg, = ax_plot.plot(n_values, proposed_avg_times, color=catppuccin['peach'], linewidth=2.5, label='Proposed (Avg Case)')
    line_proposed_worst, = ax_plot.plot(n_values, proposed_worst_times, color=catppuccin['red'], linewidth=2, linestyle='--', label='Proposed (Worst Case)')
    
    ax_plot.set_xlabel('Input Size (n)', fontsize=12)
    ax_plot.set_ylabel('Time (arbitrary units)', fontsize=12)
    ax_plot.set_title('Algorithm Complexity Comparison', fontsize=16, color=catppuccin['mauve'])
    ax_plot.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Create a styled legend
    legend = ax_plot.legend(loc='upper left', facecolor=catppuccin['surface0'], edgecolor=catppuccin['surface2'])
    for text in legend.get_texts():
        text.set_color(catppuccin['text'])
    
    # Create a frame for sliders to organize them visually
    slider_frame = plt.subplot(gs[1])
    slider_frame.axis('off')  # Hide the axes
    
    # Organize sliders in a more structured way
    # Primary parameters (top row)
    ax_n = plt.axes([0.15, 0.25, 0.75, 0.03], facecolor=catppuccin['surface1'])
    ax_p = plt.axes([0.15, 0.20, 0.75, 0.03], facecolor=catppuccin['surface1'])
    
    # Algorithm constants (organized in two rows)
    # Baseline algorithm constants (left side)
    ax_c_sort = plt.axes([0.15, 0.13, 0.30, 0.03], facecolor=catppuccin['blue'], alpha=0.3)
    ax_c_comm = plt.axes([0.15, 0.08, 0.30, 0.03], facecolor=catppuccin['blue'], alpha=0.3)
    ax_c_pivot = plt.axes([0.15, 0.03, 0.30, 0.03], facecolor=catppuccin['blue'], alpha=0.3)
    
    # Proposed algorithm constants (right side)
    ax_c_lp = plt.axes([0.60, 0.13, 0.30, 0.03], facecolor=catppuccin['peach'], alpha=0.3)
    ax_c_minmax = plt.axes([0.60, 0.08, 0.30, 0.03], facecolor=catppuccin['peach'], alpha=0.3)
    
    # Controls (right side, bottom)
    ax_log_scale = plt.axes([0.60, 0.02, 0.10, 0.05], facecolor=catppuccin['surface0'])
    ax_view_toggle = plt.axes([0.71, 0.02, 0.10, 0.05], facecolor=catppuccin['surface0'])
    
    check_log = CheckButtons(ax_log_scale, ['Log Scale'], [False])
    # Style the check box
    for rectangle in ax_log_scale.patches:
        rectangle.set_facecolor(catppuccin['surface1'])
        rectangle.set_edgecolor(catppuccin['surface2'])
    # Style the check mark
    for line in ax_log_scale.lines:
        line.set_color(catppuccin['green'])
        
    # Replace the radio button styling with this:
    radio_view = RadioButtons(ax_view_toggle, ['n', 'p'], activecolor=catppuccin['green'])

    # Style the radio buttons
    for circle in radio_view.ax.patches:
        circle.set_facecolor(catppuccin['surface1'])
        circle.set_edgecolor(catppuccin['surface2'])
        
    # Create sliders with consistent styling
    slider_n = Slider(ax_n, 'Input Size (n)', 1000, 100000, valinit=n_initial, valstep=1000, 
                      color=catppuccin['teal'])
    slider_p = Slider(ax_p, 'Number of Processors (p)', 1, 32, valinit=p_initial, valstep=1, 
                      color=catppuccin['teal'])
    
    # Baseline algorithm sliders
    slider_c_sort = Slider(ax_c_sort, 'Sorting Cost', 0.1, 5.0, valinit=c_sort_initial, valstep=0.1, 
                          color=catppuccin['blue'])
    slider_c_comm = Slider(ax_c_comm, 'Communication Cost', 0.01, 1.0, valinit=c_comm_initial, valstep=0.01, 
                          color=catppuccin['blue'])
    slider_c_pivot = Slider(ax_c_pivot, 'Pivot Cost', 0.001, 1, valinit=c_pivot_initial, valstep=0.001, 
                           color=catppuccin['blue'])
    
    # Proposed algorithm sliders
    slider_c_lp = Slider(ax_c_lp, 'LP Solver Cost', 1.0, 50.0, valinit=c_lp_initial, valstep=1.0, 
                        color=catppuccin['peach'])
    slider_c_minmax = Slider(ax_c_minmax, 'Min-Max Cost', 0.01, 0.5, valinit=c_minmax_initial, valstep=0.01, 
                            color=catppuccin['peach'])
    
    # Labels for slider groups
    # plt.figtext(0.15, 0.31, "Main Parameters:", color=catppuccin['mauve'], fontsize=12)
    plt.figtext(0.15, 0.17, "Baseline Algorithm Constants:", color=catppuccin['blue'], fontsize=12)
    plt.figtext(0.60, 0.17, "Proposed Algorithm Constants:", color=catppuccin['peach'], fontsize=12)
    
    # Update function for sliders
    def update(val):
        # Get current values from sliders
        n_max = slider_n.val
        p = slider_p.val
        c_sort = slider_c_sort.val
        c_comm = slider_c_comm.val
        c_pivot = slider_c_pivot.val
        c_lp = slider_c_lp.val
        c_minmax = slider_c_minmax.val
        
        # Check if we're viewing n or p
        if radio_view.value_selected == 'n':
            # Update n_values based on slider
            n_values = np.linspace(1000, n_max*2, 100)
            
            # Calculate new times
            baseline_avg_times = [time_complexity_baseline_avg(n, p, c_sort, c_comm, c_pivot) for n in n_values]
            baseline_worst_times = [time_complexity_baseline_worst(n, p, c_sort, c_comm, c_pivot) for n in n_values]
            proposed_avg_times = [time_complexity_proposed_avg(n, p, c_lp, c_sort, c_comm, c_minmax) for n in n_values]
            proposed_worst_times = [time_complexity_proposed_worst(n, p, c_lp, c_sort, c_comm, c_minmax) for n in n_values]
            
            # Update lines
            line_baseline_avg.set_data(n_values, baseline_avg_times)
            line_baseline_worst.set_data(n_values, baseline_worst_times)
            line_proposed_avg.set_data(n_values, proposed_avg_times)
            line_proposed_worst.set_data(n_values, proposed_worst_times)
            
            ax_plot.set_xlabel('Input Size (n)', fontsize=12)
            
        else:  # 'p' view
            # Generate p values based on slider
            p_values = np.linspace(1, p*2, 100)
            
            # Calculate times with fixed n and varying p
            baseline_avg_times = [time_complexity_baseline_avg(n_max, p_val, c_sort, c_comm, c_pivot) for p_val in p_values]
            baseline_worst_times = [time_complexity_baseline_worst(n_max, p_val, c_sort, c_comm, c_pivot) for p_val in p_values]
            proposed_avg_times = [time_complexity_proposed_avg(n_max, p_val, c_lp, c_sort, c_comm, c_minmax) for p_val in p_values]
            proposed_worst_times = [time_complexity_proposed_worst(n_max, p_val, c_lp, c_sort, c_comm, c_minmax) for p_val in p_values]
            
            # Update lines
            line_baseline_avg.set_data(p_values, baseline_avg_times)
            line_baseline_worst.set_data(p_values, baseline_worst_times)
            line_proposed_avg.set_data(p_values, proposed_avg_times)
            line_proposed_worst.set_data(p_values, proposed_worst_times)
            
            ax_plot.set_xlabel('Number of Processors (p)', fontsize=12)
        
        # Adjust y-axis limits
        all_times = baseline_avg_times + baseline_worst_times + proposed_avg_times + proposed_worst_times
        max_y = max(all_times)
        min_y = min(all_times)
        
        if ax_plot.get_yscale() == 'linear':
            ax_plot.set_ylim(0, max_y * 1.1)
        else:
            ax_plot.set_ylim(max(min_y * 0.9, 0.1), max_y * 1.1)
        
        # Adjust x-axis limits
        x_values = n_values if radio_view.value_selected == 'n' else p_values
        ax_plot.set_xlim(min(x_values), max(x_values))
        
        # # Detect and display crossover points if they exist
        # if radio_view.value_selected == 'n':
        #     # Find where baseline avg and proposed avg cross
        #     diff = np.array(baseline_avg_times) - np.array(proposed_avg_times)
        #     cross_points = np.where(np.diff(np.signbit(diff)))[0]
            
        #     # Remove any previous crossover markers
        #     for line in ax_plot.lines:
        #         if line not in [line_baseline_avg, line_baseline_worst, line_proposed_avg, line_proposed_worst]:
        #             line.remove()
                    
        #     if len(cross_points) > 0:
        #         cross_idx = cross_points[0]
        #         cross_x = n_values[cross_idx]
        #         cross_y = baseline_avg_times[cross_idx]
                
        #         # Add crossover point marker
        #         ax_plot.plot([cross_x], [cross_y], 'o', color=catppuccin['green'], markersize=8)
        #         ax_plot.axvline(x=cross_x, color=catppuccin['green'], linestyle=':', alpha=0.5)
                
        #         # Add annotation
        #         ax_plot.annotate(f'Crossover: n ≈ {int(cross_x)}',
        #                         xy=(cross_x, cross_y),
        #                         xytext=(10, 10),
        #                         textcoords='offset points',
        #                         color=catppuccin['green'],
        #                         backgroundcolor=catppuccin['surface0'],
        #                         bbox=dict(boxstyle="round,pad=0.3", facecolor=catppuccin['surface0'], 
        #                                  edgecolor=catppuccin['surface2']))
        
        # Refresh the plot
        fig.canvas.draw_idle()
    
    # Function to toggle log scale
    def toggle_log_scale(label):
        if label == 'Log Scale':
            if ax_plot.get_yscale() == 'linear':
                ax_plot.set_yscale('log')
            else:
                ax_plot.set_yscale('linear')
            update(None)  # Re-adjust the y-limits
    
    # Function to toggle between n and p views
    def toggle_view(label):
        update(None)
    
    # Register the update function with each slider
    slider_n.on_changed(update)
    slider_p.on_changed(update)
    slider_c_sort.on_changed(update)
    slider_c_comm.on_changed(update)
    slider_c_pivot.on_changed(update)
    slider_c_lp.on_changed(update)
    slider_c_minmax.on_changed(update)
    
    # Register toggle functions
    check_log.on_clicked(toggle_log_scale)
    radio_view.on_clicked(toggle_view)
    
    # Add complexity equations with proper LaTeX formatting
    props = dict(boxstyle='round', facecolor=catppuccin['surface0'], alpha=0.8, 
                edgecolor=catppuccin['surface2'])
    
    complexity_text = (
        r"$\mathbf{Baseline\;Algorithm:}$" + "\n" +
        r"$\bullet\;Average\;Case:\;O\left(\frac{n}{p}\log\left(\frac{n}{p}\right) + p^2\log(p) + n\right)$" + "\n" +
        r"$\bullet\;Worst\;Case:\;O\left(\frac{n^2}{p} + p^2\log(p) + n\right)$" + "\n\n" +
        r"$\mathbf{LP-Guided\;Hybrid\;(Option\;B):}$" + "\n" +
        r"$\bullet\;Average\;Case:\;O\left(\frac{n}{p}\log\left(\frac{n}{p}\right) + n \cdot p\right)$" + "\n" +
        r"$\bullet\;Worst\;Case:\;O\left(p^3 + \frac{n}{p}\log\left(\frac{n}{p}\right) + n \cdot p\right)$"
    )
    
    # Position text box in the upper right corner of the plot
    ax_plot.text(0.97, 0.97, complexity_text, transform=ax_plot.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props, color=catppuccin['text'])
    
    # Initial update
    update(None)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Run the visualization
visualize_complexity()