import matplotlib.pyplot as plt
import random
import numpy as np
import os

# Create an 'out' directory for plots if it doesn't exist
os.makedirs('out', exist_ok=True)

def generate_cluster_config(
    data_size,  # The fixed total dataset size
    num_nodes=3,
    memory_overprovision_factor=1.5,
    throughput_range=(1, 10),
    billing_range=(0.1, 1),
    seed=None
):
    """
    Generates a cluster configuration with a *fixed* total data size.
    """
    if seed is not None:
        random.seed(seed)

    # Generate random throughput and billing for each node
    throughput = [random.randint(*throughput_range) for _ in range(num_nodes)]
    billing_rates = [round(random.uniform(*billing_range), 4) for _ in range(num_nodes)]

    # Use random fractions that sum to 1 for memory assignment
    raw_mem_fractions = [random.random() for _ in range(num_nodes)]
    total_fraction = sum(raw_mem_fractions) if sum(raw_mem_fractions) > 0 else 1
    memory_fractions = [r / total_fraction for r in raw_mem_fractions]

    # Calculate overprovisioned total memory based on the fixed data_size
    total_memory = int(data_size * memory_overprovision_factor)

    # Assign memory, throughput, and billing to each node
    cluster_config = {}
    cumulative_mem = 0
    for i in range(num_nodes):
        node = f"node{i+1}"
        if i < num_nodes - 1:
            mem = int(memory_fractions[i] * total_memory)
            cumulative_mem += mem
        else:
            mem = total_memory - cumulative_mem

        cluster_config[node] = {
            "throughput": throughput[i],
            "memory": mem,
            "billing": billing_rates[i]
        }
    return cluster_config

def generate_data(n, skew_type='uniform', zipf_param=2.0, mean=None, std=None, filename=''):
    """Generates data with a specified distribution."""
    if mean is None:
        mean = n / 2
    if std is None:
        std = n / 6

    data = []
    if skew_type == 'uniform':
        data = [random.randint(0, n) for _ in range(n)]
    elif skew_type == 'gaussian':
        raw = np.random.normal(loc=mean, scale=std, size=n)
        capped = np.clip(raw, 0, n)
        data = capped.astype(int).tolist()
    else:
        raise ValueError(f"Unsupported skew type: {skew_type}")

    # Plot distribution for this specific dataset
    plot_data_distribution(data, n, skew_type, filename=f'{filename}.png')
    return data

def plot_data_distribution(data, n, skew_type, filename='data.png'):
    """Plots scatter, sorted, and histogram views of a single dataset."""
    if not data:
        print(f"Warning: No data to plot for {filename}.")
        return

    plt.figure(figsize=(18, 6))

    # --- Scatter plot ---
    plt.subplot(1, 3, 1)
    plt.scatter(range(len(data)), data, s=1, alpha=0.5)
    plt.title(f'Scatter: {skew_type.capitalize()} Distribution')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, max(len(data), 1))
    plt.ylim(0, n)

    # --- Sorted scatter plot ---
    plt.subplot(1, 3, 2)
    plt.scatter(range(len(data)), sorted(data), s=1, alpha=0.5, color='orange')
    plt.title(f'Sorted Scatter: {skew_type.capitalize()}')
    plt.xlabel('Index')
    plt.ylabel('Sorted Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, max(len(data), 1))
    plt.ylim(0, n)

    # --- Histogram ---
    plt.subplot(1, 3, 3)
    plt.hist(data, bins=100, density=True, alpha=0.6, color='skyblue')
    plt.title(f'Histogram: {skew_type.capitalize()} Data')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f"Dataset Distribution Analysis (n={len(data)}, type={skew_type})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join('out', filename))
    plt.close()
    print(f"Saved dataset plot as 'out/{filename}'.")


# Example Usage to use generate_dataset output as input to plot_data_distribution:
if __name__ == '__main__':
    fixed_data_size = 10000
    print("--- Generating Cluster Config with Fixed Size ---")
    cluster_config = generate_cluster_config(
        data_size=fixed_data_size,
        num_nodes=4,
        seed=42
    )

    print("\n--- Generating Gaussian Dataset ---")
    gaussian_data = generate_data(
        n=fixed_data_size,
        skew_type='gaussian',
        filename='example_gaussian_dist'
    )
