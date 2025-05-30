import matplotlib.pyplot as plt
import random
import numpy as np
from math import gcd
from functools import reduce
import os

# Original functions provided by the user (not modified)
def lcm(a, b):
    return a * b // gcd(a, b)

def lcm_list(numbers):
    return reduce(lcm, numbers, 1)

def generate_cluster_config(
    data_scale=5,  # k: used to scale total dataset size
    num_nodes=3,
    memory_overprovision_factor=1.5, 
    throughput_range=(1, 10),
    billing_range=(0.1, 1),
    seed=None
):
    if seed is not None:
        random.seed(seed)
    
    # Step 1: Generate random throughput for each node
    throughput = [random.randint(*throughput_range) for _ in range(num_nodes)]

    # Step 2: Calculate total dataset size using LCM trick
    lcm_val = lcm_list(throughput)
    data_size = data_scale * sum(p * lcm_val for p in throughput)

    # Step 3: Random fractions that sum to 1 for memory assignment
    raw = [random.random() for _ in range(num_nodes)]
    total = sum(raw)
    memory_fractions = [r / total for r in raw]

    # Step 4: Calculate overprovisioned total memory
    total_memory = int(data_size * memory_overprovision_factor)

    # Step 5: Assign memory per node
    cluster_config = {}
    cumulative = 0

    for i in range(num_nodes):
        node = f"node{i+1}"
        if i < num_nodes - 1:
            mem = int(memory_fractions[i] * total_memory)
            cumulative += mem
        else:
            mem = total_memory - cumulative  # final node gets the remainder

        cluster_config[node] = {
            "throughput": throughput[i],
            "memory": mem,
            "billing": round(random.uniform(*billing_range), 4)
        }

    # Optional: Print summary
    print(f"Total memory of cluster = {total_memory} (should be >= {data_size})\n")

    total_pct_dataset = 0
    total_pct_cluster = 0

    for node, spec in cluster_config.items():
        mem = spec["memory"]
        pct_dataset = (mem / data_size) * 100
        pct_total_mem = (mem / total_memory) * 100

        total_pct_dataset += pct_dataset
        total_pct_cluster += pct_total_mem

        print(f"{node}: {spec}")
        print(f"  - {pct_dataset:.2f}% out of {memory_overprovision_factor * 100:.2f}% of overprovisioned memory")
        print(f"  - {pct_total_mem:.2f}% out of 100.00% of total cluster memory\n")

    return data_size, cluster_config

def generate_data(n, skew_type='uniform', zipf_param=2.0, mean=5000, std=2000):
    data = []
    if skew_type == 'uniform':
        data = [random.randint(0, n) for _ in range(n)]
    elif skew_type == 'zipf':
        raw = np.random.zipf(zipf_param, size=n)
        capped = np.clip(raw, 1, n)
        data = capped.tolist()
    elif skew_type == 'exponential':
        raw = np.random.exponential(scale=n/5, size=n)
        capped = np.clip(raw, 0, n)
        data = capped.astype(int).tolist()
    elif skew_type == 'gaussian':
        raw = np.random.normal(loc=mean, scale=std, size=n)
        capped = np.clip(raw, 0, n)
        data = capped.astype(int).tolist()
    else:
        raise ValueError(f"Unsupported skew type: {skew_type}")

    plot_data_distribution(data, n, skew_type, filename=f'{skew_type}_distribution.png')
    return data

# Re-using the plot_data_distribution function from previous turn
# Updated plotting function to support bell curve for any distribution type
def plot_data_distribution(data, n, skew_type, filename='data.png'):
    os.makedirs('out', exist_ok=True)

    indices = list(range(n))
    plt.figure(figsize=(18, 6))

    # Scatter plot (Index vs Value)
    plt.subplot(1, 3, 1)
    plt.scatter(indices, data, s=1, alpha=0.5)
    plt.title(f'Scatter: {skew_type.capitalize()} Distribution')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, n)
    plt.ylim(0, n)

    # Sorted scatter plot (Index vs Sorted Value)
    plt.subplot(1, 3, 2)
    sorted_data = sorted(data)
    plt.scatter(indices, sorted_data, s=1, alpha=0.5, color='orange')
    plt.title(f'Sorted Scatter: {skew_type.capitalize()}')
    plt.xlabel('Index')
    plt.ylabel('Sorted Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, n)
    plt.ylim(0, n)

    # Histogram with bell curve
    plt.subplot(1, 3, 3)
    count, bins, ignored = plt.hist(data, bins=100, density=True, alpha=0.6, color='skyblue', label='Histogram')

    # Bell curve based on sample mean and std deviation
    mu = np.mean(data)
    sigma = np.std(data)
    bell_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bins - mu) / sigma) ** 2)
    plt.plot(bins, bell_curve, color='red', linewidth=2, label='Fitted Bell Curve')

    plt.title(f'Histogram: {skew_type.capitalize()} Data')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'out/{filename}')
    plt.close()
    print(f"Saved plot with bell curve as 'out/{filename}'.")


# Example Usage to use generate_dataset output as input to plot_data_distribution:
if __name__ == '__main__':
    # Get the dataset size (n) from generate_cluster_config
    data_size, _ = generate_cluster_config(num_nodes=4, data_scale=100, seed=42)

    print(f"Data size: {data_size} elements\n")

    # Generate data using 'gaussian' skew type
    data = generate_data(
        n=data_size,
        skew_type='gaussian',
        mean=10000,
        std=6000,
    )
    # Plot the generated gaussian data
    plot_data_distribution(
        data=data,
        n=data_size,
        skew_type='gaussian',
        filename='gaussian_distribution.png'
    )