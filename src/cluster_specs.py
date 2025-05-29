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
    dataset_size = data_scale * sum(p * lcm_val for p in throughput)

    # Step 3: Random fractions that sum to 1 for memory assignment
    raw = [random.random() for _ in range(num_nodes)]
    total = sum(raw)
    memory_fractions = [r / total for r in raw]

    # Step 4: Calculate overprovisioned total memory
    total_memory = int(dataset_size * memory_overprovision_factor)

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
    print(f"Total memory of cluster = {total_memory} (should be >= {dataset_size})\n")

    total_pct_dataset = 0
    total_pct_cluster = 0

    for node, spec in cluster_config.items():
        mem = spec["memory"]
        pct_dataset = (mem / dataset_size) * 100
        pct_total_mem = (mem / total_memory) * 100

        total_pct_dataset += pct_dataset
        total_pct_cluster += pct_total_mem

        print(f"{node}: {spec}")
        print(f"  - {pct_dataset:.2f}% out of {memory_overprovision_factor * 100:.2f}% of overprovisioned memory")
        print(f"  - {pct_total_mem:.2f}% out of 100.00% of total cluster memory\n")

    return dataset_size, cluster_config

def generate_dataset(n, skew_type='uniform', zipf_param=2.0, mean=5000, std=2000):
    if skew_type == 'uniform':
        return [random.randint(0, n) for _ in range(n)]
    
    elif skew_type == 'zipf':
        # Zipf generates values starting at 1, scale down
        raw = np.random.zipf(zipf_param, size=n)
        capped = np.clip(raw, 1, n)  # Limit to n
        return capped.tolist()
    
    elif skew_type == 'exponential':
        raw = np.random.exponential(scale=n/5, size=n)
        capped = np.clip(raw, 0, n)
        return capped.astype(int).tolist()
    
    elif skew_type == 'gaussian':
        raw = np.random.normal(loc=mean, scale=std, size=n)
        capped = np.clip(raw, 0, n)
        return capped.astype(int).tolist()
    
    else:
        raise ValueError(f"Unsupported skew type: {skew_type}")

# Re-using the plot_dataset_distribution function from previous turn
def plot_dataset_distribution(
    data,
    n,
    skew_type,
    filename='dataset_distribution.png'
):  
    # Ensure output directory exists
    os.makedirs('out', exist_ok=True)
    
    indices = list(range(n))

    plt.figure(figsize=(10, 6))
    plt.scatter(indices, data, s=1, alpha=0.5) # s controls marker size, alpha controls transparency
    plt.title(f'Distribution of {skew_type.capitalize()} Data (Total {n} Integers)')
    plt.xlabel('Index of Integer')
    plt.ylabel('Integer Value')
    plt.xlim(0, n)
    plt.ylim(0, n) # Set y-axis limit based on max possible value
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('out/'+filename)
    plt.close() # Close the plot to free up memory

    print(f"Generated a scatter plot of the {skew_type} dataset and saved it as '{filename}'.")

# Example Usage to use generate_dataset output as input to plot_dataset_distribution:
if __name__ == '__main__':
    # Get the dataset size (n) from generate_cluster_config
    dataset_size, _ = generate_cluster_config(num_nodes=4, data_scale=100, seed=42)

    print(f"Dataset size: {dataset_size} elements\n")

    # Generate data using 'gaussian' skew type
    data = generate_dataset(
        n=dataset_size,
        skew_type='uniform',
    )
    # Plot the generated gaussian data
    plot_dataset_distribution(
        data=data,
        n=dataset_size,
        skew_type='uniform',
        filename='uniform_distribution.png'
    )