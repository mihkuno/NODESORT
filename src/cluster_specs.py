import random

def generate_cluster_config(
    dataset_size,
    num_nodes=3,
    memory_overprovision_factor=1.5, 
    throughput_range=(0.5, 5.0),
    billing_range=(0.005, 0.02),
    seed=None
):
    if seed is not None:
        random.seed(seed)

    # Random fractions that sum to 1
    raw = [random.random() for _ in range(num_nodes)]
    total = sum(raw)
    memory_fractions = [r / total for r in raw]

    # Multiply dataset size by overprovision factor
    total_memory = int(dataset_size * memory_overprovision_factor)

    # Assign memory based on fractions
    cluster_config = {}
    cumulative = 0

    for i in range(num_nodes):
        node = f"node{i+1}"
        if i < num_nodes - 1:
            mem = int(memory_fractions[i] * total_memory)
            cumulative += mem
        else:
            mem = total_memory - cumulative  # final node takes remainder

        cluster_config[node] = {
            "throughput": round(random.uniform(*throughput_range), 2),
            "memory": mem,
            "billing": round(random.uniform(*billing_range), 4)
        }

    # === Printing logic ===
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

    return cluster_config


# === Example usage ===
dataset = list(range(100_000))
dataset_size = len(dataset)

config = generate_cluster_config(
    dataset_size, 
    num_nodes=5, 
    memory_overprovision_factor=1.3,
    seed=10
)
