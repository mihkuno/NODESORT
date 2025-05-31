import numpy as np
from ortools.linear_solver import pywraplp

def smart_partition(data, weights):

    n = len(data)
    num_nodes = len(weights)

    # Step 1: Compute ideal (floating point) partition sizes
    total_weight = sum(weights)
    ideal_sizes = np.array(weights) * n / total_weight

    # Step 2: Take floor of each ideal size to get initial allocation
    partition_sizes = np.floor(ideal_sizes).astype(int)

    # Step 3: Compute remaining items to allocate
    remaining = n - partition_sizes.sum()

    # Step 4: Distribute the remaining items to nodes with the largest fractional parts
    fractional_parts = ideal_sizes - partition_sizes
    indices = np.argsort(-fractional_parts)  # Sort by descending fractional parts

    for i in range(remaining):
        partition_sizes[indices[i]] += 1

    # Step 5: Slice data accordingly
    chunks = []
    start = 0
    for size in partition_sizes:
        chunks.append(data[start:start + size])
        start += size
        
    print(f"Remaining items to allocate: {remaining}")
    print(f"Partition sizes: {partition_sizes.tolist()}")
    print(f"Ideal sizes: {ideal_sizes.tolist()}")
    print(f"Fractional parts: {fractional_parts.tolist()}")

    return chunks, partition_sizes


def optimize_proportion(num_nodes, data_size, relative_throughputs, memory_sizes, costs_per_time, base_throughput, cost_weight=1e-6):
    solver = pywraplp.Solver.CreateSolver('CBC')

    # Decision variables: data items assigned to each node
    data_assigned = [solver.NumVar(0, memory_sizes[i], f'data_assigned_node_{i}') for i in range(num_nodes)]
    makespan = solver.NumVar(0, solver.infinity(), 'makespan')

    # Constraint: total data must be fully distributed
    solver.Add(solver.Sum(data_assigned) == data_size)

    # Constraints: simulated processing time per node should not exceed makespan
    for i in range(num_nodes):
        simulated_time_ms = (data_assigned[i] * base_throughput) / relative_throughputs[i]
        solver.Add(makespan >= simulated_time_ms)

    # Objective: minimize makespan with a small penalty on billing cost
    total_billing_cost_expr = solver.Sum([
        costs_per_time[i] * ((data_assigned[i] * base_throughput) / relative_throughputs[i])
        for i in range(num_nodes)
    ])
    solver.Minimize(makespan + cost_weight * total_billing_cost_expr)

    # Solve the problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        assignments = [data_assigned[i].solution_value() for i in range(num_nodes)]
        result = {
            'makespan': makespan.solution_value(),
            'total_cost': sum(
                costs_per_time[i] * ((assignments[i] * base_throughput) / relative_throughputs[i])
                for i in range(num_nodes)
            ),
            'weights': [round((assignments[i] / data_size) * 100, 2) for i in range(num_nodes)]  # percentage per node
        }
        return result
    else:
        return {'error': 'The problem does not have an optimal solution.'}


# Example usage
if __name__ == '__main__':
    num_nodes = 4
    data_size = 1000  # total items to sort
    relative_throughputs = [5, 4, 8, 7]  # relative speed of each node (1 to 10 scale)
    memory_sizes = [400, 300, 500, 350]  # how many items each node can handle
    costs_per_time = [0.005, 0.006, 0.0055, 0.0052]  # cost per millisecond of processing

    result = optimize_proportion(num_nodes, data_size, relative_throughputs, memory_sizes, costs_per_time)
    if 'error' in result:
        print(result['error'])
    else:
        print(f"Optimal makespan: {result['makespan']:.4f} ms")
        print(f"Total cost: {result['total_cost']:.4f}")
        for i, percent in enumerate(result['weights']):
            print(f"Node {i}: assigned = {percent:.2f}%")
