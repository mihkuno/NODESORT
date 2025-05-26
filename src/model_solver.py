from ortools.linear_solver import pywraplp

def optimize_partitioning(num_nodes, total_data_volume, processing_rates, latencies, bandwidths, memory_limits, usage_rates, epsilon=1e-6):
    """
    Solves the data partitioning problem using Google OR-Tools.

    Parameters:
        num_nodes (int): Number of nodes
        total_data_volume (float): Total data volume to distribute
        processing_rates (list of float): Processing rates for each node
        latencies (list of float): Network latencies for each node
        bandwidths (list of float): Bandwidths for each node
        memory_limits (list of float): Maximum memory for each node
        usage_rates (list of float): Usage billing rates for each node
        epsilon (float): Small coefficient for cost term in objective

    Returns:
        dict: Contains makespan, total cost, and data assignment per node
    """
    solver = pywraplp.Solver.CreateSolver('CBC')

    data_assigned = [solver.NumVar(0, memory_limits[i], f'data_assigned_node_{i}') for i in range(num_nodes)]
    makespan = solver.NumVar(0, solver.infinity(), 'makespan')

    solver.Add(solver.Sum(data_assigned) == total_data_volume)

    for i in range(num_nodes):
        compute_time = data_assigned[i] / processing_rates[i]
        transfer_time = latencies[i] + data_assigned[i] / bandwidths[i]
        solver.Add(makespan >= compute_time + transfer_time)

    total_cost_expr = solver.Sum([usage_rates[i] * (data_assigned[i] / processing_rates[i]) for i in range(num_nodes)])
    solver.Minimize(makespan + epsilon * total_cost_expr)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        result = {
            'makespan': makespan.solution_value(),
            'total_cost': sum(usage_rates[i] * (data_assigned[i].solution_value() / processing_rates[i]) for i in range(num_nodes)),
            'data_assignments': [data_assigned[i].solution_value() for i in range(num_nodes)]
        }
        return result
    else:
        return {'error': 'The problem does not have an optimal solution.'}

# Example usage
if __name__ == '__main__':
    num_nodes = 4
    total_data_volume = 1000
    processing_rates = [50, 40, 60, 55]
    latencies = [2, 3, 1.5, 2.5]
    bandwidths = [100, 80, 120, 90]
    memory_limits = [400, 300, 500, 350]
    usage_rates = [0.5, 0.6, 0.55, 0.52]

    result = optimize_partitioning(num_nodes, total_data_volume, processing_rates, latencies, bandwidths, memory_limits, usage_rates)
    if 'error' in result:
        print(result['error'])
    else:
        print(f"Optimal makespan: {result['makespan']:.4f}")
        print(f"Total cost: {result['total_cost']:.4f}")
        for i, data in enumerate(result['data_assignments']):
            print(f"Node {i}: assigned data = {data:.2f}")
