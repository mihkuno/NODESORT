from ortools.linear_solver import pywraplp

def optimize_partitioning(num_nodes, total_data_volume, throughputs, memory_limits, usage_rates, epsilon=1e-6):
    """
    Solves the data partitioning problem using Google OR-Tools.

    Parameters:
        num_nodes (int): Number of nodes
        total_data_volume (float): Total data volume to distribute
        throughputs (list of float): Combined throughput (1 to 10 scale) for each node
        memory_limits (list of float): Maximum memory for each node
        usage_rates (list of float): Usage billing rates for each node
        epsilon (float): Small coefficient for cost term in objective

    Returns:
        dict: Contains makespan, total cost, and data assignment per node
    """
    solver = pywraplp.Solver.CreateSolver('CBC')

    # Decision variables: amount of data assigned to each node
    data_assigned = [solver.NumVar(0, memory_limits[i], f'data_assigned_node_{i}') for i in range(num_nodes)]

    # Makespan variable
    makespan = solver.NumVar(0, solver.infinity(), 'makespan')

    # Constraint: sum of assigned data equals total volume
    solver.Add(solver.Sum(data_assigned) == total_data_volume)

    # Constraint: makespan should be at least the processing time on each node
    for i in range(num_nodes):
        solver.Add(makespan >= data_assigned[i] / throughputs[i])

    # Objective: minimize makespan plus a small weighted cost term
    total_cost_expr = solver.Sum([usage_rates[i] * (data_assigned[i] / throughputs[i]) for i in range(num_nodes)])
    solver.Minimize(makespan + epsilon * total_cost_expr)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        result = {
            'makespan': makespan.solution_value(),
            'total_cost': sum(usage_rates[i] * (data_assigned[i].solution_value() / throughputs[i]) for i in range(num_nodes)),
            'data_assignments': [data_assigned[i].solution_value() for i in range(num_nodes)]
        }
        return result
    else:
        return {'error': 'The problem does not have an optimal solution.'}


# Example usage
if __name__ == '__main__':
    num_nodes = 4
    total_data_volume = 1000
    # Throughput values between 1 and 10
    throughputs = [5, 4, 8, 7]  # Example: node 0 = 5x, node 1 = 4x, etc.
    memory_limits = [400, 300, 500, 350]
    usage_rates = [0.5, 0.6, 0.55, 0.52]

    result = optimize_partitioning(num_nodes, total_data_volume, throughputs, memory_limits, usage_rates)
    if 'error' in result:
        print(result['error'])
    else:
        print(f"Optimal makespan: {result['makespan']:.4f}")
        print(f"Total cost: {result['total_cost']:.4f}")
        for i, data in enumerate(result['data_assignments']):
            print(f"Node {i}: assigned data = {data:.2f}")
