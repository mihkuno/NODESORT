import time
import random
from multiprocessing import Queue, Process
import heapq
import sys
import io
import json
import logging
from itertools import product

# Local imports
from cluster_specs import generate_cluster_config, generate_data
from model_solver import smart_partition, optimize_proportion
import results_plot

# --- Configuration for Simulation Runs ---
DISTRIBUTIONS = ['uniform', 'gaussian']
NODE_COUNTS = [2, 5, 8]
DATASET_SIZES = [10**4, 10**5, 10**6]
SEEDS = [4, 5, 6]

LOG_FILE = 'simulation_logs.log'
RESULTS_FILE = 'results.json'

def setup_logging():
    """Sets up logging to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time):
    """Simulates the perceived delay and cost for an operation."""
    actual_time_sec = time_end - time_start
    # Ensure throughput is not zero to avoid division errors
    if relative_throughput == 0:
        return float('inf'), float('inf')
    delay_time = (actual_time_sec * scale_time) / relative_throughput
    cost_time = delay_time * cost_per_time
    return delay_time, cost_time

def format_sample(data_list, first_n=3, last_n=3):
    """Helper function to format a sample of a list for printing."""
    if not data_list:
        return "[]"
    if len(data_list) <= first_n + last_n:
        return str(data_list)
    start_str = str(data_list[:first_n])
    end_str = str(data_list[-last_n:])
    return f"{start_str[:-1]}...{end_str[1:]}"

def node_sampling(id, relative_throughput, node_mem_capacity, cost_per_time, chunk, comm_queues, results_queue, num_nodes, data_size, scale_time):
    """
    Represents the operations performed by each node in the H-PSRS algorithm.
    This function now returns its performance metrics via the results_queue.
    """
    
    # Metrics tracking
    total_time = 0
    total_cost = 0
    
    initial_chunk_len = len(chunk)
    
    # Phase 1: Initial Local Sort & Sample Selection
    time_start = time.perf_counter()
    chunk.sort()
    
    L_i = (num_nodes - 1) * relative_throughput 
    L_i = int(L_i)

    samples = []
    if num_nodes > 1 and len(chunk) > 0 and L_i > 0:
        step = max(1, len(chunk) // L_i)
        for j in range(L_i):
            sample_index = j * step
            if sample_index < len(chunk):
                samples.append(chunk[sample_index])
            else:
                break
    
    if num_nodes > 1:
        comm_queues[0].put({'from': id, 'type': 'samples', 'payload': samples})
    
    time_end = time.perf_counter()
    delay_p1, cost_p1 = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
    total_time += delay_p1
    total_cost += cost_p1
    
    percent_dataset = (initial_chunk_len / data_size) * 100 if data_size > 0 else 0
    percent_memory = (initial_chunk_len / node_mem_capacity) * 100 if node_mem_capacity > 0 else 0
    logging.info(f"N{id:<2}({relative_throughput}x) Sort: items={initial_chunk_len}/{data_size} ({percent_dataset:.1f}%) mem={initial_chunk_len}/{node_mem_capacity} ({percent_memory:.1f}%) samples={len(samples):<3} time={delay_p1:.3f}µs cost=${cost_p1:.3f}")
    
    # Phase 2: Coordinator Collects Samples & Selects Pivots
    if id == 0:
        all_samples = []
        if num_nodes > 1:
            for _ in range(num_nodes):
                msg = comm_queues[0].get()
                if msg['type'] == 'samples':
                    all_samples.extend(msg['payload'])

        time_start = time.perf_counter()
        all_samples.sort()
        
        pivots = []
        if num_nodes > 1:
            desired_pivots_count = num_nodes - 1
            if len(all_samples) > 0:
                step = max(1, len(all_samples) // num_nodes)
                for i in range(1, num_nodes):
                    pivot_index = i * step
                    if pivot_index < len(all_samples):
                        pivots.append(all_samples[pivot_index])
            
            while len(pivots) < desired_pivots_count and pivots: pivots.append(pivots[-1])
            while len(pivots) < desired_pivots_count: pivots.append(0)
            pivots = pivots[:desired_pivots_count]
        
        if num_nodes > 1:
            for node_id_target in range(num_nodes):
                comm_queues[node_id_target].put({'from': 0, 'type': 'pivots', 'payload': pivots})
    
        time_end = time.perf_counter()
        delay_p2, cost_p2 = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
        total_time += delay_p2
        total_cost += cost_p2
        logging.info(f"N0 ({relative_throughput}x) Sampling: samples={len(all_samples)}/{len(all_samples)} pivots={len(pivots)} time={delay_p2:.3f}µs cost=${cost_p2:.3f} pivots={pivots}")
        
    # Phase 3: Partition and Redistribute Data
    pivots_received = []
    if num_nodes > 1:
        while True:
            msg = comm_queues[id].get()
            if msg.get('type') == 'pivots':
                pivots_received = msg['payload']
                break
            else:
                comm_queues[id].put(msg) 
                time.sleep(0.001)

    time_start = time.perf_counter()
    partitions_to_send = [[] for _ in range(num_nodes)]
    if num_nodes > 1 and pivots_received:
        for num in chunk:
            target_node_idx = 0
            while target_node_idx < len(pivots_received) and num > pivots_received[target_node_idx]:
                target_node_idx += 1
            partitions_to_send[target_node_idx].append(num)
    else:
        partitions_to_send[0] = chunk

    if num_nodes > 1:
        for target_node_idx in range(num_nodes):
            if target_node_idx != id:
                comm_queues[target_node_idx].put({'from': id, 'type': 'partition', 'payload': partitions_to_send[target_node_idx]})
    
    time_end = time.perf_counter()
    delay_p3, cost_p3 = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)    
    total_time += delay_p3
    total_cost += cost_p3
    items_kept = len(partitions_to_send[id])
    items_sent = initial_chunk_len - items_kept
    logging.info(f"N{id:<2}({relative_throughput}x) Partition: items kept={items_kept}, sent={items_sent} time={delay_p3:.3f}µs cost=${cost_p3:.3f}")
    
    # Phase 4: Receive Partitions and Merge
    partitions_to_merge = [partitions_to_send[id]]
    if num_nodes > 1:
        for _ in range(num_nodes - 1):
             partitions_to_merge.append(comm_queues[id].get()['payload'])
    
    time_start = time.perf_counter()
    final_chunk = list(heapq.merge(*partitions_to_merge))
    time_end = time.perf_counter()
    
    delay_p4, cost_p4 = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
    total_time += delay_p4
    total_cost += cost_p4
    percent_memory_merge = (len(final_chunk) / node_mem_capacity) * 100 if node_mem_capacity > 0 else 0
    logging.info(f"N{id:<2}({relative_throughput}x) Merge: items={len(final_chunk)}/{data_size} ({percent_dataset:.1f}%) mem={len(final_chunk)}/{node_mem_capacity} ({percent_memory_merge:.1f}%) time={delay_p4:.3f}µs cost=${cost_p4:.3f}")
    
    # Final data collection at coordinator
    if id != 0 and num_nodes > 1:
        comm_queues[0].put({'from': id, 'type': 'final_sorted_chunk', 'payload': final_chunk})
    
    if id == 0:
        results_by_id = {0: final_chunk}
        if num_nodes > 1:
            for _ in range(num_nodes - 1):
                msg = comm_queues[0].get()
                results_by_id[msg['from']] = msg['payload']
        
        time_start = time.perf_counter()
        final_result = []
        for i in range(num_nodes):
            if i in results_by_id:
                final_result.extend(results_by_id[i])
        time_end = time.perf_counter()

        delay_coord_merge, cost_coord_merge = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
        total_time += delay_coord_merge
        total_cost += cost_coord_merge
        logging.info(f"N0 Concat: total={len(final_result):<4} time={delay_coord_merge:.3f}µs cost=${cost_coord_merge:.3f}")

    # Calculate peak memory utilization for this node
    peak_items = max(initial_chunk_len, len(final_chunk))
    peak_mem_util = (peak_items / node_mem_capacity) * 100 if node_mem_capacity > 0 else 0
    
    # Send final metrics back to the main process
    results_queue.put({
        'from': id,
        'total_time': total_time,
        'total_cost': total_cost,
        'peak_mem_util': peak_mem_util
    })

def node_static(id, relative_throughput, node_mem_capacity, cost_per_time, chunk, results_queue, data_size, scale_time):
    """
    Represents the operations performed by each node in the H-PSLP algorithm.
    Now returns metrics and payload via the results_queue.
    """
    initial_chunk_len = len(chunk)
    
    percent_dataset = (initial_chunk_len / data_size) * 100 if data_size > 0 else 0
    percent_memory = (initial_chunk_len / node_mem_capacity) * 100 if node_mem_capacity > 0 else 0
    
    time_start = time.perf_counter()
    chunk.sort()
    time_end = time.perf_counter()
    delay, cost = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
    
    logging.info(f"N{id:<2}({relative_throughput}x) Sort: items={initial_chunk_len}/{data_size} ({percent_dataset:.1f}%) mem={initial_chunk_len}/{node_mem_capacity} ({percent_memory:.1f}%) time={delay:.3f}µs cost=${cost:.3f}")
    
    results_queue.put({
        'from': id,
        'payload': chunk,
        'metrics': {
            'time': delay,
            'cost': cost,
            'mem_util': percent_memory
        }
    })

def run_single_simulation(params):
    """
    Executes one full simulation run for a given set of parameters.
    Aggregates metrics directly from worker processes instead of parsing logs.
    """
    dist, num_nodes, data_size, seed = params
    
    config_line = f"Config: Dist={dist}, Nodes={num_nodes}, Size={data_size}, Seed={seed}"
    logging.info('#' + '-'*35 + ' Cluster Configuration ' + '-'*34 + '#')
    logging.info(config_line)
    
    # --- Setup ---
    scale_time = 10**6
    cluster = generate_cluster_config(data_size=data_size, num_nodes=num_nodes, seed=seed)
    cluster = dict(sorted(cluster.items(), key=lambda item: item[1]['throughput'], reverse=True))
    
    relative_throughputs = [node['throughput'] for node in cluster.values()]
    memory_sizes = [node['memory'] for node in cluster.values()]
    costs_per_time = [node['billing'] for node in cluster.values()]
    
    # Corrected filename generation to include dataset_size
    filename = f'run_{dist}_{num_nodes}nodes_{data_size}size_{seed}seed'
    data = generate_data(n=data_size, skew_type=dist, filename=filename)
    total_relative_throughput = sum(relative_throughputs)
    
    logging.info(f"Nodes: {num_nodes} | Data: {data_size} elements | Sample: {format_sample(data, 5, 5)}")
    
    # --- Sequential Baseline ---
    logging.info('\n# ---------------------------- Sequential Baseline --------------------------- #')
    temp = data.copy()
    time_start = time.perf_counter()
    temp.sort()
    time_end = time.perf_counter()
    delay, _ = simulation(time_start, time_end, 1, 0, scale_time)
    base_throughput = delay / data_size if data_size > 0 else 0
    logging.info(f'Sequential: time={delay:.3f}µs throughput={base_throughput:.6f}µs/element')

    # --- H-PSRS (Regular Sampling) ---
    logging.info(f"\n# ------------------------- H-PSRS (Regular Sampling) ------------------------ #")
    comm_queues_hpsrs = [Queue() for _ in range(num_nodes)]
    results_queue_hpsrs = Queue()
    
    chunks_hpsrs = []
    last_idx = 0
    for i in range(num_nodes):
        partition_size = (data_size * relative_throughputs[i]) // total_relative_throughput if total_relative_throughput > 0 else data_size // num_nodes
        partition = data[last_idx:last_idx + partition_size]
        last_idx += partition_size
        chunks_hpsrs.append(partition)
    if last_idx < data_size: chunks_hpsrs[-1].extend(data[last_idx:])

    processes_hpsrs = [Process(target=node_sampling, args=(i, relative_throughputs[i], memory_sizes[i], costs_per_time[i], chunks_hpsrs[i], comm_queues_hpsrs, results_queue_hpsrs, num_nodes, data_size, scale_time)) for i in range(num_nodes)]
    for p in processes_hpsrs: p.start()
    
    # Collect results
    hpsrs_results = [results_queue_hpsrs.get() for _ in range(num_nodes)]
    for p in processes_hpsrs: p.join()

    # --- H-PSLP (Static Optimized) ---
    logging.info('\n# ------------------------- H-PSLP (Static Optimized) ------------------------ #')
    results_queue_hpslp = Queue()
    
    time_start = time.perf_counter()
    solver_result = optimize_proportion(num_nodes, data_size, relative_throughputs, memory_sizes, costs_per_time, base_throughput)
    time_end = time.perf_counter()
    
    weights = solver_result.get('weights', [100/num_nodes] * num_nodes)
    delay, cost = simulation(time_start, time_end, relative_throughputs[0], costs_per_time[0], scale_time)
    logging.info(f'N0 ({relative_throughputs[0]}x) Solver: time={delay:.3f}µs cost=${cost:.3f} ^makespan={solver_result.get("makespan", 0):.3f}µs ^cost=${solver_result.get("total_cost", 0):.3f}')

    chunks_hpslp, _ = smart_partition(data, weights)
    
    processes_hpslp = [Process(target=node_static, args=(i, relative_throughputs[i], memory_sizes[i], costs_per_time[i], chunks_hpslp[i], results_queue_hpslp, data_size, scale_time)) for i in range(num_nodes)]
    for p in processes_hpslp: p.start()
    
    # Collect results
    static_results = [results_queue_hpslp.get() for _ in range(num_nodes)]
    for p in processes_hpslp: p.join()
        
    time_start = time.perf_counter()
    lists_to_merge = [item['payload'] for item in sorted(static_results, key=lambda x: x['from'])]
    merged_sorted_list = list(heapq.merge(*lists_to_merge))
    time_end = time.perf_counter()
    delay_merge, cost_merge = simulation(time_start, time_end, relative_throughputs[0], costs_per_time[0], scale_time)
    logging.info(f"N0 ({relative_throughputs[0]}x) Heap Merge: total={len(merged_sorted_list):<4} time={delay_merge:.3f}µs cost=${cost_merge:.3f}")
    
    # --- Aggregate Metrics ---
    metrics = {
        'hpslp_predicted_makespan': solver_result.get('makespan', 0),
        'hpslp_predicted_cost': solver_result.get('total_cost', 0)
    }
    
    # H-PSRS Metrics
    hpsrs_mem_utils = [r['peak_mem_util'] for r in hpsrs_results]
    metrics['hpsrs_makespan'] = max(r['total_time'] for r in hpsrs_results) if hpsrs_results else 0
    metrics['hpsrs_cost'] = sum(r['total_cost'] for r in hpsrs_results)
    metrics['hpsrs_mem_util'] = max(hpsrs_mem_utils) if hpsrs_mem_utils else 0

    # H-PSLP Metrics
    hpslp_sort_metrics = [r['metrics'] for r in static_results]
    hpslp_sort_times = [m['time'] for m in hpslp_sort_metrics]
    hpslp_mem_utils = [m['mem_util'] for m in hpslp_sort_metrics]
    
    # Actual makespan and cost of just the local sort phase
    hpslp_actual_sort_makespan = max(hpslp_sort_times) if hpslp_sort_times else 0
    hpslp_actual_sort_cost = sum(m['cost'] for m in hpslp_sort_metrics)
    
    # Total makespan and cost for the entire run (sort + merge)
    metrics['hpslp_makespan'] = hpslp_actual_sort_makespan + delay_merge
    metrics['hpslp_cost'] = hpslp_actual_sort_cost + cost_merge
    metrics['hpslp_mem_util'] = max(hpslp_mem_utils) if hpslp_mem_utils else 0
    
    # Store the actual sort phase metrics for the LP accuracy plot
    metrics['hpslp_actual_sort_makespan'] = hpslp_actual_sort_makespan
    metrics['hpslp_actual_sort_cost'] = hpslp_actual_sort_cost


    return data, metrics

if __name__ == '__main__':
    setup_logging()
    
    all_params = list(product(DISTRIBUTIONS, NODE_COUNTS, DATASET_SIZES, SEEDS))
    total_runs = len(all_params)
    all_results = []
    example_datasets = {}
    
    for i, params in enumerate(all_params):
        run_header = f"--- Starting Run {i+1}/{total_runs} ---"
        logging.info("\n" + "="*len(run_header))
        logging.info(run_header)
        logging.info("="*len(run_header))

        generated_data, metrics = run_single_simulation(params)
        dist_type, _, data_size, seed = params
        
        # Capture specific datasets for plotting examples
        if seed == 4 and data_size in [10**4, 10**5, 10**6]:
            key = f"{dist_type}_{data_size}"
            if key not in example_datasets:
                example_datasets[key] = {
                    "data": generated_data,
                    "seed": seed
                }
        
        # --- Log Aggregated Metrics Comparison ---
        summary_header = "--- Aggregated Metrics Summary for this Run ---"
        logging.info("\n" + summary_header)
        logging.info("-" * len(summary_header))
        logging.info(f"{'Metric':<25} | {'H-PSRS':>18} | {'H-PSLP':>18}")
        logging.info("-" * len(summary_header))
        logging.info(f"{'Makespan (µs)':<25} | {metrics.get('hpsrs_makespan', 0):>18,.2f} | {metrics.get('hpslp_makespan', 0):>18,.2f}")
        logging.info(f"{'Total Cost ($)':<25} | {metrics.get('hpsrs_cost', 0):>18,.2f} | {metrics.get('hpslp_cost', 0):>18,.2f}")
        logging.info(f"{'Max. Peak Memory Util.(%)':<25} | {metrics.get('hpsrs_mem_util', 0):>18.2f} | {metrics.get('hpslp_mem_util', 0):>18.2f}")
        logging.info("-" * len(summary_header) + "\n")
        
        run_result = {
            'run': i + 1,
            'distribution': params[0],
            'nodes': params[1],
            'dataset_size': params[2],
            'seed': params[3],
            **metrics
        }
        all_results.append(run_result)
        
    logging.info("\n\n" + "="*40)
    logging.info("All Simulations Complete. Saving results...")
    logging.info("="*40)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"Results saved to {RESULTS_FILE}")
    
    logging.info("Generating plots from results...")
    try:
        # Correctly pass the example_datasets dictionary
        results_plot.generate_all_plots(results_file=RESULTS_FILE, example_datasets=example_datasets)
    except Exception as e:
        logging.error(f"Could not generate plots. Error: {e}")
        import traceback
        logging.error(traceback.format_exc())

    logging.info("Process finished.")
