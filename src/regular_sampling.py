import time
import random
from multiprocessing import Queue, Process
import heapq # Import for min-heap functionality
from cluster_specs import generate_cluster_config, generate_data
from model_solver import smart_partition, optimize_proportion

def simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time):
    """Simulates the perceived delay and cost for an operation."""
    actual_time_sec = time_end - time_start
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

def node_sampling(id, relative_throughput, memory_size, cost_per_time, chunk, queues, num_nodes, data_size, scale_time):
    """Represents the operations performed by each node in the parallel sorting algorithm."""
    
    initial_chunk_len = len(chunk)
    
    # Phase 1 & 2: Sort + Sample
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
        queues[0].put({'from': id, 'type': 'samples', 'payload': samples})
    
    time_end = time.perf_counter()
    delay_p1_2, cost_p1_2 = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)

    percent_dataset = (initial_chunk_len / data_size) * 100 if data_size > 0 else 0
    percent_memory = (initial_chunk_len / memory_size) * 100 if memory_size > 0 else 0
    print(f"N{id:<2}({relative_throughput}x) Sort: items={initial_chunk_len}/{data_size} ({percent_dataset:.1f}%) mem={initial_chunk_len}/{memory_size} ({percent_memory:.1f}%) samples={len(samples):<3} time={delay_p1_2:.3f}µs cost=${cost_p1_2:.3f} data={format_sample(chunk)}")
    
    # Coordinator: Collect samples & Select pivots
    if id == 0:
        all_samples = []
        if num_nodes > 1:
            for _ in range(num_nodes):
                msg = queues[0].get()
                if msg['type'] == 'samples':
                    all_samples.extend(msg['payload'])

        print('-'*80)
        time_start = time.perf_counter()
        
        all_samples.sort()
        
        pivots = []
        if num_nodes > 1 and len(all_samples) > 0:
            desired_pivots_count = num_nodes - 1
            if len(all_samples) <= desired_pivots_count:
                pivots.extend(sorted(list(set(all_samples))))
                while len(pivots) < desired_pivots_count and pivots:
                    pivots.append(pivots[-1])
                while len(pivots) < desired_pivots_count:
                    pivots.append(0)
            else:
                step = len(all_samples) // num_nodes
                if step == 0:
                    pivots.extend(all_samples[:desired_pivots_count])
                else:
                    for i in range(1, num_nodes):
                        pivot_index = i * step
                        if pivot_index < len(all_samples):
                            pivots.append(all_samples[pivot_index])
                        else:
                            if pivots: pivots.append(pivots[-1])
                            elif all_samples: pivots.append(all_samples[-1])
                            else: break
            while len(pivots) < desired_pivots_count and pivots:
                pivots.append(pivots[-1])
            while len(pivots) < desired_pivots_count:
                 pivots.append(0)
            pivots = pivots[:desired_pivots_count]

        # Broadcast pivots
        if num_nodes > 1:
            for node_id_target in range(num_nodes):
                queues[node_id_target].put({'from': 0, 'type': 'pivots', 'payload': pivots})
    
        time_end = time.perf_counter()
        delay_coord_pivots, cost_coord_pivots = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
        print(f"N0 ({relative_throughput}x) Sampling: samples={len(all_samples)}/{len(all_samples)} pivots={len(pivots)} time={delay_coord_pivots:.3f}µs cost=${cost_coord_pivots:.3f} pivots={pivots}")
        print('-'*80)
        
    # Phase 3.1: Receive pivots & Partition data
    pivots_received_payload = []
    if num_nodes > 1:
        pivots_received_flag = False
        while not pivots_received_flag:
            msg = queues[id].get()
            if msg['type'] == 'pivots':
                pivots_received_payload = msg['payload']
                pivots_received_flag = True
            else:
                queues[id].put(msg)
                time.sleep(0.001)
    
    time_start = time.perf_counter()

    partitions_to_send = [[] for _ in range(num_nodes)]
    if num_nodes > 1 and pivots_received_payload:
        for num in chunk:
            target_node_idx = 0
            while target_node_idx < len(pivots_received_payload) and num > pivots_received_payload[target_node_idx]:
                target_node_idx += 1
            target_node_idx = min(target_node_idx, num_nodes - 1)
            partitions_to_send[target_node_idx].append(num)
    elif num_nodes == 1:
        partitions_to_send[0] = chunk
    else:
        partitions_to_send[id] = chunk

    if num_nodes > 1:
        for target_node_idx in range(num_nodes):
            if target_node_idx != id:
                queues[target_node_idx].put({'from': id, 'type': 'partition', 'payload': partitions_to_send[target_node_idx]})
    
    time_end = time.perf_counter()
    delay_p3_1, cost_p3_1 = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)    
    print(f"N{id:<2}({relative_throughput}x) Partition: transfer={len(partitions_to_send[id])}/{len(chunk)} time={delay_p3_1:.3f}µs cost=${cost_p3_1:.3f}")
    
    # Phase 3.2: Receive partitions
    chunk = partitions_to_send[id] 
    if num_nodes > 1:
        received_partition_count = 0
        while received_partition_count < (num_nodes - 1):
            msg = queues[id].get()
            if msg['type'] == 'partition':
                chunk.extend(msg['payload'])
                received_partition_count += 1
            else:
                queues[id].put(msg)
                time.sleep(0.001)
    
    # Phase 4: Final sort
    time_start = time.perf_counter()
    chunk.sort()
    
    if id != 0 and num_nodes > 1:
        queues[0].put({'from': id, 'type': 'final', 'payload': chunk})
        
    time_end = time.perf_counter()
    delay_p4, cost_p4 = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
    percent_dataset = (initial_chunk_len / data_size) * 100 if data_size > 0 else 0
    percent_memory = (initial_chunk_len / memory_size) * 100 if memory_size > 0 else 0
    print(f"N{id:<2}({relative_throughput}x) Final: items={len(chunk)}/{data_size} ({percent_dataset:.1f}%) mem={len(chunk)}/{memory_size} ({percent_memory:.1f}%) time={delay_p4:.3f}µs cost=${cost_p4:.3f} data={format_sample(chunk)}")
    
    # Coordinator: Collect & Merge final results
    if id == 0:
        results_by_id = {0: chunk}
        collected_final_count = 0
        
        while collected_final_count < (num_nodes - 1):
            msg = queues[0].get()
            if msg['type'] == 'final':
                results_by_id[msg['from']] = msg['payload']
                collected_final_count += 1
            else:
                queues[0].put(msg)
                time.sleep(0.001)
        
        time_start = time.perf_counter()
        
        final_result = []
        for i in range(num_nodes):
            if i in results_by_id:
                final_result.extend(results_by_id[i])

        time_end = time.perf_counter()
        delay_coord_merge, cost_coord_merge = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
        print('-'*80)
        print(f"N0 Merge: total={len(final_result):<4} time={delay_coord_merge:.3f}µs cost=${cost_coord_merge:.3f} data={format_sample(final_result)}")
        print('-'*80)
    
def node_static(id, relative_throughput, memory_size, cost_per_time, chunk, queue, data_size, scale_time):
    initial_chunk_len = len(chunk)
    
    percent_dataset = (initial_chunk_len / data_size) * 100 if data_size > 0 else 0
    percent_memory = (initial_chunk_len / memory_size) * 100 if memory_size > 0 else 0
    
    time_start = time.perf_counter()
    chunk.sort()
    time_end = time.perf_counter()
    delay, cost = simulation(time_start, time_end, relative_throughput, cost_per_time, scale_time)
    
    queue.put({'from': id, 'type': 'result', 'payload': chunk})
    print(f"N{id:<2}({relative_throughput}x) Sort: items={initial_chunk_len}/{data_size} ({percent_dataset:.1f}%) mem={initial_chunk_len}/{memory_size} ({percent_memory:.1f}%) time={delay:.3f}µs cost=${cost:.3f} data={format_sample(chunk)}")
    

def linear_scan_merge(sorted_lists):
    result = []
    indices = [0] * len(sorted_lists)  # Track current index in each list
    
    while True:
        min_val = None
        min_list_idx = -1
        
        for i, lst in enumerate(sorted_lists):
            if indices[i] < len(lst):
                val = lst[indices[i]]
                if min_val is None or val < min_val:
                    min_val = val
                    min_list_idx = i
        
        if min_list_idx == -1:
            break  # All lists are exhausted
        
        result.append(min_val)
        indices[min_list_idx] += 1

    return result

        
if __name__ == '__main__':
    
    print('# --------------------------- Cluster Configuration -------------------------- #')
    
    scale_time = 10**6
    num_nodes = 4
    
    data_size, cluster = generate_cluster_config(
        data_scale=10, 
        num_nodes=num_nodes,
        seed=4
    )
    
    cluster = dict(sorted(cluster.items(), key=lambda item: item[1]['throughput'], reverse=True))
    
    relative_throughputs = []
    memory_sizes = []
    costs_per_time = []
    
    for key, node in cluster.items():
        memory_sizes.append(node['memory'])
        costs_per_time.append(node['billing'])
        relative_throughputs.append(node['throughput'])
    
    total_relative_throughput = sum(relative_throughputs)
    data = generate_data(n=data_size, skew_type='uniform')
    
    print(f"Nodes: {num_nodes} | Data: {data_size} elements | Sample: {format_sample(data, 5, 5)}")
    
    print('\n# ---------------------------- Sequential Baseline --------------------------- #')
    temp = data.copy()
    time_start = time.perf_counter()
    temp.sort()
    time_end = time.perf_counter()
    delay, _ = simulation(time_start, time_end, 1, 0, scale_time)
    base_throughput = delay / data_size
    print(f'Sequential: time={delay:.3f}µs throughput={base_throughput:.6f}µs/element')
    print('-'*80)
    
    
    print(f"\n# ------------------------- Regular Sampling Approach ------------------------ #")
    
    queues = [Queue() for _ in range(num_nodes)]
    
    time_start = time.perf_counter()
    
    chunks = []
    last_idx = 0
    for i in range(num_nodes):
        partition_size = (data_size * relative_throughputs[i]) // total_relative_throughput
        partition = data[last_idx:last_idx + partition_size]
        last_idx += partition_size
        chunks.append(partition)

    time_end = time.perf_counter()
    delay, cost = simulation(time_start, time_end, relative_throughputs[0], costs_per_time[0], scale_time)
    print(f"N0 ({relative_throughputs[0]}x) Partition: time={delay:.3f}µs time={cost:.3f}µs cost=${cost:.3f}")
    print('-'*80)
    
    processes = []
    for i in range(num_nodes):
        p = Process(
            target=node_sampling,
            args=(i, relative_throughputs[i], memory_sizes[i], costs_per_time[i], chunks[i], queues, num_nodes, data_size, scale_time)
        )
        processes.append(p)
        p.start()    
    
    for p in processes:
        p.join()
        
        
    print('\n# ------------------------- Static Optimized Approach ------------------------ #')
    
    queue = Queue()
    
    time_start = time.perf_counter()
    result = optimize_proportion(num_nodes, data_size, relative_throughputs, memory_sizes, costs_per_time, base_throughput)
    time_end = time.perf_counter()
    
    # Find the highest weight and set it as Node 0 (Coordinator)
    weights = result['weights']
    max_idx = weights.index(max(weights))
    if max_idx != 0:
        for arr in (weights, relative_throughputs, memory_sizes, costs_per_time):
            arr[0], arr[max_idx] = arr[max_idx], arr[0]

    delay, cost = simulation(time_start, time_end, relative_throughputs[0], costs_per_time[0], scale_time)
    print(f'N0 ({relative_throughputs[0]}x) Solver: time={delay:.3f}µs cost=${cost:.3f} ^makespan={result["makespan"]:.3f}µs ^cost=${result["total_cost"]:.3f}')

    chunks, partition_sizes = smart_partition(data, weights)
    print('-'*80)
    
    # Display static node info
    processes = []
    for i in range(num_nodes):
        p = Process(
            target=node_static,
            args=(i, relative_throughputs[i], memory_sizes[i], costs_per_time[i], chunks[i], queue, data_size, scale_time)
        )
        processes.append(p)
        p.start()
        
    # Collect results from the queue to prevent blocking
    static_results = []
    for _ in range(num_nodes):
        static_results.append(queue.get())
        
    for p in processes:
        p.join()
        
    # Merge the locally sorted results using min-heap
    time_start = time.perf_counter()
    # Extract the sorted lists from the collected results
    lists_to_merge = [item['payload'] for item in static_results]
    # Perform the min-heap merge
    merged_sorted_list = list(heapq.merge(*lists_to_merge))
    time_end = time.perf_counter()
    delay, cost = simulation(time_start, time_end, relative_throughputs[0], costs_per_time[0], scale_time) # Assuming merge is sequential and has no cost
    
    print('-'*80)
    print(f"N0 ({relative_throughputs[0]}x) Merge: total={len(merged_sorted_list):<4} time={delay:.3f}µs cost=${cost:.3f} data={format_sample(merged_sorted_list)}")
    print('-'*80)
    
    
    # Time and simulate linear scan merge
    time_start = time.perf_counter()
    merged_linear_scan = linear_scan_merge(lists_to_merge)
    time_end = time.perf_counter()

    delay_linear, cost_linear = simulation(
        time_start,
        time_end,
        relative_throughputs[0],
        costs_per_time[0],
        scale_time
    )
    
    print(f"N0 ({relative_throughputs[0]}x) Linear Scan Merge: total={len(merged_linear_scan):<4} time={delay_linear:.3f}µs cost=${cost_linear:.3f} data={format_sample(merged_linear_scan)}")
    print('-'*80)