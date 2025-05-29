import time
import random
from multiprocessing import Queue, Process
from math import gcd
from functools import reduce

def _simulate_sorting(chunk, throughput, multiplier=1000):
    time_start = time.perf_counter()
    chunk.sort()
    time_end = time.perf_counter()
    actual_time = time_end - time_start
    delay = actual_time * (1 / throughput)
    delay *= multiplier
    return round(delay * 1000, 3)

def lcm(a, b):
    return a * b // gcd(a, b)

def lcm_list(numbers):
    return reduce(lcm, numbers, 1)

# Added total_data_size as an argument to node_psrs
def node_psrs(id, perf, chunk, queues, num_nodes, total_data_size):
    
    # ---------------------------- Phase 1: Local sort --------------------------- #
    delay = _simulate_sorting(chunk, perf)
    # Calculate percentage relative to the global total_data_size
    percent_initial = (len(chunk) / total_data_size) * 100
    print(f"Node {id} ({perf}x): Initial {len(chunk)} ({percent_initial:.2f}% of total) in {delay}ms | first10={chunk[:10]} last10={chunk[-10:]}")
    
    # ------------------------ Phase 2: Sample collection ------------------------ #
    L_i = (num_nodes - 1) * perf  # Number of samples desired per node (from paper)
    samples = []
    
    if num_nodes > 1 and len(chunk) > 0:
        num_samples_to_attempt = L_i
        if num_samples_to_attempt > 0:
            step = max(1, len(chunk) // num_samples_to_attempt)
            
            for j in range(num_samples_to_attempt):
                sample_index = j * step
                if sample_index < len(chunk):
                    samples.append(chunk[sample_index])
                else:
                    break
    
    queues[0].put({'from': id, 'type': 'samples', 'payload': samples})

    # Coordinator collects samples and selects pivots
    if id == 0:
        all_samples = []
        for _ in range(num_nodes):
            msg = queues[0].get()
            all_samples.extend(msg['payload'])
        
        delay = _simulate_sorting(all_samples, perf)
        print(f"\nCoordinator samples: {len(all_samples)} in {delay}ms | first10={all_samples[:10]} last10={all_samples[-10:]}\n")
        
        # Select p-1 global pivots
        pivots = []
        if num_nodes > 1 and len(all_samples) > 0:
            desired_pivots_count = num_nodes - 1
            
            if len(all_samples) < desired_pivots_count:
                unique_sorted_samples = sorted(list(set(all_samples)))
                pivots.extend(unique_sorted_samples)
                while len(pivots) < desired_pivots_count and pivots:
                    pivots.append(pivots[-1])
            else:
                step = len(all_samples) // num_nodes
                if step == 0:
                    unique_sorted_samples = sorted(list(set(all_samples)))
                    pivots.extend(unique_sorted_samples[:desired_pivots_count])
                    while len(pivots) < desired_pivots_count and pivots:
                        pivots.append(pivots[-1])
                else:
                    for i in range(1, num_nodes):
                        pivot_index = i * step
                        if pivot_index < len(all_samples):
                            pivots.append(all_samples[pivot_index])
                        else:
                            if pivots:
                                pivots.append(pivots[-1])
                            elif all_samples:
                                pivots.append(all_samples[-1])
                            else:
                                break
        
        # Broadcast pivots to all nodes
        for node_id in range(num_nodes):
            queues[node_id].put({'from': 0, 'type': 'pivots', 'payload': pivots})
    
    # ------------------------ Phase 3: Redistribute data ------------------------ #
    # Each node waits for its pivot message
    pivots_received = False
    pivots = []
    while not pivots_received:
        msg = queues[id].get()
        if msg['type'] == 'pivots':
            pivots = msg['payload']
            pivots_received = True
        else:
            queues[id].put(msg)
            time.sleep(0.001)

    # Partition local data using pivots
    partitions = [[] for _ in range(num_nodes)]
    for num in chunk:
        nid = 0
        while nid < len(pivots) and num > pivots[nid]:
            nid += 1
        
        nid = min(nid, num_nodes - 1)
        
        partitions[nid].append(num)
    
    # Send partitions to other nodes
    for nid in range(num_nodes):
        if nid != id:
            queues[nid].put({'from': id, 'type': 'partition', 'payload': partitions[nid]})
    
    # Receive partitions from other nodes
    chunk = partitions[id] # Keep local partition
    
    received_partition_count = 0
    while received_partition_count < (num_nodes - 1):
        msg = queues[id].get()
        if msg['type'] == 'partition':
            chunk.extend(msg['payload'])
            received_partition_count += 1
        else:
            queues[id].put(msg)
            time.sleep(0.001)

    # ---------------------------- Phase 4: Final sort --------------------------- #
    delay = _simulate_sorting(chunk, perf)
    # Calculate percentage relative to the global total_data_size
    percent_final = (len(chunk) / total_data_size) * 100
    print(f"Node {id} ({perf}x): Final {len(chunk)} ({percent_final:.2f}% of total) in {delay}ms | first10={chunk[:10]} last10={chunk[-10:]}")

    
    # Send final sorted chunk to coordinator
    if id != 0:
        queues[0].put({'from': id, 'type': 'final', 'payload': chunk})
    else:
        results_by_id = {0: chunk} # Coordinator's own sorted chunk
        
        collected_final_count = 0
        while collected_final_count < (num_nodes - 1):
            msg = queues[0].get()
            if msg['type'] == 'final':
                results_by_id[msg['from']] = msg['payload']
                collected_final_count += 1
            else:
                queues[0].put(msg)
                time.sleep(0.001)
        
        final_result = []
        for nid in range(num_nodes):
            final_result.extend(results_by_id[nid])
        
        print(f"\nTotal length: {len(final_result)} | first10={final_result[:10]} last10={final_result[-10:]}\n")


if __name__ == '__main__':
    num_nodes = 4
    perf = [8, 5, 3, 1]  # Relative performance factors from paper example
    
    # Calculate optimal data size (from paper's Eq.1)
    lcm_val = lcm_list(perf)
    k = 5  # Simplest case, can be adjusted
    n = k * sum(p * lcm_val for p in perf) # This is the total data size
    
    data = [random.randint(0, 10000) for _ in range(n)]
    queues = [Queue() for _ in range(num_nodes)]
    
    print(f"Total data size: {n} | Data sample: {data[:10]} ... {data[-10:]}\n")
    
    # Data distribution proportional to performance
    total_perf = sum(perf)
    partitions_main = []
    last_idx = 0
    for i in range(num_nodes):
        partition_size = (n * perf[i]) // total_perf
        partition = data[last_idx:last_idx + partition_size]
        last_idx += partition_size
        partitions_main.append(partition)
    
    time_start = time.perf_counter()
    
    # Start processes
    processes = []
    for i in range(num_nodes):
        p = Process(
            target=node_psrs,
            # Pass the global total_data_size (n) to each node
            args=(i, perf[i], partitions_main[i], queues, num_nodes, n)
        )
        processes.append(p)
        p.start()
    
    # Wait for completion
    for p in processes:
        p.join()
        
    time_end = time.perf_counter()
    total_time = time_end - time_start
    print(f"Total execution time: {total_time*1000:.3f}ms\n")