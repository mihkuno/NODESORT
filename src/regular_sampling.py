import time
import random
from multiprocessing import Queue, Process
from cluster_specs import generate_cluster_config, generate_data
from model_solver import smart_partition

def _simulate_sorting(chunk, throughput, billing, multiplier=1000):
    time_start = time.perf_counter()
    chunk.sort()
    time_end = time.perf_counter()
    actual_time = time_end - time_start
    
    # Calculate delay based on throughput
    throughput = actual_time * (1 / throughput)
    throughput *= multiplier # Scale to milliseconds
    throughput *= 1000 # Convert to milliseconds
    throughput = round(throughput, 3) # Round to 3 decimal places
    
    # Calculate billing cost (e.g., actual_time * billing rate per unit of time)
    # Assuming 'billing' is a rate per unit of time (e.g., per second)
    billing *= throughput
    billing = round(billing, 3) # Round to 3 decimal places
    
    return throughput, billing


def node_psrs(id, throughput, memory, billing, chunk, queues, num_nodes, total_data_size):
    
    # ---------------------------- Phase 1: Local sort --------------------------- #
    delay, cost = _simulate_sorting(chunk, throughput, billing)
    percent_initial = (len(chunk) / total_data_size) * 100
    percent_in_memory_initial = (len(chunk) / memory) * 100 if memory > 0 else 0
    
    print(f"Node {id:<2} ({throughput:>2}x): Phase 1 - Initial: {len(chunk)} / {memory} elements | "
          f"{percent_initial:.2f}% of dataset | {percent_in_memory_initial:.2f}% of memory | "
          f"Time: {delay:>7.3f}ms | Cost: ${cost:>9.6f} | Sample: {str(chunk[:5])[:-1]}...{str(chunk[-5:])[1:]}")
    
    # ------------------------ Phase 2: Sample collection ------------------------ #
    L_i = (num_nodes - 1) * throughput
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
        
        delay, cost = _simulate_sorting(all_samples, throughput, billing) # Coordinator uses its own billing rate
        print(f"\n{'':<10}Coordinator: Collected {len(all_samples):<8} samples | Time: {delay:>7.3f}ms | "
              f"Cost: ${cost:>9.6f} | Sample: {str(all_samples[:5])[:-1]}...{str(all_samples[-5:])[1:]}\n")
        
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
        
        print(f"{'':<10}Coordinator: Selected {len(pivots):<8} pivots | Pivots: {pivots}\n")
        
        # Broadcast pivots to all nodes
        for node_id in range(num_nodes):
            queues[node_id].put({'from': 0, 'type': 'pivots', 'payload': pivots})
    
    # ------------------------ Phase 3: Redistribute data ------------------------ #
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

    partitions = [[] for _ in range(num_nodes)]
    for num in chunk:
        nid = 0
        while nid < len(pivots) and num > pivots[nid]:
            nid += 1
        
        nid = min(nid, num_nodes - 1)
        
        partitions[nid].append(num)
    
    for nid in range(num_nodes):
        if nid != id:
            queues[nid].put({'from': id, 'type': 'partition', 'payload': partitions[nid]})
    
    chunk = partitions[id]
    
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
    delay, cost = _simulate_sorting(chunk, throughput, billing)
    percent_final = (len(chunk) / total_data_size) * 100
    percent_in_memory_final = (len(chunk) / memory) * 100 if memory > 0 else 0
    
    print(f"Node {id:<2} ({throughput:>2}x): Phase 4 - Final: {len(chunk)} / {memory} elements | "
          f"{percent_final:.2f}% of dataset | {percent_in_memory_final:.2f}% of memory | "
          f"Time: {delay:>7.3f}ms | Cost: ${cost:>9.6f} | Sample: {str(chunk[:5])[:-1]}...{str(chunk[-5:])[1:]}")

    if id != 0:
        queues[0].put({'from': id, 'type': 'final', 'payload': chunk})
    else:
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
        
        final_result = []
        for nid in range(num_nodes):
            final_result.extend(results_by_id[nid])
        
        print(f"\n{'':<10}Coordinator: All nodes merged. Total sorted elements: {len(final_result):<8} | "
              f"Sample: {str(final_result[:5])[:-1]}...{str(final_result[-5:])[1:]}\n")


# -------- Main function to set up the environment and start processes ------- #

if __name__ == '__main__':
    
    print('\n----------------------------- Cluster Configuration ---------------------------')
    
    num_nodes = 4
    throughput = [] 
    memory = []
    billing = []
    
    # generate cluster configuration
    data_size, cluster = generate_cluster_config(
        data_scale=5, 
        num_nodes=num_nodes,
        seed=4
    )
    
    # make the fastest node the coordinator (id=0)
    cluster = dict(sorted(cluster.items(), key=lambda item: item[1]['throughput'], reverse=True))
    
    # extract throughput, memory, and billing from the cluster
    for key, node in cluster.items():
        memory.append(node['memory'])
        billing.append(node['billing'])
        throughput.append(node['throughput'])
    total_throughput = sum(throughput)
    
    # Create queues for inter-process communication
    queues = [Queue() for _ in range(num_nodes)]
    
    # Generate data for the nodes
    data = generate_data(n=data_size,skew_type='uniform')
    
    print(f"\nNumber of Nodes: {num_nodes}")
    print(f"Total Data Size: {data_size} elements")
    print(f"Data Sample: {data[:5]}...{data[-5:]}\n")
    
    
    print(f"------------------------------ Regular Sampling ------------------------------")
    
    time_start = time.perf_counter()
    
    chunks = []
    last_idx = 0
    for i in range(num_nodes):
        partition_size = (data_size * throughput[i]) // total_throughput
        partition = data[last_idx:last_idx + partition_size]
        last_idx += partition_size
        chunks.append(partition)

    time_end = time.perf_counter()
        
    total_time = time_end - time_start
    print(f"Partition time: {total_time*1000:.3f}ms")
    print("-" * 30)
    
    # Start processes for each node
    processes = []
    for i in range(num_nodes):
        p = Process(
            target=node_psrs,
            args=(i, throughput[i], memory[i], billing[i], chunks[i], queues, num_nodes, data_size)
        )
        processes.append(p)
        p.start()    
    for p in processes:
        p.join()
        
        
    print('-------------------------------- LP Guided -----------------------------------')
    
    chunks, chunk_size = smart_partition(data, [8.5, 4.21, 2.65, 3.22])
    print(f"Chunk sizes: {chunk_size}")
    
    