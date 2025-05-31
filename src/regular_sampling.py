import time
import random
from multiprocessing import Queue, Process
from cluster_specs import generate_cluster_config, generate_data
from model_solver import smart_partition, optimize_proportion

def simulate_sorting(chunk, relative_throughput, cost_per_time, scale_time):
    
    time_start = time.perf_counter()
    chunk.sort()
    time_end = time.perf_counter()

    actual_time_sec = time_end - time_start

    # Simulate the perceived delay (scaled and adjusted by throughput)
    delay_time = (actual_time_sec * scale_time) / relative_throughput
    delay_time = round(delay_time, 6)

    # Calculate billing cost based on the simulated delay
    cost_time = round(delay_time * cost_per_time, 6)

    return delay_time, cost_time



def node_psrs(id, relative_throughput, memory_size, cost_per_time, chunk, queues, num_nodes, data_size, scale_time):
    
    # ---------------------------- Phase 1: Local sort --------------------------- #
    delay, cost = simulate_sorting(chunk, relative_throughput, cost_per_time, scale_time)
    percent_initial = (len(chunk) / data_size) * 100
    percent_in_memory_initial = (len(chunk) / memory_size) * 100 if memory_size > 0 else 0
    
    print(f"Node {id:<2} ({relative_throughput:>2}x): Initial: {len(chunk)} / {memory_size} elements | "
          f"{percent_initial:.2f}% of dataset | {percent_in_memory_initial:.2f}% of memory capacity | "
          f"Time: {delay} us | Cost: $ {cost} | Sample: {str(chunk[:5])[:-1]}...{str(chunk[-5:])[1:]}")
    
    # ------------------------ Phase 2: Sample collection ------------------------ #
    L_i = (num_nodes - 1) * relative_throughput
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
        
        delay, cost = simulate_sorting(all_samples, relative_throughput, cost_per_time, scale_time) # Coordinator uses its own billing rate
        print(f"\n{'':<10}Coordinator: Collected {len(all_samples):<8} samples | Time: {delay} us | "
              f"Cost: $ {cost} | Sample: {str(all_samples[:5])[:-1]}...{str(all_samples[-5:])[1:]}\n")
        
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
    delay, cost = simulate_sorting(chunk, relative_throughput, cost_per_time, scale_time)
    percent_final = (len(chunk) / data_size) * 100
    percent_in_memory_final = (len(chunk) / memory_size) * 100 if memory_size > 0 else 0
    
    print(f"Node {id:<2} ({relative_throughput:>2}x): Final: {len(chunk)} / {memory_size} elements | "
          f"{percent_final:.2f}% of dataset | {percent_in_memory_final:.2f}% of memory capacity | "
          f"Time: {delay} us | Cost: $ {cost} | Sample: {str(chunk[:5])[:-1]}...{str(chunk[-5:])[1:]}")

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
    
    scale_time = 10**6 # Scale time basis in microseconds (10^6 or 1 millionth of a second)
    num_nodes = 4
    relative_throughputs = [] 
    memory_sizes = []
    costs_per_time = []
    
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
        memory_sizes.append(node['memory'])
        costs_per_time.append(node['billing'])
        relative_throughputs.append(node['throughput'])
    total_relative_throughput = sum(relative_throughputs)
    
    # Create queues for inter-process communication
    queues = [Queue() for _ in range(num_nodes)]
    
    # Generate data for the nodes
    data = generate_data(n=data_size,skew_type='uniform')
    
    print(f"\nNumber of Nodes: {num_nodes}")
    print(f"Total Data Size: {data_size} elements")
    print(f"Data Sample: {data[:5]}...{data[-5:]}\n")
    
    print('------------------------------ Sequential Machine ------------------------------')
    
    temp = data.copy()
    time_start = time.perf_counter()
    temp.sort()
    time_end = time.perf_counter()
    total_time = time_end - time_start
    base_throughput = (total_time * scale_time) / len(temp)
    print(f'\nTime: {total_time*scale_time:.6f} us')
    print(f'Base throughput: {base_throughput:.6f} us per element\n')
    
    
    print(f"------------------------------ Regular Sampling ------------------------------")
    
    time_start = time.perf_counter()
    
    chunks = []
    last_idx = 0
    for i in range(num_nodes):
        partition_size = (data_size * relative_throughputs[i]) // total_relative_throughput
        partition = data[last_idx:last_idx + partition_size]
        last_idx += partition_size
        chunks.append(partition)

    time_end = time.perf_counter()
    total_time = time_end - time_start
    
    print(f"Partition time: {total_time*scale_time:.3f} us")
    print("-" * 30)
    
    # Start processes for each node
    processes = []
    for i in range(num_nodes):
        p = Process(
            target=node_psrs,
            args=(i, relative_throughputs[i], memory_sizes[i], costs_per_time[i], chunks[i], queues, num_nodes, data_size, scale_time)
        )
        processes.append(p)
        p.start()    
    for p in processes:
        p.join()
        
        
    print('-------------------------------- LP Guided -----------------------------------')
    
    print('---- Optimize Solver')
    time_start = time.perf_counter()
    result = optimize_proportion(num_nodes, data_size, relative_throughputs, memory_sizes, costs_per_time, base_throughput)
    time_end = time.perf_counter()
    total_time = time_end - time_start
    
    print(f"Solver time: {total_time*scale_time:.6f} us")
    print(f"Projected makespan: {result['makespan']:.6f} us")
    print(f"Projected cost: $ {result['total_cost']:.6f}")
    
    print("-" * 30)
    print('---- Smart Partitioning')
    
    chunks, partition_sizes = smart_partition(data, result['weights'])
    
    print("-" * 30)
    
    # TODO: distribute the chunks to the nodes and sort and return a min-heap 
    
    for id in range(num_nodes):
        
        percent_initial = (len(chunks[id]) / data_size) * 100
        percent_in_memory_initial = (len(chunks[id]) / memory_sizes[id]) * 100 if memory_sizes[id] > 0 else 0
        
        print(f"Node {id:<2} ({relative_throughputs[id]:>2}x): Initial: {len(chunks[id])} / {memory_sizes[id]} elements | "
                f"{percent_initial:.2f}% of dataset | {percent_in_memory_initial:.2f}% of memory capacity | ")
                # f"Time: {delay} us | Cost: $ {cost} | Sample: {str(chunk[:5])[:-1]}...{str(chunk[-5:])[1:]}")
    