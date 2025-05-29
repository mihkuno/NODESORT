import time
import random
from multiprocessing import Queue, Process
        

def _simulate_sorting(chunk, throughput, multiplier=1000):
    # Sort the chunk and get the time taken
    time_start = time.perf_counter()
    chunk.sort()
    time_end = time.perf_counter()
    # Simulate delay based on time taken
    actual_time = time_end - time_start
    delay = actual_time * (1 / throughput)
    delay *= multiplier  # Scale by multiplier
    time.sleep(delay) # Args are in seconds
    return round(delay * 1000, 3) # Convert to milliseconds


def node_psrs(id, throughput, chunk, queues, num_nodes):
    
    # ---------------------------- Phase 1: Local sort --------------------------- #
    
    delay = _simulate_sorting(chunk, throughput)
    print(f"Node {id} ({throughput}x): Initial {len(chunk)} in {delay}ms")
        
    # ------------------------ Phase 2: Sample collection ------------------------ #
    
    samples = []
    if num_nodes > 1:
        sample_size = num_nodes - 1
        if len(chunk) >= sample_size:
            step = max(1, len(chunk) // num_nodes)
            for i in range(1, num_nodes):
                sample_index = min(i * step, len(chunk) - 1)
                samples.append(chunk[sample_index])
    
    queues[0].put({ 'from': id, 'type': 'samples', 'payload': samples })

    # Coordinator collects samples and selects pivots
    if id == 0:
        samples = []
        for _ in range(num_nodes):  # Including coordinator's own samples
            msg = queues[0].get()
            samples.extend(msg['payload'])
        delay = _simulate_sorting(samples, throughput)        
        print(f"\nCoordinator samples: {len(samples)} in {delay}ms\n")
        
        # Select global pivots
        pivots = []
        for i in range(1, num_nodes):
            pivot_index = i * len(samples) // num_nodes
            if pivot_index < len(samples):  # Ensure index is valid
                pivots.append(samples[pivot_index])
        
        # Broadcast pivots to all nodes including self
        for node_id in range(num_nodes):
            
            queues[node_id].put({ 'from': 0, 'type': 'pivots', 'payload': pivots })

    
    # ------------------------ Phase 3: Redistribute data ------------------------ #


    # All nodes receive pivots
    msg = queues[id].get()
    pivots = msg['payload']
    
    partitions = [[] for _ in range(num_nodes)]  # Fixed size
    for num in data:
        nid = 0
        while nid < len(pivots) and num > pivots[nid]:
            nid += 1
        if nid < num_nodes:  # Ensure destination is valid
            partitions[nid].append(num)
            
    # Send partitions to respective nodes
    for nid in range(num_nodes):
        if nid == id:
            continue # Skip self
        
        queues[nid].put({ 'from': id, 'type': 'partition', 'payload': partitions[nid] })
            
    # Keep local partition
    chunk = partitions[id]
        
    # Receive partitions from other nodes
    for nid in range(num_nodes):
        if nid == id:
            continue  # Skip self
        msg = queues[id].get()
        chunk.extend(msg['payload'])
    
    
    # ---------------------------- Phase 4: Final sort --------------------------- #
    
    delay = _simulate_sorting(chunk, throughput)    
    print(f"Node {id} ({throughput}x): Final {len(chunk)} in {delay}ms")
    
    # Send final sorted chunk to coordinator
    if id != 0:
        queues[0].put({ 'from': id, 'type': 'final', 'payload': chunk })
    else:
        # Coordinator collects and merges final results in order of node id
        results_by_id = {0: list(chunk)}  # Coordinator's own data
        for _ in range(num_nodes - 1):
            msg = queues[0].get()
            if msg['type'] == 'final':
                results_by_id[msg['from']] = msg['payload']
        
        # Merge results in order of node id
        final_result = []
        for nid in range(num_nodes):
            final_result.extend(results_by_id[nid])
        
        print()
        print("Total length:", len(final_result))
        print("(first 20):", final_result[:20])
        print("(last 20):", final_result[-20:])
        print()
        
        
        

# ------------------------- main configuration setup ------------------------- #

if __name__ == '__main__':
    # Configuration
    num_nodes = 4
    data_size = 10000
    data = [random.randint(0, 10000) for _ in range(data_size)]
    
    # Node setup - coordinator is node 0
    throughputs = [4, 1, 2, 3]  # Coordinator is fastest (bigger is faster)
    queues = [Queue() for _ in range(num_nodes)]
    
    # Data distribution proportional to throughput
    total_throughput = sum(throughputs)
    partitions = []
    last_idx = 0
    for i in range(num_nodes):
        if i == num_nodes - 1:
            partition = data[last_idx:]
        else:
            partition_size = int(len(data) * throughputs[i] / total_throughput)
            partition = data[last_idx:last_idx + partition_size]
            last_idx += partition_size
        partitions.append(partition)
    
    # Start processes
    processes = []
    for i in range(num_nodes):
        p = Process(
            target=node_psrs,
            args=(i, throughputs[i], partitions[i], queues, num_nodes)
        )
        processes.append(p)
        p.start()
    
    # Wait for completion
    for p in processes:
        p.join()