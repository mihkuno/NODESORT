import time
import random
from multiprocessing import Queue, Process

def _delay(ms):
    time.sleep(ms / 1000.0)
        
def _simulate_processing_delay(size, speed):
    delay = size / speed
    _delay(delay)
    
def _quicksort(arr):
    if len(arr) <= 1:
        return arr
            
    stack = [(0, len(arr) - 1)]
    arr = arr.copy()  # Work on a copy to avoid modifying original
    while stack:
        low, high = stack.pop()
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        pivot_index = i + 1
        
        if pivot_index - 1 > low:
            stack.append((low, pivot_index - 1))
        if pivot_index + 1 < high:
            stack.append((pivot_index + 1, high))
    return arr


def node_psrs(id, speed, data, queues, num_nodes):
    
    
    # ---------------------------- Phase 1: Local sort --------------------------- #
    
    sorted_data = _quicksort(data)
    print(f"Node {id} completed local sort")

    
    # ------------------------ Phase 2: Sample collection ------------------------ #
    
    
    samples = []
    if num_nodes > 1:
        sample_size = num_nodes - 1
        if len(sorted_data) >= sample_size:
            step = max(1, len(sorted_data) // num_nodes)
            for i in range(1, num_nodes):
                sample_index = min(i * step, len(sorted_data) - 1)
                samples.append(sorted_data[sample_index])
    
    queues[0].put({
        'from': id,
        'type': 'samples',
        'payload': samples
    })

    # Coordinator collects samples and selects pivots
    if id == 0:
        all_samples = []
        for _ in range(num_nodes):  # Including coordinator's own samples
            msg = queues[0].get()
            all_samples.extend(msg['payload'])
        all_samples.sort()
        
        # Select global pivots
        pivots = []
        for i in range(1, num_nodes):
            pivot_index = i * len(all_samples) // num_nodes
            if pivot_index < len(all_samples):  # Ensure index is valid
                pivots.append(all_samples[pivot_index])
        
        # Broadcast pivots to all nodes including self
        for node_id in range(num_nodes):
            queues[node_id].put({
                'from': 0,
                'type': 'pivots',
                'payload': pivots
            })

    
    # ------------------------ Phase 3: Redistribute data ------------------------ #


    # All nodes receive pivots
    msg = queues[id].get()
    pivots = msg['payload']
    
    partitions = [[] for _ in range(num_nodes)]  # Fixed size
    for num in sorted_data:
        nid = 0
        while nid < len(pivots) and num > pivots[nid]:
            nid += 1
        if nid < num_nodes:  # Ensure destination is valid
            partitions[nid].append(num)
            
    # Send partitions to respective nodes
    for nid in range(num_nodes):
        if nid == id:
            continue # Skip self
        queues[nid].put({
            'from': id,
            'type': 'partition',
            'payload': partitions[nid]
        })
            
    # Keep local partition
    final_data = partitions[id]
        
    # Receive partitions from other nodes
    for nid in range(num_nodes):
        if nid == id:
            continue  # Skip self
        msg = queues[id].get()
        final_data.extend(msg['payload'])
    
    
    # ---------------------------- Phase 4: Final sort --------------------------- #
    
    
    final_sorted = _quicksort(final_data)
    
    # Send final sorted chunk to coordinator
    if id != 0:
        queues[0].put({
            'from': id,
            'type': 'final',
            'payload': final_sorted
        })
    else:
        # Coordinator collects and merges final results in order of node id
        results_by_id = {0: list(final_sorted)}  # Coordinator's own data
        for _ in range(num_nodes - 1):
            msg = queues[0].get()
            if msg['type'] == 'final':
                results_by_id[msg['from']] = msg['payload']
        
        # Merge results in order of node id
        final_result = []
        for nid in range(num_nodes):
            final_result.extend(results_by_id[nid])
        
        print("\nFinal sorted result (first 20):", final_result[:20])
        print("Final sorted result (last 20):", final_result[-20:])
        
        
        

# ------------------------- main configuration setup ------------------------- #

if __name__ == '__main__':
    # Configuration
    num_nodes = 4
    data_size = 10000
    data = [random.randint(0, 10000) for _ in range(data_size)]
    
    # Node setup - coordinator is node 0
    speeds = [40, 10, 20, 30]  # Coordinator is fastest
    queues = [Queue() for _ in range(num_nodes)]
    
    # Data distribution proportional to speed
    total_speed = sum(speeds)
    partitions = []
    last_idx = 0
    for i in range(num_nodes):
        if i == num_nodes - 1:
            partition = data[last_idx:]
        else:
            partition_size = int(len(data) * speeds[i] / total_speed)
            partition = data[last_idx:last_idx + partition_size]
            last_idx += partition_size
        partitions.append(partition)
    
    # Start processes
    processes = []
    for i in range(num_nodes):
        p = Process(
            target=node_psrs,
            args=(i, speeds[i], partitions[i], queues, num_nodes)
        )
        processes.append(p)
        p.start()
    
    # Wait for completion
    for p in processes:
        p.join()