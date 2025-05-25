# class Node:

#     def __init__(self, name, speed, memory, latency, bandwidth, billing):
#         self.name = name
#         self.speed = speed             # items per millisecond (processing speed)
#         self.memory = memory           # max number of items this node can store
#         self.latency = latency         # fixed network latency in milliseconds
#         self.bandwidth = bandwidth     # items per millisecond (network throughput)
#         self.billing = billing         # cost per millisecond
#         self.total_billing = 0.0       # accumulated billing based on usage

#     def _delay(self, ms):
#         time.sleep(ms / 1000.0)            # time.sleep still expects seconds
#         self.total_billing += ms * self.billing  # billing is now per millisecond

#     def _simulate_network_delay(self, size):
#         delay = self.latency + (size / self.bandwidth)  # in ms
#         self._delay(delay)

#     def _simulate_processing_delay(self, size):
#         delay = size / self.speed  # in ms
#         self._delay(delay)


from ortools.linear_solver import pywraplp

import time
import random
from multiprocessing import Queue, Process


def _delay(ms):
    time.sleep(ms / 1000.0)

    
def _simulate_processing_delay(size, speed):
    delay = size / speed
    _delay(delay)

    
def _quicksort(unsorted_data):
    def _sort(unsorted_data):
        if len(unsorted_data) <= 1:
            return unsorted_data
        pivot = unsorted_data[len(unsorted_data) // 2]
        left = [x for x in unsorted_data if x < pivot]
        middle = [x for x in unsorted_data if x == pivot]
        right = [x for x in unsorted_data if x > pivot]
        return _sort(left) + middle + _sort(right)
    return _sort(unsorted_data)


def _pivot_sampling(sorted_data, num_nodes):
    if num_nodes <= 1:
        return []  # No pivots needed for single node

    sample_size = num_nodes - 1
    if len(sorted_data) < sample_size:
        return []  # Not enough data to sample

    step = max(1, len(sorted_data) // num_nodes)  # Ensure step >= 1
    pivots = []

    # Take regular samples from the sorted data
    for i in range(1, num_nodes):
        sample_index = min(i * step, len(sorted_data) - 1)  # Avoid index overflow
        pivots.append(sorted_data[sample_index])

    return pivots


def node_psrs(id, speed, data, queues):
    num_nodes = len(queues)

    # Phase 1: Local sort (all nodes including coordinator)
    sorted_data = _quicksort(data)
    _simulate_processing_delay(len(sorted_data), speed)    
    print(f"Node {id} completed local sort")

    # Phase 2: Sample collection (all nodes send samples to coordinator)
    samples = _pivot_sampling(sorted_data, num_nodes)
    
    queues[0].put({
        'from': id,
        'type': 'samples',
        'payload': samples
    })
    
    print(f"Node {id} sent samples to coordinator: {samples}")
    
    # ------------- if coordinator, collect all pivots and sort them ------------- #
    if id == 0:
        pivots = []
        samples_received = 0
        
        # Wait for samples from ALL nodes (including self)
        while samples_received < num_nodes:
            msg = queues[0].get()
            if msg['type'] == 'samples':
                pivots.extend(msg['payload'])  # Flatten immediately
                samples_received += 1
                print(f"Coordinator {id} received samples from Node {msg['from']} ({samples_received}/{num_nodes} collected)")
        
        # Sort all collected samples
        pivots.sort()
        print(f"Coordinator {id} collected all pivots: {pivots}")
        
        # Select global pivots (num_nodes-1 values)
        global_pivots = []
        for i in range(1, num_nodes):
            pivot_index = i * len(pivots) // num_nodes
            global_pivots.append(pivots[pivot_index])
        
        print(f"Selected global pivots: {global_pivots}")
        
        # Broadcast pivots to ALL nodes (including self)
        for node_id in range(num_nodes):
            queues[node_id].put({
                'from': id,
                'type': 'pivots',
                'payload': global_pivots
            })
            print(f"Sent pivots to Node {node_id}")
            
            
    # --------------------- wait for pivots to be distributed -------------------- #




# ---------------------- setup: distribute data to nodes --------------------- #


# create the dataset
data = [random.randint(0, 1000) for _ in range(10000)]

# create node configs
node_configs = [
    {'id': 1, 'speed': 10},
    {'id': 2, 'speed': 20},
    {'id': 3, 'speed': 30}    
]

# set fastest node as coordinator (id: 0)
main_node = max(node_configs, key=lambda n: n['speed'])
main_node['id'] = 0

node_queues = {
    node['id']: Queue() for node in node_configs
}

# calculate the total speed of all nodes
total_speed = sum(node['speed'] for node in node_configs)

last_index = 0
for i, node in enumerate(node_configs):
    # calculate the proportion of data for each node based on its speed
    proportion = node['speed'] / total_speed
    
    # last, takes the remaining data
    if i == len(node_configs) - 1: 
        partition = len(data) - last_index
    
    # others, take a proportion of the data
    else:
        partition = int(proportion * len(data))
    
    # assign data partition to the node
    node['data'] = data[last_index:last_index + partition]
    last_index += partition
    
    # assign queues reference (for inter-node communication)
    node['queues'] = node_queues
    
# final node configs
print("Node configurations with data partitions:")
for node in node_configs:
    print(
        f"Node {node['id']}: "
        f"Speed={node['speed']}, "
        f"Queues={len(node['queues'])}, "
        f"Data={len(node['data'])}"
    )

# ---------------------- setup: create node processes ----------------------- #

node_processes = []
for node in node_configs:
    process = Process(
        target=node_psrs,
        args=(node['id'], node['speed'], node['data'], node['queues'])
    )
    node_processes.append(process)
    process.start()
    print(f"Node {node['id']} started...")

# wait for all nodes to finish
for process in node_processes:
    process.join()
    print(f"Node {process.pid} finished processing.")

