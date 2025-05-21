import time
import random
import threading

class ArtificialNode:
    def __init__(self, name, cpu_speed_ops_per_sec, memory_limit_elements):
        self.name = name
        self.cpu_speed = cpu_speed_ops_per_sec
        self.memory_limit = memory_limit_elements

    def run_sort(self, data):
        if len(data) > self.memory_limit:
            print(f"[{self.name}] Memory limit exceeded: {len(data)} > {self.memory_limit}")
            return

        print(f"[{self.name}] Sorting {len(data)} items with CPU speed {self.cpu_speed} ops/sec...")

        for i in range(len(data)):
            for j in range(0, len(data) - i - 1):
                time.sleep(1 / self.cpu_speed)
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        
        print(f"[{self.name}] Sorted result: {data}")

def run_node(name, cpu, memory, array):
    node = ArtificialNode(name, cpu, memory)
    node.run_sort(array)

# Example arrays and configurations
arrays = [
    random.sample(range(100), 10),
    random.sample(range(100), 15),
    random.sample(range(100), 8)
]

configs = [
    ("Node-A", 100, 20),
    ("Node-B", 50, 15),
    ("Node-C", 200, 10)
]

# Launch nodes in threads
threads = []
for i in range(3):
    t = threading.Thread(target=run_node, args=(configs[i][0], configs[i][1], configs[i][2], arrays[i]))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

print("All nodes finished.")
