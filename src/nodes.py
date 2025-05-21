import time
import random

class ArtificialNode:
    def __init__(self, cpu_speed_ops_per_sec, memory_limit_elements):
        self.cpu_speed = cpu_speed_ops_per_sec
        self.memory_limit = memory_limit_elements

    def run_sort(self, data):
        if len(data) > self.memory_limit:
            raise MemoryError(f"Memory limit exceeded: {len(data)} > {self.memory_limit}")

        print(f"Sorting {len(data)} items with CPU speed {self.cpu_speed} ops/sec...")

        # Simulated bubble sort with CPU delay
        for i in range(len(data)):
            for j in range(0, len(data) - i - 1):
                # Simulate operation cost
                time.sleep(1 / self.cpu_speed)  # 1 operation
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data

# Example usage
node = ArtificialNode(cpu_speed_ops_per_sec=100, memory_limit_elements=20)
array = random.sample(range(100), 10)  # random array of size 10

print("Before:", array)
sorted_array = node.run_sort(array.copy())
print("After: ", sorted_array)
