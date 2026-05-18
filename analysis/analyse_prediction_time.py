import re
import numpy as np


logfile = "/path/to/logfile.out" 

# Pattern to capture the floating-point number before 's/it'
pattern = re.compile(r"([0-9]+\.[0-9]+)s/it")

with open(logfile, "r") as f:
    times = [float(match) for line in f for match in pattern.findall(line)]

avg_time = np.mean(times)
min_time = min(times)
max_time = max(times)
avg_time = max_time
print(f"Min time: {min_time:.2f}s")
print(f"Max time: {max_time:.2f}s")
print(f"Avg time: {avg_time:.2f}s")

NUM_FILES = 500000
NUM_GPUS = 4

if times:
    total_seconds = avg_time * NUM_FILES

    files_per_gpu = NUM_FILES / NUM_GPUS
    wall_seconds = avg_time * files_per_gpu
    
    days, remainder = divmod(total_seconds, 3600*24)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nAverage time per file: {avg_time:.2f} seconds")
    print(f"Expected duration for {NUM_FILES} files: "
          f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    days, remainder = divmod(wall_seconds, 3600*24)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nExpected time with {NUM_GPUS} gpus:")
    print(f"Average time per file: {avg_time:.2f} seconds")
    print(f"Expected duration: "
          f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")
else:
    print("No matching times found in the log.")
