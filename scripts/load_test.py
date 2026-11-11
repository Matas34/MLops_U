import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

URL = "http://localhost:8080/predict"
PAYLOAD = {"x": 5.0}
NUM_REQUESTS = 100

def make_request(_):
    start = time.perf_counter()
    response = requests.post(URL, json=PAYLOAD)
    end = time.perf_counter()
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
        
    # Return latency in milliseconds
    return (end - start) * 1000

# Warmup request
requests.post(URL, json=PAYLOAD)

start_total = time.perf_counter()

with ThreadPoolExecutor(max_workers=10) as executor:
    latencies = list(executor.map(make_request, range(NUM_REQUESTS)))

end_total = time.perf_counter()

latencies = [l for l in latencies if l is not None]

if not latencies:
    print("All requests failed.")
else:
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    throughput = NUM_REQUESTS / (end_total - start_total)
    
    print("--- Results ---")
    print(f"Total Successful Requests: {len(latencies)}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"p95 Latency: {p95_latency:.2f} ms")
    print(f"Max Latency: {max(latencies):.2f} ms")
    print(f"Throughput: {throughput:.2f} requests/sec")