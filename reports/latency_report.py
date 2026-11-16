import pandas as pd

import sys

if len(sys.argv) < 2:
    print("Usage:")
    print("python latency_report.py <csv_file>")
    exit()

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)
p50 = df["latency_seconds"].quantile(0.50)
p95 = df["latency_seconds"].quantile(0.95)

print("\nLatency Report")
print("---------------------------")
print("P50 Latency:", round(p50, 2), "seconds")
print("P95 Latency:", round(p95, 2), "seconds")
print("Average Latency:", round(df["latency_seconds"].mean(), 2), "seconds")