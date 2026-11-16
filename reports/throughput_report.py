import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage:")
    print("python throughput_report.py <csv_file>")
    exit()

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

total_requests = len(df)

average_latency = df["latency_seconds"].mean()

throughput = total_requests / df["latency_seconds"].sum()

print("\nThroughput Report")
print("------------------------")
print("Total Requests:", total_requests)
print("Average Latency:", round(average_latency, 2), "seconds")
print("Requests Per Second:", round(throughput, 2))