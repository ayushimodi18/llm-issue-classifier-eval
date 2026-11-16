import pandas as pd

import sys

if len(sys.argv) < 2:
    print("Usage:")
    print("python cost_report.py <csv_file>")
    exit()

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

total_requests = len(df)

# Approximate cost per request
# Replace with actual pricing later if needed
cost_per_request = 0.00005

total_cost = total_requests * cost_per_request

print("\nCost Report")
print("-----------------------")
print("Total Requests:", total_requests)
print("Estimated Cost Per Request: $", cost_per_request)
print("Estimated Total Cost: $", round(total_cost, 4))