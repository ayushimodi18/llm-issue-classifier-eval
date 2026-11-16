import pandas as pd
import sys

from sklearn.metrics import classification_report

if len(sys.argv) < 2:
    print("Usage:")
    print("python classification_report.py <csv_file>")
    exit()

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

print(
    classification_report(
        df["ground_truth"],
        df["predicted_label"],
        zero_division=0
    )
)