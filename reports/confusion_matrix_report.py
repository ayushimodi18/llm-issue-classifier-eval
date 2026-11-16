import pandas as pd
import sys

from sklearn.metrics import confusion_matrix

if len(sys.argv) < 2:
    print("Usage:")
    print("python confusion_matrix_report.py <csv_file>")
    exit()

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

labels = [
    "bug",
    "enhancement",
    "question",
    "documentation",
    "security",
    "other"
]

cm = confusion_matrix(
    df["ground_truth"],
    df["predicted_label"],
    labels=labels
)

print("\nConfusion Matrix\n")

print("Labels:")
print(labels)

print("\nMatrix:")
print(cm)
