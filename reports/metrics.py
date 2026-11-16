import pandas as pd
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

if len(sys.argv) < 2:
    print("Usage:")
    print("python metrics.py <csv_file>")
    exit()

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

y_true = df["ground_truth"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\nEvaluation Results")
print("-" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")